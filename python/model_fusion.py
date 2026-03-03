"""
model_fusion.py
===============
OrbitFusionNet: 3-스트림 멀티모달 융합 네트워크.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스트림 구성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stream 1 [시간 도메인]   OrbitAE1D.encode (X, Y)           → 256-dim
  - 절대 진폭 보존 (고정 스케일)
  - 파형 형태 (조화파 구조, 충격 패턴)
  - AE 재구성 오차 → 비지도 이상 점수 병행 생성

Stream 2 [주파수 도메인] SpectrogramCNN.get_features         → 256-dim
  (Sx, Sy, Re(Gxy), Im(Gxy))
  - 주파수별 에너지 분포 (고조파, 서브하모닉 탐지)
  - 교차 스펙트럼 위상차 → 와류 방향 인코딩
  - RandomHorizontalFlip 없이 와류 방향 정보 안전하게 처리

Stream 3 [다채널]         OrbitAE1D.encode (X, Y, Z 축방향)  → 256-dim
  - 축방향 진동 포함 → 정렬 불량 조기 탐지
  - Z=0 더미 채널로 축방향 센서 미연결 RCP 처리 가능

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
융합 전략: Late Fusion (특징 수준 연결)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
concat([f1, f2, f3]) → (768-dim)
  → LayerNorm          (스트림 간 스케일 정규화)
  → Linear(768→512) + GELU + Dropout(0.4)
  → Linear(512→128) + GELU + Dropout(0.3)
  → Linear(128→num_classes)

LayerNorm 선택 이유:
  BatchNorm은 배치 통계에 의존하여 소규모 배치에서 불안정.
  LayerNorm은 샘플별 정규화 → 배치 크기 무관 안정성.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
학습 전략 (Two-Stage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1a — 표현 학습 (비지도, 정상 데이터만 사용):
  Stream 1 (OrbitAE1D): L_rec = MSE(x_rec, x)
  목표: 정상 신호의 압축 표현 학습, 재구성 오차 최소화

Stage 1b — 분류 미세조정 (지도, 정상+합성 데이터):
  Stream 2 (SpectrogramCNN): CrossEntropy (주파수 분류기)
  Stream 3 (OrbitAE1D 3ch) : CrossEntropy (다채널 분류기)
  목표: 각 스트림이 독립적으로 판별적 특징 학습

Stage 2 — 융합 학습:
  Option A: Stream 1/2/3 동결 → Fusion MLP만 학습
  Option B: 스트림 매우 낮은 LR(1e-5) + Fusion 높은 LR(1e-3) end-to-end
  목표: 스트림 간 상보적 정보를 결합한 최종 분류기

Stage 3 (선택) — One-Class:
  모든 스트림 동결 → 정상 데이터 특징에 SVDD 초구 피팅
  이상 점수 = max(||z_i - c||², AE_reconstruction_error)
"""

import torch
import torch.nn as nn

from model_1d_ae import OrbitAE1D, BOTTLENECK_DIM
from model_spectrogram import SpectrogramCNN, FEATURE_DIM

# 스트림 차원 일치 확인
assert BOTTLENECK_DIM == FEATURE_DIM, (
    f"스트림 특징 차원 불일치: AE={BOTTLENECK_DIM}, Spec={FEATURE_DIM}. "
    "두 상수를 동일한 값으로 맞추세요."
)
_STREAM_DIM: int = BOTTLENECK_DIM  # = 256


class OrbitFusionNet(nn.Module):
    """
    3-스트림 융합 분류 모델.

    Args:
        num_classes   : 출력 클래스 수
        use_axial     : True이면 Stream 3 (X,Y,Z 3채널) 활성화
        spec_channels : 스펙트로그램 입력 채널 수 (기본 4)
        seq_len       : 시계열 신호 길이 (기본 40000)
    """

    def __init__(
        self,
        num_classes: int,
        use_axial: bool = True,
        spec_channels: int = 4,
        seq_len: int = 40_000,
    ):
        super().__init__()
        self.use_axial = use_axial
        self.num_classes = num_classes

        # ── Stream 1: 시간 도메인 (X, Y) ───────────────
        self.stream_time = OrbitAE1D(
            in_channels=2, seq_len=seq_len, num_classes=None,
        )

        # ── Stream 2: 주파수 도메인 (4채널 스펙트로그램) ─
        self.stream_freq = SpectrogramCNN(
            in_channels=spec_channels, num_classes=None,
        )

        # ── Stream 3: 다채널 (X, Y, Z) ─────────────────
        if use_axial:
            self.stream_axial = OrbitAE1D(
                in_channels=3, seq_len=seq_len, num_classes=None,
            )

        # ── Fusion Head ─────────────────────────────────
        n_streams = 3 if use_axial else 2
        fused_dim = _STREAM_DIM * n_streams   # 768 (axial) or 512

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    # ── 순전파 ──────────────────────────────────────────

    def forward(
        self,
        x_time:  torch.Tensor,                  # (B, 2, L)
        x_spec:  torch.Tensor,                  # (B, 4, F, T)
        x_axial: torch.Tensor | None = None,    # (B, 3, L) or None
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, num_classes)
        """
        f_time = self.stream_time.encode(x_time)          # (B, 256)
        f_freq = self.stream_freq.get_features(x_spec)    # (B, 256)

        if self.use_axial and x_axial is not None:
            f_axial = self.stream_axial.encode(x_axial)   # (B, 256)
            fused = torch.cat([f_time, f_freq, f_axial], dim=1)  # (B, 768)
        else:
            fused = torch.cat([f_time, f_freq], dim=1)    # (B, 512)

        return self.fusion_head(fused)                     # (B, num_classes)

    # ── 분석/진단 인터페이스 ────────────────────────────

    def get_all_features(
        self,
        x_time:  torch.Tensor,
        x_spec:  torch.Tensor,
        x_axial: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        각 스트림의 256-dim 특징 벡터 반환 (t-SNE 시각화, SVDD 등에 사용).
        """
        feats: dict[str, torch.Tensor] = {
            "time": self.stream_time.encode(x_time),
            "freq": self.stream_freq.get_features(x_spec),
        }
        if self.use_axial and x_axial is not None:
            feats["axial"] = self.stream_axial.encode(x_axial)
        return feats

    @torch.no_grad()
    def anomaly_scores(
        self,
        x_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stream 1 AE 재구성 오차 기반 비지도 이상 점수 (B,).
        (분류 신뢰도와 병행하여 OOD 탐지에 활용 가능)
        """
        return self.stream_time.anomaly_score(x_time)

    # ── 학습 단계별 파라미터 그룹 ────────────────────────

    def get_stage1a_params(self):
        """Stage 1a: AE 재구성 학습 파라미터 (Stream 1 전체)."""
        return list(self.stream_time.parameters())

    def get_stage1b_params(self):
        """Stage 1b: 분류 미세조정 파라미터 (Stream 2, 3)."""
        params = list(self.stream_freq.parameters())
        if self.use_axial:
            params += list(self.stream_axial.parameters())
        return params

    def get_stage2_params(self):
        """Stage 2: Fusion Head만 학습."""
        return list(self.fusion_head.parameters())

    def freeze_streams(self):
        """모든 스트림 인코더 동결 (Stage 2 Fusion 학습 시 사용)."""
        for name in ("stream_time", "stream_freq"):
            for p in getattr(self, name).parameters():
                p.requires_grad_(False)
        if self.use_axial and hasattr(self, "stream_axial"):
            for p in self.stream_axial.parameters():
                p.requires_grad_(False)

    def unfreeze_streams(self, lr_scale: float = 0.01):
        """
        스트림 해동 (end-to-end fine-tuning 시 사용).
        lr_scale: Fusion Head 대비 낮은 LR 사용을 권장하므로
                  옵티마이저에서 별도 파라미터 그룹을 만들 때 이 비율 적용.
        """
        for name in ("stream_time", "stream_freq"):
            for p in getattr(self, name).parameters():
                p.requires_grad_(True)
        if self.use_axial and hasattr(self, "stream_axial"):
            for p in self.stream_axial.parameters():
                p.requires_grad_(True)
        return lr_scale  # 호출측에서 optimizer param_group lr 설정에 사용


# ─────────────────────────────────────────────────────────────────
# 학습 루프 템플릿 (스크립트 참고용 — 실제 학습은 train_fusion.py)
# ─────────────────────────────────────────────────────────────────

def build_stage2_optimizer(
    model: OrbitFusionNet,
    fusion_lr: float = 1e-3,
    stream_lr: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Stage 2 end-to-end 옵티마이저.
    Fusion Head에 높은 LR, 스트림 인코더에 낮은 LR을 적용하여
    사전학습 표현 파괴 없이 융합 레이어를 학습합니다.
    """
    stream_params = (
        list(model.stream_time.parameters())
        + list(model.stream_freq.parameters())
        + (list(model.stream_axial.parameters()) if model.use_axial else [])
    )
    return torch.optim.AdamW(
        [
            {"params": model.fusion_head.parameters(), "lr": fusion_lr},
            {"params": stream_params,                  "lr": stream_lr},
        ],
        weight_decay=1e-4,
    )
