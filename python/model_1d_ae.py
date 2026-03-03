"""
model_1d_ae.py
==============
OrbitAE1D: 고정 스케일 원시 신호 기반 1D CNN AutoEncoder.

설계 목표
─────────
1. 고정 스케일 입력 → 절대 진폭 + 파형 형태를 동시에 인코딩
2. 재구성 오차(MSE)로 비지도 이상 점수 계산
3. Bottleneck 특징 벡터를 Stage-2 SVDD 초구 피팅에 직접 사용
4. 선택적 분류 헤드 (지도학습 미세조정 / 앙상블 참여용)

아키텍처 (in_channels=2, seq_len=40000 기준)
──────────────────────────────────────────
Encoder:
  (B, 2, 40000)
  → CBR(2→32,   k=128, s=8)  → (B, 32, 5000)   # 저주파 추세 포착
  → CBR(32→64,  k=32,  s=4)  → (B, 64, 1250)   # 회전 주기 패턴
  → CBR(64→128, k=16,  s=4)  → (B, 128, 313)   # 조화파 구조
  → CBR(128→256,k=8,   s=4)  → (B, 256, 79)    # 세부 파형
  → CBR(256→256,k=4,   s=2)  → (B, 256, 40)
  → AdaptiveAvgPool1d(1)      → (B, 256)
  → Linear(256→BOTTLENECK)    → (B, 256) ← bottleneck z

Decoder (mirror):
  z (B, 256)
  → Linear(256→256×40)        → reshape (B, 256, 40)
  → [Upsample×2  + CBR]       → (B, 256, 80)
  → [Upsample×4  + CBR]       → (B, 128, 320)
  → [Upsample×4  + CBR]       → (B, 64, 1280)
  → [Upsample×4  + CBR]       → (B, 32, 5120)
  → F.interpolate(size=40000) → (B, 32, 40000)
  → Conv1d(32→in_channels)    → (B, 2, 40000)

파라미터: ~4M (in_channels=2 기준)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BOTTLENECK_DIM: int = 256


# ─────────────────────────────────────────────
# 내부 블록 유틸
# ─────────────────────────────────────────────

def _cbr(in_ch: int, out_ch: int, k: int, s: int) -> nn.Sequential:
    """Conv1d + BatchNorm1d + ReLU (인코더용, stride > 1)."""
    return nn.Sequential(
        nn.Conv1d(
            in_ch, out_ch,
            kernel_size=k, stride=s, padding=k // 2, bias=False,
        ),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _cbr_same(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    """Conv1d + BatchNorm1d + ReLU (디코더용, stride=1, same padding)."""
    return nn.Sequential(
        nn.Conv1d(
            in_ch, out_ch,
            kernel_size=k, stride=1, padding=k // 2, bias=False,
        ),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────
# 인코더
# ─────────────────────────────────────────────

class _Encoder1D(nn.Module):
    """
    계층적 시간 압축 인코더.
    총 압축비: 8×4×4×4×2 = 1024 (40000 → ~40 프레임)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            _cbr(in_channels, 32,  128, 8),  # → (~5000)
            _cbr(32,          64,  32,  4),  # → (~1250)
            _cbr(64,          128, 16,  4),  # → (~313)
            _cbr(128,         256, 8,   4),  # → (~79)
            _cbr(256,         256, 4,   2),  # → (~40)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → (256, 1)
        self.proj = nn.Linear(256, BOTTLENECK_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)            # (B, 256, ~40)
        h = self.pool(h).squeeze(-1)  # (B, 256)
        return self.proj(h)           # (B, BOTTLENECK_DIM)


# ─────────────────────────────────────────────
# 디코더
# ─────────────────────────────────────────────

class _Decoder1D(nn.Module):
    """
    병렬 업샘플링 디코더.
    Upsample(nearest) + CBR 구조로 checkerboard 아티팩트 없이 재구성.
    최종 AdaptiveInterpolate로 정확한 출력 길이 보장.
    """

    def __init__(self, out_channels: int, out_len: int):
        super().__init__()
        self.out_len = out_len

        # Bottleneck → 시간 축 복원
        self.expand = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, 256 * 40),
            nn.ReLU(inplace=True),
        )

        # 업샘플링 단계 (×2, ×4, ×4, ×4)
        self.up1 = nn.Sequential(
            _cbr_same(256, 256, 3),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
        )  # → (~80)

        self.up2 = nn.Sequential(
            _cbr_same(256, 128, 3),
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
        )  # → (~320)

        self.up3 = nn.Sequential(
            _cbr_same(128, 64, 3),
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
        )  # → (~1280)

        self.up4 = nn.Sequential(
            _cbr_same(64, 32, 3),
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
        )  # → (~5120)

        # 최종 출력 레이어
        self.output_conv = nn.Conv1d(
            32, out_channels, kernel_size=7, padding=3, bias=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.expand(z)                    # (B, 256*40)
        h = h.view(h.size(0), 256, 40)        # (B, 256, 40)
        h = self.up1(h)                        # (B, 256, ~80)
        h = self.up2(h)                        # (B, 128, ~320)
        h = self.up3(h)                        # (B, 64, ~1280)
        h = self.up4(h)                        # (B, 32, ~5120)
        # 정확한 출력 길이 보장 (부동소수 반올림 오차 처리)
        h = F.interpolate(
            h, size=self.out_len, mode="linear", align_corners=False,
        )                                      # (B, 32, out_len)
        return self.output_conv(h)             # (B, out_channels, out_len)


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

class OrbitAE1D(nn.Module):
    """
    1D CNN AutoEncoder (재구성 + 선택적 분류).

    사용 패턴:
    ┌─────────────────────────────────────────────────────────────┐
    │ # 비지도 이상 탐지                                            │
    │ score = model.anomaly_score(x)   # (B,) MSE, 높을수록 이상  │
    │                                                              │
    │ # Stage-2 SVDD를 위한 특징 추출 (인코더 freeze 후 사용)       │
    │ z = model.encode(x)              # (B, 256)                 │
    │                                                              │
    │ # 지도학습 미세조정 (분류 헤드 활성화 시)                       │
    │ logits = model.classify(x)       # (B, num_classes)         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        in_channels : 2 = (X, Y), 3 = (X, Y, Z 축방향 포함)
        seq_len     : 신호 길이 (기본 40000 = 1초 @ 40 kHz)
        num_classes : None이면 분류 헤드 없음, 정수면 헤드 생성
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 40_000,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.encoder = _Encoder1D(in_channels)
        self.decoder = _Decoder1D(out_channels=in_channels, out_len=seq_len)

        self._classifier: nn.Module | None = None
        if num_classes is not None:
            self._classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(BOTTLENECK_DIM, num_classes),
            )

    # ── 기본 연산 ──────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) → (B, BOTTLENECK_DIM)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, BOTTLENECK_DIM) → (B, C, L)"""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_rec : (B, C, L)  재구성 신호
            z     : (B, 256)   bottleneck 특징
        """
        z = self.encode(x)
        return self.decode(z), z

    # ── 이상 탐지 ──────────────────────────────────

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample 재구성 MSE (낮을수록 정상).

        고정 스케일 입력을 사용하므로 절대 진폭이 오차에 반영됩니다.
        정상 신호: 낮은 재구성 오차 + 낮은 입력 진폭 → 이중 정상 신호
        """
        x_rec, _ = self.forward(x)
        return F.mse_loss(x_rec, x, reduction="none").mean(dim=(1, 2))

    # ── 분류 ──────────────────────────────────────

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """지도학습 분류 (num_classes 설정 시에만 사용 가능)."""
        if self._classifier is None:
            raise RuntimeError(
                "분류 헤드가 없습니다. OrbitAE1D(num_classes=N)으로 생성하세요."
            )
        z = self.encode(x)
        return self._classifier(z)

    # ── 학습 손실 ─────────────────────────────────

    @staticmethod
    def reconstruction_loss(
        x_rec: torch.Tensor,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """MSE 재구성 손실 (Stage-1a 학습용)."""
        return F.mse_loss(x_rec, x, reduction=reduction)
