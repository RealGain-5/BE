"""
model_spectrogram.py
====================
SpectrogramCNN: 주파수 도메인 궤도 분석 2D CNN.

입력 채널 설계 (make_spectrogram_4ch() 출력과 대응)
──────────────────────────────────────────────────
  Ch0: log(1 + |STFT_X|²)     — X축 전력 스펙트럼
  Ch1: log(1 + |STFT_Y|²)     — Y축 전력 스펙트럼
  Ch2: log(1 + |Re(Gxy)|)     — 교차 스펙트럼 실수부 (동위상 X-Y 성분)
  Ch3: log(1 + |Im(Gxy)|)     — 교차 스펙트럼 허수부 (위상차 = 와류 방향)

왜 4채널인가?
─────────────
1D 신호에서 생성된 궤도 플롯은 방향 정보를 잃을 수 있지만,
교차 스펙트럼의 허수부는 X-Y 위상차를 명시적으로 인코딩합니다.
  - Im(Gxy) > 0 at 1× rpm → X가 Y보다 90° 앞섬 → 순방향 와류 (불평형, 오일 훨)
  - Im(Gxy) < 0 at 1× rpm → X가 Y보다 90° 뒤짐 → 역방향 와류 (러빙)
이는 궤도 플롯의 RandomHorizontalFlip 문제 없이 와류 방향을 안전하게 인코딩합니다.

또한 고정 스케일 입력 사용으로 절대 진폭이 스펙트럼 강도에 보존됩니다.

아키텍처
────────
ResNet18 (수정된 첫 Conv2d, 4채널 입력)
  → AdaptiveAvgPool2d(1)  [임의 F×T 크기 입력 허용]
  → Linear(512 → FEATURE_DIM=256)
  → [선택] 분류 헤드 Linear(256 → num_classes)

파라미터: ~11M (ResNet18 기반)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

FEATURE_DIM: int = 256


class SpectrogramCNN(nn.Module):
    """
    4채널 스펙트로그램 기반 ResNet18 분류/특징 추출 모델.

    get_features(x) : (B, 4, F, T) → (B, FEATURE_DIM)   특징 벡터
    forward(x)      : num_classes 설정 시 logits 반환,
                      미설정 시 특징 벡터 반환 (FusionNet 연결용)

    Args:
        in_channels : 입력 채널 수 (기본 4: Sx, Sy, Re(Gxy), Im(Gxy))
        num_classes : None이면 특징 추출기, 정수이면 분류 헤드 포함
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels

        # ── ResNet18 백본 (사전학습 없음, 궤도 데이터 도메인 특화) ──
        backbone = tv_models.resnet18(weights=None)

        # 첫 Conv2d를 in_channels 입력으로 교체
        # (ImageNet 3채널 → 스펙트로그램 4채널)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # 기존 3채널 가중치를 평균하여 초기화 (학습 안정성 향상)
        with torch.no_grad():
            w = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            backbone.conv1.weight.copy_(w.expand(-1, in_channels, -1, -1))

        # 마지막 FC 레이어 제거 → (B, 512, 1, 1) AdaptiveAvgPool 출력만 사용
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # 특징 투영 레이어
        self.proj = nn.Sequential(
            nn.Flatten(),                       # (B, 512)
            nn.Linear(512, FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

        self._classifier: nn.Module | None = None
        if num_classes is not None:
            self._classifier = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(FEATURE_DIM, num_classes),
            )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, F, T) → (B, FEATURE_DIM)
        AdaptiveAvgPool2d(1) 덕분에 임의 F, T 크기 입력 허용.
        """
        h = self.backbone(x)   # (B, 512, 1, 1)
        return self.proj(h)    # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        num_classes가 설정된 경우 logits 반환.
        미설정(FusionNet 하위 모듈)이면 특징 벡터 반환.
        """
        feats = self.get_features(x)
        if self._classifier is not None:
            return self._classifier(feats)
        return feats
