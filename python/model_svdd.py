"""
model_svdd.py
=============
Deep SVDD (Support Vector Data Description) 이상 탐지 인코더.

- SVDDEncoder: OrbitCNN1D 백본 (5× Conv1d + BN + ReLU + AdaptiveAvgPool)
               + Linear(256→feature_dim), L2 정규화 없음
- 정상 데이터만으로 학습하는 단일 클래스 이상 탐지 방식.
- 이상 점수: ||z - c||²  (hypersphere 중심 c 로부터의 제곱 거리)

L2 정규화를 사용하지 않는 이유:
  - 모든 z, c가 단위 구 위에 있으면 ||z-c||² = 2(1-cos(z,c)) 로 단순화되어
    모델이 c 방향으로 모든 출력을 붕괴(collapse)시키는 degenerate solution을 찾아버린다.
  - 붕괴 방지는 백본 BatchNorm + gradient clipping(max_norm=1.0)으로 충분하다.
  - 원래 Deep SVDD 논문(Ruff et al., 2018)도 L2 정규화를 사용하지 않는다.

백본 구조는 model_1d_cnn.py의 OrbitCNN1D.features와 동일하므로
orbit_cnn1d.pth 체크포인트로 전이학습 초기화가 가능하다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch, out_ch, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class SVDDEncoder(nn.Module):
    """
    OrbitCNN1D 백본 기반 SVDD 인코더.

    입력: (batch, 2, 40000) — X_mil, Y_mil raw signal
    출력: (batch, feature_dim) — raw 피처 벡터 (정규화 없음)

    붕괴 방지: 백본 BatchNorm + gradient clipping(train_svdd.py에서 적용)
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        # OrbitCNN1D 백본과 동일한 구조 (전이학습 호환)
        self.features = nn.Sequential(
            _conv_bn_relu(2,   32,  kernel_size=128, stride=8),   # → (32, 5000)
            _conv_bn_relu(32,  64,  kernel_size=32,  stride=4),   # → (64, 1250)
            _conv_bn_relu(64,  128, kernel_size=16,  stride=4),   # → (128, 313)
            _conv_bn_relu(128, 256, kernel_size=8,   stride=4),   # → (256, 79)
            _conv_bn_relu(256, 256, kernel_size=4,   stride=2),   # → (256, 40)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)            # → (256, 1)
        self.proj = nn.Linear(256, feature_dim)        # 투영 레이어

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)           # (batch, 256)
        x = self.proj(x)           # (batch, feature_dim)
        return x


def compute_svdd_loss(features: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Compact SVDD 손실: 중심 c로부터의 평균 제곱 거리.
    L = (1/N) × Σ ||f_θ(xᵢ) - c||²
    """
    dists = torch.sum((features - center) ** 2, dim=1)  # (batch,)
    return dists.mean()


def compute_svdd_distances(features: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    샘플별 제곱 거리: ||z - c||²
    반환: (batch,) float32
    """
    return torch.sum((features - center) ** 2, dim=1)
