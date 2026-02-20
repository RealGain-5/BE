"""
model_1d_cnn.py
================
OrbitCNN1D: Raw time-series 기반 1D CNN 분류 모델.

입력: (batch, 2, 40000) — X_mil, Y_mil raw signal (sec9, 40,000 samples each)
출력: (batch, num_classes) logits

아키텍처:
  Conv1d(2→32,   k=128, stride=8)  BN ReLU → (batch, 32,  5000)
  Conv1d(32→64,  k=32,  stride=4)  BN ReLU → (batch, 64,  1250)
  Conv1d(64→128, k=16,  stride=4)  BN ReLU → (batch, 128, 313)
  Conv1d(128→256,k=8,   stride=4)  BN ReLU → (batch, 256, 79)
  Conv1d(256→256,k=4,   stride=2)  BN ReLU → (batch, 256, 40)
  AdaptiveAvgPool1d(1)                      → (batch, 256, 1)
  Flatten → Dropout(0.5) → Linear(256→num_classes)

파라미터: ~3M (ResNet18 11M 대비 경량)
AdaptiveAvgPool1d 덕분에 입력 길이 변동에 강건.
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class OrbitCNN1D(nn.Module):
    """
    Raw orbit time-series (2ch × 40000) → num_classes logits.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(2,   32,  kernel_size=128, stride=8),   # → (32, 5000)
            _conv_bn_relu(32,  64,  kernel_size=32,  stride=4),   # → (64, 1250)
            _conv_bn_relu(64,  128, kernel_size=16,  stride=4),   # → (128, 313)
            _conv_bn_relu(128, 256, kernel_size=8,   stride=4),   # → (256, 79)
            _conv_bn_relu(256, 256, kernel_size=4,   stride=2),   # → (256, 40)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
