import os
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    레거시 ResNet18 (그레이스케일 → 3ch, 224 리사이즈 기반)
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_multiscale_model(num_classes=2):
    """
    멀티스케일 ResNet18.
    - 입력: (N, 3, 256, 256) — 3채널 멀티스케일 orbit 이미지
    - AdaptiveAvgPool2d가 있으므로 256 입력 그대로 지원
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_trained_model(model_path):
    """
    체크포인트를 로드하고 모델과 메타데이터를 반환한다.

    반환:
        model       : eval 모드 PyTorch 모델
        class_names : ['normal', 'abnormal']
        meta        : {
            'model_type': str,   # 'resnet18_multiscale' | 'resnet18_legacy'
            'norm_mean':  list,  # 정규화 평균 (멀티스케일 모델만)
            'norm_std':   list,  # 정규화 표준편차 (멀티스케일 모델만)
            'img_size':   int,   # 입력 이미지 크기 (멀티스케일 모델만)
        }
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # ── class_names ──────────────────────────────────────
        if "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
        else:
            print("Warning: class_names not in checkpoint, using default.")
            last_w = checkpoint["model_state_dict"]["fc.weight"]
            class_names = [str(i) for i in range(last_w.shape[0])]

        num_classes = len(class_names)

        # ── model_type 분기 ───────────────────────────────────
        model_type = checkpoint.get("model_type", "resnet18_legacy")

        if model_type == "resnet18_multiscale":
            model = get_multiscale_model(num_classes)
            meta = {
                "model_type": model_type,
                "norm_mean":  checkpoint.get("norm_mean",  [0.5, 0.5, 0.5]),
                "norm_std":   checkpoint.get("norm_std",   [0.5, 0.5, 0.5]),
                "img_size":   checkpoint.get("img_size",   256),
            }
        else:
            model = get_model(num_classes)
            meta = {
                "model_type": "resnet18_legacy",
                "norm_mean":  [0.485, 0.456, 0.406],
                "norm_std":   [0.229, 0.224, 0.225],
                "img_size":   224,
            }

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, class_names, meta

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")
