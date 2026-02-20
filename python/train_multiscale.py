"""
train_multiscale.py
====================
멀티스케일 3채널 Orbit 이미지로 ResNet18 모델을 학습합니다.

입력 구조:
  data/
    normal/   *.BIN
    abnormal/ *.BIN

출력:
  python/model/resnet18_orbit_multiscale.pth

실행 예시:
  python train_multiscale.py --data_dir ../data --epochs 30 --batch_size 16
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

# 프로젝트 모듈
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from preprocess import (
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    make_multiscale_orbit,
    build_multiscale_transform,
)

# UTF-8 출력 (Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CLASS_NAMES = ["normal", "abnormal"]
RCP_NAMES   = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
FS          = 40_000
IMG_SIZE    = 256


# ─────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────
class OrbitMultiscaleDataset(Dataset):
    """
    각 샘플: (멀티스케일 PIL RGB, label_idx)
    한 BIN 파일에서 4 RCP × 1 sec9 구간 → 4 샘플 생성
    """

    def __init__(self, samples, transform=None):
        # samples: list of (np.ndarray HWC uint8, int label)
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        pil = Image.fromarray(arr, mode="RGB")
        if self.transform:
            tensor = self.transform(pil)
        else:
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
        return tensor, label


# ─────────────────────────────────────────────
# 2. 데이터 로딩
# ─────────────────────────────────────────────
def load_all_samples(data_dir, fs=FS, img_size=IMG_SIZE):
    """
    data_dir/normal/*.BIN, data_dir/abnormal/*.BIN 을 순회하여
    (multiscale_np, label) 리스트 반환.
    """
    samples = []
    for class_name, label in zip(CLASS_NAMES, [0, 1]):
        class_path = os.path.join(data_dir, class_name)
        bin_files  = sorted(glob.glob(os.path.join(class_path, "*.BIN")))
        if not bin_files:
            print(f"  [경고] {class_path} 에서 BIN 파일을 찾을 수 없습니다.")
            continue

        print(f"  {class_name}: {len(bin_files)} 파일 로딩 중...")
        for bin_path in bin_files:
            try:
                data     = parse_bin_legacy(bin_path, fs=fs)
                xy_pairs = extract_xy_pairs_legacy(data)

                for x, y in xy_pairs:
                    x_mil, y_mil = volt_to_mil(x, y)
                    # sec9 구간 (9~10초) 만 사용
                    # 이유: 설비 가동 초기(sec0~sec8)는 과도 상태일 수 있으므로
                    # 정상 운전 상태를 대표하는 마지막 1초를 기준으로 학습.
                    # 추론(inference_daemon)도 동일 구간(sec9)을 사용하므로 일관성 유지.
                    s, e = 9 * fs, 10 * fs
                    seg_x, seg_y = x_mil[s:e], y_mil[s:e]
                    arr = make_multiscale_orbit(seg_x, seg_y, img_size=img_size)
                    samples.append((arr, label))

            except Exception as e:
                print(f"    [오류] {os.path.basename(bin_path)}: {e}")

    return samples


def compute_dataset_stats(samples):
    """
    전체 샘플에서 채널별(R=fine, G=mid, B=wide) 평균·표준편차 계산.
    반환: (mean_list, std_list) — 각각 [ch0, ch1, ch2]
    """
    print("  데이터셋 통계 계산 중...")
    # 메모리 절약: float32 누적
    sum_   = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    n_pix  = 0

    for arr, _ in samples:
        f = arr.astype(np.float64) / 255.0   # (H, W, 3)
        sum_   += f.sum(axis=(0, 1))
        sum_sq += (f ** 2).sum(axis=(0, 1))
        n_pix  += arr.shape[0] * arr.shape[1]

    mean = sum_   / n_pix
    std  = np.sqrt(sum_sq / n_pix - mean ** 2).clip(min=1e-6)
    print(f"  mean={mean.tolist()}, std={std.tolist()}")
    return mean.tolist(), std.tolist()


# ─────────────────────────────────────────────
# 3. 모델
# ─────────────────────────────────────────────
def get_multiscale_model(num_classes=2):
    """
    ResNet18 (ImageNet pretrained).
    입력: (N, 3, 256, 256) — 3채널 멀티스케일
    AdaptiveAvgPool2d 덕분에 256×256 입력도 그대로 지원.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ─────────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────────
def train(args):
    print("\n=== 멀티스케일 Orbit ResNet18 학습 시작 ===")
    print(f"  data_dir : {args.data_dir}")
    print(f"  epochs   : {args.epochs}")
    print(f"  batch    : {args.batch_size}")
    print(f"  lr       : {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device   : {device}")

    # ── 1. 데이터 로딩 ──────────────────────────────
    print("\n[1] 데이터 로딩")
    all_samples = load_all_samples(args.data_dir)
    if not all_samples:
        print("  샘플이 없습니다. data_dir 경로를 확인하세요.")
        return

    labels = [s[1] for s in all_samples]
    n0 = labels.count(0)
    n1 = labels.count(1)
    print(f"  전체 샘플: {len(all_samples)}  (normal={n0}, abnormal={n1})")
    if n0 == 0 or n1 == 0:
        print("  [오류] 두 클래스 모두 샘플이 있어야 합니다. normal/abnormal 디렉토리를 확인하세요.")
        return

    # train/val 분리 (stratified)
    indices = list(range(len(all_samples)))
    tr_idx, va_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    train_samples = [all_samples[i] for i in tr_idx]
    val_samples   = [all_samples[i] for i in va_idx]
    print(f"  train: {len(train_samples)}, val: {len(val_samples)}")

    # ── 2. 통계 계산 ────────────────────────────────
    print("\n[2] 채널 통계 계산 (학습 셋 기준)")
    mean, std = compute_dataset_stats(train_samples)

    # ── 3. Transform ────────────────────────────────
    train_tf = build_multiscale_transform(mean=mean, std=std, augment=True)
    val_tf   = build_multiscale_transform(mean=mean, std=std, augment=False)

    train_ds = OrbitMultiscaleDataset(train_samples, transform=train_tf)
    val_ds   = OrbitMultiscaleDataset(val_samples,   transform=val_tf)

    # 클래스 불균형 대응: WeightedRandomSampler
    tr_labels = [s[1] for s in train_samples]
    class_count = [tr_labels.count(c) for c in [0, 1]]
    sample_weights = [1.0 / class_count[l] for l in tr_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # ── 4. 모델 / 손실 / 옵티마이저 ─────────────────
    print("\n[3] 모델 초기화")
    model = get_multiscale_model(num_classes=len(CLASS_NAMES)).to(device)

    # 클래스 가중치 (Focal loss 대신 단순 가중 CE)
    w = torch.tensor([1.0 / n0, 1.0 / n1], device=device)
    w = w / w.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── 5. 학습 ─────────────────────────────────────
    print("\n[4] 학습 루프")
    best_val_acc = 0.0
    model_out_dir = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(model_out_dir, exist_ok=True)
    best_path = os.path.join(model_out_dir, "resnet18_orbit_multiscale.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == lbls).sum().item()
            tr_total   += imgs.size(0)

        # --- Validation ---
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out  = model(imgs)
                loss = criterion(out, lbls)
                va_loss    += loss.item() * imgs.size(0)
                va_correct += (out.argmax(1) == lbls).sum().item()
                va_total   += imgs.size(0)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={tr_loss/tr_total:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss/va_total:.4f} acc={va_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # 최고 검증 정확도 모델 저장
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            checkpoint = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "class_names":      CLASS_NAMES,
                "norm_mean":        mean,
                "norm_std":         std,
                "img_size":         IMG_SIZE,
                "val_acc":          va_acc,
                "model_type":       "resnet18_multiscale",
            }
            torch.save(checkpoint, best_path)
            print(f"    ✓ 최고 모델 저장: val_acc={va_acc:.4f} → {best_path}")

    print(f"\n=== 학습 완료. 최고 val_acc={best_val_acc:.4f} ===")
    print(f"    모델 저장 경로: {best_path}")


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="멀티스케일 Orbit ResNet18 학습")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/normal, data/abnormal 가 위치한 상위 디렉토리",
    )
    p.add_argument("--epochs",     type=int,   default=30,    help="학습 에폭 수")
    p.add_argument("--batch_size", type=int,   default=16,    help="배치 크기")
    p.add_argument("--lr",         type=float, default=1e-4,  help="초기 학습률")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
