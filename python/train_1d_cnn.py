"""
train_1d_cnn.py
================
OrbitCNN1D 모델 학습 스크립트.
Raw time-series (X_mil, Y_mil) sec9 구간으로 1D CNN을 학습합니다.

입력 구조:
  data/
    normal/   *.BIN
    abnormal/ *.BIN

출력:
  python/model/orbit_cnn1d.pth

실행 예시:
  python train_1d_cnn.py --data_dir ../data --epochs 30 --batch_size 32
"""

import os
import sys
import glob
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# 프로젝트 모듈
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocess import (
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    prepare_1d_input,
)
from model_1d_cnn import OrbitCNN1D

# UTF-8 출력 (Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 클래스 정의: class_map.json 단일 소스 참조 (두 학습 스크립트 간 일관성 보장)
import json as _json
with open(os.path.join(SCRIPT_DIR, "class_map.json"), "r") as _f:
    CLASS_NAMES: list = _json.load(_f)["classes"]
FS          = 40_000


# ─────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────
class OrbitCNN1DDataset(Dataset):
    """
    각 샘플: (np.float32 (2, 40000), label_idx)
    torchvision transform 불필요 — 텐서 직접 반환.
    학습 시 텐서 레벨 증강 적용 (augment=True).
    """

    def __init__(self, samples, augment: bool = False):
        # samples: list of (np.ndarray (2,40000) float32, int label)
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        tensor = torch.from_numpy(arr)  # (2, 40000)

        if self.augment:
            tensor = self._augment(tensor)

        return tensor, label

    @staticmethod
    def _augment(tensor: torch.Tensor) -> torch.Tensor:
        # 1. Gaussian noise
        tensor = tensor + torch.randn_like(tensor) * 0.01

        # 2. Random circular time shift (최대 ±2000 샘플)
        shift = torch.randint(-2000, 2001, (1,)).item()
        tensor = torch.roll(tensor, shifts=shift, dims=-1)

        # 3. Random channel sign flip (측정 방향 불변성)
        if torch.rand(1).item() < 0.5:
            ch = torch.randint(0, 2, (1,)).item()
            tensor = tensor.clone()
            tensor[ch] = -tensor[ch]

        return tensor


# ─────────────────────────────────────────────
# 2. 데이터 로딩
# ─────────────────────────────────────────────
def load_all_samples_1d(data_dir, fs=FS):
    """
    data_dir/normal/*.BIN, data_dir/abnormal/*.BIN 을 순회하여
    (np.float32 (2, 40000), label) 리스트 반환.
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
                    # sec9 구간 (9~10초)
                    s, e = 9 * fs, 10 * fs
                    seg_x, seg_y = x_mil[s:e], y_mil[s:e]
                    arr = prepare_1d_input(seg_x, seg_y)  # (2, 40000) float32
                    samples.append((arr, label))

            except Exception as e:
                print(f"    [오류] {os.path.basename(bin_path)}: {e}")

    return samples


# ─────────────────────────────────────────────
# 3. 학습 루프
# ─────────────────────────────────────────────
def train(args):
    print("\n=== OrbitCNN1D 학습 시작 ===")
    print(f"  data_dir : {args.data_dir}")
    print(f"  epochs   : {args.epochs}")
    print(f"  batch    : {args.batch_size}")
    print(f"  lr       : {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device   : {device}")

    # ── 1. 데이터 로딩 ──────────────────────────────
    print("\n[1] 데이터 로딩")
    all_samples = load_all_samples_1d(args.data_dir)
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

    # ── 2. DataLoader ────────────────────────────────
    train_ds = OrbitCNN1DDataset(train_samples, augment=True)
    val_ds   = OrbitCNN1DDataset(val_samples,   augment=False)

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

    # ── 3. 모델 / 손실 / 옵티마이저 ─────────────────
    print("\n[2] 모델 초기화")
    model = OrbitCNN1D(num_classes=len(CLASS_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  파라미터: {n_params:.2f}M")

    # 클래스 가중 CrossEntropyLoss
    w = torch.tensor([1.0 / n0, 1.0 / n1], device=device)
    w = w / w.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── 4. 학습 루프 ─────────────────────────────────
    print("\n[3] 학습 루프")
    best_val_acc = 0.0
    model_out_dir = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(model_out_dir, exist_ok=True)
    best_path = os.path.join(model_out_dir, "orbit_cnn1d.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for tensors, lbls in train_loader:
            tensors, lbls = tensors.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(tensors)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * tensors.size(0)
            tr_correct += (out.argmax(1) == lbls).sum().item()
            tr_total   += tensors.size(0)

        # --- Validation ---
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for tensors, lbls in val_loader:
                tensors, lbls = tensors.to(device), lbls.to(device)
                out  = model(tensors)
                loss = criterion(out, lbls)
                va_loss    += loss.item() * tensors.size(0)
                va_correct += (out.argmax(1) == lbls).sum().item()
                va_total   += tensors.size(0)

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
                "norm_scale":       None,   # per-sample 정규화 — 저장 통계 없음
                "val_acc":          va_acc,
                "model_type":       "orbit_cnn1d",
            }
            torch.save(checkpoint, best_path)
            print(f"    ✓ 최고 모델 저장: val_acc={va_acc:.4f} → {best_path}")

    print(f"\n=== 학습 완료. 최고 val_acc={best_val_acc:.4f} ===")
    print(f"    모델 저장 경로: {best_path}")


# ─────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="OrbitCNN1D 학습")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/normal, data/abnormal 가 위치한 상위 디렉토리",
    )
    p.add_argument("--epochs",     type=int,   default=30,   help="학습 에폭 수")
    p.add_argument("--batch_size", type=int,   default=32,   help="배치 크기")
    p.add_argument("--lr",         type=float, default=1e-3, help="초기 학습률")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
