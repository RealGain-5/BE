"""
train_1d_cnn.py
================
OrbitCNN1D 모델 4-class 학습 스크립트.
Raw time-series (X_mil, Y_mil) sec9 구간으로 1D CNN을 학습합니다.

입력 구조:
  data/
    raw/
      normal/   *.BIN        → label 0 (normal)
      abnormal/ *.BIN        → 이차 검증 전용 (학습 미사용)
    synthetic/
      unbalance/    *.bin   → label 1
      misalignment/ *.bin   → label 2
      oil_whip/     *.bin   → label 3

출력:
  python/model/orbit_cnn1d.pth

실행 예시:
  python train_1d_cnn.py --data_dir ../data --epochs 50 --batch_size 32
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

# 클래스 정의: class_map.json 단일 소스 참조
import json as _json
with open(os.path.join(SCRIPT_DIR, "class_map.json"), "r") as _f:
    CLASS_NAMES: list = _json.load(_f)["classes"]

# 데이터 소스 정의
TRAIN_SOURCES = [
    ("raw/normal",                      0, "*.BIN"),  # normal
    ("synthetic/3600rpm/unbalance",     1, "*.bin"),  # unbalance  (3600 RPM)
    ("synthetic/1200rpm/unbalance",     1, "*.bin"),  # unbalance  (1200 RPM)
    ("synthetic/3600rpm/misalignment",  2, "*.bin"),  # misalignment (3600 RPM)
    ("synthetic/1200rpm/misalignment",  2, "*.bin"),  # misalignment (1200 RPM)
    ("synthetic/3600rpm/oil_whip",      3, "*.bin"),  # oil_whip   (3600 RPM)
    ("synthetic/1200rpm/oil_whip",      3, "*.bin"),  # oil_whip   (1200 RPM)
]

FS = 40_000


# ─────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────
class OrbitCNN1DDataset(Dataset):
    """각 샘플: (np.float32 (2, 40000), label_idx)"""

    def __init__(self, samples, augment: bool = False):
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
def _load_class_samples_1d(class_path, pattern, label, fs):
    """단일 클래스 디렉토리에서 1D 샘플 로딩."""
    bin_files = sorted(glob.glob(os.path.join(class_path, pattern)))
    if not bin_files:
        print(f"  [경고] {class_path} 에서 파일을 찾을 수 없습니다.")
        return []

    samples = []
    for bin_path in bin_files:
        try:
            data     = parse_bin_legacy(bin_path, fs=fs)
            xy_pairs = extract_xy_pairs_legacy(data)
            for x, y in xy_pairs:
                x_mil, y_mil = volt_to_mil(x, y)
                s, e = 9 * fs, 10 * fs
                arr = prepare_1d_input(x_mil[s:e], y_mil[s:e])  # (2, 40000) float32
                samples.append((arr, label))
        except Exception as e:
            print(f"    [오류] {os.path.basename(bin_path)}: {e}")
    return samples


def load_all_samples_1d(data_dir, fs=FS):
    """
    4-class 학습 데이터 로딩 (1D CNN용).

    반환:
        train_samples     : 학습용 (각 클래스 80%)
        val_samples       : 검증용 (각 클래스 20%)
        real_abnormal_samples : 이차 검증용 raw/abnormal
    """
    train_samples = []
    val_samples   = []

    for subdir, label, pattern in TRAIN_SOURCES:
        class_path  = os.path.join(data_dir, subdir)
        class_label = CLASS_NAMES[label]
        print(f"  [{class_label}] 로딩 중: {class_path}")

        class_samples = _load_class_samples_1d(class_path, pattern, label, fs)
        if not class_samples:
            continue

        n = len(class_samples)
        if n >= 5:
            tr_idx, va_idx = train_test_split(
                list(range(n)), test_size=0.2, random_state=42
            )
            for i in tr_idx:
                train_samples.append(class_samples[i])
            for i in va_idx:
                val_samples.append(class_samples[i])
            print(f"    → train: {len(tr_idx)}, val: {len(va_idx)}")
        else:
            train_samples.extend(class_samples)
            print(f"    → 전체 {n}개를 train에 사용 (너무 적어 val 제외)")

    # raw/abnormal: 이차 검증용만
    real_abnormal_samples = []
    abnormal_path = os.path.join(data_dir, "raw", "abnormal")
    abn_files = sorted(glob.glob(os.path.join(abnormal_path, "*.BIN")))
    if abn_files:
        print(f"  [real_abnormal] 이차 검증용 로딩: {len(abn_files)} 파일")
        for bin_path in abn_files:
            try:
                data     = parse_bin_legacy(bin_path, fs=fs)
                xy_pairs = extract_xy_pairs_legacy(data)
                for x, y in xy_pairs:
                    x_mil, y_mil = volt_to_mil(x, y)
                    s, e = 9 * fs, 10 * fs
                    arr = prepare_1d_input(x_mil[s:e], y_mil[s:e])
                    real_abnormal_samples.append((arr, -1))
            except Exception as e:
                print(f"    [오류] {os.path.basename(bin_path)}: {e}")
        print(f"    → {len(real_abnormal_samples)} 샘플")

    return train_samples, val_samples, real_abnormal_samples


# ─────────────────────────────────────────────
# 3. 이차 검증
# ─────────────────────────────────────────────
def eval_real_abnormal_1d(model, device, real_abnormal_samples, batch_size=64):
    """real_abnormal 탐지율 계산 (pred != 0 이면 올바른 탐지)."""
    model.eval()
    correct = 0
    total   = len(real_abnormal_samples)
    if total == 0:
        return 0.0

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch   = real_abnormal_samples[i : i + batch_size]
            tensors = [torch.from_numpy(arr) for arr, _ in batch]
            imgs    = torch.stack(tensors).to(device)
            preds   = model(imgs).argmax(1).cpu().numpy()
            correct += int((preds != 0).sum())

    return correct / total


# ─────────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────────
def train(args):
    # CPU 멀티코어 활성화
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))

    print("\n=== OrbitCNN1D 4-class 학습 시작 ===")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  epochs    : {args.epochs}")
    print(f"  batch     : {args.batch_size}")
    print(f"  lr        : {args.lr}")
    print(f"  CPU 스레드: {n_threads}")
    print(f"  classes   : {CLASS_NAMES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device    : {device}")

    # ── 1. 데이터 로딩 ──────────────────────────────
    print("\n[1] 데이터 로딩")
    train_samples, val_samples, real_abnormal_samples = load_all_samples_1d(args.data_dir)

    if not train_samples:
        print("  [오류] 학습 샘플이 없습니다. data_dir 경로를 확인하세요.")
        return

    tr_labels = [s[1] for s in train_samples]
    va_labels = [s[1] for s in val_samples]

    print(f"\n  train 샘플: {len(train_samples)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {tr_labels.count(i)}")
    print(f"  val 샘플:   {len(val_samples)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {va_labels.count(i)}")
    print(f"  real_abnormal (이차검증): {len(real_abnormal_samples)}")

    present_classes = set(tr_labels)
    if len(present_classes) < 2:
        print("  [오류] 2개 이상의 클래스가 있어야 합니다.")
        return

    # ── 2. DataLoader ────────────────────────────────
    train_ds = OrbitCNN1DDataset(train_samples, augment=True)
    val_ds   = OrbitCNN1DDataset(val_samples,   augment=False)

    # 클래스 불균형 대응: WeightedRandomSampler
    class_count    = {c: tr_labels.count(c) for c in range(len(CLASS_NAMES))}
    sample_weights = [1.0 / max(class_count.get(l, 1), 1) for l in tr_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # ── 3. 모델 / 손실 / 옵티마이저 ─────────────────
    print("\n[2] 모델 초기화")
    num_classes = len(CLASS_NAMES)
    model = OrbitCNN1D(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  파라미터: {n_params:.2f}M")

    counts = [class_count.get(c, 1) for c in range(num_classes)]
    w = torch.tensor([1.0 / max(c, 1) for c in counts], device=device)
    w = w / w.sum() * num_classes
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
        va_acc = va_correct / va_total if va_total > 0 else 0.0
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={tr_loss/tr_total:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss/max(va_total,1):.4f} acc={va_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # 이차 검증: real_abnormal 탐지율 (5 epoch마다, 마지막 epoch 포함)
        if real_abnormal_samples and (epoch % 5 == 0 or epoch == args.epochs):
            ra_rate = eval_real_abnormal_1d(model, device, real_abnormal_samples)
            print(
                f"    [이차검증] real_abnormal 탐지율: {ra_rate:.4f} "
                f"({int(ra_rate * len(real_abnormal_samples))}/{len(real_abnormal_samples)})"
            )

        # 최고 검증 정확도 모델 저장
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            checkpoint = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "class_names":      CLASS_NAMES,
                "norm_scale":       None,  # per-sample 정규화
                "val_acc":          va_acc,
                "model_type":       "orbit_cnn1d",
            }
            torch.save(checkpoint, best_path)
            print(f"    ✓ 최고 모델 저장: val_acc={va_acc:.4f} → {best_path}")

    print(f"\n=== 학습 완료. 최고 val_acc={best_val_acc:.4f} ===")
    print(f"    모델 저장 경로: {best_path}")


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="OrbitCNN1D 4-class 학습")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/raw, data/synthetic 가 위치한 상위 디렉토리",
    )
    p.add_argument("--epochs",     type=int,   default=50,   help="학습 에폭 수")
    p.add_argument("--batch_size", type=int,   default=32,   help="배치 크기")
    p.add_argument("--lr",         type=float, default=5e-4, help="초기 학습률")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
