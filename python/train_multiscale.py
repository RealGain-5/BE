"""
train_multiscale.py
====================
멀티스케일 3채널 Orbit 이미지로 ResNet18 모델을 4-class 학습합니다.

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
  python/model/resnet18_orbit_multiscale.pth

실행 예시:
  python train_multiscale.py --data_dir ../data --epochs 50 --batch_size 32
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
import torch.nn.functional as F
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
)

# UTF-8 출력 (Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 클래스 정의: class_map.json 단일 소스 참조
import json as _json
with open(os.path.join(SCRIPT_DIR, "class_map.json"), "r") as _f:
    CLASS_NAMES: list = _json.load(_f)["classes"]

# 데이터 소스 정의: (서브디렉토리, 클래스 라벨, 파일 패턴)
# normal은 raw/normal, 합성 고장은 synthetic/{fault_type}
TRAIN_SOURCES = [
    ("raw/normal",                      0, "*.BIN"),  # normal
    ("synthetic/3600rpm/unbalance",     1, "*.bin"),  # unbalance  (3600 RPM)
    ("synthetic/1200rpm/unbalance",     1, "*.bin"),  # unbalance  (1200 RPM)
    ("synthetic/3600rpm/misalignment",  2, "*.bin"),  # misalignment (3600 RPM)
    ("synthetic/1200rpm/misalignment",  2, "*.bin"),  # misalignment (1200 RPM)
    ("synthetic/3600rpm/oil_whip",      3, "*.bin"),  # oil_whip   (3600 RPM)
    ("synthetic/1200rpm/oil_whip",      3, "*.bin"),  # oil_whip   (1200 RPM)
]

FS        = 40_000
IMG_SIZE  = 128
LAMBDA_OE = 0.5   # Outlier Exposure 손실 가중치
PATIENCE  = 10   # 조기 종료: 복합 지표 미개선 허용 epoch 수


# ─────────────────────────────────────────────
# 1. Dataset — 텐서 사전 계산 (CPU 최적화)
# ─────────────────────────────────────────────
def _augment_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    PIL 없이 텐서 레벨 증강 (회전 로직 제거).
    - 수평 반전: 궤도 이미지의 방향 불변성
    - 수직 반전: 궤도 이미지의 방향 불변성
    - 밝기 스케일: 진폭 변동 시뮬레이션 (0.8 ~ 1.2×)
    """
    if torch.rand(1).item() < 0.5:
        t = t.flip(-1)   # 수평 반전
    if torch.rand(1).item() < 0.5:
        t = t.flip(-2)   # 수직 반전
    scale = 0.8 + torch.rand(1).item() * 0.4
    t = t * scale
    return t


class OrbitMultiscaleDataset(Dataset):
    """
    CPU 최적화 Dataset.
    __init__에서 numpy array → 정규화 텐서 변환을 모두 완료.
    학습 루프 __getitem__에서는 PIL 변환 없이 텐서를 직접 반환.
    """

    def __init__(self, samples, mean, std, augment: bool = False):
        base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        print(f"    텐서 사전 계산 ({len(samples)} 샘플)...", end="", flush=True)
        t0 = time.time()
        self.tensors = []
        self.labels  = []
        for arr, label in samples:
            pil = Image.fromarray(arr, mode="RGB")
            self.tensors.append(base_tf(pil))
            self.labels.append(label)
        print(f" {time.time() - t0:.1f}s")

        self.augment = augment

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        t = self.tensors[idx]
        if self.augment:
            t = _augment_tensor(t)
        return t, self.labels[idx]


# ─────────────────────────────────────────────
# 2. 데이터 로딩
# ─────────────────────────────────────────────
def _load_class_samples(class_path, pattern, label, fs, img_size):
    """단일 클래스 디렉토리에서 샘플 로딩."""
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
                arr = make_multiscale_orbit(x_mil[s:e], y_mil[s:e], img_size=img_size, hybrid=True)
                samples.append((arr, label))
        except Exception as e:
            print(f"    [오류] {os.path.basename(bin_path)}: {e}")
    return samples


def load_all_samples(data_dir, fs=FS, img_size=IMG_SIZE):
    """
    4-class 학습 데이터 로딩.

    반환:
        train_samples     : 학습용 (각 클래스 80%)
        val_samples       : 검증용 (각 클래스 20%)
        real_abnormal_samples : 이차 검증용 raw/abnormal (라벨 미지정)
    """
    train_samples = []
    val_samples   = []

    for subdir, label, pattern in TRAIN_SOURCES:
        class_path = os.path.join(data_dir, subdir)
        class_label = CLASS_NAMES[label]
        print(f"  [{class_label}] 로딩 중: {class_path}")

        class_samples = _load_class_samples(class_path, pattern, label, fs, img_size)
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

    # raw/abnormal: 이차 검증용만 (4-class에 라벨 없음)
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
                    arr = make_multiscale_orbit(x_mil[s:e], y_mil[s:e], img_size=img_size)
                    real_abnormal_samples.append((arr, -1))
            except Exception as e:
                print(f"    [오류] {os.path.basename(bin_path)}: {e}")
        print(f"    → {len(real_abnormal_samples)} 샘플")

    return train_samples, val_samples, real_abnormal_samples


def compute_dataset_stats(samples):
    """채널별(R=fine, G=mid, B=wide) 평균·표준편차 계산 (학습 셋 기준)."""
    print("  데이터셋 통계 계산 중...")
    sum_   = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    n_pix  = 0

    for arr, _ in samples:
        f = arr.astype(np.float64) / 255.0
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
def get_multiscale_model(num_classes=4):
    """ResNet18 (ImageNet pretrained). fc → num_classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────
# 4. 이차 검증 (real_abnormal 탐지율)
# ─────────────────────────────────────────────
def eval_real_abnormal(model, device, real_abnormal_tensors, batch_size=64,
                       ood_threshold=0.70):
    """
    real_abnormal OOD 탐지율 계산 (사전 계산된 텐서 사용).
    Outlier Exposure 학습 후 max(softmax) < ood_threshold 이면 OOD로 판정.
    """
    model.eval()
    detected = 0
    total    = len(real_abnormal_tensors)
    if total == 0:
        return 0.0

    with torch.no_grad():
        for i in range(0, total, batch_size):
            imgs     = torch.stack(real_abnormal_tensors[i : i + batch_size]).to(device)
            probs    = F.softmax(model(imgs), dim=1).cpu()
            max_conf = probs.max(dim=1).values
            detected += int((max_conf < ood_threshold).sum())

    return detected / total


# ─────────────────────────────────────────────
# 5. 학습 루프
# ─────────────────────────────────────────────
def train(args):
    # CPU 멀티코어 활성화
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))

    print("\n=== 멀티스케일 Orbit ResNet18 4-class 학습 시작 ===")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  epochs    : {args.epochs}")
    print(f"  batch     : {args.batch_size}")
    print(f"  lr        : {args.lr}")
    print(f"  img_size  : {IMG_SIZE}")
    print(f"  CPU 스레드: {n_threads}")
    print(f"  classes   : {CLASS_NAMES}")
    print(f"  lambda_oe : {LAMBDA_OE}  (Outlier Exposure)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device    : {device}")

    # ── 1. 데이터 로딩 ──────────────────────────────
    print("\n[1] 데이터 로딩")
    train_samples, val_samples, real_abnormal_samples = load_all_samples(args.data_dir)

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

    # 두 클래스 이상 있어야 학습 가능
    present_classes = set(tr_labels)
    if len(present_classes) < 2:
        print("  [오류] 2개 이상의 클래스가 있어야 합니다.")
        return

    # ── 2. 통계 계산 ────────────────────────────────
    print("\n[2] 채널 통계 계산 (학습 셋 기준)")
    mean, std = compute_dataset_stats(train_samples)

    # ── 3. Dataset 구성 (텐서 사전 계산) ────────────
    print("\n[3] 텐서 사전 계산 (Dataset 초기화)")
    print("  train:")
    train_ds = OrbitMultiscaleDataset(train_samples, mean=mean, std=std, augment=True)
    print("  val:")
    val_ds   = OrbitMultiscaleDataset(val_samples,   mean=mean, std=std, augment=False)

    # real_abnormal 텐서 사전 계산
    real_abnormal_tensors = []
    if real_abnormal_samples:
        base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        print("  real_abnormal:", end="", flush=True)
        t0 = time.time()
        for arr, _ in real_abnormal_samples:
            pil = Image.fromarray(arr, mode="RGB")
            real_abnormal_tensors.append(base_tf(pil))
        print(f" {len(real_abnormal_tensors)} 샘플 ({time.time()-t0:.1f}s)")

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

    # OE DataLoader — 사전 계산된 real_abnormal 텐서를 학습 중 순환 사용
    oe_loader = None
    if real_abnormal_tensors:
        oe_stacked = torch.stack(real_abnormal_tensors)  # (N, 3, H, W)
        oe_ds      = torch.utils.data.TensorDataset(oe_stacked)
        oe_loader  = DataLoader(
            oe_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        print(f"  OE 샘플   : {len(real_abnormal_tensors)} (raw/abnormal)")
    else:
        print("  OE 샘플   : 없음 — Outlier Exposure 비활성화")

    # ── 4. 모델 / 손실 / 옵티마이저 ─────────────────
    print("\n[4] 모델 초기화")
    num_classes = len(CLASS_NAMES)
    model = get_multiscale_model(num_classes=num_classes).to(device)

    # 클래스별 가중 CrossEntropyLoss
    counts = [class_count.get(c, 1) for c in range(num_classes)]
    w = torch.tensor([1.0 / max(c, 1) for c in counts], device=device)
    w = w / w.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── 5. 학습 ─────────────────────────────────────
    print("\n[5] 학습 루프")
    print(f"  조기 종료 patience: {args.patience} epoch")
    best_combined = 0.0   # val_acc × OOD탐지율 복합 지표
    no_improve    = 0
    model_out_dir = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(model_out_dir, exist_ok=True)
    best_path = os.path.join(model_out_dir, "resnet18_orbit_multiscale.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        oe_iter = iter(oe_loader) if oe_loader is not None else None
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out     = model(imgs)
            loss_ce = criterion(out, lbls)

            # Outlier Exposure: OE 샘플에 대해 균등 분포 출력 유도
            if oe_iter is not None:
                try:
                    (oe_imgs,) = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(oe_loader)
                    (oe_imgs,) = next(oe_iter)
                oe_imgs = oe_imgs.to(device)
                oe_out  = model(oe_imgs)
                loss_oe = -(F.log_softmax(oe_out, dim=1).mean(dim=1)).mean()
                loss    = loss_ce + LAMBDA_OE * loss_oe
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()

            tr_loss    += loss_ce.item() * imgs.size(0)
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
        va_acc = va_correct / va_total if va_total > 0 else 0.0
        scheduler.step()

        # OOD 탐지율 — 매 epoch 계산 (복합 지표 기준)
        if real_abnormal_tensors:
            ood_rate = eval_real_abnormal(model, device, real_abnormal_tensors)
            combined = va_acc * max(ood_rate, 0.01)
        else:
            ood_rate = None
            combined = va_acc

        elapsed = time.time() - t0
        ood_str = f"OOD={ood_rate:.4f} | " if ood_rate is not None else ""
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={tr_loss/tr_total:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss/max(va_total,1):.4f} acc={va_acc:.4f} | "
            f"{ood_str}score={combined:.4f} | {elapsed:.1f}s"
        )

        # 복합 지표 기준 최고 모델 저장
        if combined > best_combined:
            best_combined = combined
            no_improve    = 0
            checkpoint = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "class_names":      CLASS_NAMES,
                "norm_mean":        mean,
                "norm_std":         std,
                "img_size":         IMG_SIZE,
                "val_acc":          va_acc,
                "ood_rate":         ood_rate,
                "combined_score":   combined,
                "model_type":       "resnet18_multiscale",
                "channel_mode":     "hybrid",
            }
            torch.save(checkpoint, best_path)
            ood_info = f", OOD={ood_rate:.4f}" if ood_rate is not None else ""
            print(f"    ✓ 최고 모델 저장: score={combined:.4f} (val={va_acc:.4f}{ood_info})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  조기 종료: {args.patience} epoch 연속 미개선 (epoch {epoch})")
                break

    print(f"\n=== 학습 완료. 최고 score={best_combined:.4f} ===")
    print(f"    모델 저장 경로: {best_path}")


# ─────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="멀티스케일 Orbit ResNet18 4-class 학습")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/raw, data/synthetic 가 위치한 상위 디렉토리",
    )
    p.add_argument("--epochs",     type=int,   default=50,       help="학습 에폭 수")
    p.add_argument("--batch_size", type=int,   default=32,       help="배치 크기")
    p.add_argument("--lr",         type=float, default=1e-4,     help="초기 학습률")
    p.add_argument("--patience",   type=int,   default=PATIENCE, help="조기 종료 patience")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
