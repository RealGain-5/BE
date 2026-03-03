"""
train_svdd.py
=============
Deep SVDD 학습 스크립트.

3단계 학습:
  1. Warm-up: pretrained OrbitCNN1D 백본으로 초기화 → 전체 정상 학습 샘플 순전파 → c 계산
  2. SVDD 학습: 중심 c 고정, encoder fine-tune (patience 기반 조기 종료)
  3. 임계값 산출: 학습 완료 후 전체 학습 샘플 거리 분포의 percentile → svdd_config.json 저장

학습 데이터: data/raw/normal/ + data/raw/normal_1200rpm/ (3600rpm 제외)

사용:
  venv/Scripts/python.exe python/train_svdd.py \\
      --data_dir ../data \\
      --pretrained model/orbit_cnn1d.pth \\
      --feature_dim 128 \\
      --epochs 50 \\
      --batch_size 16 \\
      --lr 1e-4 \\
      --warmup_epochs 5 \\
      --percentile 95 \\
      --patience 15
"""

import _compat  # noqa: F401 — PyTorch/Windows/Python-3.11 호환성 패치

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_svdd import SVDDEncoder, compute_svdd_loss, compute_svdd_distances
from preprocess import prepare_1d_input
from infer_resnet_None import extract_rcp_xy_from_bin

FS = 40_000


# ─────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────

class SVDDDataset(Dataset):
    """
    data/raw/normal/ + data/raw/normal_1200rpm/ BIN 파일에서
    4 RCP × sec9 구간 추출 → 데이터 증강 포함.
    """

    def __init__(self, bin_files: list, augment: bool = True):
        self.samples = []
        self.augment = augment

        for bf in bin_files:
            try:
                rcp_xy = extract_rcp_xy_from_bin(bf, fs=FS)
                for rcp, (x_full, y_full) in rcp_xy.items():
                    x_seg = x_full[9 * FS: 10 * FS]
                    y_seg = y_full[9 * FS: 10 * FS]
                    if len(x_seg) < FS:
                        continue
                    arr = prepare_1d_input(x_seg, y_seg)  # (2, 40000) float32
                    self.samples.append(arr)
            except Exception as e:
                print(f"[Dataset] WARNING: {bf} 로드 실패 ({e}), 건너뜀.")

        print(f"[Dataset] {len(self.samples)} samples from {len(bin_files)} BIN files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr = self.samples[idx].copy()  # (2, 40000)
        if self.augment:
            arr = self._augment(arr)
        return torch.from_numpy(arr).float()

    @staticmethod
    def _augment(arr: np.ndarray) -> np.ndarray:
        """랜덤 원형 시프트 + 약한 가우시안 노이즈 + 랜덤 채널 부호 반전"""
        # 랜덤 원형 시프트 ±4000 샘플
        shift = np.random.randint(-4000, 4001)
        arr = np.roll(arr, shift, axis=1)
        # 약한 가우시안 노이즈 σ=0.005
        arr = arr + np.random.randn(*arr.shape).astype(np.float32) * 0.005
        # 랜덤 채널 부호 반전 (50% 확률)
        if np.random.rand() < 0.5:
            ch = np.random.randint(0, 2)
            arr[ch] = -arr[ch]
        return arr


def collect_bin_files(data_dir: str) -> list:
    """data/raw/normal/ + data/raw/normal_1200rpm/ BIN 파일 수집 (3600rpm 제외)"""
    include_dirs = [
        os.path.join(data_dir, "raw", "normal"),
        os.path.join(data_dir, "raw", "normal_1200rpm"),
    ]
    bin_files = []
    for d in include_dirs:
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".bin"):
                    bin_files.append(os.path.join(d, fn))
        else:
            print(f"[SVDD] WARNING: 디렉토리 없음 {d}")
    return bin_files


# ─────────────────────────────────────────────
# 전이학습 초기화
# ─────────────────────────────────────────────

def load_pretrained_backbone(encoder: SVDDEncoder, ckpt_path: str) -> bool:
    """OrbitCNN1D 체크포인트에서 features 레이어 가중치 전이학습."""
    if not os.path.exists(ckpt_path):
        print(f"[SVDD] pretrained checkpoint not found: {ckpt_path}")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        # features.* 키만 필터링하여 로드
        backbone_state = {k: v for k, v in state.items() if k.startswith("features.")}
        missing, unexpected = encoder.load_state_dict(backbone_state, strict=False)
        print(f"[SVDD] 전이학습 완료: {len(backbone_state)}개 레이어 로드, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        return True
    except Exception as e:
        print(f"[SVDD] WARNING: 전이학습 실패 ({e}), 랜덤 초기화로 진행.")
        return False


# ─────────────────────────────────────────────
# 학습 메인
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SVDD] device: {device}")

    # 1) BIN 파일 수집
    bin_files = collect_bin_files(args.data_dir)
    if not bin_files:
        raise RuntimeError(f"정상 BIN 파일을 찾을 수 없습니다: {args.data_dir}/raw/normal*/")
    print(f"[SVDD] 정상 BIN 파일 수: {len(bin_files)}")

    # train / val 분할 (80/20)
    np.random.seed(42)
    idx = np.random.permutation(len(bin_files))
    n_train = max(1, int(len(bin_files) * 0.8))
    train_files = [bin_files[i] for i in idx[:n_train]]
    val_files   = [bin_files[i] for i in idx[n_train:]]

    train_dataset = SVDDDataset(train_files, augment=True)
    val_dataset   = SVDDDataset(val_files,   augment=False)

    if len(train_dataset) == 0:
        raise RuntimeError("학습 샘플이 0개입니다. BIN 파일 경로를 확인하세요.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # 2) 모델 초기화
    encoder = SVDDEncoder(feature_dim=args.feature_dim).to(device)

    # 전이학습
    pretrained_path = os.path.join(SCRIPT_DIR, args.pretrained)
    load_pretrained_backbone(encoder, pretrained_path)

    # ── 1단계: Warm-up ─────────────────────────────────────────────
    print(f"\n[SVDD] === 1단계: Warm-up ({args.warmup_epochs} epochs) ===")
    warmup_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr * 10, weight_decay=1e-5)
    encoder.train()

    for ep in range(args.warmup_epochs):
        total_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            feat = encoder(x)
            # Warm-up: 피처 분산 최대화 (단위 행렬에 가깝게 유도)
            # feat: (B, D) — 배치 공분산이 단위 행렬에 가까워지도록
            # loss = ||cov(feat) - I||_F² → 피처가 분산되어 c 계산에 유리
            B = feat.size(0)
            feat_c = feat - feat.mean(dim=0, keepdim=True)
            cov = (feat_c.T @ feat_c) / max(B - 1, 1)           # (D, D)
            eye = torch.eye(cov.size(0), device=device)
            loss = (cov - eye).pow(2).mean()
            warmup_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            warmup_optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"  Warm-up [{ep+1}/{args.warmup_epochs}] cov_loss={total_loss/len(train_dataset):.6f}")

    # 중심 c = 모든 학습 샘플의 평균 피처 (증강 없이)
    print("[SVDD] 중심 c 계산 중...")
    encoder.eval()
    all_feats = []
    no_aug_loader = DataLoader(SVDDDataset(train_files, augment=False),
                               batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in no_aug_loader:
            feat = encoder(batch.to(device))
            all_feats.append(feat.cpu())
    center = torch.cat(all_feats, dim=0).mean(dim=0)  # (feature_dim,)
    center = center.to(device)
    print(f"[SVDD] 중심 c 계산 완료: norm={center.norm().item():.4f}")

    # ── 2단계: SVDD 학습 ─────────────────────────────────────────────
    print(f"\n[SVDD] === 2단계: SVDD 학습 ({args.epochs} epochs, patience={args.patience}) ===")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    for ep in range(args.epochs):
        # Train
        encoder.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            feat = encoder(x)
            loss = compute_svdd_loss(feat, center)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_dataset)
        scheduler.step()

        # Validation
        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                feat = encoder(x)
                loss = compute_svdd_loss(feat, center)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(len(val_dataset), 1)

        print(f"  Epoch [{ep+1}/{args.epochs}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  [조기 종료] {args.patience} epochs 개선 없음 → 중단 (epoch {ep+1})")
                break

    # 최적 가중치 복원
    if best_state is not None:
        encoder.load_state_dict(best_state)
    print(f"[SVDD] 최적 val_loss={best_val_loss:.6f}")

    # ── 3단계: 임계값 산출 ──────────────────────────────────────────
    print(f"\n[SVDD] === 3단계: 임계값 산출 (p{args.percentile}) ===")
    encoder.eval()
    all_dists = []
    train_no_aug = SVDDDataset(train_files, augment=False)
    with torch.no_grad():
        for batch in DataLoader(train_no_aug, batch_size=args.batch_size, shuffle=False):
            feat = encoder(batch.to(device))
            dists = compute_svdd_distances(feat, center)
            all_dists.append(dists.cpu().numpy())
    all_dists = np.concatenate(all_dists)
    threshold = float(np.percentile(all_dists, args.percentile))
    print(f"[SVDD] 임계값 (p{args.percentile}): {threshold:.6f}")
    print(f"[SVDD] 거리 분포: min={all_dists.min():.6f}, "
          f"mean={all_dists.mean():.6f}, max={all_dists.max():.6f}")

    # ── 저장 ────────────────────────────────────────────────────────
    model_dir = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    ckpt_path = os.path.join(model_dir, "svdd_encoder.pth")
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "center": center.cpu(),
        "threshold": threshold,
        "feature_dim": args.feature_dim,
    }, ckpt_path)
    print(f"[SVDD] 모델 저장: {ckpt_path}")

    config_path = os.path.join(SCRIPT_DIR, "svdd_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "feature_dim": args.feature_dim,
            "percentile": args.percentile,
            "n_train_samples": len(train_no_aug),
        }, f, indent=2)
    print(f"[SVDD] 설정 저장: {config_path}")
    print("\n[SVDD] 학습 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep SVDD Training")
    parser.add_argument("--data_dir",      type=str,   default="../data",
                        help="데이터 루트 디렉토리")
    parser.add_argument("--pretrained",    type=str,   default="model/orbit_cnn1d.pth",
                        help="OrbitCNN1D 체크포인트 경로 (SCRIPT_DIR 기준 상대경로)")
    parser.add_argument("--feature_dim",   type=int,   default=128)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--percentile",    type=float, default=95.0)
    parser.add_argument("--patience",      type=int,   default=15)
    args = parser.parse_args()
    train(args)
