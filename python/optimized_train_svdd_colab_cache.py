import os
import sys
import gc
import glob
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. 환경 설정
# ─────────────────────────────────────────────
PRJ_PATH = "/content/drive/MyDrive/rcp_5th/python"
LOCAL_DATA_ROOT = "/content/local_data"

if PRJ_PATH not in sys.path:
    sys.path.insert(0, PRJ_PATH)

from model_svdd import SVDDEncoder, compute_svdd_loss
from preprocess import prepare_1d_input
from infer_resnet_None import extract_rcp_xy_from_bin

FS = 40_000
WIN_SIZE = FS
STEP = FS // 5

torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────
# 2. Drive → Local 복사
# ─────────────────────────────────────────────
def copy_to_local(data_dir):
    raw_path = os.path.join(data_dir, "raw")
    local_raw = os.path.join(LOCAL_DATA_ROOT, "raw")

    if os.path.exists(local_raw):
        return local_raw

    print("Google Drive → Local SSD 복사 중...")
    shutil.copytree(raw_path, local_raw)
    print("복사 완료.")
    return local_raw


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class CachedWindowDataset(Dataset):
    def __init__(self, bin_files, augment=True):
        self.augment = augment
        self.samples = []

        print("전체 데이터 메모리 캐싱 시작...")

        for bf in tqdm(bin_files):
            rcp_xy = extract_rcp_xy_from_bin(bf, fs=FS)

            for _, (x_full, y_full) in rcp_xy.items():
                n_total = len(x_full)

                for s in range(0, n_total - WIN_SIZE + 1, STEP):
                    x_seg = x_full[s : s + WIN_SIZE]
                    y_seg = y_full[s : s + WIN_SIZE]
                    arr = prepare_1d_input(x_seg, y_seg)
                    self.samples.append(arr.astype(np.float32))

            del rcp_xy

        self.samples = np.stack(self.samples)
        print(f"캐싱 완료: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr = self.samples[idx]

        if self.augment:
            shift = np.random.randint(-2000, 2001)
            arr = np.roll(arr, shift, axis=1)
            arr = arr + np.random.randn(*arr.shape).astype(np.float32) * 0.003

        return torch.from_numpy(arr)


# ─────────────────────────────────────────────
# 4. 학습 엔진
# ─────────────────────────────────────────────
def train_engine(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    raw_path = copy_to_local(args.data_dir)

    bin_files = glob.glob(os.path.join(raw_path, "**/*.bin"), recursive=True)
    bin_files += glob.glob(os.path.join(raw_path, "**/*.BIN"), recursive=True)

    if not bin_files:
        raise RuntimeError("BIN 파일 없음")

    print("BIN 파일:", len(bin_files))

    dataset = CachedWindowDataset(bin_files, augment=True)

    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split = int(len(indices) * 0.8)

    train_dataset = Subset(dataset, indices[:split])
    val_dataset = Subset(dataset, indices[split:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    encoder = SVDDEncoder(feature_dim=args.feature_dim).to(device)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # ───────── Warm-up ─────────
    print("\n[Warm-up]")
    warmup_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr * 10)

    for ep in range(args.warmup_epochs):
        encoder.train()
        total_loss = 0

        for x in train_loader:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                feat = encoder(x)
                feat_c = feat - feat.mean(0, keepdim=True)
                cov = (feat_c.T @ feat_c) / (feat.size(0) - 1)
                loss = (cov - torch.eye(feat.size(1), device=device)).pow(2).mean()

            warmup_opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(warmup_opt)
            scaler.update()

            total_loss += loss.item()

        print(f"Warm-up {ep+1}: {total_loss/len(train_loader):.6f}")

    # ───────── Center 계산 ─────────
    print("\n[Center 계산]")
    encoder.eval()
    all_feats = []

    with torch.no_grad():
        for x in train_loader:
            x = x.to(device)
            feat = encoder(x)
            all_feats.append(feat)

    center = torch.cat(all_feats, 0).mean(0).detach()
    del all_feats
    torch.cuda.empty_cache()

    print("Center norm:", center.norm().item())

    # ───────── SVDD 학습 ─────────
    print("\n[SVDD Training]")

    best_val_loss = float("inf")
    best_path = os.path.join(PRJ_PATH, "model/svdd_best.pth")

    for ep in range(args.epochs):

        # Train
        encoder.train()
        train_loss = 0

        for x in train_loader:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                feat = encoder(x)
                loss = compute_svdd_loss(feat, center)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        encoder.eval()
        val_loss = 0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    feat = encoder(x)
                    loss = compute_svdd_loss(feat, center)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {ep+1}: Train {train_loss:.6f} | Val {val_loss:.6f}")

        # Best 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": encoder.state_dict(),
                    "center": center.cpu(),
                    "feature_dim": args.feature_dim,
                    "epoch": ep + 1,
                    "val_loss": val_loss,
                },
                best_path,
            )

            print(f"Best model updated (Epoch {ep+1})")

    # Last 모델 저장
    last_path = os.path.join(PRJ_PATH, "model/svdd_last.pth")
    os.makedirs(os.path.dirname(last_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "center": center.cpu(),
            "feature_dim": args.feature_dim,
        },
        last_path,
    )

    print("학습 완료")
    print("Best:", best_path)
    print("Last:", last_path)


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
class Args:
    data_dir = "/content/drive/MyDrive/rcp_5th/data/"
    feature_dim = 128
    epochs = 50
    batch_size = 128
    lr = 1e-4
    warmup_epochs = 5


args = Args()
train_engine(args)
