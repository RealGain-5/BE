"""
optimized_train_svdd_colab_cache.py
=====================================
Google Colab GPU 환경용 Deep SVDD 학습 스크립트.

수정 이력:
  v4 (2026-03-05):
    - [Fix] Hypersphere collapse 방지: center를 매 epoch 재계산
      (고정 center → encoder가 항상 center를 출력하는 trivial solution으로 수렴)

  v3 (2026-03-05):
    - [Perf] Stage 4 임계값 계산: 윈도우 1개씩 추론 → 파일별 배치 추론 (~91x 속도 향상)
    - [Perf] 체크포인트 저장: Drive 직접 저장 → Local SSD 임시 저장, 완료 후 Drive 1회 복사
    - [Perf] optimizer.zero_grad(set_to_none=True) 적용

  v2 (2026-03-05):
    - [Fix] bin_files 필터링: normal/ + normal_1200rpm/ 만 수집
      (abnormal/, normal_3600rpm/ 제외 — SVDD는 정상 전용 학습)
    - [Fix] Center 계산 시 augment=False 전용 패스 사용
    - [Fix] STEP = FS // 10 (90% 오버랩, train_svdd.py 일치)
    - [Fix] Early Stopping 추가 (patience 기반)
    - [Fix] OrbitCNN1D 전이학습 초기화 추가
    - [Fix] threshold 계산 + svdd_config.json 저장 추가
    - [Fix] 저장 경로 svdd_best.pth → svdd_encoder.pth (inference_daemon.py 호환)
    - [Fix] gc.collect() 실제 호출 추가
    - [Fix] val_loss 단위 샘플 수 기준으로 정정

Drive 구조 (필수):
  MyDrive/rcp_5th/
  ├── python/
  │   ├── model_svdd.py
  │   ├── preprocess.py
  │   ├── infer_resnet_None.py
  │   ├── _compat.py
  │   └── model/
  │       └── orbit_cnn1d.pth          <- 전이학습용 (없으면 랜덤 초기화)
  └── data/
      └── raw/
          ├── normal/         *.BIN
          └── normal_1200rpm/ *.BIN

Colab 런타임: GPU (T4 이상 권장)
"""

import gc
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. 환경 설정
# ─────────────────────────────────────────────
PRJ_PATH        = "/content/drive/MyDrive/rcp_5th/python"
DATA_DRIVE_ROOT = "/content/drive/MyDrive/rcp_5th/data"
LOCAL_DATA_ROOT = "/content/local_data"

# 학습 중 체크포인트는 Local SSD에 저장 (Drive 직접 쓰기보다 ~10x 빠름)
# 학습 완료 후 Drive로 1회 복사
LOCAL_CKPT_PATH = "/content/svdd_best.pth"

# 정상 데이터 디렉토리만 명시 (abnormal, normal_3600rpm 제외)
NORMAL_DIRS = ["normal", "normal_1200rpm"]

if PRJ_PATH not in sys.path:
    sys.path.insert(0, PRJ_PATH)

from model_svdd import SVDDEncoder, compute_svdd_loss, compute_svdd_distances
from preprocess import prepare_1d_input
from infer_resnet_None import extract_rcp_xy_from_bin

FS       = 40_000
WIN_SIZE = FS           # 1초 윈도우
STEP     = FS // 10     # 90% 오버랩 (train_svdd.py 일치)

torch.backends.cudnn.benchmark = True


def _count_windows(bin_files: list) -> int:
    """np.stack 전 총 윈도우 수를 미리 계산 (GPU 사전 할당용)."""
    total = 0
    for bf in bin_files:
        try:
            rcp_xy = extract_rcp_xy_from_bin(bf, fs=FS)
            for _, (x_full, _) in rcp_xy.items():
                n = max(0, (len(x_full) - WIN_SIZE) // STEP + 1)
                total += n
        except Exception:
            pass
    return total


# ─────────────────────────────────────────────
# 2. Drive → Local 복사 (정상 데이터만)
# ─────────────────────────────────────────────
def copy_to_local() -> str:
    """
    Google Drive의 정상 데이터 디렉토리만 Local SSD로 복사.
    이미 복사된 경우 스킵.
    Returns: local raw 경로
    """
    local_raw = os.path.join(LOCAL_DATA_ROOT, "raw")

    if os.path.exists(local_raw):
        print("[Copy] 로컬 캐시 이미 존재, 복사 생략.")
        return local_raw

    os.makedirs(local_raw, exist_ok=True)
    print("[Copy] Google Drive -> Local SSD 복사 중 (정상 데이터만)...")

    for d in NORMAL_DIRS:
        src = os.path.join(DATA_DRIVE_ROOT, "raw", d)
        dst = os.path.join(local_raw, d)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            n_files = len(glob.glob(os.path.join(dst, "*.BIN")))
            print(f"  OK {d}: {n_files}개 BIN")
        else:
            print(f"  SKIP {d}: 디렉토리 없음 ({src})")

    print("[Copy] 복사 완료.")
    return local_raw


def collect_bin_files(local_raw: str) -> list:
    """정상 디렉토리에서만 BIN 파일 수집."""
    bin_files = []
    for d in NORMAL_DIRS:
        d_path = os.path.join(local_raw, d)
        found  = sorted(glob.glob(os.path.join(d_path, "*.BIN")))
        found += sorted(glob.glob(os.path.join(d_path, "*.bin")))
        bin_files.extend(found)
        print(f"  [{d}] {len(found)}개 BIN")
    return bin_files


# ─────────────────────────────────────────────
# 3. Dataset (메모리 캐싱 + 슬라이딩 윈도우)
# ─────────────────────────────────────────────
class CachedWindowDataset(Dataset):
    """
    전체 BIN 파일 -> 슬라이딩 윈도우 -> GPU VRAM 직접 캐싱.

    CPU RAM OOM 방지 전략:
      1단계: 총 윈도우 수를 미리 계산
      2단계: GPU에 빈 텐서 사전 할당 (N, 2, 40000)
      3단계: BIN 파일별로 읽어 GPU 텐서에 직접 기록 → CPU에 대형 배열 잔류 없음

    augment=True : torch.roll + torch.randn_like (GPU 연산)
    augment=False: 원본 그대로 (Center 계산/임계값 산출용)

    DataLoader는 num_workers=0 으로 설정 필요
    (데이터가 이미 GPU에 있어 IPC 불필요, pin_memory=False)
    """

    def __init__(self, bin_files: list, device: torch.device, augment: bool = True):
        self.augment = augment
        self.device  = device

        # ── 1단계: 총 윈도우 수 사전 계산 ────────────
        print(f"[Dataset] 윈도우 수 계산 중...")
        n_total_wins = _count_windows(bin_files)
        vram_gb = n_total_wins * 2 * WIN_SIZE * 4 / 1e9
        print(f"[Dataset] 총 {n_total_wins}개 윈도우 / 예상 VRAM {vram_gb:.2f} GB")

        # ── 2단계: GPU에 빈 텐서 사전 할당 ──────────
        self.samples = torch.empty(
            (n_total_wins, 2, WIN_SIZE), dtype=torch.float32, device=device
        )

        # ── 3단계: BIN 파일별 즉시 GPU 기록 ──────────
        print(f"[Dataset] GPU 직접 로딩 중 ({len(bin_files)}개 BIN)...")
        idx = 0
        for bf in tqdm(bin_files):
            try:
                rcp_xy = extract_rcp_xy_from_bin(bf, fs=FS)
                for _, (x_full, y_full) in rcp_xy.items():
                    n = len(x_full)
                    for s in range(0, n - WIN_SIZE + 1, STEP):
                        arr = prepare_1d_input(
                            x_full[s: s + WIN_SIZE],
                            y_full[s: s + WIN_SIZE],
                        )   # (2, 40000) float32, CPU numpy
                        self.samples[idx] = torch.from_numpy(arr)  # CPU → GPU 즉시 전송
                        idx += 1
                del rcp_xy
                gc.collect()
            except Exception as e:
                print(f"  WARNING: {os.path.basename(bf)} 건너뜀 ({e})")

        # 실제 기록된 수로 트리밍 (예외로 인한 공백 제거)
        self.samples = self.samples[:idx]
        print(f"[Dataset] GPU 캐싱 완료: {len(self.samples)}개 샘플 "
              f"({self.samples.element_size() * self.samples.nelement() / 1e9:.2f} GB VRAM)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 데이터가 이미 GPU에 있으므로 복사 최소화
        x = self.samples[idx].clone()
        if self.augment:
            x = self._augment_gpu(x)
        return x

    @staticmethod
    def _augment_gpu(x: torch.Tensor) -> torch.Tensor:
        """GPU 텐서 기반 증강 (np 연산 불필요)."""
        shift = int(torch.randint(-4000, 4001, (1,)).item())
        x = torch.roll(x, shift, dims=1)
        x = x + torch.randn_like(x) * 0.003
        return x


# ─────────────────────────────────────────────
# 4. 전이학습 초기화
# ─────────────────────────────────────────────
def load_pretrained_backbone(encoder: SVDDEncoder) -> bool:
    """
    OrbitCNN1D 체크포인트에서 features.* 레이어 가중치 전이학습.
    파일이 없으면 랜덤 초기화로 진행.
    """
    ckpt_path = os.path.join(PRJ_PATH, "model", "orbit_cnn1d.pth")
    if not os.path.exists(ckpt_path):
        print("[Transfer] orbit_cnn1d.pth 없음 -> 랜덤 초기화로 진행.")
        return False
    try:
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {k: v for k, v in state.items() if k.startswith("features.")}
        missing, unexpected = encoder.load_state_dict(backbone_state, strict=False)
        print(f"[Transfer] 전이학습 완료: {len(backbone_state)}개 레이어 로드, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        return True
    except Exception as e:
        print(f"[Transfer] 전이학습 실패 ({e}) -> 랜덤 초기화로 진행.")
        return False


# ─────────────────────────────────────────────
# 5. 학습 엔진
# ─────────────────────────────────────────────
def train_engine(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SVDD] Device: {device}")

    # ── 데이터 준비 ─────────────────────────────
    local_raw = copy_to_local()
    bin_files = collect_bin_files(local_raw)
    if not bin_files:
        raise RuntimeError("BIN 파일 없음 — NORMAL_DIRS 경로 확인 필요.")
    print(f"[SVDD] 정상 BIN 파일 총 {len(bin_files)}개")

    dataset = CachedWindowDataset(bin_files, device, augment=True)

    # Train / Val 분할 (샘플 단위 80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split   = int(len(indices) * 0.8)
    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    # 데이터가 이미 GPU에 있으므로 num_workers=0, pin_memory=False
    _loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )
    train_loader = DataLoader(train_subset, shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_subset,   shuffle=False, **_loader_kwargs)

    # ── 모델 초기화 ───────────────────────────
    encoder = SVDDEncoder(feature_dim=args.feature_dim).to(device)
    load_pretrained_backbone(encoder)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── 1단계: Warm-up ────────────────────────
    print(f"\n[SVDD] === 1단계: Warm-up ({args.warmup_epochs} epochs) ===")
    warmup_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr * 10)

    for ep in range(args.warmup_epochs):
        encoder.train()
        total_loss, n = 0.0, 0

        for x in train_loader:
            # 데이터가 이미 GPU에 있으므로 .to(device) 생략
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                feat   = encoder(x)
                feat_c = feat - feat.mean(0, keepdim=True)
                cov    = (feat_c.T @ feat_c) / max(feat.size(0) - 1, 1)
                loss   = (cov - torch.eye(feat.size(1), device=device)).pow(2).mean()

            warmup_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(warmup_opt)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            n          += x.size(0)

        print(f"  Warm-up [{ep+1}/{args.warmup_epochs}] cov_loss={total_loss/n:.6f}")

    # ── 2단계: Center 초기 계산 (augment=False) ────
    def _compute_center(enc, loader):
        """train_subset 전체 forward → feature mean 반환 (augment=False 상태에서 호출)."""
        enc.eval()
        feats = []
        with torch.no_grad():
            for x in loader:
                feats.append(enc(x).cpu())
        return torch.cat(feats, 0).mean(0).to(device)

    center_loader = DataLoader(train_subset, batch_size=args.batch_size,
                               num_workers=0, pin_memory=False)

    print("\n[SVDD] Center 초기 계산 중 (augment=False)...")
    dataset.augment = False
    center = _compute_center(encoder, center_loader)
    dataset.augment = True
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[SVDD] Center norm: {center.norm().item():.4f}")

    # ── 3단계: SVDD 학습 (Early Stopping) ─────
    print(f"\n[SVDD] === 3단계: SVDD 학습 "
          f"({args.epochs} epochs, patience={args.patience}) ===")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    drive_ckpt_path = os.path.join(PRJ_PATH, "model", "svdd_encoder.pth")
    best_val_loss   = float("inf")
    patience_cnt    = 0

    os.makedirs(os.path.dirname(drive_ckpt_path), exist_ok=True)

    for ep in range(args.epochs):
        # ── Center 재계산 (매 epoch, collapse 방지) ──
        dataset.augment = False
        center = _compute_center(encoder, center_loader)
        dataset.augment = True

        # Train
        encoder.train()
        train_loss, n_tr = 0.0, 0
        for x in train_loader:
            # 이미 GPU
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                feat = encoder(x)
                loss = compute_svdd_loss(feat, center)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            n_tr       += x.size(0)
        train_loss /= max(n_tr, 1)
        scheduler.step()

        # Validation
        encoder.eval()
        val_loss, n_va = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                # 이미 GPU
                feat = encoder(x)
                loss = compute_svdd_loss(feat, center)
                val_loss += loss.item() * x.size(0)
                n_va     += x.size(0)
        val_loss /= max(n_va, 1)

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch [{ep+1:3d}/{args.epochs}] "
              f"train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            # [Perf] Local SSD에 저장 (Drive 직접 쓰기 대비 ~10x 빠름)
            torch.save({
                "model_state_dict": encoder.state_dict(),
                "center":           center.cpu(),
                "feature_dim":      args.feature_dim,
                "epoch":            ep + 1,
                "val_loss":         val_loss,
            }, LOCAL_CKPT_PATH)
            print(f"  -> Best 저장 (epoch {ep+1}, local SSD)")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[SVDD] 조기 종료: val_loss 미개선 {args.patience} epochs "
                      f"-> 중단 (epoch {ep+1})")
                break

    # ── 4단계: 최적 가중치 복원 + Threshold 계산 ─
    print(f"\n[SVDD] === 4단계: Threshold 계산 "
          f"(p{args.percentile}, 전체 {len(bin_files)}개 파일) ===")
    ckpt_data = torch.load(LOCAL_CKPT_PATH, map_location=device)
    encoder.load_state_dict(ckpt_data["model_state_dict"])
    encoder.eval()

    # [Perf] 파일별 전체 윈도우를 배치로 묶어 단일 forward pass
    # 기존: 윈도우 1개씩 91회 추론 → 변경: 한 파일의 ~91 윈도우를 한 번에 추론
    file_max_dists = []
    with torch.no_grad():
        for bf in tqdm(bin_files, desc="Threshold calc"):
            try:
                rcp_xy = extract_rcp_xy_from_bin(bf, fs=FS)
                for _, (x_full, y_full) in rcp_xy.items():
                    n_total  = len(x_full)
                    windows  = []
                    for s in range(0, n_total - WIN_SIZE + 1, STEP):
                        arr = prepare_1d_input(
                            x_full[s: s + WIN_SIZE],
                            y_full[s: s + WIN_SIZE],
                        )
                        windows.append(arr)

                    if not windows:
                        continue

                    # 파일 내 모든 윈도우를 배치로 한 번에 추론
                    batch_t = torch.from_numpy(
                        np.stack(windows)          # (N_win, 2, 40000)
                    ).float().to(device, non_blocking=True)
                    feats = encoder(batch_t)        # (N_win, feature_dim)
                    dists = compute_svdd_distances(feats, center)  # (N_win,)
                    file_max_dists.append(dists.max().item())

                del rcp_xy
                gc.collect()
            except Exception as e:
                print(f"  WARNING: {os.path.basename(bf)} 건너뜀 ({e})")

    all_dists = np.array(file_max_dists)
    threshold = float(np.percentile(all_dists, args.percentile))
    print(f"[SVDD] 파일 max 분포: min={all_dists.min():.6f}  "
          f"mean={all_dists.mean():.6f}  max={all_dists.max():.6f}")
    print(f"[SVDD] Threshold (p{args.percentile}): {threshold:.6f}")

    # threshold를 체크포인트에 추가 후 Drive로 1회 복사
    ckpt_data["threshold"] = threshold
    drive_model_dir = os.path.dirname(drive_ckpt_path)
    os.makedirs(drive_model_dir, exist_ok=True)
    torch.save(ckpt_data, drive_ckpt_path)
    print(f"[SVDD] 체크포인트 Drive 복사 완료: {drive_ckpt_path}")

    # svdd_config.json 저장 (inference_daemon.py 호환)
    cfg_path = os.path.join(PRJ_PATH, "svdd_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "threshold":           threshold,
            "feature_dim":         args.feature_dim,
            "percentile":          args.percentile,
            "n_calibration_files": len(bin_files),
        }, f, indent=2)

    print(f"\n[SVDD] 학습 완료.")
    print(f"  모델 체크포인트 : {drive_ckpt_path}")
    print(f"  설정 파일       : {cfg_path}")
    print(f"  Best val_loss   : {best_val_loss:.6f}")


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
class Args:
    feature_dim   = 128
    epochs        = 50
    batch_size    = 128
    lr            = 1e-4
    warmup_epochs = 5
    patience      = 15      # Early Stopping patience
    percentile    = 95.0    # Threshold 산출 백분위 (train_svdd.py 일치)


args = Args()
train_engine(args)
