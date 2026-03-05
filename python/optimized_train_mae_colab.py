"""
optimized_train_mae_colab.py
=====================================
Google Colab GPU 환경용 OrbitMAE 비지도 학습 스크립트.

SVDD Colab 코드와 동일한 최적화 적용:
  - [Perf] Google Drive -> Local SSD 복사 후 학습 (I/O 병목 제거)
  - [Perf] 전체 데이터 GPU VRAM 직접 캐싱 (CPU RAM OOM 방지)
    * np.stack → GPU 사전 할당 텐서에 창별 즉시 기록
  - [Perf] AMP (GradScaler + autocast)
  - [Perf] num_workers=0, pin_memory=False (데이터 이미 GPU)
  - [Perf] Stage 4 임계값 계산: 파일별 전체 윈도우 배치 추론
  - [Perf] 체크포인트 Local SSD 저장 -> 완료 후 Drive 1회 복사
  - [Perf] zero_grad(set_to_none=True)
  - [Fix]  scale_mil 계산: sec9 편향 제거 (전체 신호에서 균등 샘플링)

Drive 구조 (필수):
  MyDrive/rcp_5th/
  ├── python/
  │   ├── model_mae.py
  │   ├── preprocess.py
  │   └── model/          <- 없으면 자동 생성
  └── data/
      └── raw/
          └── normal_1200rpm/  *.BIN

Colab 런타임: GPU (T4 이상 권장)
실행:
  shutil.copy(".../optimized_train_mae_colab.py", "/content/optimized_train_mae_colab.py")
  %run /content/optimized_train_mae_colab.py
"""

import gc
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. 환경 설정
# ─────────────────────────────────────────────
PRJ_PATH        = "/content/drive/MyDrive/rcp_5th/python"
DATA_DRIVE_ROOT = "/content/drive/MyDrive/rcp_5th/data"
LOCAL_DATA_ROOT = "/content/local_data"

# train_mae.py 기준: normal_1200rpm 만 사용
# (raw/normal은 3600rpm 데이터 품질 문제로 제외)
NORMAL_DIRS = ["normal_1200rpm"]

# 학습 중 체크포인트는 Local SSD에 저장, 완료 후 Drive로 1회 복사
LOCAL_CKPT_PATH = "/content/mae_best.pth"

if PRJ_PATH not in sys.path:
    sys.path.insert(0, PRJ_PATH)

from model_mae import OrbitMAE, MASK_RATIO
from preprocess import (
    FIXED_1D_SCALE_MIL,
    compute_dataset_scale,
    extract_xyz_triplets_legacy,
    parse_bin_legacy,
    prepare_1d_input_fixed,
    volt_to_mil,
)

FS       = 40_000
WIN_SIZE = FS           # 1초 윈도우
STEP     = FS // 10     # 90% 오버랩 (train_mae.py 일치)

torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────
# 2. Drive → Local 복사 (정상 데이터만)
# ─────────────────────────────────────────────
def copy_to_local() -> str:
    local_raw = os.path.join(LOCAL_DATA_ROOT, "raw")

    if os.path.exists(local_raw):
        print("[Copy] 로컬 캐시 이미 존재, 복사 생략.")
        return local_raw

    os.makedirs(local_raw, exist_ok=True)
    print("[Copy] Google Drive -> Local SSD 복사 중...")

    for d in NORMAL_DIRS:
        src = os.path.join(DATA_DRIVE_ROOT, "raw", d)
        dst = os.path.join(local_raw, d)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            n = len(glob.glob(os.path.join(dst, "*.BIN")))
            print(f"  OK {d}: {n}개 BIN")
        else:
            print(f"  SKIP {d}: 디렉토리 없음 ({src})")

    print("[Copy] 복사 완료.")
    return local_raw


def collect_bin_files(local_raw: str) -> list:
    bin_files = []
    for d in NORMAL_DIRS:
        found  = sorted(glob.glob(os.path.join(local_raw, d, "*.BIN")))
        found += sorted(glob.glob(os.path.join(local_raw, d, "*.bin")))
        bin_files.extend(found)
        print(f"  [{d}] {len(found)}개 BIN")
    return bin_files


def _count_windows_mae(bin_files: list) -> int:
    """GPU 사전 할당을 위한 총 윈도우 수 계산."""
    total = 0
    for bf in bin_files:
        try:
            data = parse_bin_legacy(bf, fs=FS)
            for x_r, _, _ in extract_xyz_triplets_legacy(data):
                n = max(0, (len(x_r) - WIN_SIZE) // STEP + 1)
                total += n
        except Exception:
            pass
    return total


# ─────────────────────────────────────────────
# 3. scale_mil 계산 (sec9 편향 제거)
# ─────────────────────────────────────────────
def compute_scale_unbiased(bin_files: list, n_files: int = 20) -> float:
    """
    전체 신호에서 균등 샘플링하여 scale_mil 계산.
    기존 train_mae.py의 sec9 고정 참조 버그를 수정한 버전.

    전체 10초 신호를 1초 단위로 분할하여 고르게 샘플링하므로
    특정 구간 편향 없이 데이터셋 전체의 진폭 분포를 반영.
    """
    xy_pairs = []
    sample_files = bin_files[:n_files]

    print(f"[Scale] {len(sample_files)}개 파일에서 scale_mil 계산 중...")
    for bp in sample_files:
        try:
            data = parse_bin_legacy(bp, fs=FS)
            for x_r, y_r, _ in extract_xyz_triplets_legacy(data):
                xm, ym = volt_to_mil(x_r, y_r)
                n = len(xm)
                # 전체 신호를 1초 단위 비겹침 구간으로 분할 → 균등 대표
                for s in range(0, n - FS + 1, FS):
                    xy_pairs.append((xm[s: s + FS], ym[s: s + FS]))
        except Exception as e:
            print(f"  WARNING: {os.path.basename(bp)} 건너뜀 ({e})")

    scale = compute_dataset_scale(xy_pairs) if xy_pairs else FIXED_1D_SCALE_MIL
    print(f"[Scale] scale_mil = {scale:.4f} mil ({len(xy_pairs)}개 구간 기반)")
    return scale


# ─────────────────────────────────────────────
# 4. Dataset (GPU VRAM 직접 캐싱 + 슬라이딩 윈도우)
# ─────────────────────────────────────────────
class CachedMAEDataset(Dataset):
    """
    전체 BIN -> 슬라이딩 윈도우 -> GPU VRAM 직접 캐싱.

    CPU RAM OOM 방지 전략 (SVDD Colab과 동일):
      1단계: 총 윈도우 수 사전 계산
      2단계: GPU에 빈 텐서 사전 할당 (N, 2, 40000)
      3단계: BIN 파일별 즉시 GPU 기록 → CPU 대형 배열 잔류 없음

    DataLoader 설정: num_workers=0, pin_memory=False 필수
    """

    def __init__(self, bin_files: list, scale_mil: float,
                 device: torch.device, augment: bool = True):
        self.scale_mil = scale_mil
        self.augment   = augment
        self.device    = device

        # ── 1단계: 총 윈도우 수 사전 계산 ────────────
        print("[Dataset] 윈도우 수 계산 중...")
        n_total_wins = _count_windows_mae(bin_files)
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
                data = parse_bin_legacy(bf, fs=FS)
                for x_r, y_r, _ in extract_xyz_triplets_legacy(data):
                    xm, ym  = volt_to_mil(x_r, y_r)
                    n_total = len(xm)
                    for s in range(0, n_total - WIN_SIZE + 1, STEP):
                        arr = prepare_1d_input_fixed(
                            xm[s: s + WIN_SIZE],
                            ym[s: s + WIN_SIZE],
                            scale_mil,
                        )   # (2, 40000) float32, CPU numpy
                        self.samples[idx] = torch.from_numpy(arr)
                        idx += 1
                del data
                gc.collect()
            except Exception as e:
                print(f"  WARNING: {os.path.basename(bf)} 건너뜀 ({e})")

        self.samples = self.samples[:idx]
        print(f"[Dataset] GPU 캐싱 완료: {len(self.samples)}개 샘플 "
              f"({self.samples.element_size() * self.samples.nelement() / 1e9:.2f} GB VRAM)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.samples[idx].clone()
        if self.augment:
            x = self._augment_gpu(x)
        return x

    @staticmethod
    def _augment_gpu(x: torch.Tensor) -> torch.Tensor:
        """GPU 텐서 기반 증강."""
        shift = int(torch.randint(-4000, 4001, (1,)).item())
        x = torch.roll(x, shift, dims=1)
        x = x + torch.randn_like(x) * 0.003
        return x


class _GPUSplitDataset(Dataset):
    """
    원본 GPU 텐서를 복사 없이 공유 참조 + 인덱스 배열로 분할.

    fancy index(복사) 대신 __getitem__ 시 단일 인덱스 접근 → peak VRAM = 1x (복사본 없음)
    """

    def __init__(self, full_tensor: torch.Tensor,
                 indices: np.ndarray, augment: bool):
        self.data    = full_tensor                                  # 원본 참조 (복사 없음)
        self.indices = torch.as_tensor(indices, dtype=torch.long)  # CPU에 보관
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[self.indices[idx]].clone()
        if self.augment:
            x = CachedMAEDataset._augment_gpu(x)
        return x


# ─────────────────────────────────────────────
# 5. 학습 루프
# ─────────────────────────────────────────────
def _run_epoch(
    model:     OrbitMAE,
    loader:    DataLoader,
    optimizer,
    scaler,
    device:    torch.device,
) -> tuple:
    """
    단일 epoch 실행.
    optimizer=None 이면 validation 모드.
    Returns: (total_loss, loss_1d, loss_spec) — 샘플 수 기준 평균
    """
    training = optimizer is not None
    model.train(training)
    if not training:
        model.eval()

    total, l1d, lsp, n = 0.0, 0.0, 0.0, 0
    nan_skipped = 0

    with torch.set_grad_enabled(training):
        for x_1d in loader:
            # 데이터가 이미 GPU에 있으므로 .to(device) 생략
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss, loss_1d, loss_spec = model(x_1d, x_spec=None)

            if not torch.isfinite(loss):
                nan_skipped += 1
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                has_nan = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if has_nan:
                    nan_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            bs     = x_1d.size(0)
            total += loss.item()      * bs
            l1d   += loss_1d.item()   * bs
            lsp   += loss_spec.item() * bs
            n     += bs

    if nan_skipped:
        print(f"    [NaN skip] {nan_skipped}개 배치 건너뜀")

    n = max(n, 1)
    return total / n, l1d / n, lsp / n


# ─────────────────────────────────────────────
# 6. Threshold 계산 (배치 추론 최적화)
# ─────────────────────────────────────────────
@torch.no_grad()
def compute_threshold_batched(
    model:      OrbitMAE,
    bin_files:  list,
    scale_mil:  float,
    device:     torch.device,
    percentile: float = 90.0,
    n_eval:     int   = 10,
) -> tuple:
    """
    전체 BIN 파일 슬라이딩 윈도우 2단계 평가로 임계값 계산.

    Stage 1 (배치 추론):
      파일별 전체 윈도우를 한 번에 forward_masked (n_eval=1)
      → 최고 점수 윈도우 선정
    Stage 2:
      최고 점수 윈도우에 n_eval=10 Monte Carlo → 최종 점수

    기존 방식 대비: Stage 1에서 윈도우 1개씩 추론 → 전체 배치 추론 (~91x 단축)
    """
    model.eval()
    file_scores = []

    for bf in tqdm(bin_files, desc="Threshold calc"):
        try:
            data = parse_bin_legacy(bf, fs=FS)
            for x_r, y_r, _ in extract_xyz_triplets_legacy(data):
                xm, ym  = volt_to_mil(x_r, y_r)
                n_total = len(xm)
                starts  = list(range(0, n_total - WIN_SIZE + 1, STEP))
                if not starts:
                    continue

                # ── Stage 1: 전체 윈도우 배치 추론 ──────────
                windows = []
                for s in starts:
                    w = prepare_1d_input_fixed(
                        xm[s: s + WIN_SIZE],
                        ym[s: s + WIN_SIZE],
                        scale_mil,
                    )
                    windows.append(w)

                batch_t = torch.from_numpy(
                    np.stack(windows)           # (N_win, 2, 40000)
                ).float().to(device, non_blocking=True)

                # forward_masked 1회 = n_eval=1 에 해당
                _, per_sample, _ = model.branch_1d.forward_masked(batch_t)
                win_scores = per_sample.cpu().numpy()   # (N_win,)

                # ── Stage 2: 최고 점수 윈도우 → n_eval 반복 ─
                best_i  = int(np.argmax(win_scores))
                best_t  = batch_t[best_i: best_i + 1]  # (1, 2, 40000)
                final   = model.branch_1d.anomaly_score(best_t, n_eval=n_eval).item()
                file_scores.append(final)

        except Exception as e:
            print(f"  WARNING: {os.path.basename(bf)} 건너뜀 ({e})")
        gc.collect()

    scores_np = np.array(file_scores, dtype=np.float32)
    threshold = float(np.percentile(scores_np, percentile))
    return threshold, float(scores_np.mean()), float(scores_np.std())


# ─────────────────────────────────────────────
# 7. 학습 엔진
# ─────────────────────────────────────────────
def train_engine(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAE] Device: {device}")

    # ── 데이터 준비 ─────────────────────────────
    local_raw = copy_to_local()
    bin_files = collect_bin_files(local_raw)
    if not bin_files:
        raise RuntimeError("BIN 파일 없음 — NORMAL_DIRS 경로 확인 필요.")
    print(f"[MAE] 정상 BIN 파일 총 {len(bin_files)}개")

    # ── scale_mil 계산 (sec9 버그 수정 버전) ────
    if args.scale_mil > 0.0:
        scale_mil = args.scale_mil
        print(f"[MAE] 스케일 (CLI 고정): {scale_mil:.4f} mil")
    else:
        scale_mil = compute_scale_unbiased(bin_files)

    # ── 데이터셋 전처리 및 GPU 캐싱 ─────────────
    print("[MAE] 전처리 및 GPU 직접 캐싱 중...")
    t0 = time.time()
    dataset = CachedMAEDataset(bin_files, scale_mil, device, augment=True)
    print(f"[MAE] 전처리 완료: {time.time()-t0:.1f}s")

    if len(dataset) < 4:
        raise RuntimeError(f"샘플 수 부족 ({len(dataset)}개)")

    # ── Train / Val 분할 (원본 텐서 공유 참조, 복사 없음) ──
    idx_all         = np.arange(len(dataset))
    idx_tr, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42)
    print(f"[MAE] train={len(idx_tr)}, val={len(idx_val)}")

    # 원본 텐서를 공유 참조 → peak VRAM = 1x (fancy index 복사본 생성 없음)
    full_tensor = dataset.samples
    del dataset   # Dataset 객체 해제 (텐서는 full_tensor가 참조하므로 유지)
    gc.collect()

    train_ds = _GPUSplitDataset(full_tensor, idx_tr,  augment=True)
    val_ds   = _GPUSplitDataset(full_tensor, idx_val, augment=False)

    # 데이터가 이미 GPU에 있으므로 num_workers=0, pin_memory=False
    _loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_loader_kw)

    # ── 모델 ─────────────────────────────────────
    model = OrbitMAE(use_spec=False, alpha=args.alpha).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MAE] 파라미터: {n_params:,}")

    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-5
    )

    # 워밍업 + 코사인 감소 스케줄러
    warmup_epochs = max(1, args.epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_epochs),
                eta_min=args.lr * 1e-2,
            ),
        ],
        milestones=[warmup_epochs],
    )

    # ── 학습 루프 ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[MAE] 학습 시작  epochs={args.epochs}  "
          f"warmup={warmup_epochs}  patience={args.patience}")
    print(f"{'='*60}")

    drive_ckpt_path = os.path.join(PRJ_PATH, "model", "orbit_mae.pth")
    os.makedirs(os.path.dirname(drive_ckpt_path), exist_ok=True)

    best_val_loss = float("inf")
    patience_cnt  = 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_1d, _ = _run_epoch(model, train_loader, optimizer, scaler, device)
        va_loss, va_1d, _ = _run_epoch(model, val_loader,   None,      scaler, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        print(f"  [{ep:4d}/{args.epochs}] "
              f"tr={tr_loss:.5f}(1d={tr_1d:.5f})  "
              f"va={va_loss:.5f}(1d={va_1d:.5f})  "
              f"lr={lr_now:.2e}")

        if va_loss < best_val_loss - 1e-7:
            best_val_loss = va_loss
            patience_cnt  = 0
            # [Perf] Local SSD에 저장
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            ep,
                "val_loss":         va_loss,
                "scale_mil":        scale_mil,
            }, LOCAL_CKPT_PATH)
            print(f"  -> Best 저장 (epoch {ep}, local SSD)")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[MAE] 조기 종료: val_loss 미개선 {args.patience} epochs "
                      f"-> 중단 (epoch {ep})")
                break

    # ── 최적 가중치 복원 ──────────────────────────
    print("\n[MAE] 최적 체크포인트 복원...")
    ckpt_meta = torch.load(LOCAL_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt_meta["model_state_dict"])
    print(f"[MAE] 최적 val_loss = {ckpt_meta['val_loss']:.6f}  "
          f"(epoch {ckpt_meta['epoch']})")

    # ── 임계값 계산 (배치 추론 최적화) ────────────
    print(f"\n[MAE] 임계값 계산 중 "
          f"(n_eval={args.n_eval}, p{args.threshold_pct}, "
          f"전체 {len(bin_files)}개 파일)...")
    threshold, sc_mean, sc_std = compute_threshold_batched(
        model, bin_files, scale_mil, device,
        percentile=args.threshold_pct,
        n_eval=args.n_eval,
    )
    print(f"[MAE] 학습 세트 이상 점수: mean={sc_mean:.6f}  std={sc_std:.6f}")
    print(f"[MAE] 임계값 (p{args.threshold_pct:.0f}): {threshold:.6f}")

    # ── Drive로 1회 복사 ──────────────────────────
    shutil.copy(LOCAL_CKPT_PATH, drive_ckpt_path)
    print(f"[MAE] 체크포인트 Drive 복사 완료: {drive_ckpt_path}")

    # ── mae_config.json 저장 ──────────────────────
    cfg_path = os.path.join(PRJ_PATH, "mae_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "scale_mil":     scale_mil,
            "mask_ratio":    MASK_RATIO,
            "threshold":     threshold,
            "score_mean":    sc_mean,
            "score_std":     sc_std,
            "threshold_pct": args.threshold_pct,
            "use_spec":      False,
            "alpha":         args.alpha,
            "n_eval":        args.n_eval,
            "val_loss":      float(ckpt_meta["val_loss"]),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[MAE] 학습 완료.")
    print(f"  모델 체크포인트 : {drive_ckpt_path}")
    print(f"  설정 파일       : {cfg_path}")
    print(f"  Best val_loss   : {best_val_loss:.6f}")


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
class Args:
    scale_mil     = 0.0     # 0.0 = 자동 계산 (전체 신호 기반, sec9 버그 수정)
                            # 고정값 사용 시 예: 1.42 (mae_config.json 기존값)
    epochs        = 100
    batch_size    = 128
    lr            = 1e-4
    patience      = 15
    threshold_pct = 90.0    # 이상 임계값 백분위 (train_mae.py 기본값 일치)
    n_eval        = 10      # Monte Carlo 마스크 반복 횟수
    alpha         = 0.5     # 통합 이상 점수 1D 가중치 (use_spec=False → 무관)


args = Args()
train_engine(args)

