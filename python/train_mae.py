"""
train_mae.py
============
OrbitMAE 비지도 학습 스크립트.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
데이터: 정상 데이터만 사용
  data/raw/normal/          *.BIN
  data/raw/normal_1200rpm/  *.BIN
  (abnormal 데이터 불필요)

학습 방식:
  1D MAE + 스펙트로그램 MAE 공동 학습
  손실 = MSE(재구성, 원본) on 마스킹된 패치만

이상 임계값:
  학습 후 전체 학습 샘플의 이상 점수 분포에서 percentile 계산
  → mae_config.json에 저장

실행 예시:
  venv/Scripts/python.exe python/train_mae.py \\
      --data_dir ../data \\
      --epochs 100 \\
      --batch_size 16 \\
      --patience 15 \\
      --threshold_pct 95
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import _compat  # noqa: F401 — PyTorch/Windows/Python-3.11 호환성 패치 (torch import 전 실행)

import argparse
import glob as _glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_mae import OrbitMAE, MASK_RATIO
from preprocess import (
    FIXED_1D_SCALE_MIL,
    compute_dataset_scale,
    extract_xyz_triplets_legacy,
    make_spectrogram_4ch,
    parse_bin_legacy,
    prepare_1d_input_fixed,
    volt_to_mil,
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

FS = 40_000

# 정상 데이터 소스 (1200rpm만 사용 — raw/normal은 3600rpm 데이터 문제로 제외)
NORMAL_SOURCES = [
    ("raw/normal_1200rpm", "*.BIN"),
]

# 스크립트 위치 기준 데이터 경로 (python/ 디렉토리 기준 ../data)
_DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

_CKPT_PATH = os.path.join(SCRIPT_DIR, "model", "orbit_mae.pth")
_CFG_PATH  = os.path.join(SCRIPT_DIR, "mae_config.json")


# ─────────────────────────────────────────────
# 1. 데이터 수집
# ─────────────────────────────────────────────

def collect_bin_files(data_dir: str) -> list[str]:
    """정상 BIN 파일 경로 수집."""
    files: list[str] = []
    for subdir, pattern in NORMAL_SOURCES:
        d = os.path.join(data_dir, subdir)
        if not os.path.isdir(d):
            print(f"[MAE] WARNING: 디렉토리 없음 — {d}")
            continue
        found = sorted(_glob.glob(os.path.join(d, pattern)))
        files.extend(found)
        print(f"[MAE]   {subdir}: {len(found)}개 BIN")
    return files


def load_samples(
    bin_files:  list[str],
    scale_mil:  float,
    use_spec:   bool = True,
) -> list[dict]:
    """
    BIN → (x_1d, x_spec) 전처리.
    각 RCP 트리플렛을 독립 샘플로 분리.

    Returns: list of {'x_1d': np.float32 (2,L), 'x_spec': np.float32 (4,F,T) or None}
    """
    samples: list[dict] = []
    for bin_path in bin_files:
        try:
            data = parse_bin_legacy(bin_path, fs=FS)
            triplets = extract_xyz_triplets_legacy(data)
            for x_raw, y_raw, _ in triplets:
                x_mil, y_mil = volt_to_mil(x_raw, y_raw)

                # 9~10초 구간 (안정 운전)
                x_seg = x_mil[9 * FS : 10 * FS]
                y_seg = y_mil[9 * FS : 10 * FS]
                if len(x_seg) < FS:
                    continue

                x_1d  = prepare_1d_input_fixed(x_seg, y_seg, scale_mil)  # (2, L)
                x_spec = make_spectrogram_4ch(x_seg, y_seg, scale_mil) if use_spec else None

                samples.append({"x_1d": x_1d, "x_spec": x_spec})

        except Exception as e:
            print(f"[MAE] WARNING: {os.path.basename(bin_path)} 로드 실패 ({e})")

    print(f"[MAE] 총 {len(samples)}개 샘플 로드.")
    return samples


# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────

class MAEDataset(Dataset):
    """
    정상 신호 (x_1d, x_spec) 쌍 Dataset.

    augment=True 시 물리적으로 타당한 증강:
      ✅ 랜덤 원형 시프트: 위상 불변성 (시작 위치 무관)
      ✅ 약한 가우시안 노이즈: 센서 측정 잡음 모사
      ❌ 채널별 부호 반전: 와류 방향 소거 → 사용 안 함
      ❌ 진폭 스케일 지터: 절대 진폭 소거 → 사용 안 함
    """

    def __init__(self, samples: list[dict], augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.use_spec = samples[0]["x_spec"] is not None if samples else False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x_1d  = s["x_1d"].copy()                    # (2, L)
        x_spec = s["x_spec"].copy() if self.use_spec else None  # (4, F, T) or None

        if self.augment:
            shift = np.random.randint(-4000, 4001)
            x_1d = np.roll(x_1d, shift, axis=1)
            x_1d = x_1d + np.random.randn(*x_1d.shape).astype(np.float32) * 0.003
            # x_spec은 x_1d와 파생 관계지만 이미 전처리된 스펙트로그램이므로
            # 시프트 증강을 별도로 적용하지 않음 (패턴 불일치 방지)
            if x_spec is not None:
                x_spec = x_spec + np.random.randn(*x_spec.shape).astype(np.float32) * 0.01

        x_1d_t = torch.from_numpy(x_1d)

        if x_spec is not None:
            return x_1d_t, torch.from_numpy(x_spec)
        return x_1d_t, None


def _collate_fn(batch):
    """x_spec=None 인 경우를 처리하는 커스텀 collate."""
    x_1d_list  = [b[0] for b in batch]
    x_spec_list = [b[1] for b in batch]

    x_1d = torch.stack(x_1d_list)

    if x_spec_list[0] is None:
        return x_1d, None
    return x_1d, torch.stack(x_spec_list)


# ─────────────────────────────────────────────
# 3. 학습 루프
# ─────────────────────────────────────────────

def _run_epoch(
    model:     OrbitMAE,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
) -> tuple[float, float, float]:
    """
    단일 epoch 실행.
    Returns: (total_loss, loss_1d, loss_spec) — 배치 평균
    """
    training = optimizer is not None
    model.train(training)
    if not training:
        model.eval()

    total, l1d, lsp, n = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x_1d, x_spec in loader:
            x_1d = x_1d.to(device)
            if x_spec is not None:
                x_spec = x_spec.to(device)

            loss, loss_1d, loss_spec = model(x_1d, x_spec)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = x_1d.size(0)
            total += loss.item()      * bs
            l1d   += loss_1d.item()   * bs
            lsp   += loss_spec.item() * bs
            n += bs

    n = max(n, 1)
    return total / n, l1d / n, lsp / n


# ─────────────────────────────────────────────
# 4. 이상 임계값 계산
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_threshold(
    model:      OrbitMAE,
    loader:     DataLoader,
    device:     torch.device,
    percentile: float = 95.0,
    n_eval:     int   = 10,
) -> tuple[float, float, float]:
    """
    학습 세트 전체에서 이상 점수를 계산하여 임계값을 산출합니다.

    Returns:
        threshold    : percentile 기반 임계값
        score_mean   : 학습 데이터 점수 평균
        score_std    : 학습 데이터 점수 표준편차
    """
    model.eval()
    all_scores: list[torch.Tensor] = []

    for x_1d, x_spec in loader:
        x_1d = x_1d.to(device)
        if x_spec is not None:
            x_spec = x_spec.to(device)
        scores = model.anomaly_score(x_1d, x_spec, n_eval=n_eval)
        all_scores.append(scores.cpu())

    scores_np = torch.cat(all_scores).numpy()
    threshold = float(np.percentile(scores_np, percentile))
    return threshold, float(scores_np.mean()), float(scores_np.std())


# ─────────────────────────────────────────────
# 5. 체크포인트
# ─────────────────────────────────────────────

def _save_checkpoint(model: OrbitMAE, meta: dict) -> None:
    torch.save({"model_state_dict": model.state_dict(), **meta}, _CKPT_PATH)
    print(f"  → 저장: {_CKPT_PATH}  {meta}")


def _restore_checkpoint(model: OrbitMAE, device: torch.device) -> dict:
    if not os.path.exists(_CKPT_PATH):
        return {}
    ckpt = torch.load(_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("  ← 최적 체크포인트 복원.")
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


# ─────────────────────────────────────────────
# 6. 학습 진입점
# ─────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAE] device: {device}")

    # ── data_dir 경로 정규화 ──────────────────────
    # 상대 경로는 CWD 기준으로 먼저 시도, 없으면 스크립트 위치 기준으로 재시도
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        alt = os.path.normpath(os.path.join(SCRIPT_DIR, args.data_dir))
        if os.path.isdir(alt):
            data_dir = alt
            print(f"[MAE] data_dir 재해석: {data_dir}")

    # ── 파일 수집 ─────────────────────────────────
    bin_files = collect_bin_files(data_dir)
    if not bin_files:
        raise RuntimeError(f"정상 BIN 파일 없음: {data_dir}")

    # ── 고정 스케일 계산 ──────────────────────────
    if args.scale_mil > 0.0:
        scale_mil = args.scale_mil
        print(f"[MAE] 스케일 (CLI): {scale_mil:.3f} mil")
    else:
        print("[MAE] 스케일 자동 계산 중...")
        xy_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for bp in bin_files:
            try:
                data = parse_bin_legacy(bp, fs=FS)
                for x_r, y_r, _ in extract_xyz_triplets_legacy(data):
                    xm, ym = volt_to_mil(x_r, y_r)
                    xy_pairs.append((xm[9*FS:10*FS], ym[9*FS:10*FS]))
            except Exception:
                pass
        scale_mil = compute_dataset_scale(xy_pairs) if xy_pairs else FIXED_1D_SCALE_MIL
        print(f"[MAE] 계산된 스케일: {scale_mil:.3f} mil")

    # ── 샘플 전처리 ───────────────────────────────
    print("[MAE] 전처리 중...")
    t0 = time.time()
    all_samples = load_samples(bin_files, scale_mil, use_spec=not args.no_spec)
    print(f"[MAE] 전처리 완료: {time.time()-t0:.1f}s / {len(all_samples)}개")

    if len(all_samples) < 4:
        raise RuntimeError(
            f"샘플 수 부족 ({len(all_samples)}개). BIN 파일/경로를 확인하세요."
        )

    # ── Train / Val 분할 (80/20) ──────────────────
    idx = np.arange(len(all_samples))
    idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=42)
    train_samples = [all_samples[i] for i in idx_tr]
    val_samples   = [all_samples[i] for i in idx_val]
    print(f"[MAE] train={len(train_samples)}, val={len(val_samples)}")

    # ── DataLoader ────────────────────────────────
    loader_tr  = DataLoader(
        MAEDataset(train_samples, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    loader_val = DataLoader(
        MAEDataset(val_samples, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_fn,
    )
    # 임계값 계산용: 전체 학습 샘플 (증강 없음)
    loader_full = DataLoader(
        MAEDataset(train_samples, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_fn,
    )

    # ── 모델 ──────────────────────────────────────
    use_spec = not args.no_spec
    model = OrbitMAE(use_spec=use_spec).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MAE] 파라미터: {n_params:,} "
          f"(1D={sum(p.numel() for p in model.branch_1d.parameters()):,}"
          + (f", spec={sum(p.numel() for p in model.branch_spec.parameters()):,})"
             if use_spec else ")"))

    os.makedirs(os.path.join(SCRIPT_DIR, "model"), exist_ok=True)

    # ── 최적화기 / 스케줄러 ───────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # 워밍업 + 코사인 감소 스케줄
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

    best_val_loss = float("inf")
    patience_cnt  = 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_1d, tr_sp = _run_epoch(model, loader_tr,  optimizer, device)
        va_loss, va_1d, va_sp = _run_epoch(model, loader_val, None,      device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if use_spec:
            print(f"  [{ep:4d}/{args.epochs}] "
                  f"tr={tr_loss:.5f}(1d={tr_1d:.5f} sp={tr_sp:.5f})  "
                  f"va={va_loss:.5f}(1d={va_1d:.5f} sp={va_sp:.5f})  "
                  f"lr={lr_now:.2e}")
        else:
            print(f"  [{ep:4d}/{args.epochs}] "
                  f"tr={tr_loss:.5f}  va={va_loss:.5f}  lr={lr_now:.2e}")

        if va_loss < best_val_loss - 1e-7:
            best_val_loss = va_loss
            patience_cnt  = 0
            _save_checkpoint(model, {
                "epoch": ep, "val_loss": va_loss, "scale_mil": scale_mil,
            })
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[MAE] 조기 종료 "
                      f"(val_loss 미개선 {args.patience} epochs, best={best_val_loss:.6f})")
                break

    # ── 최적 체크포인트 복원 ──────────────────────
    meta = _restore_checkpoint(model, device)
    print(f"[MAE] 최적 val_loss = {meta.get('val_loss', best_val_loss):.6f}  "
          f"(epoch {meta.get('epoch', '?')})")

    # ── 이상 임계값 계산 ──────────────────────────
    print(f"\n[MAE] 이상 임계값 계산 중 "
          f"(n_eval={args.n_eval}, percentile={args.threshold_pct})...")
    threshold, sc_mean, sc_std = compute_threshold(
        model, loader_full, device,
        percentile=args.threshold_pct,
        n_eval=args.n_eval,
    )
    print(f"[MAE] 학습 세트 이상 점수: mean={sc_mean:.6f}  std={sc_std:.6f}")
    print(f"[MAE] 임계값 (p{args.threshold_pct:.0f}): {threshold:.6f}")

    # ── 설정 저장 ─────────────────────────────────
    cfg = {
        "scale_mil":      scale_mil,
        "mask_ratio":     MASK_RATIO,
        "threshold":      threshold,
        "score_mean":     sc_mean,
        "score_std":      sc_std,
        "threshold_pct":  args.threshold_pct,
        "use_spec":       use_spec,
        "alpha":          0.5,        # anomaly_score alpha (1d vs spec 가중치)
        "n_eval":         args.n_eval,
        "val_loss":       float(meta.get("val_loss", best_val_loss)),
    }
    with open(_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[MAE] 완료.")
    print(f"  체크포인트 : {_CKPT_PATH}")
    print(f"  설정 파일  : {_CFG_PATH}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OrbitMAE 비지도 학습 (정상 데이터 전용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",      type=str,   default=_DEFAULT_DATA_DIR,
                   help="데이터 루트 디렉토리 (기본값: 스크립트 위치 기준 ../data)")
    p.add_argument("--epochs",        type=int,   default=100,
                   help="최대 학습 epoch")
    p.add_argument("--batch_size",    type=int,   default=16,
                   help="배치 크기")
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="AdamW 학습률 (MAE는 낮은 LR 권장)")
    p.add_argument("--patience",      type=int,   default=15,
                   help="조기 종료 patience epoch 수")
    p.add_argument("--scale_mil",     type=float, default=0.0,
                   help="고정 스케일 [mil]. 0이면 자동 계산.")
    p.add_argument("--threshold_pct", type=float, default=95.0,
                   help="이상 임계값 percentile (95 권장, 엄격하게는 99)")
    p.add_argument("--n_eval",        type=int,   default=10,
                   help="임계값 계산 시 Monte Carlo 마스크 반복 횟수")
    p.add_argument("--no_spec",       action="store_true", default=True,
                   help="스펙트로그램 브랜치 비활성화 (기본값: 비활성화)")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
