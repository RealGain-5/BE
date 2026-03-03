"""
train_fusion.py
===============
OrbitFusionNet 3단계 학습 스크립트.

데이터 소스:
  data/
    raw/
      normal/          *.BIN  → label 0 (정상)
      normal_1200rpm/  *.BIN  → label 0 (정상)
      abnormal/        *.BIN  → label 1 (비정상)

3단계 학습 전략:
  Stage 1a — AE 재구성 (비지도, 정상 데이터만):
    stream_time (OrbitAE1D) 학습: L_rec = MSE(x_rec, x_fixed_scale)
    정상 신호의 컴팩트 표현 학습. 재구성 오차가 이상 점수로 사용됨.

  Stage 1b — 분류 미세조정 (지도, 정상+이상):
    stream_time 동결 + stream_freq (SpectrogramCNN) + stream_axial (OrbitAE1D 3ch) 학습.
    fusion_head를 함께 학습하여 스트림 간 정렬을 유도.

  Stage 2 — Fusion 최종 학습 (지도, 정상+이상):
    스트림 매우 낮은 LR(1e-5) + fusion_head 높은 LR(1e-3) end-to-end 학습.
    선택: --freeze_streams 플래그 사용 시 스트림 동결, fusion_head만 학습.

실행 예시:
  venv/Scripts/python.exe python/train_fusion.py \\
      --data_dir ../data \\
      --epochs_1a 30 \\
      --epochs_1b 30 \\
      --epochs_2  40 \\
      --batch_size 8 \\
      --patience 10
"""

import _compat  # noqa: F401 — PyTorch/Windows/Python-3.11 호환성 패치

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_1d_ae import OrbitAE1D
from model_fusion import OrbitFusionNet, build_stage2_optimizer
from preprocess import (
    FIXED_1D_SCALE_MIL,
    compute_dataset_scale,
    extract_xyz_triplets_legacy,
    make_spectrogram_4ch,
    parse_bin_legacy,
    prepare_1d_input_fixed,
    prepare_3ch_input_fixed,
    volt_to_mil,
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

FS = 40_000

# 데이터 소스 (train_multiscale.py 와 동일 구조 유지)
TRAIN_SOURCES = [
    ("raw/normal",         0, "*.BIN"),
    ("raw/normal_1200rpm", 0, "*.BIN"),
    # raw/normal_3600rpm 제외 — 데이터 품질 문제
    ("raw/abnormal",       1, "*.BIN"),
]


# ─────────────────────────────────────────────
# 1. 데이터 수집 및 전처리
# ─────────────────────────────────────────────

def collect_bin_files(data_dir: str) -> list[tuple[str, int]]:
    """BIN 파일 경로와 라벨을 수집한다."""
    items: list[tuple[str, int]] = []
    for subdir, label, pattern in TRAIN_SOURCES:
        d = os.path.join(data_dir, subdir)
        if not os.path.isdir(d):
            print(f"[Fusion] WARNING: 디렉토리 없음 — {d}")
            continue
        files = sorted(_glob.glob(os.path.join(d, pattern)))
        items.extend((f, label) for f in files)
        print(f"[Fusion]   {subdir}: {len(files)}개 BIN")
    return items


def load_samples(
    bin_items: list[tuple[str, int]],
    scale_mil: float,
) -> list[dict]:
    """
    BIN 파일에서 RCP별 트리플렛을 추출하고
    고정 스케일 전처리를 적용하여 샘플 딕셔너리 리스트를 반환한다.

    각 샘플: {
        'x1d':  np.float32 (2, L)   — stream_time 입력 (X, Y 2채널)
        'x3ch': np.float32 (3, L)   — stream_axial 입력 (X, Y, Z 3채널)
        'spec': np.float32 (4, F, T)— stream_freq 입력 (4채널 스펙트로그램)
        'label': int                — 0=normal, 1=abnormal
    }
    """
    samples: list[dict] = []
    for bin_path, label in bin_items:
        try:
            data = parse_bin_legacy(bin_path, fs=FS)
            triplets = extract_xyz_triplets_legacy(data)
            for x_raw, y_raw, z_raw in triplets:
                # DC 제거 + 단위 변환 (volt → mil)
                x_mil, y_mil = volt_to_mil(x_raw, y_raw)
                if z_raw is not None:
                    z_mil = (z_raw - z_raw.mean()) * 10.0
                else:
                    z_mil = None

                # 9~10초 구간 (안정 운전 구간)
                x_seg = x_mil[9 * FS : 10 * FS]
                y_seg = y_mil[9 * FS : 10 * FS]
                z_seg = z_mil[9 * FS : 10 * FS] if z_mil is not None else None

                if len(x_seg) < FS:
                    continue

                samples.append({
                    "x1d":   prepare_1d_input_fixed(x_seg, y_seg, scale_mil),
                    "x3ch":  prepare_3ch_input_fixed(x_seg, y_seg, z_seg, scale_mil),
                    "spec":  make_spectrogram_4ch(x_seg, y_seg, scale_mil),
                    "label": int(label),
                })
        except Exception as e:
            print(f"[Fusion] WARNING: {os.path.basename(bin_path)} 로드 실패 ({e})")

    print(f"[Fusion] 총 {len(samples)}개 샘플 로드.")
    return samples


# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    (x1d, x3ch, spec, label) 튜플을 반환하는 Dataset.

    augment=True 시 물리적으로 타당한 시계열 증강 적용:
      ✅ 랜덤 원형 시프트: 위상 불변성 모사
      ✅ 약한 가우시안 노이즈: 센서 측정 잡음 모사
      ❌ 부호 반전 (채널별 flip): 와류 방향 정보 소거 → 사용 안 함
      ❌ 진폭 스케일 지터: 절대 진폭이 핵심 특징 → 사용 안 함
    """

    def __init__(self, samples: list[dict], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x1d  = s["x1d"].copy()   # (2, L)
        x3ch = s["x3ch"].copy()  # (3, L)
        spec = s["spec"].copy()  # (4, F, T)

        if self.augment:
            # 랜덤 원형 시프트: x1d / x3ch 동일 시프트 (정렬 유지)
            shift = np.random.randint(-4000, 4001)
            x1d  = np.roll(x1d,  shift, axis=1)
            x3ch = np.roll(x3ch, shift, axis=1)
            # 약한 가우시안 노이즈 (σ = 0.003, scale=3.0 기준 ≈ 0.009 mil)
            noise = 0.003
            x1d  = x1d  + np.random.randn(*x1d.shape ).astype(np.float32) * noise
            x3ch = x3ch + np.random.randn(*x3ch.shape).astype(np.float32) * noise

        return (
            torch.from_numpy(x1d),           # (2, L)
            torch.from_numpy(x3ch),          # (3, L)
            torch.from_numpy(spec),          # (4, F, T)
            s["label"],                      # int — collation → LongTensor
        )


def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """클래스 불균형 보정 WeightedRandomSampler."""
    arr = np.array(labels, dtype=np.int64)
    counts = np.bincount(arr)
    weights = (1.0 / counts[arr]).astype(np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(labels),
        replacement=True,
    )


# ─────────────────────────────────────────────
# 3. 학습 루프
# ─────────────────────────────────────────────

def _active_params(model: nn.Module) -> list[torch.nn.Parameter]:
    """requires_grad=True인 파라미터만 반환 (clip_grad_norm_ 용)."""
    return [p for p in model.parameters() if p.requires_grad]


def run_ae_epoch(
    model: OrbitFusionNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """
    Stage 1a: stream_time AE 재구성 손실.
    optimizer=None 이면 검증 모드.
    """
    training = optimizer is not None
    model.stream_time.train(training)
    total_loss, n = 0.0, 0

    with torch.set_grad_enabled(training):
        for x1d, _x3ch, _spec, _lbl in loader:
            x1d = x1d.to(device)
            x_rec, _ = model.stream_time(x1d)
            loss = OrbitAE1D.reconstruction_loss(x_rec, x1d)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.stream_time.parameters(), max_norm=1.0
                )
                optimizer.step()

            total_loss += loss.item() * x1d.size(0)
            n += x1d.size(0)

    return total_loss / max(n, 1)


def run_cls_epoch(
    model: OrbitFusionNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """
    Stage 1b: stream_freq + stream_axial + fusion_head 분류 손실.
    stream_time은 freeze된 상태를 가정 (requires_grad=False).
    optimizer=None 이면 검증 모드.
    """
    training = optimizer is not None
    model.stream_freq.train(training)
    model.fusion_head.train(training)
    if model.use_axial:
        model.stream_axial.train(training)
    # stream_time은 eval 유지 (BatchNorm 추론 모드)
    model.stream_time.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x1d, x3ch, x_spec, labels in loader:
            x1d    = x1d.to(device)
            x3ch   = x3ch.to(device)
            x_spec = x_spec.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            # stream_time은 동결 — 특징만 추출 (no grad)
            with torch.no_grad():
                f_time = model.stream_time.encode(x1d)          # (B, 256)

            f_freq = model.stream_freq.get_features(x_spec)     # (B, 256)

            if model.use_axial:
                f_axial = model.stream_axial.encode(x3ch)       # (B, 256)
                fused = torch.cat([f_time, f_freq, f_axial], dim=1)  # (B, 768)
            else:
                fused = torch.cat([f_time, f_freq], dim=1)      # (B, 512)

            logits = model.fusion_head(fused)                   # (B, C)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_active_params(model), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x1d.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            n          += x1d.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


def run_fusion_epoch(
    model: OrbitFusionNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """
    Stage 2: 전체 모델 end-to-end 분류 손실.
    optimizer=None 이면 검증 모드.
    """
    training = optimizer is not None
    model.train(training)
    if not training:
        model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x1d, x3ch, x_spec, labels in loader:
            x1d    = x1d.to(device)
            x3ch   = x3ch.to(device)
            x_spec = x_spec.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            logits = model(x1d, x_spec, x3ch if model.use_axial else None)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_active_params(model), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x1d.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            n          += x1d.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


# ─────────────────────────────────────────────
# 4. 체크포인트 저장/복원
# ─────────────────────────────────────────────

_CKPT_PATH = os.path.join(SCRIPT_DIR, "model", "orbit_fusion.pth")
_CFG_PATH  = os.path.join(SCRIPT_DIR, "fusion_config.json")


def save_checkpoint(model: OrbitFusionNet, meta: dict) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes":      model.num_classes,
        "use_axial":        model.use_axial,
        **meta,
    }, _CKPT_PATH)
    print(f"  → 체크포인트 저장: {_CKPT_PATH}  {meta}")


def restore_checkpoint(model: OrbitFusionNet, device: torch.device) -> None:
    if not os.path.exists(_CKPT_PATH):
        return
    ckpt = torch.load(_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  ← 최적 체크포인트 복원.")


# ─────────────────────────────────────────────
# 5. 학습 진입점
# ─────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Fusion] device: {device}")

    # ── 데이터 수집 ──────────────────────────────
    bin_items = collect_bin_files(args.data_dir)
    if not bin_items:
        raise RuntimeError(f"BIN 파일 없음: {args.data_dir}")

    # ── 고정 스케일 ───────────────────────────────
    if args.scale_mil > 0.0:
        scale_mil = args.scale_mil
        print(f"[Fusion] 고정 스케일 (CLI 지정): {scale_mil:.3f} mil")
    else:
        print("[Fusion] 정상 학습 데이터에서 스케일 자동 계산 중...")
        xy_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for bin_path, label in bin_items:
            if label != 0:
                continue
            try:
                data = parse_bin_legacy(bin_path, fs=FS)
                for x_raw, y_raw, _ in extract_xyz_triplets_legacy(data):
                    x_m, y_m = volt_to_mil(x_raw, y_raw)
                    xy_pairs.append((x_m[9*FS:10*FS], y_m[9*FS:10*FS]))
            except Exception:
                pass
        scale_mil = compute_dataset_scale(xy_pairs) if xy_pairs else FIXED_1D_SCALE_MIL
        print(f"[Fusion] 계산된 스케일: {scale_mil:.3f} mil  (기본값: {FIXED_1D_SCALE_MIL})")

    # ── 샘플 전처리 ───────────────────────────────
    t0 = time.time()
    print("[Fusion] 신호 전처리 중 (스펙트로그램 포함, 시간 소요 가능)...")
    all_samples = load_samples(bin_items, scale_mil)
    print(f"[Fusion] 전처리 완료: {time.time()-t0:.1f}s")

    if not all_samples:
        raise RuntimeError("로드된 샘플 없음. BIN 파일/경로 확인 필요.")

    labels_all = [s["label"] for s in all_samples]
    for lbl, name in enumerate(["normal", "abnormal"]):
        print(f"[Fusion]   class {lbl} ({name}): {labels_all.count(lbl)}개")

    # ── Train/Val 분할 (stratified 80/20) ─────────
    idx = np.arange(len(all_samples))
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels_all,
    )
    train_all  = [all_samples[i] for i in idx_tr]
    val_all    = [all_samples[i] for i in idx_val]
    train_norm = [s for s in train_all  if s["label"] == 0]
    val_norm   = [s for s in val_all    if s["label"] == 0]
    print(f"[Fusion] train={len(train_all)}, val={len(val_all)}, "
          f"normal_train={len(train_norm)}, normal_val={len(val_norm)}")

    # ── DataLoader ────────────────────────────────
    def _loader(samples, augment, sampler=None, shuffle=False):
        return DataLoader(
            FusionDataset(samples, augment=augment),
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    sampler_train = make_weighted_sampler([s["label"] for s in train_all])
    loader_train  = _loader(train_all,  augment=True,  sampler=sampler_train)
    loader_val    = _loader(val_all,    augment=False)
    loader_norm_tr = _loader(train_norm, augment=True,  shuffle=True)
    loader_norm_val = _loader(val_norm,  augment=False) if val_norm else loader_val

    # ── 모델 ──────────────────────────────────────
    num_classes = 2  # normal / abnormal
    model = OrbitFusionNet(
        num_classes=num_classes,
        use_axial=args.use_axial,
        spec_channels=4,
        seq_len=FS,
    ).to(device)
    print(f"[Fusion] 모델 파라미터: "
          f"{sum(p.numel() for p in model.parameters()):,} "
          f"(use_axial={args.use_axial})")

    os.makedirs(os.path.join(SCRIPT_DIR, "model"), exist_ok=True)

    # ══════════════════════════════════════════════
    # Stage 1a — AE 재구성 (비지도, 정상 데이터)
    # ══════════════════════════════════════════════
    if train_norm:
        print(f"\n{'='*60}")
        print(f"[Stage 1a] AE 재구성 학습  (epochs={args.epochs_1a}, patience={args.patience})")
        print(f"{'='*60}")

        opt_1a  = torch.optim.AdamW(
            model.stream_time.parameters(), lr=args.lr, weight_decay=1e-4
        )
        sch_1a  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_1a, T_max=args.epochs_1a
        )
        best_ae  = float("inf")
        pat_cnt  = 0

        for ep in range(1, args.epochs_1a + 1):
            tr_loss = run_ae_epoch(model, loader_norm_tr,  opt_1a, device)
            va_loss = run_ae_epoch(model, loader_norm_val, None,   device)
            sch_1a.step()
            lr_now = sch_1a.get_last_lr()[0]
            print(f"  1a [{ep:3d}/{args.epochs_1a}]  "
                  f"rec_train={tr_loss:.6f}  rec_val={va_loss:.6f}  lr={lr_now:.2e}")

            if va_loss < best_ae - 1e-7:
                best_ae, pat_cnt = va_loss, 0
                save_checkpoint(model, {"stage": "1a", "scale_mil": scale_mil})
            else:
                pat_cnt += 1
                if pat_cnt >= args.patience:
                    print(f"  1a 조기 종료 (no improvement for {args.patience} epochs)")
                    break

        restore_checkpoint(model, device)
    else:
        print("[Stage 1a] 정상 학습 샘플 없음 — 건너뜀.")

    # ══════════════════════════════════════════════
    # Stage 1b — 분류 미세조정 (stream_freq + axial)
    # ══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Stage 1b] 분류 학습  (epochs={args.epochs_1b}, patience={args.patience})")
    print(f"{'='*60}")

    # stream_time 동결
    for p in model.stream_time.parameters():
        p.requires_grad_(False)
    frozen_count = sum(
        1 for p in model.stream_time.parameters() if not p.requires_grad
    )
    print(f"  stream_time 동결 ({frozen_count} param tensors)")

    stage1b_params = model.get_stage1b_params() + list(model.fusion_head.parameters())
    opt_1b = torch.optim.AdamW(stage1b_params, lr=args.lr, weight_decay=1e-4)
    sch_1b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_1b, T_max=args.epochs_1b)

    best_acc_1b = 0.0
    pat_cnt     = 0

    for ep in range(1, args.epochs_1b + 1):
        tr_loss, tr_acc = run_cls_epoch(model, loader_train, opt_1b, device)
        va_loss, va_acc = run_cls_epoch(model, loader_val,   None,   device)
        sch_1b.step()
        print(f"  1b [{ep:3d}/{args.epochs_1b}]  "
              f"loss={tr_loss:.4f}|{va_loss:.4f}  "
              f"acc={tr_acc:.3f}|{va_acc:.3f}")

        if va_acc > best_acc_1b + 1e-5:
            best_acc_1b, pat_cnt = va_acc, 0
            save_checkpoint(model, {
                "stage": "1b", "scale_mil": scale_mil, "val_acc": va_acc,
            })
        else:
            pat_cnt += 1
            if pat_cnt >= args.patience:
                print(f"  1b 조기 종료 (no improvement for {args.patience} epochs)")
                break

    restore_checkpoint(model, device)

    # stream_time 해동 (Stage 2 end-to-end)
    for p in model.stream_time.parameters():
        p.requires_grad_(True)

    # ══════════════════════════════════════════════
    # Stage 2 — Fusion 최종 학습
    # ══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[Stage 2] Fusion 학습  (epochs={args.epochs_2}, patience={args.patience})")
    print(f"{'='*60}")

    if args.freeze_streams:
        model.freeze_streams()
        print("  스트림 동결 (fusion_head만 학습)")
        opt_2 = torch.optim.AdamW(
            model.fusion_head.parameters(), lr=args.lr, weight_decay=1e-4
        )
    else:
        print(f"  end-to-end (fusion_lr={args.lr:.2e}, stream_lr={args.lr*0.01:.2e})")
        opt_2 = build_stage2_optimizer(
            model, fusion_lr=args.lr, stream_lr=args.lr * 0.01
        )

    sch_2   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_2, T_max=args.epochs_2)
    best_acc_2 = 0.0
    pat_cnt    = 0

    for ep in range(1, args.epochs_2 + 1):
        tr_loss, tr_acc = run_fusion_epoch(model, loader_train, opt_2,  device)
        va_loss, va_acc = run_fusion_epoch(model, loader_val,   None, device)
        sch_2.step()
        print(f"  2  [{ep:3d}/{args.epochs_2}]  "
              f"loss={tr_loss:.4f}|{va_loss:.4f}  "
              f"acc={tr_acc:.3f}|{va_acc:.3f}")

        if va_acc > best_acc_2 + 1e-5:
            best_acc_2, pat_cnt = va_acc, 0
            save_checkpoint(model, {
                "stage": "final", "scale_mil": scale_mil, "val_acc": va_acc,
            })
        else:
            pat_cnt += 1
            if pat_cnt >= args.patience:
                print(f"  2 조기 종료 (no improvement for {args.patience} epochs)")
                break

    restore_checkpoint(model, device)

    # ── 최종 검증 ─────────────────────────────────
    _, final_acc = run_fusion_epoch(model, loader_val, None, device)
    print(f"\n[Fusion] 최종 val_acc = {final_acc:.4f}")

    # ── 설정 파일 저장 ─────────────────────────────
    cfg = {
        "scale_mil":   scale_mil,
        "num_classes": num_classes,
        "use_axial":   args.use_axial,
        "seq_len":     FS,
        "val_acc":     float(final_acc),
        "classes":     ["normal", "abnormal"],
    }
    with open(_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Fusion] 설정 저장: {_CFG_PATH}")
    print(f"[Fusion] 체크포인트: {_CKPT_PATH}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OrbitFusionNet 3단계 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",      type=str,   default="../data",
                   help="데이터 루트 디렉토리 (raw/normal, raw/abnormal 포함)")
    p.add_argument("--epochs_1a",     type=int,   default=30,
                   help="Stage 1a (AE 재구성) 최대 epoch")
    p.add_argument("--epochs_1b",     type=int,   default=30,
                   help="Stage 1b (스트림 분류) 최대 epoch")
    p.add_argument("--epochs_2",      type=int,   default=40,
                   help="Stage 2 (Fusion) 최대 epoch")
    p.add_argument("--batch_size",    type=int,   default=8,
                   help="배치 크기 (GPU 8 GB 기준 8~16 권장)")
    p.add_argument("--lr",            type=float, default=1e-3,
                   help="기본 학습률 (AdamW, CosineAnnealingLR 스케줄)")
    p.add_argument("--patience",      type=int,   default=10,
                   help="조기 종료 patience epoch 수")
    p.add_argument("--scale_mil",     type=float, default=0.0,
                   help="고정 스케일 [mil]. 0이면 정상 데이터에서 자동 계산.")
    p.add_argument("--use_axial",     action="store_true",  default=True,
                   help="축방향(Z) 스트림 활성화")
    p.add_argument("--no_axial",      dest="use_axial", action="store_false",
                   help="축방향 스트림 비활성화 (Z 채널 없는 환경)")
    p.add_argument("--freeze_streams", action="store_true", default=False,
                   help="Stage 2에서 스트림 동결, fusion_head만 학습 (Option A)")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
