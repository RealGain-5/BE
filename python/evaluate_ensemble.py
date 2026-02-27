"""
evaluate_ensemble.py
=====================
ResNet18(multiscale) + OrbitCNN1D 앙상블 검증 스크립트 (4-class 지원).

평가 항목:
  1) ResNet18 단독
  2) OrbitCNN1D 단독
  3) 가중 앙상블 (ensemble_config.json 기준)
  4) [OOD 오탐율] val split에서 max(ens_prob) < ood_threshold 비율
     → 학습 분포 내 샘플이 OOD로 잘못 판정되는 비율 (낮을수록 좋음)
  5) [OOD 탐지율] raw/abnormal에서 max(ens_prob) < ood_threshold 비율
     → 실제 분포 외 샘플을 OOD로 올바르게 판정하는 비율 (높을수록 좋음)

실행 예시:
  python evaluate_ensemble.py --data_dir ../data
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocess import (
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    make_multiscale_orbit,
    prepare_1d_input,
)
from model_loader import load_trained_model
from infer_resnet_None import build_transform_from_meta, predict_from_multiscale

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 클래스 정의
_CLASS_MAP_PATH = os.path.join(SCRIPT_DIR, "class_map.json")
with open(_CLASS_MAP_PATH, "r") as _f:
    CLASS_NAMES: list = json.load(_f)["classes"]

FS = 40_000

# 데이터 소스 (train_multiscale.py / train_1d_cnn.py 와 동일 구조) — raw 데이터만
TRAIN_SOURCES = [
    ("raw/normal",         0, "*.BIN"),
    ("raw/normal_1200rpm", 0, "*.BIN"),
    # raw/normal_3600rpm 제외 — 데이터 품질 문제로 학습에서 배제
    ("raw/abnormal",       1, "*.BIN"),
]

OOD_CLASS_NAME = "unknown_abnormal"

RESNET_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
if not os.path.exists(RESNET_PATH):
    RESNET_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")
CNN1D_PATH  = os.path.join(SCRIPT_DIR, "model", "orbit_cnn1d.pth")
CFG_PATH    = os.path.join(SCRIPT_DIR, "ensemble_config.json")


# ─────────────────────────────────────────────
# 1. val split 재현 (train_*.py 와 동일 random_state=42)
# ─────────────────────────────────────────────
def load_val_samples(data_dir, fs=FS):
    """
    각 클래스 80/20 split 재현 → val 20% 반환.
    반환: list of (x_seg, y_seg, label)
    """
    val = []

    for subdir, label, pattern in TRAIN_SOURCES:
        class_path = os.path.join(data_dir, subdir)
        bin_files  = sorted(glob.glob(os.path.join(class_path, pattern)))
        if not bin_files:
            print(f"  [경고] {class_path} 에 파일 없음")
            continue

        class_label = CLASS_NAMES[label]
        print(f"  [{class_label}] {len(bin_files)} 파일 로딩...")

        class_raw = []
        for bp in bin_files:
            try:
                data     = parse_bin_legacy(bp, fs=fs)
                xy_pairs = extract_xy_pairs_legacy(data)
                for x, y in xy_pairs:
                    xm, ym = volt_to_mil(x, y)
                    s, e   = 9 * fs, 10 * fs
                    class_raw.append((xm[s:e], ym[s:e], label))
            except Exception as ex:
                print(f"    [오류] {os.path.basename(bp)}: {ex}")

        n = len(class_raw)
        if n < 5:
            print(f"    → {n}개 (val 제외)")
            continue

        _, va_idx = train_test_split(
            list(range(n)), test_size=0.2, random_state=42
        )
        for i in va_idx:
            val.append(class_raw[i])
        print(f"    → val: {len(va_idx)}")

    n0 = sum(1 for _, _, l in val if l == 0)
    print(f"\n  val 총합: {len(val)}  (normal={n0}, fault={len(val)-n0})")
    return val





# ─────────────────────────────────────────────
# 2. 모델별 예측
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_resnet(model, transform, x_seg, y_seg, class_names, img_size=128, hybrid=False):
    ms_arr = make_multiscale_orbit(x_seg, y_seg, img_size=img_size, hybrid=hybrid)
    _, prob = predict_from_multiscale(model, class_names, ms_arr, transform)
    return prob  # (num_classes,)


@torch.no_grad()
def predict_1d(model, device, x_seg, y_seg):
    arr    = prepare_1d_input(x_seg, y_seg)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    logits = model(tensor)
    return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────
# 3. 평가 출력
# ─────────────────────────────────────────────
def report(name, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cr  = classification_report(
        y_true, y_pred, target_names=class_names,
        labels=list(range(len(class_names))), digits=4, zero_division=0
    )
    print(f"\n{'='*60}")
    print(f"  [{name}]  accuracy = {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)})")
    print(f"{'='*60}")

    # Confusion matrix 출력
    header = "         " + "".join(f"{n:>14}" for n in class_names)
    print(f"  Confusion Matrix (행=실제, 열=예측):")
    print(f"  {header}")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>8}: {row}")
    print(f"\n{cr}")
    return acc


def _is_ood_composite(conf, tv, ood_thr, tv_thr):
    """TV Distance 복합 OOD 판정. TV가 None이면 conf 단독 사용."""
    cond_conf = conf < ood_thr
    cond_tv   = (tv  > tv_thr) if tv is not None else False
    return cond_conf or cond_tv


def report_ood(name,
               val_confs, ood_threshold,
               val_tvs=None, tv_threshold=None):
    """
    OOD 탐지 성능 리포트 (TV Distance 복합 기준).

    val_confs    : list[float]       — val split 앙상블 max 확률
    ood_threshold: float             — max_conf 기준 임계값
    val_tvs      : list[float|None]  — val split TV Distance (없으면 None)
    tv_threshold : float|None        — TV 기준 임계값

    OOD 판정 기준:
        (TV > tv_threshold) OR (max_conf < ood_threshold)
    """
    has_tv = (val_tvs is not None and tv_threshold is not None
              and any(v is not None for v in val_tvs))

    print(f"\n{'='*60}")
    if has_tv:
        print(f"  [OOD 평가 — {name}]")
        print(f"    기준: (TV > {tv_threshold:.2f}) OR (max_conf < {ood_threshold:.2f})")
    else:
        print(f"  [OOD 평가 — {name}]  max_conf < {ood_threshold:.2f}")
    print(f"{'='*60}")

    fp_rate = None

    if val_confs:
        n     = len(val_confs)
        tvs_  = val_tvs if val_tvs else [None] * n
        flags = [_is_ood_composite(c, t, ood_threshold, tv_threshold or 1.0)
                 for c, t in zip(val_confs, tvs_)]
        fp_rate  = sum(flags) / n
        mean_c   = float(np.mean(val_confs))
        p10_c    = float(np.percentile(val_confs, 10))

        print(f"  [val split — 학습 분포 내]  n={n}")
        print(f"    복합 OOD율 : {fp_rate:.4f}  ({sum(flags)}/{n})  ← 낮을수록 좋음")
        print(f"    max_conf   : 평균={mean_c:.4f}  10th={p10_c:.4f}")

        if has_tv:
            tv_valid = [t for t in tvs_ if t is not None]
            if tv_valid:
                mean_tv  = float(np.mean(tv_valid))
                p10_tv   = float(np.percentile(tv_valid, 10))
                only_tv  = sum(1 for c, t in zip(val_confs, tvs_)
                               if t is not None and t > tv_threshold and c >= ood_threshold)
                only_conf= sum(1 for c, t in zip(val_confs, tvs_)
                               if t is not None and t <= tv_threshold and c < ood_threshold)
                both     = sum(1 for c, t in zip(val_confs, tvs_)
                               if t is not None and t > tv_threshold and c < ood_threshold)
                print(f"    TV dist    : 평균={mean_tv:.4f}  10th={p10_tv:.4f}")
                print(f"    OOD 근거   : TV만={only_tv}건  conf만={only_conf}건  둘다={both}건")

    return fp_rate


# ─────────────────────────────────────────────
# 4. 메인
# ─────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")
    print(f"[Info] classes: {CLASS_NAMES}")

    # ── 앙상블 가중치 + OOD 임계값 ──────────────
    try:
        with open(CFG_PATH) as f:
            cfg = json.load(f)
        rw            = float(cfg.get("resnet_weight", 0.5))
        cw            = float(cfg.get("cnn1d_weight",  0.5))
        ood_threshold = float(cfg.get("ood_threshold", 0.65))
        tv_threshold  = float(cfg.get("tv_threshold",  0.30))
    except Exception:
        rw, cw, ood_threshold, tv_threshold = 0.5, 0.5, 0.65, 0.30
    total = rw + cw
    rw, cw = rw / total, cw / total
    print(f"[Info] ensemble weights: resnet={rw:.2f}, cnn1d={cw:.2f}")
    print(f"[Info] ood_threshold   : {ood_threshold:.2f}")
    print(f"[Info] tv_threshold    : {tv_threshold:.2f}")

    # ── 모델 로드 ────────────────────────────────
    print(f"\n[1] ResNet 로드: {RESNET_PATH}")
    resnet, cn_r, meta_r = load_trained_model(RESNET_PATH)
    resnet.to(device).eval()
    transform     = build_transform_from_meta(meta_r)
    img_size      = int(meta_r.get("img_size", 128))
    channel_mode  = meta_r.get("channel_mode", "dynamic")
    use_hybrid    = (channel_mode == "hybrid")
    print(f"    model_type={meta_r['model_type']}, classes={cn_r}, img_size={img_size}, channel_mode={channel_mode}")

    model_1d = None
    if os.path.exists(CNN1D_PATH):
        print(f"\n[2] 1D CNN 로드: {CNN1D_PATH}")
        model_1d, cn_1d, _ = load_trained_model(CNN1D_PATH)
        model_1d.to(device).eval()
        print(f"    classes={cn_1d}")
    else:
        print(f"\n[2] orbit_cnn1d.pth 없음 — ResNet 단독 평가만 수행")

    # ── val 데이터 로드 ──────────────────────────
    print(f"\n[3] val 데이터 로드: {args.data_dir}")
    val_samples = load_val_samples(args.data_dir)
    if not val_samples:
        print("  [오류] val 샘플이 없습니다.")
        return

    # ── 예측 수집 ────────────────────────────────
    print("\n[4] 예측 중...")
    y_true, preds_r, preds_1d, preds_ens = [], [], [], []
    ens_confs_val = []   # 앙상블 max 확률
    tv_vals_val   = []   # TV Distance (val)

    for i, (x_seg, y_seg, label) in enumerate(val_samples):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(val_samples)}")

        y_true.append(label)
        prob_r = predict_resnet(resnet, transform, x_seg, y_seg, cn_r, img_size, hybrid=use_hybrid)
        preds_r.append(int(prob_r.argmax()))

        if model_1d is not None:
            prob_1d = predict_1d(model_1d, device, x_seg, y_seg)
            preds_1d.append(int(prob_1d.argmax()))
            ens = rw * prob_r + cw * prob_1d
            preds_ens.append(int(ens.argmax()))
            ens_confs_val.append(float(ens.max()))
            tv = float(0.5 * np.sum(np.abs(prob_r - prob_1d)))
            tv_vals_val.append(tv)
        else:
            ens_confs_val.append(float(prob_r.max()))
            tv_vals_val.append(None)

    # ── 결과 출력 ────────────────────────────────
    acc_r   = report("ResNet18 (multiscale)", y_true, preds_r, CLASS_NAMES)
    acc_1d  = None
    acc_ens = None

    if model_1d is not None:
        acc_1d  = report("OrbitCNN1D (1D CNN)", y_true, preds_1d, CLASS_NAMES)
        acc_ens = report(
            f"Ensemble (resnet×{rw:.2f} + cnn1d×{cw:.2f})",
            y_true, preds_ens, CLASS_NAMES
        )

    # ── OOD 평가 ─────────────────────────────────
    ens_label = (f"Ensemble (resnet×{rw:.2f} + cnn1d×{cw:.2f})"
                 if model_1d is not None else "ResNet18 (multiscale)")
    ood_fp = report_ood(
        ens_label,
        ens_confs_val, ood_threshold,
        tv_vals_val,   tv_threshold,
    )

    # ── 요약 ─────────────────────────────────────
    print("\n" + "="*60)
    print("  최종 요약")
    print("="*60)
    print(f"\n  [4-class 분류 정확도 — val split]")
    print(f"  ResNet18  단독: {acc_r:.4f}")
    if acc_1d is not None:
        delta_1d  = acc_1d  - acc_r
        delta_ens = acc_ens - acc_r
        print(f"  1D CNN    단독: {acc_1d:.4f}  ({delta_1d:+.4f} vs ResNet)")
        print(f"  앙상블        : {acc_ens:.4f}  ({delta_ens:+.4f} vs ResNet)")
        if acc_ens >= max(acc_r, acc_1d):
            print("  → 앙상블이 두 단독 모델을 모두 상회합니다. ✓")
        else:
            best = "ResNet" if acc_r >= acc_1d else "1D CNN"
            print(f"  → 앙상블이 {best} 단독보다 낮습니다. 가중치 조정 검토 필요.")

    print(f"\n  [OOD 탐지 — TV>{tv_threshold:.2f} OR conf<{ood_threshold:.2f}]")
    if ood_fp is not None:
        status_fp = "✓" if ood_fp < 0.10 else "✗ (목표: < 0.10)"
        print(f"  OOD 오탐율 (val → OOD 오분류) : {ood_fp:.4f}  {status_fp}")
        if ood_fp >= 0.10:
            print(f"  → threshold 조정 필요 여부: 검토 권장")


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="4-class ResNet + 1D CNN 앙상블 검증")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/raw, data/synthetic 상위 디렉토리",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
