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

# 데이터 소스 (train_multiscale.py / train_1d_cnn.py 와 동일 구조)
TRAIN_SOURCES = [
    ("raw/normal",                      0, "*.BIN"),
    ("synthetic/3600rpm/unbalance",     1, "*.bin"),
    ("synthetic/1200rpm/unbalance",     1, "*.bin"),
    ("synthetic/3600rpm/misalignment",  2, "*.bin"),
    ("synthetic/1200rpm/misalignment",  2, "*.bin"),
    ("synthetic/3600rpm/oil_whip",      3, "*.bin"),
    ("synthetic/1200rpm/oil_whip",      3, "*.bin"),
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


def load_real_abnormal(data_dir, fs=FS):
    """raw/abnormal 이차 검증용 로딩."""
    abnormal_path = os.path.join(data_dir, "raw", "abnormal")
    bin_files = sorted(glob.glob(os.path.join(abnormal_path, "*.BIN")))
    if not bin_files:
        print(f"  [경고] {abnormal_path} 에 BIN 파일 없음")
        return []

    print(f"  [real_abnormal] {len(bin_files)} 파일 로딩...")
    samples = []
    for bp in bin_files:
        try:
            data     = parse_bin_legacy(bp, fs=fs)
            xy_pairs = extract_xy_pairs_legacy(data)
            for x, y in xy_pairs:
                xm, ym = volt_to_mil(x, y)
                s, e   = 9 * fs, 10 * fs
                samples.append((xm[s:e], ym[s:e]))
        except Exception as ex:
            print(f"    [오류] {os.path.basename(bp)}: {ex}")

    print(f"    → {len(samples)} 샘플")
    return samples


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


def report_ood(name, val_confs, ra_confs, ood_threshold):
    """
    OOD 탐지 성능 리포트.

    val_confs : list[float] — val split 앙상블 최대 확률 (학습 분포 내)
    ra_confs  : list[float] — real_abnormal 앙상블 최대 확률 (분포 외)
    ood_threshold : float   — ensemble_config.json 기준

    출력:
      OOD 오탐율 : val 샘플 중 max_conf < threshold 비율 (낮을수록 좋음)
      OOD 탐지율 : real_abnormal 중 max_conf < threshold 비율 (높을수록 좋음)
    """
    print(f"\n{'='*60}")
    print(f"  [OOD 평가 — {name}]  threshold = {ood_threshold:.2f}")
    print(f"{'='*60}")

    # ── val split: OOD 오탐율 (False Positive) ────────────────
    if val_confs:
        fp      = sum(1 for c in val_confs if c < ood_threshold)
        fp_rate = fp / len(val_confs)
        mean_c  = float(np.mean(val_confs))
        p10     = float(np.percentile(val_confs, 10))
        print(f"  [val split — 학습 분포 내]  n={len(val_confs)}")
        print(f"    OOD 오탐율  : {fp_rate:.4f}  ({fp}/{len(val_confs)})  ← 낮을수록 좋음")
        print(f"    max_conf 평균: {mean_c:.4f}  /  10th percentile: {p10:.4f}")

    # ── real_abnormal: OOD 탐지율 (True Positive) ────────────
    if ra_confs:
        tp      = sum(1 for c in ra_confs if c < ood_threshold)
        tp_rate = tp / len(ra_confs)
        mean_c  = float(np.mean(ra_confs))
        p90     = float(np.percentile(ra_confs, 90))
        print(f"  [real_abnormal — 분포 외]  n={len(ra_confs)}")
        print(f"    OOD 탐지율  : {tp_rate:.4f}  ({tp}/{len(ra_confs)})  ← 높을수록 좋음")
        print(f"    max_conf 평균: {mean_c:.4f}  /  90th percentile: {p90:.4f}")

    fp_rate = (sum(1 for c in val_confs if c < ood_threshold) / len(val_confs)
               if val_confs else None)
    tp_rate = (sum(1 for c in ra_confs if c < ood_threshold) / len(ra_confs)
               if ra_confs else None)
    return fp_rate, tp_rate


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
        rw          = float(cfg.get("resnet_weight", 0.5))
        cw          = float(cfg.get("cnn1d_weight",  0.5))
        ood_threshold = float(cfg.get("ood_threshold", 0.70))
    except Exception:
        rw, cw, ood_threshold = 0.5, 0.5, 0.70
    total = rw + cw
    rw, cw = rw / total, cw / total
    print(f"[Info] ensemble weights: resnet={rw:.2f}, cnn1d={cw:.2f}")
    print(f"[Info] ood_threshold   : {ood_threshold:.2f}")

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

    # ── real_abnormal 로드 ───────────────────────
    print(f"\n[4] real_abnormal 이차검증 데이터 로드")
    real_abnormal_samples = load_real_abnormal(args.data_dir)

    # ── 예측 수집 ────────────────────────────────
    print("\n[5] 예측 중...")
    y_true, preds_r, preds_1d, preds_ens = [], [], [], []
    ens_confs_val = []  # 앙상블 max 확률 (OOD 오탐율 측정용)

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
        else:
            ens_confs_val.append(float(prob_r.max()))

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
    ra_ens_confs = []  # real_abnormal 앙상블 max 확률 (OOD 탐지율 측정용)

    if real_abnormal_samples:
        print("\n[6] real_abnormal 신뢰도 수집 중...")
        for j, (x_seg, y_seg) in enumerate(real_abnormal_samples):
            if (j + 1) % 200 == 0:
                print(f"  {j+1}/{len(real_abnormal_samples)}")

            prob_r = predict_resnet(resnet, transform, x_seg, y_seg, cn_r, img_size, hybrid=use_hybrid)
            if model_1d is not None:
                prob_1d = predict_1d(model_1d, device, x_seg, y_seg)
                ens = rw * prob_r + cw * prob_1d
            else:
                ens = prob_r
            ra_ens_confs.append(float(ens.max()))

    ens_label = (f"Ensemble (resnet×{rw:.2f} + cnn1d×{cw:.2f})"
                 if model_1d is not None else "ResNet18 (multiscale)")
    ood_fp, ood_tp = report_ood(ens_label, ens_confs_val, ra_ens_confs, ood_threshold)

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

    print(f"\n  [OOD 탐지 — threshold={ood_threshold:.2f}]")
    if ood_fp is not None:
        status_fp = "✓" if ood_fp < 0.10 else "✗ (목표: < 0.10)"
        print(f"  OOD 오탐율 (val → OOD 오분류) : {ood_fp:.4f}  {status_fp}")
    if ood_tp is not None:
        status_tp = "✓" if ood_tp >= 0.75 else "△ (목표: ≥ 0.75)"
        print(f"  OOD 탐지율 (abnormal → OOD)  : {ood_tp:.4f}  {status_tp}")
    if ood_fp is not None and ood_tp is not None:
        print(f"  → threshold 조정 필요 여부: "
              f"{'없음' if ood_fp < 0.10 and ood_tp >= 0.75 else '검토 권장'}")


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
