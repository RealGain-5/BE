"""
evaluate_ensemble.py
=====================
ResNet18(multiscale) + OrbitCNN1D 앙상블 검증 스크립트.

train_multiscale.py / train_1d_cnn.py 와 동일한 val split(random_state=42)을
재현하여 세 가지를 동시에 평가한다:
  1) ResNet18 단독
  2) OrbitCNN1D 단독
  3) 가중 앙상블 (ensemble_config.json 기준)

출력:
  - 모델별 accuracy / confusion matrix / classification report
  - 앙상블 vs 단독 모델 비교 요약

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

CLASS_NAMES = ["normal", "abnormal"]
FS          = 40_000

RESNET_PATH  = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
if not os.path.exists(RESNET_PATH):
    RESNET_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")
CNN1D_PATH   = os.path.join(SCRIPT_DIR, "model", "orbit_cnn1d.pth")
CFG_PATH     = os.path.join(SCRIPT_DIR, "ensemble_config.json")


# ─────────────────────────────────────────────
# 1. 공통 val split 재현
# ─────────────────────────────────────────────
def load_val_samples(data_dir, fs=FS):
    """
    train_multiscale.py / train_1d_cnn.py 와 동일한 BIN 순회 순서 + split 재현.
    반환: list of (x_seg, y_seg, label)  — 원시 신호 보존, 이미지 변환은 평가 시 수행
    """
    raw = []  # (x_seg, y_seg, label)
    for class_name, label in zip(CLASS_NAMES, [0, 1]):
        class_path = os.path.join(data_dir, class_name)
        bin_files  = sorted(glob.glob(os.path.join(class_path, "*.BIN")))
        if not bin_files:
            print(f"  [경고] {class_path} 에 BIN 파일 없음")
            continue
        print(f"  {class_name}: {len(bin_files)} 파일 로딩...")
        for bp in bin_files:
            try:
                data     = parse_bin_legacy(bp, fs=fs)
                xy_pairs = extract_xy_pairs_legacy(data)
                for x, y in xy_pairs:
                    xm, ym  = volt_to_mil(x, y)
                    s, e    = 9 * fs, 10 * fs
                    raw.append((xm[s:e], ym[s:e], label))
            except Exception as ex:
                print(f"    [오류] {os.path.basename(bp)}: {ex}")

    labels  = [r[2] for r in raw]
    indices = list(range(len(raw)))
    _, va_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    val = [raw[i] for i in va_idx]
    n0 = sum(1 for _, _, l in val if l == 0)
    n1 = sum(1 for _, _, l in val if l == 1)
    print(f"  val 샘플: {len(val)}  (normal={n0}, abnormal={n1})")
    return val


# ─────────────────────────────────────────────
# 2. 모델별 예측
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_resnet(model, transform, x_seg, y_seg, class_names):
    ms_arr = make_multiscale_orbit(x_seg, y_seg, img_size=256)
    _, prob = predict_from_multiscale(model, class_names, ms_arr, transform)
    return prob  # np.ndarray (num_classes,)


@torch.no_grad()
def predict_1d(model, device, x_seg, y_seg):
    arr    = prepare_1d_input(x_seg, y_seg)                       # (2, 40000)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)        # (1, 2, 40000)
    logits = model(tensor)                                         # (1, C)
    return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()       # (C,)


# ─────────────────────────────────────────────
# 3. 평가 출력
# ─────────────────────────────────────────────
def report(name, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    cr  = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\n{'='*50}")
    print(f"  [{name}]  accuracy = {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)})")
    print(f"{'='*50}")
    print(f"  Confusion Matrix (행=실제, 열=예측):")
    print(f"              normal  abnormal")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>8}: {row}")
    print(f"\n{cr}")
    return acc


# ─────────────────────────────────────────────
# 4. 메인
# ─────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")

    # ── 앙상블 가중치 ────────────────────────────
    try:
        with open(CFG_PATH) as f:
            cfg = json.load(f)
        rw = float(cfg.get("resnet_weight", 0.5))
        cw = float(cfg.get("cnn1d_weight",  0.5))
    except Exception:
        rw, cw = 0.5, 0.5
    total = rw + cw
    rw, cw = rw / total, cw / total
    print(f"[Info] ensemble weights: resnet={rw:.2f}, cnn1d={cw:.2f}")

    # ── 모델 로드 ────────────────────────────────
    print(f"\n[1] ResNet 로드: {RESNET_PATH}")
    resnet, cn_r, meta_r = load_trained_model(RESNET_PATH)
    resnet.to(device).eval()
    transform = build_transform_from_meta(meta_r)
    print(f"    model_type={meta_r['model_type']}, classes={cn_r}")

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

    # ── 예측 수집 ────────────────────────────────
    print("\n[4] 예측 중...")
    y_true, preds_r, preds_1d, preds_ens = [], [], [], []

    for i, (x_seg, y_seg, label) in enumerate(val_samples):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val_samples)}")

        y_true.append(label)

        # ResNet
        prob_r = predict_resnet(resnet, transform, x_seg, y_seg, cn_r)
        preds_r.append(int(prob_r.argmax()))

        if model_1d is not None:
            # 1D CNN
            prob_1d = predict_1d(model_1d, device, x_seg, y_seg)
            preds_1d.append(int(prob_1d.argmax()))

            # 앙상블
            ens = rw * prob_r + cw * prob_1d
            preds_ens.append(int(ens.argmax()))

    # ── 결과 출력 ────────────────────────────────
    acc_r = report("ResNet18 (multiscale)", y_true, preds_r, CLASS_NAMES)

    acc_1d  = None
    acc_ens = None
    if model_1d is not None:
        acc_1d  = report("OrbitCNN1D (1D CNN)", y_true, preds_1d, CLASS_NAMES)
        acc_ens = report(f"Ensemble (resnet×{rw:.2f} + cnn1d×{cw:.2f})",
                         y_true, preds_ens, CLASS_NAMES)

    # ── 요약 ─────────────────────────────────────
    print("\n" + "="*50)
    print("  최종 요약")
    print("="*50)
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


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="ResNet + 1D CNN 앙상블 검증")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/normal, data/abnormal 상위 디렉토리",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
