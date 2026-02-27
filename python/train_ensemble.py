"""
train_ensemble.py
==================
ResNet18(multiscale) + OrbitCNN1D 앙상블 가중치 및 OOD 임계값 최적화 스크립트.

사전 학습된 두 모델의 val split 예측값을 수집한 뒤,
격자 탐색(Grid Search)으로 최적 파라미터를 찾아 ensemble_config.json에 저장합니다.

최적화 목표 (복합 지표):
    score = val_acc × (1 − ood_fp_rate)
    ood_fp_rate : val 분포 내 샘플 중 OOD로 오분류되는 비율 (낮을수록 좋음)
    OOD 판정   : (max_conf < ood_threshold) OR (tv_dist > tv_threshold)

선행 조건:
    train_multiscale.py 및 train_1d_cnn.py 실행 완료 후 사용하세요.

실행 예시:
    python train_ensemble.py --data_dir ../data
"""

import os
import sys
import glob
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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

# 데이터 소스 — evaluate_ensemble.py / train_*.py 와 동일 구조
TRAIN_SOURCES = [
    ("raw/normal",         0, "*.BIN"),
    ("raw/normal_1200rpm", 0, "*.BIN"),
    # raw/normal_3600rpm 제외 — 데이터 품질 문제로 학습에서 배제
    ("raw/abnormal",       1, "*.BIN"),
]

RESNET_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
if not os.path.exists(RESNET_PATH):
    RESNET_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")
CNN1D_PATH  = os.path.join(SCRIPT_DIR, "model", "orbit_cnn1d.pth")
CFG_PATH    = os.path.join(SCRIPT_DIR, "ensemble_config.json")


# ─────────────────────────────────────────────
# 1. val split 재현 (train_*.py / evaluate_ensemble.py 와 동일 random_state=42)
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

        class_label = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"label{label}"
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
# 2. 두 모델 예측값 일괄 수집
# ─────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(
    resnet, transform, model_1d, device,
    val_samples, class_names, img_size=128, use_hybrid=False,
):
    """
    val 샘플 전체에 대해 두 모델의 softmax 확률을 수집.

    반환:
        probs_r  : (N, C) ndarray — ResNet18 확률
        probs_1d : (N, C) ndarray — OrbitCNN1D 확률
        y_true   : (N,)  list    — 정답 레이블
    """
    probs_r  = []
    probs_1d = []
    y_true   = []

    print(f"  총 {len(val_samples)} 샘플 추론 중...")
    t0 = time.time()

    for i, (x_seg, y_seg, label) in enumerate(val_samples):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(val_samples)}  ({elapsed:.1f}s)")

        y_true.append(label)

        # ResNet18 예측
        ms_arr = make_multiscale_orbit(x_seg, y_seg, img_size=img_size, hybrid=use_hybrid)
        _, prob_r = predict_from_multiscale(resnet, class_names, ms_arr, transform)
        probs_r.append(prob_r)

        # OrbitCNN1D 예측
        arr    = prepare_1d_input(x_seg, y_seg)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        logits = model_1d(tensor)
        prob_1d = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs_1d.append(prob_1d)

    elapsed = time.time() - t0
    print(f"  추론 완료: {elapsed:.1f}s")

    return np.array(probs_r), np.array(probs_1d), y_true


# ─────────────────────────────────────────────
# 3. 격자 탐색 (Grid Search)
# ─────────────────────────────────────────────
def grid_search(probs_r, probs_1d, y_true, verbose=True):
    """
    최적 (alpha, ood_threshold, tv_threshold) 탐색.

    탐색 공간:
        alpha          : ResNet 가중치   [0.10 .. 0.90]  step=0.05
        ood_threshold  : max_conf 임계값 [0.45 .. 0.85]  step=0.025
        tv_threshold   : TV 거리 임계값  [0.15 .. 0.55]  step=0.05

    OOD 판정: (max_conf < ood_threshold) OR (tv_dist > tv_threshold)

    score = val_acc × (1 − ood_fp_rate)
        val_acc      : argmax(ens_prob) 정확도
        ood_fp_rate  : val 분포 내 샘플 중 OOD 오분류 비율 (낮을수록 좋음)

    반환: (best_params dict, best_score float, top_results list)
    """
    y_true_arr = np.array(y_true)

    alphas         = np.round(np.arange(0.10, 0.91, 0.05), 4)
    ood_thresholds = np.round(np.arange(0.45, 0.86, 0.025), 4)
    tv_thresholds  = np.round(np.arange(0.15, 0.56, 0.05), 4)

    total = len(alphas) * len(ood_thresholds) * len(tv_thresholds)
    if verbose:
        print(f"  탐색 조합 수: {total:,}  "
              f"(alpha={len(alphas)} × ood_thr={len(ood_thresholds)} × tv_thr={len(tv_thresholds)})")

    best_score  = -1.0
    best_params = {}
    all_results = []

    t0 = time.time()

    for alpha in alphas:
        # 앙상블 확률 — alpha × ResNet + (1-alpha) × CNN1D
        ens   = alpha * probs_r + (1.0 - alpha) * probs_1d  # (N, C)
        preds = ens.argmax(axis=1)                           # (N,)
        confs = ens.max(axis=1)                              # (N,)  max softmax 확률
        tvs   = 0.5 * np.abs(probs_r - probs_1d).sum(axis=1)  # (N,)  TV Distance

        val_acc = float((preds == y_true_arr).mean())

        # ood_threshold × tv_threshold 2중 루프 — 내부에서 벡터 연산
        for ood_thr in ood_thresholds:
            conf_flags = confs < ood_thr                     # (N,) bool

            for tv_thr in tv_thresholds:
                ood_flags = conf_flags | (tvs > tv_thr)      # (N,) bool
                fp_rate   = float(ood_flags.mean())
                score     = val_acc * (1.0 - fp_rate)

                if score > best_score:
                    best_score  = score
                    best_params = {
                        "alpha":         float(alpha),
                        "ood_threshold": float(ood_thr),
                        "tv_threshold":  float(tv_thr),
                        "val_acc":       round(val_acc, 6),
                        "ood_fp_rate":   round(fp_rate, 6),
                        "score":         round(score, 6),
                    }

                all_results.append((score, val_acc, fp_rate, float(alpha), float(ood_thr), float(tv_thr)))

    elapsed = time.time() - t0
    if verbose:
        print(f"  탐색 완료: {elapsed:.1f}s")

    # 상위 10개 결과 정렬
    all_results.sort(key=lambda x: -x[0])
    top_results = all_results[:10]

    return best_params, best_score, top_results


# ─────────────────────────────────────────────
# 4. 결과 출력
# ─────────────────────────────────────────────
def _print_top_results(top_results):
    """상위 탐색 결과 테이블 출력."""
    print(f"\n  {'순위':>4}  {'score':>7}  {'val_acc':>7}  {'fp_rate':>7}  "
          f"{'alpha(R)':>8}  {'1-alpha(C)':>10}  {'ood_thr':>7}  {'tv_thr':>6}")
    print("  " + "-" * 72)
    for rank, (sc, acc, fp, alpha, ood_thr, tv_thr) in enumerate(top_results, 1):
        print(f"  {rank:>4}  {sc:>7.4f}  {acc:>7.4f}  {fp:>7.4f}  "
              f"{alpha:>8.2f}  {1-alpha:>10.2f}  {ood_thr:>7.3f}  {tv_thr:>6.3f}")


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────
def train_ensemble(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device : {device}")
    print(f"[Info] classes: {CLASS_NAMES}")

    # ── 모델 로드 ────────────────────────────────
    if not os.path.exists(RESNET_PATH):
        print(f"[오류] ResNet 체크포인트 없음: {RESNET_PATH}")
        print("  → train_multiscale.py 를 먼저 실행하세요.")
        return

    if not os.path.exists(CNN1D_PATH):
        print(f"[오류] OrbitCNN1D 체크포인트 없음: {CNN1D_PATH}")
        print("  → train_1d_cnn.py 를 먼저 실행하세요.")
        return

    print(f"\n[1] ResNet 로드: {RESNET_PATH}")
    resnet, cn_r, meta_r = load_trained_model(RESNET_PATH)
    resnet.to(device).eval()
    transform    = build_transform_from_meta(meta_r)
    img_size     = int(meta_r.get("img_size", 128))
    channel_mode = meta_r.get("channel_mode", "dynamic")
    use_hybrid   = (channel_mode == "hybrid")
    print(f"    model_type={meta_r.get('model_type')}, classes={cn_r}, "
          f"img_size={img_size}, channel_mode={channel_mode}")

    print(f"\n[2] OrbitCNN1D 로드: {CNN1D_PATH}")
    model_1d, cn_1d, _ = load_trained_model(CNN1D_PATH)
    model_1d.to(device).eval()
    print(f"    classes={cn_1d}")

    # ── val 데이터 로드 ──────────────────────────
    print(f"\n[3] val 데이터 로드: {args.data_dir}")
    val_samples = load_val_samples(args.data_dir)
    if not val_samples:
        print("  [오류] val 샘플이 없습니다. data_dir 경로를 확인하세요.")
        return

    # ── 예측 수집 ────────────────────────────────
    print("\n[4] 예측값 수집 (두 모델 추론)")
    probs_r, probs_1d, y_true = collect_predictions(
        resnet, transform, model_1d, device,
        val_samples, cn_r, img_size, use_hybrid,
    )
    print(f"  ResNet 예측 shape : {probs_r.shape}")
    print(f"  CNN1D  예측 shape : {probs_1d.shape}")

    # 각 모델 단독 정확도 출력 (기준선)
    acc_r  = float((probs_r.argmax(axis=1)  == np.array(y_true)).mean())
    acc_1d = float((probs_1d.argmax(axis=1) == np.array(y_true)).mean())
    print(f"\n  [기준선] ResNet18 단독 val_acc : {acc_r:.4f}")
    print(f"  [기준선] CNN1D   단독 val_acc : {acc_1d:.4f}")

    # ── 격자 탐색 ────────────────────────────────
    print("\n[5] 격자 탐색 (Grid Search)...")
    best_params, best_score, top_results = grid_search(probs_r, probs_1d, y_true)

    # ── 결과 출력 ────────────────────────────────
    alpha = best_params["alpha"]

    print(f"\n{'='*65}")
    print("  상위 10개 결과")
    print(f"{'='*65}")
    _print_top_results(top_results)

    print(f"\n{'='*65}")
    print("  최적 앙상블 파라미터")
    print(f"{'='*65}")
    print(f"  resnet_weight  : {alpha:.2f}")
    print(f"  cnn1d_weight   : {1 - alpha:.2f}")
    print(f"  ood_threshold  : {best_params['ood_threshold']:.3f}")
    print(f"  tv_threshold   : {best_params['tv_threshold']:.3f}")
    print(f"  val_acc        : {best_params['val_acc']:.4f}")
    print(f"  ood_fp_rate    : {best_params['ood_fp_rate']:.4f}"
          f"  {'✓' if best_params['ood_fp_rate'] < 0.10 else '✗ (목표: < 0.10)'}")
    print(f"  복합 score     : {best_score:.4f}")

    # 앙상블이 단독 모델보다 좋은지 확인
    ens_acc = best_params["val_acc"]
    if ens_acc >= max(acc_r, acc_1d):
        print(f"\n  → 앙상블({ens_acc:.4f})이 ResNet({acc_r:.4f}) / CNN1D({acc_1d:.4f}) 모두 상회. ✓")
    else:
        best_solo = "ResNet" if acc_r >= acc_1d else "CNN1D"
        best_solo_acc = max(acc_r, acc_1d)
        print(f"\n  → 앙상블({ens_acc:.4f})이 {best_solo} 단독({best_solo_acc:.4f})보다 낮습니다. "
              f"가중치 탐색 범위 확장 또는 모델 재학습을 검토하세요.")

    # ── ensemble_config.json 저장 ─────────────────
    new_cfg = {
        "resnet_weight": round(float(alpha), 4),
        "cnn1d_weight":  round(float(1 - alpha), 4),
        "ood_threshold": round(best_params["ood_threshold"], 4),
        "tv_threshold":  round(best_params["tv_threshold"], 4),
    }

    # 기존 config 백업
    if os.path.exists(CFG_PATH):
        backup_path = CFG_PATH.replace(".json", "_backup.json")
        with open(CFG_PATH, "r") as f:
            old_cfg = json.load(f)
        with open(backup_path, "w") as f:
            json.dump(old_cfg, f, indent=2)
        print(f"\n  기존 config 백업: {backup_path}")
        print(f"  이전 값: {old_cfg}")

    with open(CFG_PATH, "w") as f:
        json.dump(new_cfg, f, indent=2)
    print(f"  ensemble_config.json 저장 완료: {new_cfg}")
    print(f"\n  다음 단계: python evaluate_ensemble.py --data_dir {args.data_dir}")


# ─────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="ResNet18 + OrbitCNN1D 앙상블 가중치 / OOD 임계값 최적화"
    )
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/raw 상위 디렉토리 (기본값: ../data)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train_ensemble(_parse_args())
