"""
inference_daemon.py
====================
Electron 앱과 stdin/stdout JSON 통신으로 추론을 수행하는 데몬.

멀티스케일 ResNet18 + OrbitCNN1D 앙상블 지원.
레거시 단일 채널 ResNet18도 자동 감지하여 지원.

앙상블 가중치: python/ensemble_config.json (없으면 균등 0.5/0.5 기본값)
1D CNN 모델:   python/model/orbit_cnn1d.pth   (없으면 ResNet 단독 추론)

명령:
  analyze  - sec9 궤도 추론 + 동적 스케일 이미지 반환
  timeline - 초단위 궤도 이미지 생성 (동적 스케일)
"""

import sys
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_loader import load_trained_model
from preprocess import (
    make_multiscale_orbit,
    make_orbit_image_v2,
    compute_dynamic_axis_lim,
    build_multiscale_transform,
    prepare_1d_input,
)
from infer_resnet_None import (
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    predict_from_multiscale,
    predict_rcp_single,          # 레거시 호환
    extract_rcp_xy_from_bin,
    generate_gradcam_on_display,
    generate_gradcam_images,     # 레거시 호환
    render_with_axes,
    build_transform_from_meta,
)
from utils import image_to_base64

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
FS = 40_000
RCP_NAMES = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]

MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
# 새 모델 없으면 레거시 fallback
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")

CNN1D_MODEL_PATH    = os.path.join(SCRIPT_DIR, "model", "orbit_cnn1d.pth")
ENSEMBLE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "ensemble_config.json")
CLASS_MAP_PATH       = os.path.join(SCRIPT_DIR, "class_map.json")

print(f"[Daemon] resnet model path: {MODEL_PATH}", file=sys.stderr)
print(f"[Daemon] 1d cnn model path: {CNN1D_MODEL_PATH}", file=sys.stderr)

# ─────────────────────────────────────────────
# class_map.json — 설계 단계 클래스 정렬 소스
# ─────────────────────────────────────────────
try:
    with open(CLASS_MAP_PATH, "r") as _f:
        CANONICAL_CLASS_NAMES: list = json.load(_f)["classes"]
    print(f"[Daemon] class_map: {CANONICAL_CLASS_NAMES}", file=sys.stderr)
except Exception as e:
    print(f"[Daemon] WARNING: class_map.json 로드 실패 ({e}). 기본값 사용.", file=sys.stderr)
    CANONICAL_CLASS_NAMES = ["normal", "abnormal"]

# ─────────────────────────────────────────────
# 앙상블 가중치 로드
# ─────────────────────────────────────────────
def _load_ensemble_config():
    try:
        with open(ENSEMBLE_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        rw = float(cfg.get("resnet_weight", 0.5))
        cw = float(cfg.get("cnn1d_weight",  0.5))
        total = rw + cw
        if total <= 0:
            print(
                "[Daemon] Warning: Using default weights (0.5/0.5) — "
                "ensemble_config.json 의 가중치 합이 0 이하입니다.",
                file=sys.stderr,
            )
            return 0.5, 0.5
        return rw / total, cw / total  # 정규화
    except FileNotFoundError:
        print(
            "[Daemon] Warning: Using default weights (0.5/0.5) — "
            f"ensemble_config.json 파일을 찾을 수 없습니다: {ENSEMBLE_CONFIG_PATH}",
            file=sys.stderr,
        )
        return 0.5, 0.5
    except Exception as e:
        print(
            f"[Daemon] Warning: Using default weights (0.5/0.5) — "
            f"ensemble_config.json 파싱 실패: {e}",
            file=sys.stderr,
        )
        return 0.5, 0.5

resnet_weight, cnn1d_weight = _load_ensemble_config()
print(f"[Daemon] ensemble weights: resnet={resnet_weight:.2f}, cnn1d={cnn1d_weight:.2f}", file=sys.stderr)

# ─────────────────────────────────────────────
# 콜드 스타트: 모델 로드
# ─────────────────────────────────────────────
try:
    model, class_names, model_meta = load_trained_model(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 모델 타입에 맞는 transform 빌드
    transform = build_transform_from_meta(model_meta)
    is_multiscale = (model_meta.get("model_type") == "resnet18_multiscale")

    # ResNet 클래스 정렬 검증 (class_map.json 기준)
    if class_names != CANONICAL_CLASS_NAMES:
        print(
            f"[Daemon] FATAL: ResNet class_names={class_names} 가 "
            f"class_map.json({CANONICAL_CLASS_NAMES})과 다릅니다. "
            "모델을 재학습하거나 class_map.json을 확인하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[Daemon] resnet loaded: {model_meta.get('model_type', 'unknown')} | "
        f"classes={class_names}",
        file=sys.stderr,
    )
except SystemExit:
    raise
except Exception as e:
    print(f"[Daemon] ERROR loading resnet model: {e}", file=sys.stderr)
    sys.exit(1)

# 1D CNN 모델 (옵션)
model_1d = None
class_names_1d = None

if os.path.exists(CNN1D_MODEL_PATH):
    try:
        model_1d, class_names_1d, _ = load_trained_model(CNN1D_MODEL_PATH)
        model_1d.to(device)
        model_1d.eval()

        # 1D CNN 클래스 정렬 검증 — 불일치 시 시스템 중단 없이 비활성화
        if class_names_1d != CANONICAL_CLASS_NAMES:
            print(
                f"[Daemon] Warning: 1D CNN class_names={class_names_1d} 가 "
                f"class_map.json({CANONICAL_CLASS_NAMES})과 다릅니다. "
                "앙상블을 비활성화하고 ResNet 단독 모드로 계속합니다.",
                file=sys.stderr,
            )
            model_1d = None
        elif class_names_1d != class_names:
            print(
                f"[Daemon] Warning: 1D CNN class_names={class_names_1d} 가 "
                f"ResNet class_names={class_names}와 다릅니다. "
                "앙상블을 비활성화하고 ResNet 단독 모드로 계속합니다.",
                file=sys.stderr,
            )
            model_1d = None
        else:
            print(f"[Daemon] 1d cnn loaded | classes={class_names_1d}", file=sys.stderr)
    except Exception as e:
        print(f"[Daemon] WARNING: 1D CNN load failed ({e}), falling back to single-model.", file=sys.stderr)
        model_1d = None
else:
    print("[Daemon] orbit_cnn1d.pth not found — single-model mode.", file=sys.stderr)

# DaemonPool 준비 완료 신호 (PythonDaemonPool.ts가 이 문자열을 감지)
print("model loaded successfully", file=sys.stderr)


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
def _make_display_pil(x_seg, y_seg, axis_lim):
    """단일 채널 display PIL 이미지 (동적 스케일)"""
    arr = make_orbit_image_v2(x_seg, y_seg, axis_lim=axis_lim, img_size=256)
    return Image.fromarray(arr, mode='L')


def _predict_resnet(x_seg, y_seg, ms_arr_cache=None):
    """
    멀티스케일 또는 레거시 ResNet 예측.
    반환: (pred_class, prob_array)
    """
    if is_multiscale:
        ms_arr = ms_arr_cache if ms_arr_cache is not None else make_multiscale_orbit(x_seg, y_seg, img_size=256)
        return predict_from_multiscale(model, class_names, ms_arr, transform)
    else:
        from infer_resnet_None import make_orbit_image
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=256)
        pil = Image.fromarray(arr, mode='L')
        return predict_rcp_single(model, class_names, pil, transform)


def _predict_1d_cnn(x_seg, y_seg):
    """
    1D CNN 예측.
    반환: (pred_class, prob_array)
    """
    arr = prepare_1d_input(x_seg, y_seg)                          # (2, 40000)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)        # (1, 2, 40000)
    with torch.no_grad():
        logits = model_1d(tensor)                                  # (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (num_classes,)
    pred_idx = int(probs.argmax())
    return class_names_1d[pred_idx], probs


def _ensemble_predict(x_seg, y_seg, ms_arr_cache=None):
    """
    ResNet + 1D CNN 가중 앙상블 예측.
    1D CNN 없으면 ResNet 단독 결과 반환.

    반환:
        pred_class  : str
        ens_probs   : np.ndarray (num_classes,)  — 앙상블 확률
        resnet_pred : str
        resnet_probs: np.ndarray
        cnn1d_pred  : str | None
        cnn1d_probs : np.ndarray | None
    """
    resnet_pred, resnet_probs = _predict_resnet(x_seg, y_seg, ms_arr_cache)

    if model_1d is None:
        return resnet_pred, resnet_probs, resnet_pred, resnet_probs, None, None

    cnn1d_pred, cnn1d_probs = _predict_1d_cnn(x_seg, y_seg)

    # 가중 평균 — class_names 순서가 동일하다고 가정 (모두 ["normal", "abnormal"])
    ens_probs = resnet_weight * resnet_probs + cnn1d_weight * cnn1d_probs
    pred_idx  = int(ens_probs.argmax())
    pred_class = class_names[pred_idx]

    return pred_class, ens_probs, resnet_pred, resnet_probs, cnn1d_pred, cnn1d_probs


def _gradcam(x_seg, y_seg, display_pil, axis_lim, ms_arr_cache=None, class_idx=None):
    """
    GradCAM 생성.
    class_idx: 앙상블의 최종 예측 클래스 인덱스.
               전달 시 "앙상블이 결정한 클래스" 기준으로 ResNet 활성화 맵 생성.
               None이면 ResNet 자체 예측 클래스 사용.
    """
    if is_multiscale:
        ms_arr = ms_arr_cache if ms_arr_cache is not None else make_multiscale_orbit(x_seg, y_seg, img_size=256)
        return generate_gradcam_on_display(
            model, class_names, ms_arr, display_pil, transform,
            class_idx=class_idx,
        )
    else:
        from infer_resnet_None import make_orbit_image
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=256)
        pil = Image.fromarray(arr, mode='L')
        return generate_gradcam_images(model, class_names, pil, transform,
                                       class_idx=class_idx)


# ─────────────────────────────────────────────
# 데몬 루프
# ─────────────────────────────────────────────
def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF

            req     = json.loads(line)
            command = req.get("command")
            payload = req.get("payload", {})
            bin_path = payload.get("bin_path")

            print(f"[Daemon] command={command}", file=sys.stderr)

            response = {"status": "error", "data": None}

            if command in ("analyze", "timeline") and not bin_path:
                response["message"] = "payload.bin_path is required"
                print(json.dumps(response))
                sys.stdout.flush()
                continue

            # ── analyze ──────────────────────────────────────
            if command == "analyze":
                rcp_xy = extract_rcp_xy_from_bin(bin_path, fs=FS)

                results    = {}
                images_b64 = {}

                for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                    # sec9
                    x_seg = x_mil_full[9 * FS : 10 * FS]
                    y_seg = y_mil_full[9 * FS : 10 * FS]

                    if len(x_seg) < FS:
                        raise ValueError(
                            f"{rcp}: sec9 구간이 너무 짧습니다 "
                            f"({len(x_seg)} samples, 필요: {FS}). "
                            "BIN 파일이 10초 미만일 수 있습니다."
                        )

                    # 동적 표시 스케일
                    display_axis_lim = compute_dynamic_axis_lim(x_seg, y_seg)

                    # 멀티스케일 배열 1회 생성 → 예측 + GradCAM에서 재사용
                    ms_arr_cache = make_multiscale_orbit(x_seg, y_seg, img_size=256) if is_multiscale else None

                    # 앙상블 예측
                    (pred_class, ens_probs,
                     resnet_pred, resnet_probs,
                     cnn1d_pred, cnn1d_probs) = _ensemble_predict(x_seg, y_seg, ms_arr_cache)

                    result_entry = {
                        "prediction": pred_class,
                        "probabilities": {
                            name: float(p) for name, p in zip(class_names, ens_probs)
                        },
                        "display_axis_lim": display_axis_lim,
                        "model_predictions": {
                            "resnet": {
                                "prediction":   resnet_pred,
                                "probabilities": {
                                    name: float(p) for name, p in zip(class_names, resnet_probs)
                                },
                            },
                        },
                    }

                    if cnn1d_pred is not None:
                        result_entry["model_predictions"]["cnn1d"] = {
                            "prediction":   cnn1d_pred,
                            "probabilities": {
                                name: float(p) for name, p in zip(class_names_1d, cnn1d_probs)
                            },
                        }

                    results[rcp] = result_entry

                    # 앙상블 클래스 인덱스 (Grad-CAM 타겟)
                    ens_class_idx = int(ens_probs.argmax())

                    # 표시용 단일 채널 이미지 (동적 스케일)
                    display_pil = _make_display_pil(x_seg, y_seg, display_axis_lim)

                    # GradCAM — 앙상블 예측 클래스 기준으로 ResNet 활성화 맵 생성
                    gradcam_imgs = _gradcam(
                        x_seg, y_seg, display_pil, display_axis_lim,
                        ms_arr_cache, class_idx=ens_class_idx,
                    )

                    # 렌더링 레이블
                    target_cls   = gradcam_imgs.get("target_class", pred_class)
                    scale_label  = f"±{display_axis_lim:.1f} mil"
                    gcam_label   = f"{target_cls} · Grad-CAM (ensemble)"

                    images_b64[rcp] = {
                        "orbit": image_to_base64(
                            render_with_axes(display_pil, display_axis_lim,
                                             cmap='gray', label=scale_label)
                        ),
                        "heatmap": image_to_base64(
                            render_with_axes(gradcam_imgs["heatmap"], display_axis_lim,
                                             label=gcam_label)
                        ),
                        "overlay": image_to_base64(
                            render_with_axes(gradcam_imgs["overlay"], display_axis_lim,
                                             label=gcam_label)
                        ),
                    }

                final_label = (
                    "abnormal"
                    if any(r["prediction"] == "abnormal" for r in results.values())
                    else "normal"
                )

                # 어떤 모델이 활성화됐는지 표시
                active_models = [model_meta.get("model_type", "resnet18_legacy")]
                if model_1d is not None:
                    active_models.append("orbit_cnn1d")

                response = {
                    "status": "ok",
                    "type":   "analysis_result",
                    "data": {
                        "final_label": final_label,
                        "model_info":  " + ".join(active_models),
                        "results":     results,
                        "images":      images_b64,
                    },
                }

            # ── timeline ─────────────────────────────────────
            elif command == "timeline":
                rcp_xy = extract_rcp_xy_from_bin(bin_path, fs=FS)

                timeline_b64 = {}
                for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                    # 전체 신호 기준으로 스케일 결정 (초별 일관성 유지)
                    full_axis_lim = compute_dynamic_axis_lim(x_mil_full, y_mil_full)

                    sec_images = []
                    duration_sec = 10
                    for sec in range(duration_sec):
                        x_seg = x_mil_full[sec * FS : (sec + 1) * FS]
                        y_seg = y_mil_full[sec * FS : (sec + 1) * FS]
                        display_pil = _make_display_pil(x_seg, y_seg, full_axis_lim)
                        rendered = render_with_axes(display_pil, full_axis_lim, cmap='gray')
                        sec_images.append(image_to_base64(rendered))

                    timeline_b64[rcp] = sec_images

                response = {
                    "status": "ok",
                    "type":   "timeline_result",
                    "data":   timeline_b64,
                }

            else:
                response["message"] = f"Unknown command: {command}"

            print(json.dumps(response))
            sys.stdout.flush()

        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            err_response = {"status": "error", "message": str(e)}
            print(json.dumps(err_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
