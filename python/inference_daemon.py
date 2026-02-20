"""
inference_daemon.py
====================
Electron 앱과 stdin/stdout JSON 통신으로 추론을 수행하는 데몬.

멀티스케일 모델 (resnet18_orbit_multiscale.pth) 전용.
레거시 모델도 자동 감지하여 지원.

명령:
  analyze  - sec9 궤도 추론 + 동적 스케일 이미지 반환
  timeline - 초단위 궤도 이미지 생성 (동적 스케일)
"""

import sys
import json
import os
import torch
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

print(f"[Daemon] model path: {MODEL_PATH}", file=sys.stderr)

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

    print(
        f"[Daemon] model loaded: {model_meta.get('model_type', 'unknown')} | "
        f"classes={class_names}",
        file=sys.stderr,
    )
except Exception as e:
    print(f"[Daemon] ERROR loading model: {e}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
def _render_display(display_arr_or_pil, axis_lim, cmap='gray'):
    """동적 axis_lim으로 render_with_axes 호출"""
    return render_with_axes(display_arr_or_pil, axis_lim=axis_lim, cmap=cmap)


def _make_display_pil(x_seg, y_seg, axis_lim):
    """단일 채널 display PIL 이미지 (동적 스케일)"""
    arr = make_orbit_image_v2(x_seg, y_seg, axis_lim=axis_lim, img_size=256)
    return Image.fromarray(arr, mode='L')


def _predict(x_seg, y_seg):
    """
    멀티스케일 또는 레거시 방식으로 예측.
    반환: (pred_class, prob_array)
    """
    if is_multiscale:
        ms_arr = make_multiscale_orbit(x_seg, y_seg, img_size=256)
        return predict_from_multiscale(model, class_names, ms_arr, transform)
    else:
        # 레거시: 단일 채널 PIL
        from infer_resnet_None import make_orbit_image
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=256)
        pil = Image.fromarray(arr, mode='L')
        return predict_rcp_single(model, class_names, pil, transform)


def _gradcam(x_seg, y_seg, display_pil, axis_lim):
    """
    GradCAM 생성.
    멀티스케일: generate_gradcam_on_display (모델 입력 = 3ch, 오버레이 = display)
    레거시: generate_gradcam_images (단일 채널)
    """
    if is_multiscale:
        ms_arr = make_multiscale_orbit(x_seg, y_seg, img_size=256)
        return generate_gradcam_on_display(
            model, class_names, ms_arr, display_pil, transform
        )
    else:
        from infer_resnet_None import make_orbit_image
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=256)
        pil = Image.fromarray(arr, mode='L')
        return generate_gradcam_images(model, class_names, pil, transform)


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

            # ── analyze ──────────────────────────────────────
            if command == "analyze":
                rcp_xy = extract_rcp_xy_from_bin(bin_path, fs=FS)

                results    = {}
                images_b64 = {}

                for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                    # sec9
                    x_seg = x_mil_full[9 * FS : 10 * FS]
                    y_seg = y_mil_full[9 * FS : 10 * FS]

                    # 동적 표시 스케일
                    display_axis_lim = compute_dynamic_axis_lim(x_seg, y_seg)

                    # 예측
                    pred_class, prob = _predict(x_seg, y_seg)
                    results[rcp] = {
                        "prediction": pred_class,
                        "probabilities": {
                            name: float(p) for name, p in zip(class_names, prob)
                        },
                        "display_axis_lim": display_axis_lim,
                    }

                    # 표시용 단일 채널 이미지 (동적 스케일)
                    display_pil = _make_display_pil(x_seg, y_seg, display_axis_lim)

                    # GradCAM
                    gradcam_imgs = _gradcam(x_seg, y_seg, display_pil, display_axis_lim)

                    # 렌더링 (동적 axis_lim으로 축 표시)
                    images_b64[rcp] = {
                        "orbit": image_to_base64(
                            _render_display(display_pil, display_axis_lim, cmap='gray')
                        ),
                        "heatmap": image_to_base64(
                            _render_display(gradcam_imgs["heatmap"], display_axis_lim)
                        ),
                        "overlay": image_to_base64(
                            _render_display(gradcam_imgs["overlay"], display_axis_lim)
                        ),
                    }

                final_label = (
                    "abnormal"
                    if any(r["prediction"] == "abnormal" for r in results.values())
                    else "normal"
                )

                response = {
                    "status": "ok",
                    "type":   "anlysis_result",
                    "data": {
                        "final_label": final_label,
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
                        rendered = _render_display(display_pil, full_axis_lim, cmap='gray')
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
