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
import io
import json
import os

# ── stdin/stdout UTF-8 강제 설정 (Windows에서 한글 파일명 등 비ASCII 문자 처리) ──
# PYTHONIOENCODING 환경변수는 PyInstaller 번들 / 일부 subprocess 초기화 시 적용이
# 보장되지 않으므로, 프로그램 시작 시 명시적으로 인코딩을 재설정한다.
try:
    sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding='utf-8', errors='replace')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace',
                                  line_buffering=True)
except AttributeError:
    # 이미 TextIOWrapper이거나 buffer 속성이 없는 환경 (테스트 harness 등) — 무시
    pass
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.signal import spectrogram as _scipy_spectrogram
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_loader import load_trained_model
from model_mae import OrbitMAE, OrbitMAE1D
from rcpvms_parser import RcpvmsParser
from preprocess import (
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    make_orbit_image,
    make_multiscale_orbit,
    make_orbit_image_v2,
    make_orbit_display_image,
    compute_dynamic_axis_lim,
    build_multiscale_transform,
    prepare_1d_input,
    prepare_1d_input_fixed,
    make_spectrogram_4ch,
    SPEC_NPERSEG, SPEC_NOVERLAP, SPEC_F_MAX_HZ,
    estimate_1x_freq,
    filter_1x_bandpass,
    filter_2x_bandpass,
    filter_broadband,
)
from infer_resnet_None import (
    predict_from_multiscale,
    predict_rcp_single,          # 레거시 호환
    extract_rcp_xy_from_bin,
    generate_gradcam_on_display,
    generate_gradcam_images,     # 레거시 호환
    render_with_axes,
    build_transform_from_meta,
)
from utils import image_to_base64

# Integrated Gradients (옵션 — 로드 실패 시 graceful disable)
try:
    from integrated_gradients import render_ig_resnet
    IG_AVAILABLE = True
except Exception as _ig_import_err:
    print(f"[Daemon] WARNING: integrated_gradients 로드 실패 ({_ig_import_err}), IG 비활성화.",
          file=sys.stderr)
    IG_AVAILABLE = False

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
FS      = 40_000
SW_STEP = FS // 10  # 슬라이딩 윈도우 스텝 (90% 오버랩)
RCP_NAMES = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
_FILTER_LABELS = {"raw": "Raw", "1x": "1X", "2x": "2X", "broadband": "BB", "overlay": "Overlay"}

MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
# 새 모델 없으면 레거시 fallback
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")

CNN1D_MODEL_PATH     = os.path.join(SCRIPT_DIR, "model", "orbit_cnn1d.pth")
ENSEMBLE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "ensemble_config.json")
CLASS_MAP_PATH       = os.path.join(SCRIPT_DIR, "class_map.json")
MAE_MODEL_PATH       = os.path.join(SCRIPT_DIR, "model", "orbit_mae.pth")
MAE_CONFIG_PATH      = os.path.join(SCRIPT_DIR, "mae_config.json")

# 앙상블 최대 확률이 임계값 미만이면 OOD(분포 외) 판정
OOD_CLASS_NAME = "unknown_abnormal"

# ── RCPVMS 파일 헤더 캐시 ────────────────────────────────
# filepath → (mtime, info, orbit_map)
# scale mode 전환 시 동일 파일 헤더를 반복 파싱하지 않도록 캐싱.
# mtime이 바뀌면 캐시를 무효화해 파일 교체를 반영한다.
_rcpvms_header_cache: dict = {}

# rcpvms_orbit_single 반복 호출 시 read_orbit_data 재실행 방지 캐시.
# 키: (filepath, mtime, window_sec) — mtime 변경 또는 window_sec 변경 시 무효화.
# 메모리 절감: 1개 항목만 유지 (clear 후 저장).
_rcpvms_orbit_cache: dict = {}

# dmd_info / dmd_orbit_timeline 연속 호출 시 블록 체인 재스캔 방지 캐시.
# key: dmd_path, value: (mtime, DmdFileInfo)
_dmd_info_cache: dict = {}

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
        rw      = float(cfg.get("resnet_weight", 0.5))
        cw      = float(cfg.get("cnn1d_weight",  0.5))
        ood_thr = float(cfg.get("ood_threshold", 0.65))
        tv_thr  = float(cfg.get("tv_threshold",  0.30))
        total = rw + cw
        if total <= 0:
            print(
                "[Daemon] Warning: Using default weights (0.5/0.5) — "
                "ensemble_config.json 의 가중치 합이 0 이하입니다.",
                file=sys.stderr,
            )
            return 0.5, 0.5, ood_thr, tv_thr
        return rw / total, cw / total, ood_thr, tv_thr  # 정규화
    except FileNotFoundError:
        print(
            "[Daemon] Warning: Using default weights (0.5/0.5) — "
            f"ensemble_config.json 파일을 찾을 수 없습니다: {ENSEMBLE_CONFIG_PATH}",
            file=sys.stderr,
        )
        return 0.5, 0.5, 0.65, 0.40
    except Exception as e:
        print(
            f"[Daemon] Warning: Using default weights (0.5/0.5) — "
            f"ensemble_config.json 파싱 실패: {e}",
            file=sys.stderr,
        )
        return 0.5, 0.5, 0.65, 0.40

resnet_weight, cnn1d_weight, ood_threshold, tv_threshold = _load_ensemble_config()
print(
    f"[Daemon] ensemble weights: resnet={resnet_weight:.2f}, cnn1d={cnn1d_weight:.2f}, "
    f"ood_threshold={ood_threshold:.2f}, tv_threshold={tv_threshold:.2f}",
    file=sys.stderr,
)

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

    # 체크포인트에 저장된 학습 img_size — 추론 입력과 반드시 일치해야 함.
    # evaluate_ensemble.py 와 동일하게 meta에서 읽으며, 절대 하드코딩 금지.
    INFERENCE_IMG_SIZE = int(model_meta.get("img_size", 128))
    # channel_mode: "hybrid" → Ch2 고정 6.0 mil / "dynamic" → 전채널 동적 (레거시)
    CHANNEL_HYBRID = (model_meta.get("channel_mode", "dynamic") == "hybrid")
    print(
        f"[Daemon] inference img_size={INFERENCE_IMG_SIZE}, "
        f"channel_mode={model_meta.get('channel_mode', 'dynamic')}",
        file=sys.stderr,
    )

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

# ─────────────────────────────────────────────
# MAE 모델 로드 (옵션 — 없으면 graceful disable)
# ─────────────────────────────────────────────
mae_model            = None
mae_threshold        = None
mae_threshold_1d     = None   # OR 로직용 1D 브랜치 독립 임계값
mae_threshold_spec   = None   # OR 로직용 spec 브랜치 독립 임계값
mae_scale_mil        = None
mae_use_spec         = False
mae_alpha            = 0.5
mae_spec_mask_ratio  = 0.85
mae_topk_ratio       = 1.0    # 이상 점수 상위 K% 패치 사용 (1.0=전체 평균)

if os.path.exists(MAE_MODEL_PATH):
    try:
        if os.path.exists(MAE_CONFIG_PATH):
            with open(MAE_CONFIG_PATH, "r") as _f:
                _mae_cfg = json.load(_f)
            mae_threshold       = float(_mae_cfg.get("threshold", 0.0))
            mae_threshold_1d    = (_mae_cfg.get("threshold_1d")   and
                                   float(_mae_cfg["threshold_1d"])) or None
            mae_threshold_spec  = (_mae_cfg.get("threshold_spec") and
                                   float(_mae_cfg["threshold_spec"])) or None
            mae_scale_mil       = float(_mae_cfg.get("scale_mil", 1.0))
            mae_use_spec        = bool(_mae_cfg.get("use_spec", False))
            mae_alpha           = float(_mae_cfg.get("alpha", 0.5))
            mae_spec_mask_ratio = float(_mae_cfg.get("spec_mask_ratio", 0.85))
            mae_topk_ratio      = float(_mae_cfg.get("topk_ratio", 1.0))
            _or_status = (f"1D={mae_threshold_1d:.6f} spec={mae_threshold_spec:.6f}"
                          if mae_threshold_1d and mae_threshold_spec else "비활성(config 없음)")
            print(f"[Daemon] mae_config: threshold={mae_threshold:.6f}, "
                  f"scale_mil={mae_scale_mil:.4f}, use_spec={mae_use_spec}, "
                  f"alpha={mae_alpha:.2f}, spec_mask_ratio={mae_spec_mask_ratio:.2f}, "
                  f"topk_ratio={mae_topk_ratio:.2f}, OR-logic={_or_status}",
                  file=sys.stderr)
        else:
            print("[Daemon] WARNING: mae_config.json 없음 — MAE 비활성화.", file=sys.stderr)
            raise FileNotFoundError("mae_config.json not found")

        _mae_ckpt = torch.load(MAE_MODEL_PATH, map_location="cpu")
        _full_sd  = _mae_ckpt["model_state_dict"]

        if mae_use_spec:
            # OrbitMAE 래퍼(1D + spec) 직접 로드 — spec_mask_ratio 복원
            mae_model = OrbitMAE(
                use_spec=True, alpha=mae_alpha,
                spec_mask_ratio=mae_spec_mask_ratio,
            )
            mae_model.load_state_dict(_full_sd)
        else:
            # 1D 브랜치만 사용: branch_1d.* 접두사 제거
            _prefix     = "branch_1d."
            _has_prefix = any(k.startswith(_prefix) for k in _full_sd)
            _branch_sd  = (
                {k[len(_prefix):]: v for k, v in _full_sd.items() if k.startswith(_prefix)}
                if _has_prefix else _full_sd
            )
            mae_model = OrbitMAE1D()
            mae_model.load_state_dict(_branch_sd)

        mae_model.to(device)
        mae_model.eval()
        print(f"[Daemon] MAE model loaded: threshold={mae_threshold:.6f}, "
              f"mode={'1D+Spec' if mae_use_spec else '1D-only'}", file=sys.stderr)
    except Exception as e:
        print(f"[Daemon] WARNING: MAE 모델 로드 실패 ({e}), MAE 비활성화.", file=sys.stderr)
        mae_model = None
else:
    print("[Daemon] orbit_mae.pth not found — MAE disabled.", file=sys.stderr)

# MAE OR-로직 활성 여부 — 모듈 로드 시 1회 계산 (globals 불변)
_or_active: bool = bool(mae_use_spec and mae_threshold_1d and mae_threshold_spec)

# DaemonPool 준비 완료 신호 (PythonDaemonPool.ts가 이 문자열을 감지)
print("model loaded successfully", file=sys.stderr)


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
# sosfiltfilt 과도 응답 제거 후 정수 사이클 트리밍에 쓸 엣지 마진 (사이클 수)
_FILTER_EDGE_CYCLES = 5

# Minimum 1X confidence for bandpass filtering. Below this threshold the frequency
# estimate is unreliable (flat/noisy spectrum) and broadband fallback is applied instead.
_1X_CONF_THRESHOLD = 0.35


def _trim_to_integer_cycles(
    x: np.ndarray, y: np.ndarray, f_ref: float, fs: float
) -> tuple:
    """
    sosfiltfilt 엣지 과도 응답 제거 + 정수 사이클 트리밍.

    1. 양단 엣지 사이클 제거 (filtfilt 과도 응답 구간) → 나선형 궤도 방지
       - 최대 _FILTER_EDGE_CYCLES 사이클 제거하되, 2사이클 이상 남도록 적응 조절.
       - 300~12000 RPM, 1초 윈도우에서도 동작 보장.
    2. 남은 구간에서 정수 사이클로 후단 트리밍 → 미폐합 궤도 방지

    Returns:
        (x, y, edge_trimmed: bool)
        edge_trimmed=False 이면 엣지 트리밍이 건너뛰어짐 (filtfilt 과도 응답 잔존 가능).
        f_ref <= 0 이면 원신호를 그대로 반환한다.
    """
    if f_ref <= 0:
        return x, y, False

    n_samples_cycle = fs / f_ref
    n_cycles_total = len(x) / n_samples_cycle

    # 적응형 엣지 사이클: 트리밍 후 최소 2사이클이 남도록 동적 조절
    # max_edge_cycles = (총 사이클 - 2) / 2  (양단 균등 분배)
    max_edge_cycles = max(0.0, (n_cycles_total - 2.0) / 2.0)
    actual_edge_cycles = min(float(_FILTER_EDGE_CYCLES), max_edge_cycles)

    edge_trimmed = False
    if actual_edge_cycles >= 0.5:  # 최소 반 사이클 이상 트리밍해야 의미 있음
        n_edge = int(n_samples_cycle * actual_edge_cycles)
        if n_edge > 0 and n_edge * 2 < len(x):
            x = x[n_edge:-n_edge]
            y = y[n_edge:-n_edge]
            edge_trimmed = True
    else:
        print(
            f"[trim_cycles] 엣지 트리밍 건너뜀: 총 {n_cycles_total:.1f}사이클 "
            f"(f_ref={f_ref:.1f}Hz) — 2사이클 최소 잔존을 보장할 수 없음",
            file=sys.stderr,
        )

    n_cycles = int(len(x) / n_samples_cycle)
    if n_cycles >= 1:
        n_trim = int(n_cycles * n_samples_cycle)
        x = x[:n_trim]
        y = y[:n_trim]

    return x, y, edge_trimmed


def _freq_estimate_payload(est):
    if est is None:
        return None
    return {
        "freq_hz": float(est.freq_hz),
        "confidence": float(est.confidence),
        "flags": list(est.flags),
    }


def _draw_crosshair(pil_img, color=(80, 80, 80), alpha=180):
    """Draw a faint center crosshair on a PIL image in-place (PIL-only, no matplotlib)."""
    from PIL import ImageDraw as _IDraw
    w, h = pil_img.size
    cx, cy = w // 2, h // 2
    overlay = pil_img.convert("RGBA")
    draw = _IDraw.Draw(overlay)
    draw.line([(0, cy), (w - 1, cy)], fill=(*color, alpha), width=1)
    draw.line([(cx, 0), (cx, h - 1)], fill=(*color, alpha), width=1)
    return overlay.convert(pil_img.mode)


def _make_display_pil(x_seg, y_seg, axis_lim, fs=None, filter_mode="1x", img_size=256, f1x_hint=None):
    """Single-channel display PIL image with optional filtering and frequency metadata.

    f1x_hint: pre-computed FrequencyEstimate for this position (skips per-cell FFT).
              Pass None to compute fresh from x_seg + y_seg.
    """
    x_seg = x_seg - x_seg.mean()
    y_seg = y_seg - y_seg.mean()
    actual_filter = "raw" if (fs is None or fs <= 0) else filter_mode
    edge_trim_applied = False
    freq_estimate = None

    if fs is not None and fs > 0 and filter_mode != "raw":
        try:
            if filter_mode in ("1x", "2x"):
                # Reuse pre-computed estimate when available (avoids per-cell FFT in batch)
                if f1x_hint is not None:
                    freq_estimate = f1x_hint
                else:
                    freq_estimate = estimate_1x_freq(x_seg, fs, y_mil=y_seg)

                # Low-confidence or harmonics-absent → signal is noisy; bandpass would be meaningless.
                # Fall back to broadband (DC-drift removed, full spectrum retained).
                if (freq_estimate.confidence < _1X_CONF_THRESHOLD
                        or "no_harmonic_support" in freq_estimate.flags):
                    actual_filter = "broadband"
                    x_seg, y_seg = filter_broadband(x_seg, y_seg, fs)
                else:
                    f1x = freq_estimate.freq_hz
                    if filter_mode == "2x":
                        x_seg, y_seg, f_ref = filter_2x_bandpass(x_seg, y_seg, fs, f1x=f1x)
                    else:
                        x_seg, y_seg, f_ref = filter_1x_bandpass(x_seg, y_seg, fs, f1x=f1x)
                    x_seg, y_seg, edge_trim_applied = _trim_to_integer_cycles(x_seg, y_seg, f_ref, fs)
            elif filter_mode == "broadband":
                x_seg, y_seg = filter_broadband(x_seg, y_seg, fs)
        except Exception as _fe:
            print(f"[{filter_mode} filter fallback] {_fe}", file=sys.stderr)
            actual_filter = "raw"

    if actual_filter != "raw":
        used_axis_lim = compute_dynamic_axis_lim(x_seg, y_seg)
    else:
        used_axis_lim = axis_lim
    arr = make_orbit_display_image(x_seg, y_seg, axis_lim=used_axis_lim, img_size=img_size)
    return Image.fromarray(arr, mode='L'), actual_filter, used_axis_lim, edge_trim_applied, _freq_estimate_payload(freq_estimate)


def _make_overlay_pil(x_seg, y_seg, axis_lim, fs=None, img_size=256):
    """Raw / 1X / 2X / BB 4가지 필터를 색상으로 구분하여 단일 캔버스에 오버레이.
    axis_lim은 raw 신호 기준으로 공통 적용하므로 각 성분의 상대 진폭 비교 가능.
    반환: PIL RGB Image
    """
    from PIL import ImageDraw as _IDraw, Image as _PILImg
    from scipy.ndimage import gaussian_filter as _gfilt

    x0 = x_seg - x_seg.mean()
    y0 = y_seg - y_seg.mean()

    # 검정 배경 기준 최대 대비 4색 (Hue 간격 ~90°, 채도·명도 균일)
    # Raw: 흰색-회색(무채색 기준), BB: 노란색, 2X: 진홍, 1X: 하늘색
    _OVERLAY_COLORS = {
        'raw':       (210, 210, 210),   # 밝은 회백 — 원신호 기준선
        'broadband': (255, 210,   0),   # 선명한 황색 — BB 전대역
        '2x':        (255,  50,  90),   # 진홍/빨강 — 2X 성분
        '1x':        (  0, 210, 255),   # 하늘색/시안 — 1X 동기 성분
    }
    _DRAW_ORDER = ['raw', 'broadband', '2x', '1x']

    scale  = (img_size - 1) / (2.0 * axis_lim)
    center = (img_size - 1) / 2.0

    def _to_px(xm, ym):
        px = int(round(center + xm * scale))
        py = int(round(center + ym * scale))
        return (max(0, min(img_size - 1, px)), max(0, min(img_size - 1, py)))

    canvas = _PILImg.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = _IDraw.Draw(canvas)
    f1x_est = None
    if fs is not None and fs > 0:
        try:
            f1x_est = estimate_1x_freq(x0, fs).freq_hz
        except Exception:
            f1x_est = None

    for fmode in _DRAW_ORDER:
        xf, yf = x0.copy(), y0.copy()
        if fs is not None and fs > 0 and fmode != 'raw':
            try:
                if fmode == '1x':
                    xf, yf, f1x = filter_1x_bandpass(xf, yf, fs, f1x=f1x_est)
                    xf, yf, _ = _trim_to_integer_cycles(xf, yf, f1x, fs)
                elif fmode == '2x':
                    xf, yf, f2x = filter_2x_bandpass(xf, yf, fs, f1x=f1x_est)
                    xf, yf, _ = _trim_to_integer_cycles(xf, yf, f2x, fs)
                elif fmode == 'broadband':
                    xf, yf = filter_broadband(xf, yf, fs)
            except Exception:
                xf, yf = x0.copy(), y0.copy()

        n = len(xf)
        max_points = max(1200, img_size * 20)
        step = max(1, n // max_points)
        pts = [_to_px(xf[i], yf[i]) for i in range(0, n, step)]
        if len(pts) >= 2:
            draw.line(pts, fill=_OVERLAY_COLORS[fmode], width=1)

    arr = np.array(canvas, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = _gfilt(arr[:, :, c], sigma=0.6)
    return _PILImg.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='RGB')


def _predict_resnet(x_seg, y_seg, ms_arr_cache=None):
    """
    멀티스케일 또는 레거시 ResNet 예측.
    반환: (pred_class, prob_array)
    """
    if is_multiscale:
        ms_arr = ms_arr_cache if ms_arr_cache is not None else make_multiscale_orbit(x_seg, y_seg, img_size=INFERENCE_IMG_SIZE, hybrid=CHANNEL_HYBRID)
        return predict_from_multiscale(model, class_names, ms_arr, transform)
    else:
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=INFERENCE_IMG_SIZE)
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
    ResNet + 1D CNN 가중 앙상블 예측 + OOD 탐지.
    1D CNN 없으면 ResNet 단독 결과 반환.

    OOD 판정 (복합 기준 — CNN1D 있을 때):
        조건 A: TV Distance(resnet_probs, cnn1d_probs) > tv_threshold
                두 모달리티(이미지/시계열)가 서로 크게 다른 판단 → 이상 의심
        조건 B: max(앙상블 확률) < ood_threshold
                앙상블 자체가 어떤 클래스에도 자신 없음 → 이상 의심
        A OR B 중 하나라도 충족 시 OOD 판정.
        → OE 없이도 heterogeneous 모달리티 간 불일치로 OOD를 탐지 가능.

    CNN1D 없을 때: 조건 B(max_conf)만 사용 (레거시 동작 유지).

    반환:
        pred_class  : str   — 예측 클래스명 또는 OOD_CLASS_NAME
        ens_probs   : ndarray(num_classes,) — 앙상블 확률 (OOD여도 원본 반환)
        resnet_pred : str
        resnet_probs: ndarray
        cnn1d_pred  : str | None
        cnn1d_probs : ndarray | None
        is_ood      : bool
        tv_distance : float | None  — TV Distance (CNN1D 없으면 None)
        ood_reason  : str   — OOD 판정 근거 ("tv" | "conf" | "tv+conf" | "none")
    """
    resnet_pred, resnet_probs = _predict_resnet(x_seg, y_seg, ms_arr_cache)

    # ── CNN1D 없음: max_conf 단독 판정 (레거시) ───────────────
    if model_1d is None:
        ens_probs  = resnet_probs
        pred_idx   = int(ens_probs.argmax())
        max_conf   = float(ens_probs[pred_idx])
        is_ood     = max_conf < ood_threshold
        ood_reason = "conf" if is_ood else "none"
        pred_class = OOD_CLASS_NAME if is_ood else class_names[pred_idx]
        return pred_class, ens_probs, resnet_pred, resnet_probs, None, None, is_ood, None, ood_reason

    cnn1d_pred, cnn1d_probs = _predict_1d_cnn(x_seg, y_seg)

    # ── 가중 평균 앙상블 ──────────────────────────────────────
    ens_probs = resnet_weight * resnet_probs + cnn1d_weight * cnn1d_probs
    pred_idx  = int(ens_probs.argmax())
    max_conf  = float(ens_probs[pred_idx])

    # ── TV Distance: 두 모달리티 간 확률 분포 불일치 ──────────
    # TV(P, Q) = 0.5 * Σ|P_i - Q_i|   범위: [0, 1]
    tv_distance = float(0.5 * np.sum(np.abs(resnet_probs - cnn1d_probs)))

    # ── 복합 OOD 판정 ─────────────────────────────────────────
    cond_tv   = tv_distance > tv_threshold
    cond_conf = max_conf    < ood_threshold

    if cond_tv and cond_conf:
        ood_reason = "tv+conf"
    elif cond_tv:
        ood_reason = "tv"
    elif cond_conf:
        ood_reason = "conf"
    else:
        ood_reason = "none"

    is_ood     = cond_tv or cond_conf
    pred_class = OOD_CLASS_NAME if is_ood else class_names[pred_idx]

    return (pred_class, ens_probs, resnet_pred, resnet_probs,
            cnn1d_pred, cnn1d_probs, is_ood, tv_distance, ood_reason)


# ─────────────────────────────────────────────
# MAE 시각화 헬퍼
# ─────────────────────────────────────────────

def _viridis(t):
    """Viridis colormap (0→1 float → RGB uint8 tuple)."""
    t = float(np.clip(t, 0.0, 1.0))
    stops = [
        (68,1,84),(72,35,116),(64,67,135),(52,96,141),
        (41,123,142),(32,150,139),(34,176,126),(74,194,107),
        (135,207,81),(194,219,56),(253,231,37),
    ]
    idx = t * (len(stops) - 1)
    i = min(int(idx), len(stops) - 2)
    f = idx - i
    a, b = stops[i], stops[i + 1]
    return tuple(int(a[j] + f * (b[j] - a[j])) for j in range(3))

def _inferno(t):
    """Inferno colormap (0→1 float → RGB uint8 tuple)."""
    t = float(np.clip(t, 0.0, 1.0))
    stops = [
        (0,0,4),(22,11,57),(67,15,117),(115,25,130),
        (161,44,120),(203,71,100),(237,105,73),(252,148,44),
        (251,197,30),(252,255,165),
    ]
    idx = t * (len(stops) - 1)
    i = min(int(idx), len(stops) - 2)
    f = idx - i
    a, b = stops[i], stops[i + 1]
    return tuple(int(a[j] + f * (b[j] - a[j])) for j in range(3))

# 모듈 수준 LUT 캐시 — 매 호출마다 256회 Python 함수 호출을 방지
_LUT_VIRIDIS = np.array([_viridis(i / 255.0) for i in range(256)], dtype=np.uint8)
_LUT_INFERNO = np.array([_inferno(i / 255.0) for i in range(256)], dtype=np.uint8)

def _colorize(arr2d, lut):
    """(H, W) float [0,1] → (H, W, 3) uint8 PIL Image — LUT 벡터화."""
    idx = (np.clip(arr2d, 0.0, 1.0) * 255.0).astype(np.uint8)  # (H, W)
    return Image.fromarray(lut[idx])  # (H, W, 3)

def _stft_matrix(signal, fs=40_000, nperseg=512, noverlap=448, max_freq=1000):
    """1D signal → log-power spectrogram (H×W float [0,1]), freq axis limited."""
    f, _t, Sxx = _scipy_spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    mask = f <= max_freq
    Sxx = Sxx[mask]                                    # (F, T)
    Sxx = np.log10(Sxx + 1e-12)
    vmin, vmax = Sxx.min(), Sxx.max()
    if vmax > vmin:
        Sxx = (Sxx - vmin) / (vmax - vmin)
    else:
        Sxx = np.zeros_like(Sxx)
    return Sxx[::-1].copy()                            # 저주파 → 하단

def _stft_to_pil(stft_mat, lut, out_size=(360, 200)):
    """log-power STFT matrix → PIL Image (resized to out_size)."""
    img = _colorize(stft_mat, lut)
    return img.resize(out_size, Image.BILINEAR)

def _stft_error_overlay(stft_input, stft_recon, out_size=(360, 200), threshold=0.30):
    """입력 STFT 위에 재구성 오차가 큰 영역을 빨간 오버레이로 표시."""
    base = _stft_to_pil(stft_input, _LUT_VIRIDIS, out_size)
    base = base.convert("RGBA")
    W, H = base.size
    err = np.abs(stft_input - stft_recon)
    err_norm = err / (err.max() + 1e-10)
    err_resized = np.array(Image.fromarray(
        (err_norm * 255).astype(np.uint8)
    ).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    ov_data = np.zeros((H, W, 4), dtype=np.uint8)
    mask = err_resized > threshold
    alpha = np.clip((err_resized - threshold) / (1.0 - threshold) * 200, 0, 255).astype(np.uint8)
    ov_data[mask, :3] = [255, 64, 96]
    ov_data[mask, 3]  = alpha[mask]
    overlay = Image.fromarray(ov_data, "RGBA")
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")

def _stft_error_heatmap(stft_input, stft_recon, out_size=(360, 200)):
    """절대 오차 STFT → Inferno heatmap PIL Image."""
    err = np.abs(stft_input - stft_recon)
    mx = err.max()
    err_norm = err / mx if mx > 0 else err
    return _stft_to_pil(err_norm, _LUT_INFERNO, out_size)

def _compute_spec_gpu_batch(x_1d_batch: torch.Tensor) -> torch.Tensor:
    """
    배치 1D 신호 → 4채널 스펙트로그램 (GPU on-the-fly).
    x_1d_batch : (B, 2, L) GPU tensor (고정 스케일 정규화 완료)
    Returns    : (B, 4, F_bins, T_frames) float32 GPU tensor
    """
    n_fft     = SPEC_NPERSEG                        # 1024
    hop       = SPEC_NPERSEG - SPEC_NOVERLAP        # 256
    pad       = n_fft // 2                          # 512
    f_res     = FS / n_fft                          # Hz/bin
    f_bin_max = int(np.ceil(SPEC_F_MAX_HZ / f_res)) + 1  # ~257

    win = torch.hann_window(n_fft, device=x_1d_batch.device)

    x_ch = F.pad(x_1d_batch[:, 0, :], (pad, pad))  # (B, L+2*pad)
    y_ch = F.pad(x_1d_batch[:, 1, :], (pad, pad))

    # torch.stft: (B, L) → (B, n_fft//2+1, T) complex
    Zx = torch.stft(x_ch, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    Zy = torch.stft(y_ch, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    Zx = Zx[:, :f_bin_max, :]
    Zy = Zy[:, :f_bin_max, :]

    Sx  = torch.log1p(Zx.abs() ** 2)
    Sy  = torch.log1p(Zy.abs() ** 2)
    Gxy = Zx.conj() * Zy
    Cre = torch.log1p(Gxy.real.abs())
    Cim = torch.log1p(Gxy.imag.abs())

    return torch.stack([Sx, Sy, Cre, Cim], dim=1).float()  # (B, 4, F, T)


def _mae_stage1_sweep(x_mil_full, y_mil_full, n_total):
    """
    Stage 1 슬라이딩 윈도우 스윕 (배치 처리).
    - 모든 윈도우를 한 번에 GPU 텐서로 적재
    - GPU STFT로 spec 배치 생성 (CPU SciPy STFT 루프 대체)
    - 단일 배치 forward pass로 이상 점수 계산
    - OR 로직 활성화 시: max(norm_1d, norm_spec) 기준으로 최악 윈도우 선정
    Returns: (best_x_seg, best_y_seg)
    """
    arr_list = []
    seg_list = []
    for s in range(0, n_total - FS + 1, SW_STEP):
        xs = x_mil_full[s: s + FS]
        ys = y_mil_full[s: s + FS]
        seg_list.append((xs, ys))
        arr_list.append(prepare_1d_input_fixed(xs, ys, mae_scale_mil))  # (2, L)

    t_1d = torch.from_numpy(np.stack(arr_list, axis=0)).to(device)  # (N, 2, L)

    with torch.no_grad():
        if mae_use_spec:
            t_spec = _compute_spec_gpu_batch(t_1d)              # (N, 4, F, T)
            if _or_active:
                # OR 로직: max(norm_1d, norm_spec) 기준 윈도우 선정
                scores_1d   = mae_model.branch_1d.anomaly_score(
                    t_1d,   n_eval=1, topk_ratio=mae_topk_ratio).cpu().numpy()
                scores_spec = mae_model.branch_spec.anomaly_score(
                    t_spec, n_eval=1, topk_ratio=mae_topk_ratio).cpu().numpy()
                sweep_scores = np.maximum(
                    scores_1d   / mae_threshold_1d,
                    scores_spec / mae_threshold_spec,
                )
            else:
                sweep_scores = mae_model.anomaly_score(
                    t_1d, t_spec, n_eval=1, topk_ratio=mae_topk_ratio).cpu().numpy()
        else:
            sweep_scores = mae_model.anomaly_score(
                t_1d, n_eval=1, topk_ratio=mae_topk_ratio).cpu().numpy()

    best_idx = int(np.argmax(sweep_scores))
    return seg_list[best_idx]


def _mae_predict(x_seg, y_seg, n_eval: int = 10, viz: bool = True):
    """
    MAE 재구성 오차 기반 이상 탐지.
    n_eval : Monte Carlo 마스크 반복 횟수 (1=빠른 스코어, 10=최종 판정)
    viz    : True → 4종 시각화 이미지 생성 (Stage 2), False → 점수만 (Stage 1 skip용)
    반환   : dict { score, threshold, is_anomaly, normalized_score, [images] }
    """
    score_1d = score_spec = None   # _or_active 분기에서만 할당 — 방어적 초기화
    arr    = prepare_1d_input_fixed(x_seg, y_seg, mae_scale_mil)     # (2, L)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)           # (1, 2, L)

    if mae_use_spec:
        x_spec_arr = make_spectrogram_4ch(x_seg, y_seg, mae_scale_mil)  # (4, F, T)
        x_spec_t   = torch.from_numpy(x_spec_arr).unsqueeze(0).to(device)
        with torch.no_grad():
            if _or_active:
                # OR 로직: 브랜치별 점수를 독립적으로 계산
                score_1d   = float(mae_model.branch_1d.anomaly_score(
                    tensor,    n_eval=n_eval, topk_ratio=mae_topk_ratio).item())
                score_spec = float(mae_model.branch_spec.anomaly_score(
                    x_spec_t,  n_eval=n_eval, topk_ratio=mae_topk_ratio).item())
                score = mae_alpha * score_1d + (1.0 - mae_alpha) * score_spec
            else:
                score = float(mae_model.anomaly_score(
                    tensor, x_spec_t, n_eval=n_eval, topk_ratio=mae_topk_ratio).item())
            if viz:
                recon, _err_map, _mask = mae_model.branch_1d.reconstruct_once(tensor)
    else:
        with torch.no_grad():
            score = float(mae_model.anomaly_score(
                tensor, n_eval=n_eval, topk_ratio=mae_topk_ratio).item())
            if viz:
                recon, _err_map, _mask = mae_model.reconstruct_once(tensor)

    # ── 이상 판정 ─────────────────────────────────────────────────────
    if _or_active:
        # OR 로직: 두 브랜치 중 하나라도 독립 임계값 초과 → 이상
        norm_1d        = score_1d   / mae_threshold_1d
        norm_spec      = score_spec / mae_threshold_spec
        is_anomaly     = (score_1d > mae_threshold_1d) or (score_spec > mae_threshold_spec)
        normalized_score = round(max(norm_1d, norm_spec), 4)
    else:
        norm_1d = norm_spec = None
        is_anomaly       = score > mae_threshold
        normalized_score = round(score / mae_threshold, 4) if mae_threshold > 0 else float('inf')

    result = {
        "score":            round(score, 6),
        "threshold":        round(mae_threshold, 6),
        "is_anomaly":       is_anomaly,
        "normalized_score": normalized_score,
    }
    if _or_active:
        result["score_1d"]        = round(score_1d,   6)
        result["score_spec"]      = round(score_spec, 6)
        result["threshold_1d"]    = round(mae_threshold_1d,   6)
        result["threshold_spec"]  = round(mae_threshold_spec, 6)
        result["norm_1d"]         = round(norm_1d,   4)
        result["norm_spec"]       = round(norm_spec, 4)

    if viz:
        recon_np = recon.squeeze(0).cpu().numpy()   # (2, L)
        x_orig   = arr[0]                           # 입력 X (정규화)
        x_recon  = recon_np[0]                      # 재구성 X

        stft_in = _stft_matrix(x_orig)
        stft_rc = _stft_matrix(x_recon)

        img1 = _stft_to_pil(stft_in, _LUT_VIRIDIS)
        img2 = _stft_to_pil(stft_rc, _LUT_VIRIDIS)
        img3 = _stft_error_overlay(stft_in, stft_rc)
        img4 = _stft_error_heatmap(stft_in, stft_rc)

        result["images"] = {
            "input_spec":    image_to_base64(img1),
            "recon_spec":    image_to_base64(img2),
            "error_overlay": image_to_base64(img3),
            "error_heatmap": image_to_base64(img4),
        }

    return result


def _run_mae_batch(bin_path: str, viz: bool):
    """
    MAE 배치 처리 공통 로직.

    bin_path : RCPVMS BIN 파일 경로
    viz      : True → 시각화 이미지 생성 (mae_analyze), False → 점수만 (mae_fp_check)

    반환 tuple:
        mae_results   : {rcp: {score, threshold, is_anomaly, normalized_score, [amplitude_mil]}}
        mae_images_b64: {rcp: images} (viz=True 시), 또는 None (viz=False 시)
        final_verdict : "anomaly" | "normal"
        max_norm      : float — 전체 RCP 중 최대 normalized_score
    """
    rcp_xy = extract_rcp_xy_from_bin(bin_path, fs=FS)

    mae_results    = {}
    mae_images_b64 = {} if viz else None
    any_anomaly    = False

    for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
        n_total = len(x_mil_full)
        if n_total < FS:
            raise ValueError(
                f"{rcp}: 신호가 너무 짧습니다 "
                f"({n_total} samples, 필요: {FS})."
            )

        # Stage 1: 배치 스윕 → 최고 점수 윈도우 선정
        best_x_seg, best_y_seg = _mae_stage1_sweep(x_mil_full, y_mil_full, n_total)

        # Stage 2: 최고 점수 윈도우 → 최종 판정 (viz=True 시 시각화 포함)
        n_eval = 10 if viz else 1
        result = _mae_predict(best_x_seg, best_y_seg, n_eval=n_eval, viz=viz)
        if result["is_anomaly"]:
            any_anomaly = True

        rcp_entry = {
            "score":            result["score"],
            "threshold":        result["threshold"],
            "is_anomaly":       result["is_anomaly"],
            "normalized_score": result["normalized_score"],
        }

        if viz:
            amplitude_mil = float(np.percentile(
                np.abs(np.concatenate([best_x_seg, best_y_seg])), 99.5
            ))
            rcp_entry["amplitude_mil"] = round(amplitude_mil, 4)
            mae_images_b64[rcp] = result["images"]

        mae_results[rcp] = rcp_entry

    final_verdict = "anomaly" if any_anomaly else "normal"
    max_norm = max(r["normalized_score"] for r in mae_results.values())
    return mae_results, mae_images_b64, final_verdict, max_norm


def _ig(x_seg, y_seg, display_pil, ms_arr_cache=None, class_idx=None):
    """
    Integrated Gradients 시각화 생성.
    class_idx: 앙상블 최종 예측 클래스 인덱스 (None이면 생략).
    반환: {"resnet_heatmap": PIL, "resnet_overlay": PIL}
          또는 빈 dict(비다중스케일 모델), 실패 시 None
    """
    if not IG_AVAILABLE or class_idx is None:
        return None
    try:
        result = {}

        if is_multiscale:
            ms_arr = (ms_arr_cache if ms_arr_cache is not None
                      else make_multiscale_orbit(x_seg, y_seg,
                                                 img_size=INFERENCE_IMG_SIZE,
                                                 hybrid=CHANNEL_HYBRID))
            ig_resnet = render_ig_resnet(model, ms_arr, display_pil, transform,
                                         class_idx, steps=30)
            result["resnet_heatmap"] = ig_resnet["heatmap"]
            result["resnet_overlay"] = ig_resnet["overlay"]

        return result if result else None
    except Exception as _e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"[Daemon] WARNING: IG 생성 실패 ({_e}), 건너뜀.", file=sys.stderr)
        return None


def _gradcam(x_seg, y_seg, display_pil, axis_lim, ms_arr_cache=None, class_idx=None):
    """
    GradCAM 생성.
    class_idx: 앙상블의 최종 예측 클래스 인덱스.
               전달 시 "앙상블이 결정한 클래스" 기준으로 ResNet 활성화 맵 생성.
               None이면 ResNet 자체 예측 클래스 사용.
    """
    if is_multiscale:
        ms_arr = ms_arr_cache if ms_arr_cache is not None else make_multiscale_orbit(x_seg, y_seg, img_size=INFERENCE_IMG_SIZE, hybrid=CHANNEL_HYBRID)
        return generate_gradcam_on_display(
            model, class_names, ms_arr, display_pil, transform,
            class_idx=class_idx,
        )
    else:
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=INFERENCE_IMG_SIZE)
        pil = Image.fromarray(arr, mode='L')
        return generate_gradcam_images(model, class_names, pil, transform,
                                       class_idx=class_idx)


# ─────────────────────────────────────────────
# RCPVMS 헤더 캐시 헬퍼
# ─────────────────────────────────────────────
def _get_rcpvms_header(filepath: str):
    """filepath의 (info, orbit_map)을 반환. mtime 불변 시 캐시 재사용."""
    mtime = os.path.getmtime(filepath)
    cached = _rcpvms_header_cache.get(filepath)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    info = RcpvmsParser.read_info(filepath)
    orbit_map = RcpvmsParser.resolve_orbit_channels(info)
    _rcpvms_header_cache[filepath] = (mtime, info, orbit_map)
    return info, orbit_map


def _get_rcpvms_orbit_data(filepath: str, window_sec: float):
    """
    RcpvmsParser.read_orbit_data 결과를 캐시에서 반환.
    (filepath, mtime, window_sec) 3-tuple이 동일한 동안 파일 재읽기 없이 재사용.
    rcpvms_orbit에서 선행 호출 후 rcpvms_orbit_single이 반복 요청할 때 효과적이다.
    """
    mtime = os.path.getmtime(filepath)
    key = (filepath, mtime, window_sec)
    cached = _rcpvms_orbit_cache.get(key)
    if cached is not None:
        return cached
    _rcpvms_orbit_cache.clear()  # 항목 1개 유지 (메모리 절감)
    info, orbit_map = _get_rcpvms_header(filepath)
    orbit_data = RcpvmsParser.read_orbit_data(info, orbit_map, window_sec)
    _rcpvms_orbit_cache[key] = orbit_data
    return orbit_data


def _get_items_windows(
    filepath: str,
    info,
    orbit_map: dict,
    items: list,
    window_sec: float,
) -> dict:
    """Return {(pos, wi): {"x": ndarray, "y": ndarray}} for the requested items.

    Cache hit  → slice from pre-materialized data (no file I/O).
    Cache miss → open the file once and seek only the windows needed,
                 avoiding full-file materialization.
    """
    mtime = os.path.getmtime(filepath)
    key = (filepath, mtime, window_sec)
    cached = _rcpvms_orbit_cache.get(key)

    result: dict = {}

    if cached is not None:
        for item in items:
            pos = item.get("pos")
            wi = int(item.get("wi", 0))
            if not pos or pos not in orbit_map:
                continue
            windows = cached["data"].get(pos, [])
            if 0 <= wi < len(windows):
                result[(pos, wi)] = windows[wi]
        return result

    # Cache miss: compute window geometry from info and seek per (pos, wi).
    window_samples = int(info.sampling_rate * window_sec)
    if window_samples <= 0:
        return result
    n_windows_total = info.samples_per_ch // window_samples
    ch_bytes = info.samples_per_ch * 4
    byte_count = window_samples * 4

    def _read_win(f, ch_idx: int, wi_val: int) -> np.ndarray:
        f.seek(info.data_offset + ch_idx * ch_bytes + wi_val * window_samples * 4)
        raw = f.read(byte_count)
        usable = len(raw) - (len(raw) % 4)
        arr = np.frombuffer(raw[:usable], dtype=np.float32).astype(np.float64)
        if len(arr) < window_samples:
            padded = np.zeros(window_samples, dtype=np.float64)
            padded[:len(arr)] = arr
            arr = padded
        else:
            arr = arr[:window_samples]
        arr = np.nan_to_num(arr * info.mils_per_v)
        arr -= arr.mean()
        return arr

    # Deduplicate requests: same (pos, wi) may appear multiple times in items.
    needed = {
        (item.get("pos"), int(item.get("wi", 0)))
        for item in items
        if item.get("pos") in orbit_map
        and 0 <= int(item.get("wi", 0)) < n_windows_total
    }

    with open(filepath, "rb") as f:
        for (pos, wi) in needed:
            x_idx = orbit_map[pos]["x"]
            y_idx = orbit_map[pos]["y"]
            result[(pos, wi)] = {
                "x": _read_win(f, x_idx, wi),
                "y": _read_win(f, y_idx, wi),
            }

    return result


def _get_rcpvms_orbit_window(filepath: str, pos: str, wi: int, window_sec: float):
    """Return one orbit window, using full-data cache when present and direct seek otherwise."""
    mtime = os.path.getmtime(filepath)
    key = (filepath, mtime, window_sec)
    cached = _rcpvms_orbit_cache.get(key)
    if cached is not None:
        windows = cached["data"].get(pos, [])
        if 0 <= wi < len(windows):
            return windows[wi]

    info, orbit_map = _get_rcpvms_header(filepath)
    return RcpvmsParser.read_orbit_window(info, orbit_map, pos, wi, window_sec)


# ─────────────────────────────────────────────
# 데몬 루프
# ─────────────────────────────────────────────
def main():
    while True:
        try:
            raw_line = sys.stdin.buffer.readline()
            if not raw_line:
                break  # EOF
            line = raw_line.decode('utf-8')

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

                    # 원신호 기반 절대 진폭 지표 (스냅핑 전 99.5th percentile, mil)
                    # 모델 입력(동적 정규화)에서는 이 정보가 소거되므로, 심각도 평가용으로 별도 보존.
                    amplitude_mil = float(np.percentile(
                        np.abs(np.concatenate([x_seg, y_seg])), 99.5
                    ))

                    # 멀티스케일 배열 1회 생성 → 예측 + GradCAM에서 재사용
                    ms_arr_cache = make_multiscale_orbit(x_seg, y_seg, img_size=INFERENCE_IMG_SIZE, hybrid=CHANNEL_HYBRID) if is_multiscale else None

                    # 앙상블 예측 + OOD 판정 (TV Distance 복합 기준)
                    (pred_class, ens_probs,
                     resnet_pred, resnet_probs,
                     cnn1d_pred, cnn1d_probs,
                     is_ood, tv_distance, ood_reason) = _ensemble_predict(x_seg, y_seg, ms_arr_cache)

                    result_entry = {
                        "prediction":      pred_class,
                        "is_ood":          is_ood,
                        "confidence":      float(ens_probs.max()),
                        "ood_threshold":   ood_threshold,
                        "probabilities": {
                            name: float(p) for name, p in zip(class_names, ens_probs)
                        },
                        "display_axis_lim": display_axis_lim,
                        # 절대 진폭 (mil) — 모델 입력의 동적 정규화로 소거된 진폭 정보를 UI 표시용으로 복원.
                        # ISO 20816-7 심각도 구간 판단, 경보 임계값 비교 등에 활용.
                        "amplitude_mil": amplitude_mil,
                        # TV Distance 복합 OOD 진단 정보
                        "tv_distance":  round(tv_distance, 4) if tv_distance is not None else None,
                        "tv_threshold": tv_threshold,
                        "ood_reason":   ood_reason,   # "tv" | "conf" | "tv+conf" | "none"
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

                    # GradCAM 타겟: OOD여도 가장 가까운 알려진 클래스 기준으로 시각화
                    ens_class_idx = int(ens_probs.argmax())

                    # 표시용 단일 채널 이미지 (동적 스케일)
                    display_pil, _, used_display_lim, _, _ = _make_display_pil(x_seg, y_seg, display_axis_lim)

                    # GradCAM — 앙상블 예측 클래스 기준으로 ResNet 활성화 맵 생성
                    gradcam_imgs = _gradcam(
                        x_seg, y_seg, display_pil, used_display_lim,
                        ms_arr_cache, class_idx=ens_class_idx,
                    )

                    # 렌더링 레이블
                    target_cls  = gradcam_imgs.get("target_class", pred_class)
                    scale_label = f"±{used_display_lim:.1f} mil"
                    if is_ood:
                        gcam_label = f"OOD(closest: {target_cls}) · Grad-CAM (ensemble)"
                    else:
                        gcam_label = f"{target_cls} · Grad-CAM (ensemble)"

                    images_b64[rcp] = {
                        "orbit": image_to_base64(
                            render_with_axes(display_pil, used_display_lim,
                                             cmap='gray', label=scale_label)
                        ),
                        "heatmap": image_to_base64(
                            render_with_axes(gradcam_imgs["heatmap"], used_display_lim,
                                             label=gcam_label)
                        ),
                        "overlay": image_to_base64(
                            render_with_axes(gradcam_imgs["overlay"], used_display_lim,
                                             label=gcam_label)
                        ),
                    }

                    # IG (Integrated Gradients)
                    ig_label_base = (f"OOD(closest: {target_cls})"
                                     if is_ood else target_cls)
                    ig_imgs = _ig(x_seg, y_seg, display_pil,
                                  ms_arr_cache, class_idx=ens_class_idx)
                    if ig_imgs:
                        if "resnet_heatmap" in ig_imgs:
                            images_b64[rcp]["ig_resnet_heatmap"] = image_to_base64(
                                render_with_axes(ig_imgs["resnet_heatmap"],
                                                 used_display_lim,
                                                 label=f"{ig_label_base} · IG (ResNet)")
                            )
                            images_b64[rcp]["ig_resnet_overlay"] = image_to_base64(
                                render_with_axes(ig_imgs["resnet_overlay"],
                                                 used_display_lim,
                                                 label=f"{ig_label_base} · IG (ResNet)")
                            )

                # 4-class: 하나라도 비정상이면 가장 많이 예측된 고장 유형 반환
                non_normal = [
                    r["prediction"]
                    for r in results.values()
                    if r["prediction"] != "normal"
                ]
                if non_normal:

                    final_label = Counter(non_normal).most_common(1)[0][0]
                else:
                    final_label = "normal"

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
                        display_pil, _, used_seg_lim, _, _ = _make_display_pil(x_seg, y_seg, full_axis_lim)
                        rendered = render_with_axes(display_pil, used_seg_lim, cmap='gray')
                        sec_images.append(image_to_base64(rendered))

                    timeline_b64[rcp] = sec_images

                response = {
                    "status": "ok",
                    "type":   "timeline_result",
                    "data":   timeline_b64,
                }

            # ── mae_analyze ──────────────────────────────────────
            elif command == "mae_analyze":
                if mae_model is None:
                    response = {
                        "status": "error",
                        "message": "MAE 모델이 로드되지 않았습니다. "
                                   "python/train_mae.py를 실행하여 모델을 학습하세요."
                    }
                elif not bin_path:
                    response = {
                        "status": "error",
                        "message": "payload.bin_path is required"
                    }
                else:
                    mae_results, mae_images_b64, final_verdict, max_norm = \
                        _run_mae_batch(bin_path, viz=True)

                    response = {
                        "status": "ok",
                        "type":   "mae_result",
                        "data": {
                            "final_verdict":        final_verdict,
                            "max_normalized_score": round(max_norm, 4),
                            "threshold":            round(mae_threshold, 6),
                            "results":              mae_results,
                            "images":               mae_images_b64,
                        },
                    }

            # ── mae_fp_check ─────────────────────────────────────
            # mae_analyze와 동일하나 이미지 생성 생략 → 배치 FP 평가용
            elif command == "mae_fp_check":
                if mae_model is None:
                    response = {
                        "status": "error",
                        "message": "MAE 모델이 로드되지 않았습니다.",
                    }
                elif not bin_path:
                    response = {"status": "error", "message": "payload.bin_path is required"}
                else:
                    mae_results, _images, final_verdict, max_norm = \
                        _run_mae_batch(bin_path, viz=False)

                    response = {
                        "status": "ok",
                        "type":   "mae_fp_result",
                        "data": {
                            "final_verdict":        final_verdict,
                            "max_normalized_score": round(max_norm, 4),
                            "threshold":            round(mae_threshold, 6),
                            "results":              mae_results,
                        },
                    }

            # ── dmd_info ──────────────────────────────────────────
            elif command == "dmd_info":
                dmd_path = payload.get("dmd_path")
                if not dmd_path:
                    response = {"status": "error", "message": "payload.dmd_path is required"}
                else:
                    from dmd_parser import DmdParser
                    _dmd_mtime = os.path.getmtime(dmd_path)
                    _dmd_cached = _dmd_info_cache.get(dmd_path)
                    if _dmd_cached and _dmd_cached[0] == _dmd_mtime:
                        info = _dmd_cached[1]
                    else:
                        info = DmdParser.read_info(dmd_path)
                        _dmd_info_cache[dmd_path] = (_dmd_mtime, info)
                    ch_list = [
                        {
                            "index":       ch.index,
                            "name":        ch.name,
                            "unit":        ch.unit,
                            "sample_rate": ch.sample_rate,
                            "segment_id":  ch.segment_id,
                        }
                        for ch in info.channels
                    ]
                    orbit_map = {
                        rcp: {"x_name": rd["x_name"], "y_name": rd["y_name"]}
                        for rcp, rd in info.orbit_channels.items()
                    }
                    response = {
                        "status": "ok",
                        "type":   "dmd_info",
                        "data": {
                            "n_channels":   info.n_channels,
                            "has_orbit":    info.has_orbit,
                            "channels":     ch_list,
                            "orbit_map":    orbit_map,
                        },
                    }

            # ── dmd_orbit_timeline ────────────────────────────────
            elif command == "dmd_orbit_timeline":
                dmd_path    = payload.get("dmd_path")
                window_sec  = int(payload.get("window_sec", 10))
                mil_per_volt = float(payload.get("mil_per_volt", 10.0))

                if not dmd_path:
                    response = {"status": "error", "message": "payload.dmd_path is required"}
                else:
                    from dmd_parser import DmdParser
                    _dmd_mtime2 = os.path.getmtime(dmd_path)
                    _dmd_cached2 = _dmd_info_cache.get(dmd_path)
                    if _dmd_cached2 and _dmd_cached2[0] == _dmd_mtime2:
                        info = _dmd_cached2[1]
                    else:
                        info = DmdParser.read_info(dmd_path)
                        _dmd_info_cache[dmd_path] = (_dmd_mtime2, info)
                    windows = DmdParser.read_orbit_windows(
                        dmd_path, info,
                        window_sec=window_sec,
                        mil_per_volt=mil_per_volt,
                    )

                    # 윈도우별 궤도 이미지 생성
                    # 전체 신호 기준 동적 스케일 결정 (초간 일관성 유지)
                    # np.concatenate 없이 per-window p99.5 running max 사용
                    _snap = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
                    global_p99 = 0.0
                    for win in windows:
                        for rcp_data in win["rcp"].values():
                            x_, y_ = rcp_data["x"], rcp_data["y"]
                            if len(x_) > 0:
                                global_p99 = max(
                                    global_p99,
                                    float(np.percentile(np.abs(x_), 99.5)),
                                    float(np.percentile(np.abs(y_), 99.5)),
                                )
                    if global_p99 > 0:
                        _raw = global_p99 * 1.2
                        global_axis_lim = next((bp for bp in _snap if _raw <= bp), round(_raw, 1))
                    else:
                        global_axis_lim = 3.0

                    timeline_b64 = {}
                    for rcp in info.orbit_channels:
                        timeline_b64[rcp] = []

                    for win in windows:
                        for rcp_name, rcp_data in win["rcp"].items():
                            x_seg = rcp_data["x"]
                            y_seg = rcp_data["y"]
                            if len(x_seg) == 0:
                                timeline_b64[rcp_name].append(None)
                                continue
                            display_pil, _, used_seg_lim, _, _ = _make_display_pil(x_seg, y_seg, global_axis_lim)
                            label = (
                                f"{rcp_name} · {win['start_sec']:.0f}~{win['end_sec']:.0f}s"
                                f" · ±{used_seg_lim:.1f} mil"
                            )
                            rendered = render_with_axes(
                                display_pil, used_seg_lim, cmap="gray", label=label
                            )
                            timeline_b64[rcp_name].append(image_to_base64(rendered))

                    response = {
                        "status": "ok",
                        "type":   "dmd_orbit_timeline",
                        "data": {
                            "n_windows":   len(windows),
                            "window_sec":  window_sec,
                            "axis_lim":    global_axis_lim,
                            "orbit_map":   {
                                rcp: {"x_name": rd["x_name"], "y_name": rd["y_name"]}
                                for rcp, rd in info.orbit_channels.items()
                            },
                            "timeline":    timeline_b64,
                        },
                    }

            # ── dmd_convert_to_rcpvms ───────────────────────────
            elif command == "dmd_convert_to_rcpvms":
                dmd_path    = payload.get("dmd_path")
                output_dir  = payload.get("output_dir")
                window_sec  = int(payload.get("window_sec", 10))
                mils_per_v  = float(payload.get("mils_per_v", 10.0))
                g_per_v     = float(payload.get("g_per_v", 1.0))
                site_id     = payload.get("site_id", "")
                base_name   = payload.get("base_name", "")

                if not dmd_path:
                    response = {"status": "error", "message": "payload.dmd_path is required"}
                elif not output_dir:
                    response = {"status": "error", "message": "payload.output_dir is required"}
                else:
                    from dmd_to_rcpvms import DmdToRcpvmsConverter

                    def _progress(current, total):
                        sys.stderr.write(json.dumps({
                            "type": "convert_progress",
                            "current": current,
                            "total": total,
                        }) + "\n")
                        sys.stderr.flush()

                    result = DmdToRcpvmsConverter.convert(
                        dmd_path=dmd_path,
                        output_dir=output_dir,
                        window_sec=window_sec,
                        mils_per_v=mils_per_v,
                        g_per_v=g_per_v,
                        site_id=site_id,
                        base_name=base_name,
                        progress_callback=_progress,
                    )
                    response = {
                        "status": "ok",
                        "type":   "dmd_convert_to_rcpvms",
                        "data":   result,
                    }

            # ── rcpvms_info ────────────────────────────────────
            elif command == "rcpvms_info":
                filepath = payload.get("filepath")
                if not filepath:
                    response = {"status": "error", "message": "payload.filepath is required"}
                else:
                    info, orbit_map = _get_rcpvms_header(filepath)
                    ch_list = [
                        {
                            "index":    ch.index,
                            "ch_no":    ch.ch_no,
                            "ch_name":  ch.ch_name,
                            "ch_type":  ch.ch_type,
                        }
                        for ch in info.channels
                    ]
                    response = {
                        "status": "ok",
                        "type":   "rcpvms_info",
                        "data": {
                            "site_id":          info.site_id,
                            "total_ch":         info.total_ch,
                            "sampling_rate":    info.sampling_rate,
                            "event_duration_ms": info.event_duration_ms,
                            "event_date":       info.event_date,
                            "g_per_v":          info.g_per_v,
                            "mils_per_v":       info.mils_per_v,
                            "is_legacy":        info.is_legacy,
                            "has_orbit":        len(orbit_map) > 0,
                            "channels":         ch_list,
                            "orbit_map":        {
                                pos: {"x_name": om["x_name"], "y_name": om["y_name"]}
                                for pos, om in orbit_map.items()
                            },
                        },
                    }

            # ── rcpvms_orbit ───────────────────────────────────
            elif command == "rcpvms_orbit":
                filepath   = payload.get("filepath")
                window_sec = float(payload.get("window_sec", 1.0))

                # 궤도 별 사용자 스케일 맵: {pos: float} 형태
                # 구형 단일값(user_axis_lim)도 fallback으로 지원
                _ual_map_raw = payload.get("user_axis_lim_map")
                user_axis_lim_map: dict = {}
                if isinstance(_ual_map_raw, dict):
                    for _pos, _val in _ual_map_raw.items():
                        try:
                            _v = float(_val)
                            if _v > 0:
                                user_axis_lim_map[_pos] = _v
                        except (TypeError, ValueError):
                            pass
                elif payload.get("user_axis_lim") is not None:
                    # 구형 클라이언트 호환: 단일값을 모든 위치에 적용
                    try:
                        _v = float(payload["user_axis_lim"])
                        if _v > 0:
                            user_axis_lim_map = {"__all__": _v}
                    except (TypeError, ValueError):
                        pass

                if not filepath:
                    response = {"status": "error", "message": "payload.filepath is required"}
                elif window_sec <= 0:
                    response = {"status": "error", "message": "window_sec must be positive"}
                else:
                    # 헤더(info, orbit_map)는 캐시에서 가져옴
                    info, orbit_map = _get_rcpvms_header(filepath)
                    orbit_data = _get_rcpvms_orbit_data(filepath, window_sec)

                    positions = orbit_data["positions"]
                    n_windows = orbit_data["n_windows"]

                    # per-window axis_lim 1회 산출
                    per_window_lim = {
                        pos: [
                            compute_dynamic_axis_lim(wd["x"], wd["y"]) if len(wd["x"]) > 0 else None
                            for wd in orbit_data["data"][pos]
                        ]
                        for pos in positions
                    }

                    # fixed scale: 전체 윈도우 중 최대값
                    all_lims = [lim for lims in per_window_lim.values() for lim in lims if lim is not None]
                    fixed_axis_lim = max(all_lims) if all_lims else 3.0

                    has_user_scale = len(user_axis_lim_map) > 0

                    response = {
                        "status": "ok",
                        "type":   "rcpvms_orbit",
                        "data": {
                            "positions":           positions,
                            "n_windows":           n_windows,
                            "window_sec":          window_sec,
                            "fixed_axis_lim":      fixed_axis_lim,
                            "user_axis_lim_map":   user_axis_lim_map if has_user_scale else None,
                            "mils_per_v":          orbit_data["mils_per_v"],
                            "event_date":          info.event_date,
                            "orbit_map":           {
                                pos: {"x_name": orbit_map[pos]["x_name"],
                                      "y_name": orbit_map[pos]["y_name"]}
                                for pos in positions
                            },
                            "per_window_axis_lim": {
                                pos: per_window_lim[pos]
                                for pos in positions
                            },
                        },
                    }

            # ── rcpvms_orbit_single ────────────────────────────
            elif command == "rcpvms_orbit_single":
                filepath    = payload.get("filepath")
                pos         = payload.get("pos")
                wi          = int(payload.get("wi", 0))
                window_sec  = float(payload.get("window_sec", 1.0))
                axis_lim    = float(payload.get("axis_lim", 0.0))
                filter_mode = str(payload.get("filter_mode", "1x")).lower()
                # 허용 값 검증 (미지정 또는 오타 시 1x로 fallback)
                if filter_mode not in ("raw", "1x", "2x", "broadband", "overlay"):
                    filter_mode = "1x"

                if not filepath or not pos:
                    response = {"status": "error", "message": "filepath and pos are required"}
                elif window_sec <= 0:
                    response = {"status": "error", "message": "window_sec must be positive"}
                elif axis_lim <= 0:
                    response = {"status": "error", "message": "axis_lim must be positive"}
                else:
                    # 필터 엣지 트리밍 보장 최소 창 공식: (2×_FILTER_EDGE_CYCLES+2) / f_ref
                    # f_ref는 신호에서 동적으로 추정되므로 정적 검증 불가;
                    # 런타임에 _trim_to_integer_cycles 내부에서 경고를 stderr로 출력한다.
                    info, orbit_map = _get_rcpvms_header(filepath)
                    if pos not in orbit_map:
                        response = {"status": "error", "message": f"position '{pos}' not in orbit_map"}
                    else:
                        try:
                            wd = _get_rcpvms_orbit_window(filepath, pos, wi, window_sec)
                        except Exception as _win_err:
                            response = {"status": "error", "message": str(_win_err)}
                        else:
                            x_seg = wd["x"]
                            y_seg = wd["y"]
                            t_start = wi * window_sec
                            t_end   = (wi + 1) * window_sec
                            # Defaults ensure all variables are bound regardless of which branch runs.
                            actual_filter = filter_mode
                            used_axis_lim = axis_lim
                            edge_trim_applied = False
                            freq_estimate = None
                            if filter_mode == "overlay":
                                display_pil = _make_overlay_pil(
                                    x_seg, y_seg, axis_lim, fs=info.sampling_rate
                                )
                                label = (
                                    f"{pos} · {t_start:.0f}~{t_end:.0f}s"
                                    f" · ±{axis_lim:.1f} mil · Overlay (Raw/BB/2X/1X)"
                                )
                                rendered = render_with_axes(display_pil, used_axis_lim, label=label)
                            else:
                                display_pil, actual_filter, used_axis_lim, edge_trim_applied, freq_estimate = _make_display_pil(
                                    x_seg, y_seg, axis_lim,
                                    fs=info.sampling_rate, filter_mode=filter_mode
                                )
                                label = (
                                    f"{pos} · {t_start:.0f}~{t_end:.0f}s"
                                    f" · ±{used_axis_lim:.1f} mil · {_FILTER_LABELS[actual_filter]}"
                                )
                                rendered = render_with_axes(display_pil, used_axis_lim, cmap="gray", label=label)
                            response = {
                                "status": "ok",
                                "type":   "rcpvms_orbit_single",
                                "data": {
                                    "image_b64":        image_to_base64(rendered),
                                    "axis_lim":         used_axis_lim,
                                    "pos":              pos,
                                    "wi":               wi,
                                    "filter_used":      actual_filter,
                                    "edge_trim_applied": edge_trim_applied,
                                    "freq_estimate":    freq_estimate,
                                },
                            }

            # ── rcpvms_orbit_multi ─────────────────────────────
            # 다수의 (pos, wi) 셀을 한 번의 IPC 왕복으로 처리.
            # 썸네일용 소형 이미지(render_with_axes 생략)를 반환하여 전송량을 최소화.
            elif command == "rcpvms_orbit_multi":
                filepath    = payload.get("filepath")
                window_sec  = float(payload.get("window_sec", 1.0))
                filter_mode = str(payload.get("filter_mode", "1x")).lower()
                items       = payload.get("items", [])   # [{pos, wi, axis_lim}, ...]
                thumb_size  = int(payload.get("thumb_size", 96))
                if filter_mode not in ("raw", "1x", "2x", "broadband", "overlay"):
                    filter_mode = "1x"

                if not filepath:
                    response = {"status": "error", "message": "filepath is required"}
                elif window_sec <= 0:
                    response = {"status": "error", "message": "window_sec must be positive"}
                else:
                    info, orbit_map = _get_rcpvms_header(filepath)
                    fs_val = info.sampling_rate

                    # Issue 4: fetch only the requested windows (cache hit → free slice,
                    # cache miss → single file open with targeted seeks, no full materialization)
                    windows_data = _get_items_windows(filepath, info, orbit_map, items, window_sec)

                    # Issue 3: pre-compute f1x once per position from a representative window
                    # so the per-cell FFT is skipped in _make_display_pil.
                    f1x_by_pos: dict = {}
                    if filter_mode in ("1x", "2x") and fs_val > 0:
                        pos_seen: set = set()
                        for item in items:
                            pos = item.get("pos")
                            if not pos or pos not in orbit_map or pos in pos_seen:
                                continue
                            pos_seen.add(pos)
                            rep_wi = int(item.get("wi", 0))
                            rep_wd = windows_data.get((pos, rep_wi))
                            if rep_wd is not None:
                                try:
                                    f1x_by_pos[pos] = estimate_1x_freq(
                                        rep_wd["x"], fs_val, y_mil=rep_wd["y"]
                                    )
                                except Exception:
                                    pass

                    images = []
                    for item in items:
                        pos      = item.get("pos")
                        wi       = int(item.get("wi", 0))
                        axis_lim = float(item.get("axis_lim") or 3.0)
                        if not pos or pos not in orbit_map:
                            continue
                        wd = windows_data.get((pos, wi))
                        if wd is None:
                            continue
                        x_seg = wd["x"]
                        y_seg = wd["y"]
                        if filter_mode == "overlay":
                            freq_estimate = None
                            thumb_pil = _make_overlay_pil(
                                x_seg, y_seg, axis_lim, fs=fs_val, img_size=thumb_size
                            )
                            thumb_pil = _draw_crosshair(thumb_pil)
                            used_axis_lim = axis_lim
                            actual_filter = "overlay"
                        else:
                            thumb_pil, actual_filter, used_axis_lim, _, freq_estimate = _make_display_pil(
                                x_seg, y_seg, axis_lim,
                                fs=fs_val, filter_mode=filter_mode, img_size=thumb_size,
                                f1x_hint=f1x_by_pos.get(pos),
                            )
                            thumb_pil = _draw_crosshair(thumb_pil.convert("RGB"))
                        images.append({
                            "pos":         pos,
                            "wi":          wi,
                            "image_b64":   image_to_base64(thumb_pil),
                            "axis_lim":    used_axis_lim,
                            "filter_used": actual_filter,
                            "freq_estimate": freq_estimate,
                        })
                    response = {
                        "status": "ok",
                        "type":   "rcpvms_orbit_multi",
                        "data":   {"images": images},
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
