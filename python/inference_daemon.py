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
from model_svdd import SVDDEncoder, compute_svdd_distances
from model_mae import OrbitMAE, OrbitMAE1D
from preprocess import (
    make_multiscale_orbit,
    make_orbit_image_v2,
    compute_dynamic_axis_lim,
    build_multiscale_transform,
    prepare_1d_input,
    prepare_1d_input_fixed,
    make_spectrogram_4ch,
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

MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
# 새 모델 없으면 레거시 fallback
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")

CNN1D_MODEL_PATH     = os.path.join(SCRIPT_DIR, "model", "orbit_cnn1d.pth")
ENSEMBLE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "ensemble_config.json")
CLASS_MAP_PATH       = os.path.join(SCRIPT_DIR, "class_map.json")
SVDD_MODEL_PATH      = os.path.join(SCRIPT_DIR, "model", "svdd_encoder.pth")
SVDD_CONFIG_PATH     = os.path.join(SCRIPT_DIR, "svdd_config.json")
MAE_MODEL_PATH       = os.path.join(SCRIPT_DIR, "model", "orbit_mae.pth")
MAE_CONFIG_PATH      = os.path.join(SCRIPT_DIR, "mae_config.json")

# 앙상블 최대 확률이 임계값 미만이면 OOD(분포 외) 판정
OOD_CLASS_NAME = "unknown_abnormal"

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
# SVDD 모델 로드 (옵션 — 없으면 graceful disable)
# ─────────────────────────────────────────────
svdd_encoder = None
svdd_center   = None
svdd_threshold = None
svdd_feature_dim = 128

if os.path.exists(SVDD_MODEL_PATH):
    try:
        # svdd_config.json 로드
        if os.path.exists(SVDD_CONFIG_PATH):
            with open(SVDD_CONFIG_PATH, "r") as _f:
                _svdd_cfg = json.load(_f)
            svdd_threshold   = float(_svdd_cfg.get("threshold", 0.0))
            svdd_feature_dim = int(_svdd_cfg.get("feature_dim", 128))
            print(f"[Daemon] svdd_config: threshold={svdd_threshold:.6f}, "
                  f"feature_dim={svdd_feature_dim}", file=sys.stderr)
        else:
            print("[Daemon] WARNING: svdd_config.json 없음 — threshold를 체크포인트에서 로드.",
                  file=sys.stderr)

        _svdd_ckpt = torch.load(SVDD_MODEL_PATH, map_location="cpu")
        svdd_feature_dim = int(_svdd_ckpt.get("feature_dim", svdd_feature_dim))
        svdd_encoder = SVDDEncoder(feature_dim=svdd_feature_dim)
        svdd_encoder.load_state_dict(_svdd_ckpt["model_state_dict"])
        svdd_encoder.to(device)
        svdd_encoder.eval()

        svdd_center = _svdd_ckpt["center"].to(device)
        if svdd_threshold is None:
            svdd_threshold = float(_svdd_ckpt.get("threshold", 0.0))

        print(f"[Daemon] SVDD encoder loaded: feature_dim={svdd_feature_dim}, "
              f"threshold={svdd_threshold:.6f}", file=sys.stderr)
    except Exception as e:
        print(f"[Daemon] WARNING: SVDD 모델 로드 실패 ({e}), SVDD 비활성화.", file=sys.stderr)
        svdd_encoder = None
else:
    print("[Daemon] svdd_encoder.pth not found — SVDD disabled.", file=sys.stderr)

# ─────────────────────────────────────────────
# MAE 모델 로드 (옵션 — 없으면 graceful disable)
# ─────────────────────────────────────────────
mae_model            = None
mae_threshold        = None
mae_scale_mil        = None
mae_use_spec         = False
mae_alpha            = 0.5
mae_spec_mask_ratio  = 0.85

if os.path.exists(MAE_MODEL_PATH):
    try:
        if os.path.exists(MAE_CONFIG_PATH):
            with open(MAE_CONFIG_PATH, "r") as _f:
                _mae_cfg = json.load(_f)
            mae_threshold       = float(_mae_cfg.get("threshold", 0.0))
            mae_scale_mil       = float(_mae_cfg.get("scale_mil", 1.0))
            mae_use_spec        = bool(_mae_cfg.get("use_spec", False))
            mae_alpha           = float(_mae_cfg.get("alpha", 0.5))
            mae_spec_mask_ratio = float(_mae_cfg.get("spec_mask_ratio", 0.85))
            print(f"[Daemon] mae_config: threshold={mae_threshold:.6f}, "
                  f"scale_mil={mae_scale_mil:.4f}, use_spec={mae_use_spec}, "
                  f"alpha={mae_alpha:.2f}, spec_mask_ratio={mae_spec_mask_ratio:.2f}",
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
        ms_arr = ms_arr_cache if ms_arr_cache is not None else make_multiscale_orbit(x_seg, y_seg, img_size=INFERENCE_IMG_SIZE, hybrid=CHANNEL_HYBRID)
        return predict_from_multiscale(model, class_names, ms_arr, transform)
    else:
        from infer_resnet_None import make_orbit_image
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


def _svdd_predict(x_seg, y_seg):
    """
    SVDD 이상 점수 계산.
    반환: (score, threshold, is_anomaly, normalized_score)
      score            : ||z - c||² 원시 거리
      threshold        : 학습 시 산출된 임계값
      is_anomaly       : score > threshold
      normalized_score : score / threshold  (1.0 기준)
    """
    arr = prepare_1d_input(x_seg, y_seg)                          # (2, 40000)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)        # (1, 2, 40000)
    with torch.no_grad():
        feat  = svdd_encoder(tensor)                              # (1, feature_dim)
        dists = compute_svdd_distances(feat, svdd_center)         # (1,)
    score = float(dists.squeeze().item())
    is_anomaly = score > svdd_threshold
    normalized_score = score / svdd_threshold if svdd_threshold > 0 else float('inf')
    return score, svdd_threshold, is_anomaly, normalized_score


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

def _colorize(arr2d, colormap_fn):
    """(H, W) float [0,1] → (H, W, 3) uint8 PIL Image."""
    H, W = arr2d.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            rgb[y, x] = colormap_fn(arr2d[y, x])
    return Image.fromarray(rgb)

def _stft_matrix(signal, fs=40_000, nperseg=512, noverlap=448, max_freq=1000):
    """1D signal → log-power spectrogram (H×W float [0,1]), freq axis limited."""
    from scipy.signal import spectrogram as _spec
    f, _t, Sxx = _spec(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    mask = f <= max_freq
    Sxx = Sxx[mask]                                    # (F, T)
    Sxx = np.log10(Sxx + 1e-12)
    vmin, vmax = Sxx.min(), Sxx.max()
    if vmax > vmin:
        Sxx = (Sxx - vmin) / (vmax - vmin)
    else:
        Sxx = np.zeros_like(Sxx)
    return Sxx[::-1].copy()                            # 저주파 → 하단

def _stft_to_pil(stft_mat, colormap_fn, out_size=(360, 200)):
    """log-power STFT matrix → PIL Image (resized to out_size)."""
    img = _colorize(stft_mat, colormap_fn)
    return img.resize(out_size, Image.BILINEAR)

def _stft_error_overlay(stft_input, stft_recon, out_size=(360, 200), threshold=0.30):
    """입력 STFT 위에 재구성 오차가 큰 영역을 빨간 오버레이로 표시."""
    base = _stft_to_pil(stft_input, _viridis, out_size)
    base = base.convert("RGBA")
    W, H = base.size
    err = np.abs(stft_input - stft_recon)
    err_norm = err / (err.max() + 1e-10)
    err_resized = np.array(Image.fromarray(
        (err_norm * 255).astype(np.uint8)
    ).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_data = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            v = err_resized[y, x]
            if v > threshold:
                alpha = min(255, int((v - threshold) / (1 - threshold) * 200))
                ov_data[y, x] = [255, 64, 96, alpha]
    overlay = Image.fromarray(ov_data, "RGBA")
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")

def _stft_error_heatmap(stft_input, stft_recon, out_size=(360, 200)):
    """절대 오차 STFT → Inferno heatmap PIL Image."""
    err = np.abs(stft_input - stft_recon)
    mx = err.max()
    err_norm = err / mx if mx > 0 else err
    return _stft_to_pil(err_norm, _inferno, out_size)

def _mae_predict(x_seg, y_seg, n_eval: int = 10):
    """
    MAE 재구성 오차 기반 이상 탐지 + 4종 시각화 이미지 생성.
    n_eval: Monte Carlo 마스크 반복 횟수 (1=빠른 스윕, 10=최종 판정)
    반환: dict { score, threshold, is_anomaly, normalized_score, images }
    """
    arr    = prepare_1d_input_fixed(x_seg, y_seg, mae_scale_mil)     # (2, L)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)           # (1, 2, L)

    if mae_use_spec:
        # 1D + 스펙트로그램 통합 이상 점수
        x_spec_arr = make_spectrogram_4ch(x_seg, y_seg, mae_scale_mil)  # (4, F, T)
        x_spec_t   = torch.from_numpy(x_spec_arr).unsqueeze(0).to(device)
        score = float(mae_model.anomaly_score(tensor, x_spec_t, n_eval=n_eval).item())
        # 시각화는 1D 브랜치 재구성 기반 유지
        recon, _err_map, _mask = mae_model.branch_1d.reconstruct_once(tensor)
    else:
        # 1D 브랜치 단독
        score = float(mae_model.anomaly_score(tensor, n_eval=n_eval).item())
        recon, _err_map, _mask = mae_model.reconstruct_once(tensor)

    is_anomaly       = score > mae_threshold
    normalized_score = score / mae_threshold if mae_threshold > 0 else float('inf')

    recon_np = recon.squeeze(0).cpu().numpy()   # (2, L)

    # X 채널 STFT (정규화된 신호 기준)
    x_orig  = arr[0]                            # 입력 X (정규화)
    x_recon = recon_np[0]                       # 재구성 X

    stft_in    = _stft_matrix(x_orig)
    stft_rc    = _stft_matrix(x_recon)

    img1 = _stft_to_pil(stft_in, _viridis)             # 1열: 입력 스펙트로그램
    img2 = _stft_to_pil(stft_rc, _viridis)             # 2열: MAE 재구성
    img3 = _stft_error_overlay(stft_in, stft_rc)        # 3열: 오차 오버레이
    img4 = _stft_error_heatmap(stft_in, stft_rc)        # 4열: 오차 히트맵

    return {
        "score":            round(score, 6),
        "threshold":        round(mae_threshold, 6),
        "is_anomaly":       is_anomaly,
        "normalized_score": round(normalized_score, 4),
        "images": {
            "input_spec":    image_to_base64(img1),
            "recon_spec":    image_to_base64(img2),
            "error_overlay": image_to_base64(img3),
            "error_heatmap": image_to_base64(img4),
        },
    }


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
        from infer_resnet_None import make_orbit_image
        arr = make_orbit_image(x_seg, y_seg, axis_lim=3.0, img_size=INFERENCE_IMG_SIZE)
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
                    display_pil = _make_display_pil(x_seg, y_seg, display_axis_lim)

                    # GradCAM — 앙상블 예측 클래스 기준으로 ResNet 활성화 맵 생성
                    gradcam_imgs = _gradcam(
                        x_seg, y_seg, display_pil, display_axis_lim,
                        ms_arr_cache, class_idx=ens_class_idx,
                    )

                    # 렌더링 레이블
                    target_cls  = gradcam_imgs.get("target_class", pred_class)
                    scale_label = f"±{display_axis_lim:.1f} mil"
                    if is_ood:
                        gcam_label = f"OOD(closest: {target_cls}) · Grad-CAM (ensemble)"
                    else:
                        gcam_label = f"{target_cls} · Grad-CAM (ensemble)"

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

                    # IG (Integrated Gradients)
                    ig_label_base = (f"OOD(closest: {target_cls})"
                                     if is_ood else target_cls)
                    ig_imgs = _ig(x_seg, y_seg, display_pil,
                                  ms_arr_cache, class_idx=ens_class_idx)
                    if ig_imgs:
                        if "resnet_heatmap" in ig_imgs:
                            images_b64[rcp]["ig_resnet_heatmap"] = image_to_base64(
                                render_with_axes(ig_imgs["resnet_heatmap"],
                                                 display_axis_lim,
                                                 label=f"{ig_label_base} · IG (ResNet)")
                            )
                            images_b64[rcp]["ig_resnet_overlay"] = image_to_base64(
                                render_with_axes(ig_imgs["resnet_overlay"],
                                                 display_axis_lim,
                                                 label=f"{ig_label_base} · IG (ResNet)")
                            )

                # 4-class: 하나라도 비정상이면 가장 많이 예측된 고장 유형 반환
                non_normal = [
                    r["prediction"]
                    for r in results.values()
                    if r["prediction"] != "normal"
                ]
                if non_normal:
                    from collections import Counter
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
                        display_pil = _make_display_pil(x_seg, y_seg, full_axis_lim)
                        rendered = render_with_axes(display_pil, full_axis_lim, cmap='gray')
                        sec_images.append(image_to_base64(rendered))

                    timeline_b64[rcp] = sec_images

                response = {
                    "status": "ok",
                    "type":   "timeline_result",
                    "data":   timeline_b64,
                }

            # ── svdd_analyze ─────────────────────────────────────
            elif command == "svdd_analyze":
                if svdd_encoder is None:
                    response = {
                        "status": "error",
                        "message": "SVDD 모델이 로드되지 않았습니다. "
                                   "python/train_svdd.py를 실행하여 모델을 학습하세요."
                    }
                elif not bin_path:
                    response = {
                        "status": "error",
                        "message": "payload.bin_path is required"
                    }
                else:
                    rcp_xy = extract_rcp_xy_from_bin(bin_path, fs=FS)

                    svdd_results    = {}
                    svdd_images_b64 = {}
                    any_anomaly     = False

                    for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                        n_total = len(x_mil_full)
                        if n_total < FS:
                            raise ValueError(
                                f"{rcp}: 신호가 너무 짧습니다 "
                                f"({n_total} samples, 필요: {FS})."
                            )

                        # 슬라이딩 윈도우 → 최대 거리 윈도우 선정
                        best_score_sw = -1.0
                        best_x_seg   = None
                        best_y_seg   = None
                        for s in range(0, n_total - FS + 1, SW_STEP):
                            xs = x_mil_full[s: s + FS]
                            ys = y_mil_full[s: s + FS]
                            sc, _, _, _ = _svdd_predict(xs, ys)
                            if sc > best_score_sw:
                                best_score_sw = sc
                                best_x_seg = xs
                                best_y_seg = ys

                        x_seg, y_seg = best_x_seg, best_y_seg

                        # SVDD 이상 점수 (최고 점수 윈도우 기준)
                        score, thr, is_anomaly, norm_score = _svdd_predict(x_seg, y_seg)
                        if is_anomaly:
                            any_anomaly = True

                        # 동적 표시 스케일
                        display_axis_lim = compute_dynamic_axis_lim(x_seg, y_seg)

                        # 절대 진폭
                        amplitude_mil = float(np.percentile(
                            np.abs(np.concatenate([x_seg, y_seg])), 99.5
                        ))

                        svdd_results[rcp] = {
                            "score":            round(score, 6),
                            "threshold":        round(thr, 6),
                            "is_anomaly":       is_anomaly,
                            "normalized_score": round(norm_score, 4),
                            "amplitude_mil":    round(amplitude_mil, 4),
                            "display_axis_lim": display_axis_lim,
                        }

                        # orbit 이미지 (최고 점수 윈도우 기준)
                        display_pil = _make_display_pil(x_seg, y_seg, display_axis_lim)
                        scale_label = f"±{display_axis_lim:.1f} mil"
                        svdd_images_b64[rcp] = {
                            "orbit": image_to_base64(
                                render_with_axes(display_pil, display_axis_lim,
                                                 cmap='gray', label=scale_label)
                            )
                        }

                    # 하나라도 이상이면 anomaly 판정
                    final_verdict = "anomaly" if any_anomaly else "normal"

                    # 최대 normalized_score
                    max_norm = max(
                        r["normalized_score"] for r in svdd_results.values()
                    )

                    response = {
                        "status": "ok",
                        "type":   "svdd_result",
                        "data": {
                            "final_verdict":      final_verdict,
                            "max_normalized_score": round(max_norm, 4),
                            "threshold":          round(svdd_threshold, 6),
                            "results":            svdd_results,
                            "images":             svdd_images_b64,
                        },
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
                    rcp_xy = extract_rcp_xy_from_bin(bin_path, fs=FS)

                    mae_results    = {}
                    mae_images_b64 = {}
                    any_anomaly    = False

                    for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                        n_total = len(x_mil_full)
                        if n_total < FS:
                            raise ValueError(
                                f"{rcp}: 신호가 너무 짧습니다 "
                                f"({n_total} samples, 필요: {FS})."
                            )

                        # Stage 1: n_eval=1 스윕 → 최고 점수 윈도우 선정
                        best_score1  = -1.0
                        best_x_seg   = None
                        best_y_seg   = None
                        for s in range(0, n_total - FS + 1, SW_STEP):
                            xs = x_mil_full[s: s + FS]
                            ys = y_mil_full[s: s + FS]
                            sc = _mae_predict(xs, ys, n_eval=1)["score"]
                            if sc > best_score1:
                                best_score1 = sc
                                best_x_seg  = xs
                                best_y_seg  = ys

                        # Stage 2: 최고 점수 윈도우 → n_eval=10 (최종 판정 + 시각화)
                        x_seg, y_seg = best_x_seg, best_y_seg
                        result = _mae_predict(x_seg, y_seg, n_eval=10)
                        if result["is_anomaly"]:
                            any_anomaly = True

                        amplitude_mil = float(np.percentile(
                            np.abs(np.concatenate([x_seg, y_seg])), 99.5
                        ))

                        mae_results[rcp] = {
                            "score":            result["score"],
                            "threshold":        result["threshold"],
                            "is_anomaly":       result["is_anomaly"],
                            "normalized_score": result["normalized_score"],
                            "amplitude_mil":    round(amplitude_mil, 4),
                        }
                        mae_images_b64[rcp] = result["images"]

                    final_verdict = "anomaly" if any_anomaly else "normal"
                    max_norm = max(r["normalized_score"] for r in mae_results.values())

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
                    rcp_xy      = extract_rcp_xy_from_bin(bin_path, fs=FS)
                    mae_results = {}
                    any_anomaly = False

                    for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                        n_total = len(x_mil_full)
                        if n_total < FS:
                            raise ValueError(f"{rcp}: 신호가 너무 짧습니다.")

                        # 슬라이딩 윈도우 n_eval=1 스윕 → 최대 점수
                        best_score_sw = -1.0
                        best_result   = None
                        for s in range(0, n_total - FS + 1, SW_STEP):
                            xs = x_mil_full[s: s + FS]
                            ys = y_mil_full[s: s + FS]
                            r = _mae_predict(xs, ys, n_eval=1)
                            if r["score"] > best_score_sw:
                                best_score_sw = r["score"]
                                best_result   = r

                        result = best_result
                        if result["is_anomaly"]:
                            any_anomaly = True

                        mae_results[rcp] = {
                            "score":            result["score"],
                            "threshold":        result["threshold"],
                            "is_anomaly":       result["is_anomaly"],
                            "normalized_score": result["normalized_score"],
                        }

                    final_verdict = "anomaly" if any_anomaly else "normal"
                    max_norm = max(r["normalized_score"] for r in mae_results.values())

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

            # ── svdd_fp_check ────────────────────────────────────
            # svdd_analyze와 동일하나 이미지 생성 생략 → 배치 FP 평가용
            elif command == "svdd_fp_check":
                if svdd_encoder is None:
                    response = {
                        "status": "error",
                        "message": "SVDD 모델이 로드되지 않았습니다.",
                    }
                elif not bin_path:
                    response = {"status": "error", "message": "payload.bin_path is required"}
                else:
                    rcp_xy       = extract_rcp_xy_from_bin(bin_path, fs=FS)
                    svdd_results = {}
                    any_anomaly  = False

                    for rcp, (x_mil_full, y_mil_full) in rcp_xy.items():
                        n_total = len(x_mil_full)
                        if n_total < FS:
                            raise ValueError(f"{rcp}: 신호가 너무 짧습니다.")

                        # 슬라이딩 윈도우 → 최대 거리 윈도우
                        best_score_sw = -1.0
                        best_x_seg    = None
                        best_y_seg    = None
                        for s in range(0, n_total - FS + 1, SW_STEP):
                            xs = x_mil_full[s: s + FS]
                            ys = y_mil_full[s: s + FS]
                            sc, _, _, _ = _svdd_predict(xs, ys)
                            if sc > best_score_sw:
                                best_score_sw = sc
                                best_x_seg    = xs
                                best_y_seg    = ys

                        score, thr, is_anomaly, norm_score = _svdd_predict(
                            best_x_seg, best_y_seg
                        )
                        if is_anomaly:
                            any_anomaly = True

                        svdd_results[rcp] = {
                            "score":            round(score, 6),
                            "threshold":        round(thr, 6),
                            "is_anomaly":       is_anomaly,
                            "normalized_score": round(norm_score, 4),
                        }

                    final_verdict = "anomaly" if any_anomaly else "normal"
                    max_norm = max(r["normalized_score"] for r in svdd_results.values())

                    response = {
                        "status": "ok",
                        "type":   "svdd_fp_result",
                        "data": {
                            "final_verdict":        final_verdict,
                            "max_normalized_score": round(max_norm, 4),
                            "threshold":            round(svdd_threshold, 6),
                            "results":              svdd_results,
                        },
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
