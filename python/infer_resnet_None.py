import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter  # 없으면 !pip install scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# 공유 전처리 모듈
SCRIPT_DIR_IMPORT = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR_IMPORT not in sys.path:
    sys.path.insert(0, SCRIPT_DIR_IMPORT)
from preprocess import (
    make_multiscale_orbit,
    make_orbit_image_v2,
    compute_dynamic_axis_lim,
    build_multiscale_transform,
)

# UTF-8 출력 강제 (Windows 한글 깨짐 방지)
# reconfigure()는 표준 TextIOWrapper 전용 — Colab OutStream 등에서는 스킵
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure') and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# =========================
# 설정
# =========================
DEVICE = torch.device("cpu")   # Grad-CAM까지 전부 CPU에서 수행
print("Using device:", DEVICE)

# 가중치 경로 (스크립트 디렉토리 기준 상대 경로)
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 새 멀티스케일 모델 경로 (학습 후 생성됨)
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_multiscale.pth")
# 레거시 모델 경로 (호환성 유지)
MODEL_PATH_LEGACY = os.path.join(SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth")


# =============================
# 1) 이전 기수 스타일 BIN 파서
# =============================
def parse_bin_legacy(bin_path,
                     fs=40_000,
                     duration_sec=10,
                     num_channels=24,
                     bytes_per_sample=4):
    """
    - 파일 맨 끝에서 24채널 × 10초 신호만 잘라서 읽는 이전 기수 방식.
    - float32 little-endian으로 파싱.
    - 반환: (24, 400000)
    """
    num_samples_total = fs * duration_sec
    block_bytes = num_channels * num_samples_total * bytes_per_sample

    with open(bin_path, "rb") as f:
        content = f.read()

    # 맨 끝에서 필요한 데이터 크기만큼 자르기
    signal_bytes = content[-block_bytes:]

    # float32 LE 변환
    data = np.frombuffer(signal_bytes, dtype="<f4")
    data = data.reshape(num_channels, num_samples_total)

    return data


# =============================
# 2) 이전 기수 방식 XY 선택
# =============================
def extract_xy_pairs_legacy(data,
                            target_pairs=((4, 6), (10, 12), (16, 18), (22, 24))):
    """
    - (4,6) → 채널 4,5 페어
    - (10,12) → 채널 10,11
    - ...
    """
    xy_list = []

    for (start_idx, end_idx) in target_pairs:
        ch_x = start_idx      # ex: 4
        ch_y = start_idx + 1  # ex: 5

        x = data[ch_x].copy()
        y = data[ch_y].copy()
        xy_list.append((x, y))

    return xy_list


# =============================
# 3) Volt -> mil 변환
# =============================
def volt_to_mil(x, y, mil_per_volt=10.0):
    """
    평균 제거 + mil 변환
    """
    x_ac = x - x.mean()
    y_ac = y - y.mean()
    return x_ac * mil_per_volt, y_ac * mil_per_volt


# =============================
# 4) CNN 입력용 orbit grid 생성
# =============================
def make_orbit_image(x_mil,
                     y_mil,
                     axis_lim=3.0,
                     img_size=256):
    """
    학습용 orbit_dataset_v2와 동일한 방식:
    - [-axis_lim, axis_lim] 범위를 img_size x img_size grid로 매핑
    - 점 개수(count)를 누적
    - Gaussian blur + log1p로 대비 강화
    - 0~255 uint8 이미지 반환
    """
    # 좌표 → [0, img_size-1] 인덱스
    x_norm = (x_mil + axis_lim) / (2 * axis_lim) * (img_size - 1)
    y_norm = (y_mil + axis_lim) / (2 * axis_lim) * (img_size - 1)

    x_idx = np.clip(x_norm.astype(int), 0, img_size - 1)
    y_idx = np.clip(y_norm.astype(int), 0, img_size - 1)

    grid = np.zeros((img_size, img_size), dtype=np.float32)
    grid[y_idx, x_idx] += 1.0

    # blur + log scaling
    grid = gaussian_filter(grid, sigma=1.2)
    grid = np.log1p(grid)
    grid = grid / (grid.max() + 1e-8)

    return (grid * 255).astype(np.uint8)


# =========================
# 5) 모델 정의 & 로드
# =========================
def get_model(num_classes):
    # 학습 때는 ResNet18_Weights.DEFAULT 썼지만,
    # 추론에서는 weights=None 후 state_dict로 덮어써도 무방
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_trained_model(model_path):
    """
    모델을 로드하고 (model, class_names, meta) 를 반환한다.
    meta에는 norm_mean, norm_std, model_type 등이 포함된다.
    이전 코드와의 호환을 위해 2-tuple 언패킹도 동작하도록 클래스를 사용한다.
    """
    from model_loader import load_trained_model as _load
    model, class_names, meta = _load(model_path)
    model.to(DEVICE)
    # 3-튜플을 반환하되, 오래된 코드(2-언패킹)도 지원하기 위해
    # (model, class_names) 로 사용하는 호출 사이트는 그대로 동작한다.
    return model, class_names, meta


def build_transform_from_meta(meta):
    """checkpoint meta → 올바른 transform 빌드"""
    if meta.get("model_type") == "resnet18_multiscale":
        return build_multiscale_transform(
            mean=meta["norm_mean"], std=meta["norm_std"], augment=False
        )
    # 레거시: Grayscale→3ch, 224 리사이즈, ImageNet 정규화
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# 레거시 transform (구버전 호환)
transform_for_model = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class GradCAM:
    """
    backward hook 안 쓰고,
    forward hook에서 feature map을 저장하고 retain_grad() 걸어둔 다음
    backward 후에 activations.grad를 사용하는 방식.
    → PyTorch 2.x에서도 안정적으로 동작하고, deprecation 경고도 없음.
    """
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()

        # 예: model.layer4
        self.target_layer = getattr(self.model, target_layer_name)
        self.activations = None  # forward 결과 (feature map)
        self.hook_handle = None

        # ----- forward hook 등록 -----
        def forward_hook(module, input, output):
            # output: (N, C, H, W)
            # backward 때 gradient를 받기 위해 retain_grad()
            self.activations = output
            self.activations.retain_grad()

        self.hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, 3, 224, 224) on DEVICE
        class_idx: 타겟 클래스 인덱스 (None이면 모델이 예측한 클래스 사용)
        """
        # 이전 gradient 초기화
        self.model.zero_grad()

        # forward
        output = self.model(input_tensor)  # (1, num_classes)

        # 타겟 클래스 선택
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        # backward: activations.grad에 gradient가 쌓임
        target.backward()

        # forward hook에서 저장해둔 feature & gradient
        acts = self.activations            # (1, C, H, W)
        grads = self.activations.grad      # (1, C, H, W)

        # 채널별 평균 gradient → weight
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)    # (H, W)

        # ReLU + 정규화
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), class_idx

    def close(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def show_gradcam_for_pil(model, class_names, pil_img, title_prefix=""):
    """
    한 장의 orbit PIL 이미지에 대해 Grad-CAM overlay 시각화
    """
    # model input
    inp = transform_for_model(pil_img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, target_layer_name="layer4")
    cam, class_idx = gradcam.generate(inp)
    gradcam.close()

    pred_label = class_names[class_idx]

    # cam resize to original
    raw_img = pil_img.convert("L")
    cam_img = Image.fromarray(np.uint8(cam * 255), mode="L")
    cam_img = cam_img.resize(raw_img.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_img) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_resized)[:, :, :3]

    raw_arr = np.array(raw_img).astype(np.float32)
    raw_arr = raw_arr / (raw_arr.max() + 1e-8)
    raw_rgb = np.stack([raw_arr]*3, axis=-1)

    overlay = 0.4 * raw_rgb + 0.6 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(8, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(raw_img, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.imshow(cam_resized, cmap="jet")

    plt.subplot(1, 3, 3)
    plt.title(f"{title_prefix} pred: {pred_label}")
    plt.axis("off")
    plt.imshow(overlay)

    plt.tight_layout()
    plt.show()

# =========================
# 6-1) 축/스케일 포함 이미지 렌더링 (PIL-only 컴포지팅)
# =========================
# axis_lim 값별로 프레임을 캐시 (동적 스케일 지원)
_frame_cache: dict = {}   # axis_lim → (overlay, bounds, size)


def _init_frame(axis_lim=3.0):
    """
    matplotlib으로 축/눈금/레이블/십자선만 있는 투명 프레임을 axis_lim당 1회 렌더링하여
    딕셔너리 캐시에 저장한다. 동적 스케일(여러 axis_lim)에서도 캐시 재사용 가능.
    """
    global _frame_cache
    if axis_lim in _frame_cache:
        return

    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=90)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)

    ax.axhline(0, color='white', linewidth=1.3, linestyle='--', alpha=0.9, zorder=5)
    ax.axvline(0, color='white', linewidth=1.3, linestyle='--', alpha=0.9, zorder=5)

    ax.set_xlabel('X (mil)', fontsize=9)
    ax.set_ylabel('Y (mil)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.canvas.draw()

    bbox = ax.get_position()
    w_fig, h_fig = fig.canvas.get_width_height()
    px = round(bbox.x0 * w_fig)
    py = round((1 - bbox.y1) * h_fig)
    pw = round(bbox.x1 * w_fig) - px
    ph = round((1 - bbox.y0) * h_fig) - py

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_fig, w_fig, 4)
    overlay = Image.fromarray(buf.copy(), mode='RGBA')
    _frame_cache[axis_lim] = (overlay, (px, py, pw, ph), (w_fig, h_fig))

    plt.close(fig)


def render_with_axes(pil_or_array, axis_lim=3.0, cmap='gray', label=None):
    """
    PIL Image 또는 numpy 배열을 받아 물리적 축(mil 단위)이 포함된 이미지를 반환한다.
    axis_lim은 동적으로 지정 가능하며, 스냅된 값별로 캐시를 재사용한다.

    Args:
        label: 이미지 좌상단에 표시할 텍스트 (예: "abnormal · Grad-CAM").
               None이면 표시 안 함.
    """
    _init_frame(axis_lim)
    frame_overlay, (px, py, pw, ph), frame_size = _frame_cache[axis_lim]

    if isinstance(pil_or_array, Image.Image):
        arr = np.array(pil_or_array)
    else:
        arr = pil_or_array

    if arr.ndim == 2:
        if cmap == 'gray':
            rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            cm = plt.get_cmap(cmap)
            normed = arr.astype(np.float32) / (arr.max() + 1e-8)
            rgb = (cm(normed)[:, :, :3] * 255).astype(np.uint8)
    else:
        rgb = arr if arr.dtype == np.uint8 else (arr * 255).astype(np.uint8)

    data_img = Image.fromarray(rgb).transpose(Image.FLIP_TOP_BOTTOM)
    data_img = data_img.resize((pw, ph), Image.BILINEAR)

    canvas = Image.new('RGBA', frame_size, (255, 255, 255, 255))
    canvas.paste(data_img, (px, py))
    canvas = Image.alpha_composite(canvas, frame_overlay)
    result = canvas.convert('RGB')

    # 레이블 텍스트 오버레이 (좌상단)
    if label:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            try:
                import matplotlib as _mpl
                _dejavu = os.path.join(
                    os.path.dirname(_mpl.__file__),
                    "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"
                )
                font = ImageFont.truetype(_dejavu, 13)
            except Exception:
                font = ImageFont.load_default(size=13)
        # 반투명 배경을 위한 bbox 계산
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        bg = Image.new('RGBA', result.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg)
        bg_draw.rectangle(
            [px + 4, py + 4, px + tw + pad * 2 + 4, py + th + pad * 2 + 4],
            fill=(0, 0, 0, 140)
        )
        result = Image.alpha_composite(result.convert('RGBA'), bg).convert('RGB')
        draw = ImageDraw.Draw(result)
        draw.text((px + 4 + pad, py + 4 + pad), label, fill=(255, 255, 255), font=font)

    return result


# =========================
# 7) BIN → RCP별 sec9 orbit PIL 생성
# =========================
def make_orbit_pils_sec9_from_bin(bin_path,
                                  fs=40_000,
                                  duration_sec=10,
                                  mil_per_volt=10.0,
                                  axis_lim_mil=3.0,
                                  img_size=256):
    """
    BIN 하나에서:
      - parse_bin_legacy
      - extract_xy_pairs_legacy
      - volt_to_mil
      - sec9 (9~10초) orbit grid 이미지 1장 생성
    반환:
      { "RCP1A": PIL, "RCP1B": PIL, "RCP2A": PIL, "RCP2B": PIL }
    """
    data = parse_bin_legacy(bin_path, fs=fs, duration_sec=duration_sec)
    xy_pairs = extract_xy_pairs_legacy(data)

    rcp_names = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
    samples_per_sec = fs

    rcp_to_pil = {}

    for i, (x, y) in enumerate(xy_pairs):
        rcp = rcp_names[i]
        x_mil, y_mil = volt_to_mil(x, y, mil_per_volt=mil_per_volt)

        # ---- sec9 구간 ----
        s = 9 * samples_per_sec
        e = 10 * samples_per_sec
        seg_x = x_mil[s:e]
        seg_y = y_mil[s:e]

        grid = make_orbit_image(seg_x, seg_y,
                                axis_lim=axis_lim_mil,
                                img_size=img_size)

        pil_img = Image.fromarray(grid)  # mode="L"
        rcp_to_pil[rcp] = pil_img

    return rcp_to_pil   # {"RCP1A": PIL, ...}


# =========================
# 8) RCP별 orbit 이미지 1장으로 추론 (레거시)
# =========================
def predict_rcp_single(model, class_names, pil_img, transform=None):
    """pil_img: PIL 이미지 (레거시: Grayscale, 신규: RGB 3ch 멀티스케일)"""
    tf = transform if transform is not None else transform_for_model
    inp = tf(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(prob))
    pred_class = class_names[pred_idx]
    return pred_class, prob


# =========================
# 8-2) 멀티스케일 numpy 배열 → 예측 (신규)
# =========================
def predict_from_multiscale(model, class_names, ms_arr, transform):
    """
    ms_arr: (H, W, 3) uint8 멀티스케일 배열
    transform: build_transform_from_meta() 로 얻은 transform
    """
    pil = Image.fromarray(ms_arr, mode='RGB')
    inp = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(prob))
    return class_names[pred_idx], prob


# =========================
# 8-3) BIN → RCP별 XY 신호 추출
# =========================
def extract_rcp_xy_from_bin(bin_path, fs=40_000, duration_sec=10, mil_per_volt=10.0):
    """
    BIN 파일 전체를 파싱하여 RCP별 전체 구간 XY(mil) 반환.
    반환: {"RCP1A": (x_mil, y_mil), ...}
    """
    data = parse_bin_legacy(bin_path, fs=fs, duration_sec=duration_sec)
    xy_pairs = extract_xy_pairs_legacy(data)
    rcp_names = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
    return {
        rcp_names[i]: volt_to_mil(x, y, mil_per_volt=mil_per_volt)
        for i, (x, y) in enumerate(xy_pairs)
        if i < len(rcp_names)
    }


# =========================
# 9) 시간대별 Orbit 이미지 생성
# =========================
def make_temporal_orbit_pils(bin_path,
                             fs=40_000,
                             duration_sec=10,
                             mil_per_volt=10.0,
                             axis_lim_mil=3.0,
                             img_size=256):
    """
    BIN 파일에서 시간대별(초단위) Orbit 이미지 생성
    반환: {
        "RCP1A": [PIL_0s, PIL_1s, ..., PIL_9s],
        ...
    }
    """
    data = parse_bin_legacy(bin_path, fs=fs, duration_sec=duration_sec)
    xy_pairs = extract_xy_pairs_legacy(data)
    
    rcp_names = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
    samples_per_sec = fs
    
    rcp_to_temporal = {}
    
    for i, (x, y) in enumerate(xy_pairs):
        rcp = rcp_names[i]
        x_mil, y_mil = volt_to_mil(x, y, mil_per_volt=mil_per_volt)
        
        temporal_pils = []
        for sec in range(duration_sec):
            s = sec * samples_per_sec
            e = (sec + 1) * samples_per_sec
            seg_x = x_mil[s:e]
            seg_y = y_mil[s:e]
            
            grid = make_orbit_image(seg_x, seg_y,
                                   axis_lim=axis_lim_mil,
                                   img_size=img_size)
            pil_img = Image.fromarray(grid)
            temporal_pils.append(pil_img)
        
        rcp_to_temporal[rcp] = temporal_pils
    
    return rcp_to_temporal


# =========================
# 10) Grad-CAM 이미지 생성
# =========================
def generate_gradcam_images(model, class_names, pil_img, transform=None, class_idx=None):
    """
    Grad-CAM 3종 이미지 생성.
    pil_img: 모델 입력용 PIL (레거시: Grayscale, 신규: RGB 3ch 멀티스케일)
    transform: 적절한 transform (None이면 레거시 transform 사용)
    class_idx: Grad-CAM 타겟 클래스 인덱스 (None이면 모델 예측 클래스 사용)
    반환: {"original": PIL, "heatmap": PIL, "overlay": PIL, "target_class": str}
    """
    tf = transform if transform is not None else transform_for_model
    inp = tf(pil_img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, target_layer_name="layer4")
    cam, class_idx = gradcam.generate(inp, class_idx=class_idx)
    gradcam.close()

    target_class = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)

    # 표시용 배경: 멀티스케일이면 mid 채널(axis_lim=3.0), 레거시면 그레이스케일
    if pil_img.mode == 'RGB':
        arr = np.array(pil_img)
        raw_img = Image.fromarray(arr[:, :, 1], mode='L')  # mid 채널
    else:
        raw_img = pil_img.convert("L")

    cam_img = Image.fromarray(np.uint8(cam * 255), mode="L")
    cam_img = cam_img.resize(raw_img.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_img) / 255.0

    cmap_jet = plt.get_cmap("jet")
    heatmap = cmap_jet(cam_resized)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))

    raw_arr = np.array(raw_img).astype(np.float32)
    raw_arr = raw_arr / (raw_arr.max() + 1e-8)
    raw_rgb = np.stack([raw_arr] * 3, axis=-1)

    overlay = 0.4 * raw_rgb + 0.6 * heatmap
    overlay = np.clip(overlay, 0, 1)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    return {"original": raw_img, "heatmap": heatmap_pil, "overlay": overlay_pil,
            "target_class": target_class}


def generate_gradcam_on_display(model, class_names, ms_arr, display_pil, transform,
                                class_idx=None):
    """
    멀티스케일 배열로 GradCAM을 실행하고, 동적 스케일 display_pil에 오버레이한다.

    Args:
        class_idx: Grad-CAM 타겟 클래스 인덱스.
                   None이면 ResNet 자체 예측 클래스 사용.
                   앙상블 추론 시 앙상블의 예측 클래스 인덱스를 전달하면
                   "앙상블이 최종 결정한 클래스"에 대한 ResNet 활성화 맵을 얻는다.
    반환: {"original": PIL(L), "heatmap": PIL(RGB), "overlay": PIL(RGB),
           "target_class": str}
    """
    model_pil = Image.fromarray(ms_arr, mode='RGB')
    inp = transform(model_pil).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, target_layer_name="layer4")
    cam, used_idx = gradcam.generate(inp, class_idx=class_idx)
    gradcam.close()

    target_class = class_names[used_idx] if used_idx < len(class_names) else str(used_idx)

    raw_img = display_pil.convert("L")
    cam_img = Image.fromarray(np.uint8(cam * 255), mode="L")
    cam_img = cam_img.resize(raw_img.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_img) / 255.0

    cmap_jet = plt.get_cmap("jet")
    heatmap = cmap_jet(cam_resized)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))

    raw_arr = np.array(raw_img).astype(np.float32)
    raw_arr = raw_arr / (raw_arr.max() + 1e-8)
    raw_rgb = np.stack([raw_arr] * 3, axis=-1)

    overlay = np.clip(0.4 * raw_rgb + 0.6 * heatmap, 0, 1)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    return {
        "original":     raw_img,
        "heatmap":      heatmap_pil,
        "overlay":      overlay_pil,
        "target_class": target_class,
    }


# =========================
# 11) 이미지 저장 및 경로 반환
# =========================
def save_visualization_images(bin_path, model, class_names, rcp_to_pil, rcp_to_temporal):
    """
    모든 시각화 이미지를 임시 디렉토리에 저장하고 경로 반환
    """
    import tempfile
    import time
    
    # 임시 디렉토리 생성 (타임스탬프 포함)
    timestamp = int(time.time() * 1000)
    temp_dir = os.path.join(tempfile.gettempdir(), f"rcp_viz_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)
    
    visualization_data = {}
    
    for rcp, pil_img in rcp_to_pil.items():
        rcp_dir = os.path.join(temp_dir, rcp)
        os.makedirs(rcp_dir, exist_ok=True)
        
        # 1. Orbit 이미지 (sec9)
        orbit_path = os.path.join(rcp_dir, "orbit_sec9.png")
        pil_img.save(orbit_path)
        
        # 2. Grad-CAM 이미지들
        gradcam_imgs = generate_gradcam_images(model, class_names, pil_img)
        gradcam_paths = {}
        for img_type, img in gradcam_imgs.items():
            path = os.path.join(rcp_dir, f"gradcam_{img_type}.png")
            img.save(path)
            gradcam_paths[img_type] = path
        
        # 3. 시간대별 Orbit 이미지들
        temporal_paths = []
        if rcp in rcp_to_temporal:
            for sec, temporal_pil in enumerate(rcp_to_temporal[rcp]):
                path = os.path.join(rcp_dir, f"orbit_sec{sec}.png")
                temporal_pil.save(path)
                temporal_paths.append(path)
        
        visualization_data[rcp] = {
            "orbit": orbit_path,
            "gradcam": gradcam_paths,
            "temporal": temporal_paths
        }
    
    return visualization_data, temp_dir


# =========================
# 12) BIN 하나에 대해 RCP별 sec9 추론 + 이미지 저장
# =========================
def infer_bin_sec9_with_images(bin_path,
                                model_path=MODEL_PATH):
    """
    이미지 포함 전체 추론
    """
    print(f"\n=== Inference with Visualization for BIN : {bin_path} ===")
    
    # 1) 모델 로드
    model, class_names, _meta = load_trained_model(model_path)
    
    # 2) BIN → RCP별 sec9 orbit PIL
    rcp_to_pil = make_orbit_pils_sec9_from_bin(bin_path)
    
    # 3) 시간대별 Orbit 생성
    rcp_to_temporal = make_temporal_orbit_pils(bin_path)
    
    # 4) 추론
    results = {}
    for rcp, pil9 in rcp_to_pil.items():
        pred_class, prob = predict_rcp_single(model, class_names, pil9)
        results[rcp] = {
            "prediction": pred_class,
            "probabilities": {
                class_names[i]: float(prob[i]) for i in range(len(class_names))
            }
        }
        print(f"[{rcp}] → pred: {pred_class}, probs={results[rcp]['probabilities']}")
    
    # 5) 이미지 저장
    visualization_data, temp_dir = save_visualization_images(
        bin_path, model, class_names, rcp_to_pil, rcp_to_temporal
    )
    
    # 6) 최종 라벨
    if any(res["prediction"] == "abnormal" for res in results.values()):
        final_label = "abnormal"
    else:
        final_label = "normal"
    
    print(f"\n>>> Final BIN decision: {final_label}")
    print(f">>> Visualization images saved to: {temp_dir}")
    
    return results, final_label, visualization_data, temp_dir


# =========================
# 13) BIN 하나에 대해 RCP별 sec9 추론 + (선택) Grad-CAM
# =========================
def infer_bin_sec9(bin_path,
                   model_path=MODEL_PATH,
                   show_cam=False):
    """
    bin_path 하나에 대해:
      - RCP1A/1B/2A/2B 별 normal/abnormal 확률 (sec9만 사용)
      - 필요하면 각 RCP sec9에 대해 Grad-CAM 시각화
    """
    print(f"\n=== Inference for BIN : {bin_path} ===")

    # 1) 모델 로드
    model, class_names, _meta = load_trained_model(model_path)

    # 2) BIN → RCP별 sec9 orbit PIL
    rcp_to_pil = make_orbit_pils_sec9_from_bin(bin_path)

    results = {}

    for rcp, pil9 in rcp_to_pil.items():
        pred_class, prob = predict_rcp_single(model, class_names, pil9)
        results[rcp] = {
            "prediction": pred_class,
            "probabilities": {
                class_names[i]: float(prob[i]) for i in range(len(class_names))
            }
        }

        print(f"[{rcp}] → pred: {pred_class}, probs={results[rcp]['probabilities']}")

        # Grad-CAM
        if show_cam:
            show_gradcam_for_pil(model, class_names, pil9,
                                 title_prefix=f"{rcp} ")

    # BIN 전체 라벨: 하나라도 abnormal이면 abnormal
    if any(res["prediction"] == "abnormal" for res in results.values()):
        final_label = "abnormal"
    else:
        final_label = "normal"

    print(f"\n>>> Final BIN decision: {final_label}")

    return results, final_label


# =========================
# 10) 실행 예시
# =========================

# 위에서 BIN_PATH, MODEL_PATH 설정해두고 실행

#정상버전

# =========================
# CLI Entry
# =========================
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="ResNet (orbit sec9) inference for a single BIN file"
    )
    p.add_argument(
        "--bin_path",
        required=True,
        help="Path to the .BIN file"
    )
    p.add_argument(
        "--device",
        default="cpu",
        help='Device string (default: "cpu")'
    )
    p.add_argument(
        "--show_cam",
        action="store_true",
        help="If set, visualize Grad-CAM per RCP"
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="If set, print only JSON result"
    )
    p.add_argument(
        "--with-images",
        action="store_true",
        help="If set, save visualization images and include paths in JSON"
    )
    return p.parse_args()


def _set_device(device_str: str):
    # Many notebooks use a global DEVICE; keep that convention if present.
    global DEVICE
    try:
        import torch
        DEVICE = torch.device(device_str)
    except Exception:
        DEVICE = device_str  # fallback; functions may ignore DEVICE anyway.

if __name__ == "__main__":
    args = _parse_args()
    _set_device(args.device)

    # MODEL_PATH는 이미 위에서 고정됨

    # 이미지 포함 모드
    if args.with_images or (args.json and not args.show_cam):
        results, final_label, visualization_data, temp_dir = infer_bin_sec9_with_images(
            args.bin_path
        )
        
        if args.json:
            import json
            out = {
                "bin_path": args.bin_path,
                "model_path": MODEL_PATH,
                "final_label": final_label,
                "results": results,
                "visualization": visualization_data,
                "temp_dir": temp_dir
            }
            print(json.dumps(out, ensure_ascii=False))
        else:
            print("\n=== Inference Done ===")
            print(f"BIN: {args.bin_path}")
            print(f"FINAL LABEL: {final_label}")
            print(f"Images saved to: {temp_dir}")
    # 기본 모드 (Grad-CAM 플롯 표시)
    else:
        results, final_label = infer_bin_sec9(
            args.bin_path,
            show_cam=args.show_cam,
        )
        
        if args.json:
            import json
            out = {
                "bin_path": args.bin_path,
                "model_path": MODEL_PATH,
                "final_label": final_label,
                "results": results,
            }
            print(json.dumps(out, ensure_ascii=False))
        else:
            print("\n=== Inference Done ===")
            print(f"BIN: {args.bin_path}")
            print(f"FINAL LABEL: {final_label}")