"""
integrated_gradients.py
=======================
Integrated Gradients (IG) 시각화 모듈.

ResNet18 (멀티스케일 이미지) 및 OrbitCNN1D (1D 신호) 모두 지원.

참고: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
  IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F_c(x' + α(x-x'))/∂x_i dα

Option A: steps=30 (ResNet), steps=15 (1D CNN)
  - 배치 방식으로 단일 순전파로 모든 단계 처리 (속도 최적화)
  - 기준선(baseline): 영 텐서
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def compute_ig(model, inp_tensor, baseline_tensor, steps, target_idx):
    """
    Integrated Gradients 계산 (배치 방식).

    Args:
        model          : eval 모드의 PyTorch 모델
        inp_tensor     : (1, C, H, W) 또는 (1, C, L) — 입력 (device 할당 완료)
        baseline_tensor: inp_tensor와 동일 shape — 기준선 (보통 zeros)
        steps          : 적분 단계 수
        target_idx     : int — 타겟 클래스 인덱스 (logit 기준)

    반환:
        ig: (C, H, W) 또는 (C, L) numpy array — 픽셀/샘플별 귀속값
    """
    device = inp_tensor.device

    # 보간 계수 α ∈ [1/steps, 2/steps, ..., 1.0]
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)  # (steps,)

    diff = inp_tensor - baseline_tensor  # (1, C, ...)
    # 보간 입력: (steps, C, ...)
    extra_dims = [1] * (inp_tensor.dim() - 1)
    interp = baseline_tensor + alphas.view(-1, *extra_dims) * diff   # broadcast
    interp = interp.detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(interp)                  # (steps, num_classes)
        score = logits[:, target_idx].sum()     # scalar
        score.backward()

    grads = interp.grad.detach()                # (steps, C, ...)
    avg_grads = grads.mean(dim=0, keepdim=True) # (1, C, ...)
    ig = (diff * avg_grads).squeeze(0).cpu().numpy()  # (C, ...)
    return ig


def render_ig_resnet(model, ms_arr, display_pil, transform, target_idx, steps=30):
    """
    ResNet18 Integrated Gradients 시각화.

    Args:
        model      : eval 모드의 ResNet18
        ms_arr     : (H, W, 3) uint8 — 멀티스케일 RGB 배열
        display_pil: PIL(L) — 동적 스케일 표시용 이미지 (오버레이 배경)
        transform  : 모델 입력 transform
        target_idx : 타겟 클래스 인덱스
        steps      : 적분 단계 수 (Option A: 30)

    반환: {"heatmap": PIL(RGB), "overlay": PIL(RGB)}
    """
    device = next(model.parameters()).device

    model_pil = Image.fromarray(ms_arr, mode='RGB')
    inp_tensor = transform(model_pil).unsqueeze(0).to(device)  # (1, 3, H, W)
    baseline = torch.zeros_like(inp_tensor)

    ig = compute_ig(model, inp_tensor, baseline, steps, target_idx)  # (3, H, W)

    # 채널 통합: 절대값 합산
    attr = np.abs(ig).sum(axis=0)  # (H, W)

    # p99 클리핑 + 정규화
    vmax = np.percentile(attr, 99)
    if vmax > 0:
        attr = np.clip(attr / vmax, 0, 1)

    # display_pil 크기에 맞게 리사이즈
    disp_w, disp_h = display_pil.size
    attr_img = Image.fromarray((attr * 255).astype(np.uint8), mode="L")
    attr_img = attr_img.resize((disp_w, disp_h), resample=Image.BILINEAR)
    attr_resized = np.array(attr_img) / 255.0

    # Jet 컬러맵 (GradCAM과 동일한 스타일)
    cmap_jet = plt.get_cmap("jet")
    heatmap = cmap_jet(attr_resized)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))

    # 오버레이 (GradCAM과 동일한 0.4/0.6 블렌드)
    raw_arr = np.array(display_pil.convert("L")).astype(np.float32)
    raw_arr = raw_arr / (raw_arr.max() + 1e-8)
    raw_rgb = np.stack([raw_arr] * 3, axis=-1)
    overlay = np.clip(0.4 * raw_rgb + 0.6 * heatmap, 0, 1)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    return {"heatmap": heatmap_pil, "overlay": overlay_pil}


def render_ig_signal(model_1d, x_seg, y_seg, target_idx, steps=15):
    """
    OrbitCNN1D Integrated Gradients 신호 시각화.

    Args:
        model_1d  : eval 모드의 OrbitCNN1D
        x_seg     : (40000,) numpy — X 방향 신호 (mil)
        y_seg     : (40000,) numpy — Y 방향 신호 (mil)
        target_idx: 타겟 클래스 인덱스
        steps     : 적분 단계 수 (Option A: 15)

    반환: PIL(RGB) — matplotlib 시각화 이미지
    """
    from preprocess import prepare_1d_input

    device = next(model_1d.parameters()).device

    arr = prepare_1d_input(x_seg, y_seg)                        # (2, 40000)
    inp_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 2, 40000)
    baseline = torch.zeros_like(inp_tensor)

    ig = compute_ig(model_1d, inp_tensor, baseline, steps, target_idx)  # (2, 40000)

    # 다운샘플링 (시각화: 1000 포인트)
    DS = 40
    x_plot = arr[0, ::DS]       # (1000,)
    y_plot = arr[1, ::DS]       # (1000,)
    ig_x   = ig[0, ::DS]       # (1000,)
    ig_y   = ig[1, ::DS]       # (1000,)
    t      = np.arange(len(x_plot)) * DS / 40000.0  # 시간축 (초)

    def _normalize(a):
        vmax = np.percentile(np.abs(a), 99) + 1e-8
        return np.clip(a / vmax, -1, 1)

    ig_x_n = _normalize(ig_x)
    ig_y_n = _normalize(ig_y)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), dpi=100, tight_layout=True)
    fig.patch.set_facecolor("#0f172a")

    for ax, sig, attr, label in [
        (axes[0], x_plot, ig_x_n, "X (mil)"),
        (axes[1], y_plot, ig_y_n, "Y (mil)"),
    ]:
        ax.set_facecolor("#1e293b")
        ax.plot(t, sig, color="#94a3b8", linewidth=0.6, alpha=0.9)
        # 귀속값 음영 (양수=빨강, 음수=파랑)
        ax.fill_between(t, sig, where=(attr > 0.1),  alpha=0.4, color="#dc2626")
        ax.fill_between(t, sig, where=(attr < -0.1), alpha=0.4, color="#3b82f6")
        ax.set_ylabel(label, color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#64748b", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(0.5)

    axes[0].set_title("Integrated Gradients — 1D CNN", color="#e2e8f0", fontsize=9, pad=4)
    axes[1].set_xlabel("time (s)", color="#94a3b8", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="PNG", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
