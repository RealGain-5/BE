import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

# 멀티스케일 고정 채널 (fine / mid / wide)
MULTISCALE_AXIS_LIMS = (1.0, 3.0, 6.0)


# ==========================================
# 1. 바이너리 파싱 및 신호 추출
# ==========================================
def parse_bin_legacy(
    bin_path, fs=40_000, duration_sec=10, num_channels=24, bytes_per_sample=4
):
    """
    이전 기수 방식 BIN 파서:
    - 파일 맨 끝에서 24채널 × 10초 신호만 잘라서 읽음.
    - float32 little-endian으로 파싱.
    - 반환: (24, 400000) NumPy Array
    """
    num_samples_total = fs * duration_sec
    block_bytes = num_channels * num_samples_total * bytes_per_sample

    try:
        with open(bin_path, "rb") as f:
            content = f.read()

        # 데이터가 부족한 경우 예외 처리
        if len(content) < block_bytes:
            # 혹은 0으로 패딩하는 로직을 넣을 수도 있음
            raise ValueError(
                f"File size too small. Expected at least {block_bytes} bytes."
            )

        # 맨 끝에서 필요한 데이터 크기만큼 자르기
        signal_bytes = content[-block_bytes:]

        # float32 LE 변환
        data = np.frombuffer(signal_bytes, dtype="<f4")
        data = data.reshape(num_channels, num_samples_total)

        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse BIN file: {e}")


def extract_xy_pairs_legacy(data, target_pairs=((4, 6), (10, 12), (16, 18), (22, 24))):
    """
    채널 데이터에서 RCP별 X, Y 페어를 추출합니다.
    - (4,6) → 채널 4,5 페어 (RCP1A)
    - (10,12) → 채널 10,11 (RCP1B)
    ...
    """
    xy_list = []

    for start_idx, end_idx in target_pairs:
        ch_x = start_idx  # ex: 4
        ch_y = start_idx + 1  # ex: 5

        # 인덱스 범위 체크
        if ch_x >= data.shape[0] or ch_y >= data.shape[0]:
            continue

        x = data[ch_x].copy()
        y = data[ch_y].copy()
        xy_list.append((x, y))

    return xy_list


def volt_to_mil(x, y, mil_per_volt=10.0):
    """
    전압(Volt) 신호를 변위(mil) 단위로 변환합니다.
    - 평균 제거(DC Offset Removal) 수행
    """
    x_ac = x - x.mean()
    y_ac = y - y.mean()
    return x_ac * mil_per_volt, y_ac * mil_per_volt


# ==========================================
# 2. 이미지 생성 (Orbit Image)
# ==========================================
def make_orbit_image(x_mil, y_mil, axis_lim=3.0, img_size=256):
    """
    X, Y 진동 신호를 2D Orbit 히스토그램 이미지로 변환합니다. (레거시 호환용)
    - int() 절삭 방식 (기존 동작 유지)
    """
    x_norm = (x_mil + axis_lim) / (2 * axis_lim) * (img_size - 1)
    y_norm = (y_mil + axis_lim) / (2 * axis_lim) * (img_size - 1)

    x_idx = np.clip(x_norm.astype(int), 0, img_size - 1)
    y_idx = np.clip(y_norm.astype(int), 0, img_size - 1)

    grid = np.zeros((img_size, img_size), dtype=np.float32)
    grid[y_idx, x_idx] += 1.0

    grid = gaussian_filter(grid, sigma=1.2)
    grid = np.log1p(grid)
    grid = grid / (grid.max() + 1e-8)

    return (grid * 255).astype(np.uint8)


def make_orbit_image_v2(x_mil, y_mil, axis_lim=3.0, img_size=256):
    """
    np.histogram2d 기반 서브픽셀 정밀도 orbit 이미지.
    - 클리핑 없이 범위 내 포인트만 집계
    - Gaussian Blur + log1p + min-max 정규화
    - 반환: (img_size, img_size) uint8
    """
    grid, _, _ = np.histogram2d(
        y_mil, x_mil,
        bins=img_size,
        range=[[-axis_lim, axis_lim], [-axis_lim, axis_lim]],
    )
    grid = grid.astype(np.float32)
    grid = gaussian_filter(grid, sigma=1.2)
    grid = np.log1p(grid)
    grid = grid / (grid.max() + 1e-8)
    return (grid * 255).astype(np.uint8)


def make_multiscale_orbit(x_mil, y_mil, img_size=256):
    """
    3채널 멀티스케일 orbit 이미지 생성.
    채널 0: axis_lim=1.0 mil (fine)
    채널 1: axis_lim=3.0 mil (mid)
    채널 2: axis_lim=6.0 mil (wide)
    반환: (img_size, img_size, 3) uint8 — PIL RGB Image로 바로 변환 가능
    """
    ch_fine = make_orbit_image_v2(x_mil, y_mil, axis_lim=MULTISCALE_AXIS_LIMS[0], img_size=img_size)
    ch_mid  = make_orbit_image_v2(x_mil, y_mil, axis_lim=MULTISCALE_AXIS_LIMS[1], img_size=img_size)
    ch_wide = make_orbit_image_v2(x_mil, y_mil, axis_lim=MULTISCALE_AXIS_LIMS[2], img_size=img_size)
    return np.stack([ch_fine, ch_mid, ch_wide], axis=-1)


def compute_dynamic_axis_lim(x_mil, y_mil, percentile=99.5, margin=1.2):
    """
    실제 신호 범위에서 표시용 axis_lim을 자동 산정한다.
    결과값을 '보기 좋은' 이산 구간으로 스냅핑하여 캐시 재사용률을 높인다.
    """
    max_range = max(
        np.percentile(np.abs(x_mil), percentile),
        np.percentile(np.abs(y_mil), percentile),
    )
    raw_lim = float(max_range * margin)
    # 이산 스냅: 캐시 히트율 향상
    snap_breakpoints = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    for bp in snap_breakpoints:
        if raw_lim <= bp:
            return bp
    return round(raw_lim, 1)


# ==========================================
# 1D CNN 전처리
# ==========================================
def prepare_1d_input(x_mil, y_mil):
    """
    Raw orbit 신호 (X_mil, Y_mil) → 1D CNN 입력 텐서.

    Per-sample 정규화: 99.5 퍼센타일 스케일 사용.
    체크포인트에 통계 저장 불필요 — train/infer 일관성 자동 보장.

    Args:
        x_mil: np.ndarray (40000,) — X 변위 (mil)
        y_mil: np.ndarray (40000,) — Y 변위 (mil)

    Returns:
        np.ndarray (2, 40000) float32
    """
    combined = np.concatenate([x_mil, y_mil])
    scale = float(np.percentile(np.abs(combined), 99.5)) + 1e-8
    return np.stack([x_mil / scale, y_mil / scale], axis=0).astype(np.float32)


# ==========================================
# 3. 모델 입력용 Transform
# ==========================================
# 레거시 모델용 (그레이스케일 → 3ch 복사, 224 리사이즈, ImageNet 정규화)
transform_for_model = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def build_multiscale_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), augment=False):
    """
    멀티스케일 모델용 transform 빌더.
    - 입력: PIL RGB 이미지 (256×256, 3채널 멀티스케일)
    - 리사이즈 불필요 (ResNet18 AdaptiveAvgPool이 처리)
    - mean/std는 학습 데이터셋에서 계산한 값을 사용 (체크포인트에 저장됨)
    """
    ops = []
    if augment:
        ops += [
            transforms.RandomRotation(degrees=360),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ]
    return transforms.Compose(ops)


# ==========================================
# 4. 고수준 파이프라인 (High-level Functions)
# ==========================================
def make_orbit_pils_sec9_from_bin(
    bin_path,
    fs=40_000,
    duration_sec=10,
    mil_per_volt=10.0,
    axis_lim_mil=3.0,
    img_size=256,
):
    """
    [분석용] BIN 파일 하나를 처리하여 RCP별 'sec9 (마지막 1초)' Orbit 이미지를 생성합니다.
    반환: { "RCP1A": PIL.Image, ... }
    """
    # 1. 파싱
    data = parse_bin_legacy(bin_path, fs=fs, duration_sec=duration_sec)

    # 2. 채널 추출
    xy_pairs = extract_xy_pairs_legacy(data)
    rcp_names = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
    samples_per_sec = fs

    rcp_to_pil = {}

    for i, (x, y) in enumerate(xy_pairs):
        if i >= len(rcp_names):
            break

        rcp = rcp_names[i]
        x_mil, y_mil = volt_to_mil(x, y, mil_per_volt=mil_per_volt)

        # 3. 구간 선택 (9초~10초)
        s = 9 * samples_per_sec
        e = 10 * samples_per_sec
        seg_x = x_mil[s:e]
        seg_y = y_mil[s:e]

        # 4. 이미지 생성
        grid = make_orbit_image(seg_x, seg_y, axis_lim=axis_lim_mil, img_size=img_size)

        pil_img = Image.fromarray(grid)  # mode="L" (Grayscale)
        rcp_to_pil[rcp] = pil_img

    return rcp_to_pil


def make_temporal_orbit_pils(
    bin_path,
    fs=40_000,
    duration_sec=10,
    mil_per_volt=10.0,
    axis_lim_mil=3.0,
    img_size=256,
):
    """
    [타임라인용] BIN 파일에서 시간대별(0~9초) Orbit 이미지를 모두 생성합니다.
    반환: { "RCP1A": [PIL_0s, PIL_1s, ...], ... }
    """
    data = parse_bin_legacy(bin_path, fs=fs, duration_sec=duration_sec)
    xy_pairs = extract_xy_pairs_legacy(data)

    rcp_names = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
    samples_per_sec = fs

    rcp_to_temporal = {}

    for i, (x, y) in enumerate(xy_pairs):
        if i >= len(rcp_names):
            break

        rcp = rcp_names[i]
        x_mil, y_mil = volt_to_mil(x, y, mil_per_volt=mil_per_volt)

        temporal_pils = []
        for sec in range(duration_sec):
            # 1초 단위 슬라이싱
            s = sec * samples_per_sec
            e = (sec + 1) * samples_per_sec
            seg_x = x_mil[s:e]
            seg_y = y_mil[s:e]

            grid = make_orbit_image(
                seg_x, seg_y, axis_lim=axis_lim_mil, img_size=img_size
            )
            pil_img = Image.fromarray(grid)
            temporal_pils.append(pil_img)

        rcp_to_temporal[rcp] = temporal_pils

    return rcp_to_temporal
