import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

# 멀티스케일 고정 채널 (fine / mid / wide)
MULTISCALE_AXIS_LIMS = (1.0, 3.0, 6.0)

# 하이브리드 채널의 Ch2(wide) 고정 스케일
# 합성 고장 데이터 최대 진폭(3.0 mil)과 일치 → Ch2 캔버스 전체를 실제 데이터 범위가 커버.
# - 정상 (0.3 mil) → 캔버스 10% → 경미 고장과 구별 가능
# - 심각 고장 (3.0 mil) → 캔버스 100% → 최대 진폭 표현
# 이 값을 변경하면 반드시 모델을 재학습해야 함.
HYBRID_WIDE_LIM = 3.0  # mil

# 1D 고정 스케일 기본값 (학습 데이터에서 compute_dataset_scale()로 계산 가능)
# HYBRID_WIDE_LIM과 동일한 물리적 참조점 사용.
# 정상 신호: ~0.1–0.5 mil → 이 스케일로 나누면 [0, ~0.17] 범위
# 심각 고장: ~3.0 mil → 이 스케일로 나누면 ~1.0 (포화 직전)
FIXED_1D_SCALE_MIL: float = 3.0  # mil — 체크포인트에 저장하여 학습/추론 일관성 보장

# 스펙트로그램 STFT 파라미터 (40 kHz 샘플링 기준)
# nperseg=1024 → 주파수 분해능 39.1 Hz, 윈도우 길이 25.6 ms
# noverlap=768  → hop 256 샘플 (6.4 ms), 약 75% overlap
# → 1초(40000 샘플) 기준 시간 프레임: ~154개
# → 주파수 빈: 513 (0–20 kHz), 진단 유효 범위 0–10 kHz 만 사용 (256 빈)
SPEC_NPERSEG:  int   = 1024
SPEC_NOVERLAP: int   = 768
SPEC_F_MAX_HZ: float = 10000.0  # 진단 유효 주파수 상한 (Hz) — 공동/베어링 고주파 대역 포함


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
            f.seek(0, 2)
            file_size = f.tell()
            if file_size < block_bytes:
                raise ValueError(
                    f"File size too small. Expected at least {block_bytes} bytes."
                )
            f.seek(-block_bytes, 2)
            signal_bytes = f.read(block_bytes)

        # float32 LE 변환
        data = np.frombuffer(signal_bytes, dtype="<f4")
        data = data.reshape(num_channels, num_samples_total)

        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse BIN file: {e}")


def extract_xyz_triplets_legacy(
    data,
    target_triplets=((4, 5, 6), (10, 11, 12), (16, 17, 18), (22, 23, 24)),
):
    """
    채널 데이터에서 RCP별 X, Y, Z(축방향) 트리플렛을 추출합니다.
    각 RCP 블록 레이아웃 가정:
      offset+0: X 수평 진동
      offset+1: Y 수직 진동
      offset+2: Z 축방향 진동
    Z 채널이 없거나(채널 수 초과) 모두 0이면 None을 반환합니다.

    반환: [(x, y, z_or_None), ...] 길이 4 리스트
    """
    triplets = []
    for ch_x, ch_y, ch_z in target_triplets:
        if ch_x >= data.shape[0] or ch_y >= data.shape[0]:
            continue
        x = data[ch_x].copy()
        y = data[ch_y].copy()
        if ch_z < data.shape[0]:
            z_raw = data[ch_z].copy()
            # 유효 신호 여부 확인 (분산이 거의 0이면 미연결 채널로 간주)
            z = z_raw if z_raw.std() > 1e-6 else None
        else:
            z = None
        triplets.append((x, y, z))
    return triplets


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


def detect_1x_freq(x_mil: np.ndarray, fs: int, rpm_min: float = 300, rpm_max: float = 24000) -> float:
    """
    FFT 피크 탐지로 1X(기본 회전 주파수)를 자동 검출합니다.

    Args:
        x_mil : X 변위 신호 (mils) — X 또는 Y 어느 쪽이든 사용 가능
        fs    : 샘플링 주파수 (Hz)
        rpm_min, rpm_max : 탐지 대상 RPM 범위

    Returns:
        f1x (float) : 탐지된 1X 주파수 (Hz)
    """
    f_min = rpm_min / 60.0
    f_max = rpm_max / 60.0
    freqs = np.fft.rfftfreq(len(x_mil), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x_mil))
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return (f_min + f_max) / 2.0
    peak_rel = int(np.argmax(spectrum[mask]))
    return float(freqs[mask][peak_rel])


def filter_1x_bandpass(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    fs: int,
    rpm_min: float = 300,
    rpm_max: float = 24000,
    bw_ratio: float = 0.15,
) -> tuple:
    """
    1X(기본 회전 주파수) 동기 성분만 추출하는 밴드패스 필터.

    탐지 절차:
      1. X 신호 FFT에서 rpm_min~rpm_max 범위의 최대 피크 → f1x
      2. 대역폭 bw = max(f1x × bw_ratio, 1.0) Hz
      3. Butterworth 4차 band-pass [f1x-bw, f1x+bw] 적용

    Args:
        x_mil, y_mil : 변위 신호 (mils)
        fs           : 샘플링 주파수 (Hz)
        rpm_min, rpm_max : 1X 탐지 RPM 범위
        bw_ratio     : 필터 반폭 = f1x × bw_ratio  (기본 ±15%)

    Returns:
        (x_filt, y_filt, f1x)
          x_filt, y_filt : 1X 성분 신호 (mils)
          f1x            : 탐지된 1X 주파수 (Hz)
    """
    from scipy.signal import butter, filtfilt

    f1x = detect_1x_freq(x_mil, fs, rpm_min=rpm_min, rpm_max=rpm_max)
    bw  = max(f1x * bw_ratio, 1.0)
    low  = max(f1x - bw, 0.5)
    high = min(f1x + bw, fs / 2.0 - 0.5)

    if low >= high:
        raise ValueError(
            f"1X 필터 대역 무효: low={low:.2f} >= high={high:.2f}Hz "
            f"(f1x={f1x:.2f}Hz, fs={fs}Hz) — 샘플레이트가 너무 낮습니다"
        )

    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    x_filt = filtfilt(b, a, x_mil)
    y_filt = filtfilt(b, a, y_mil)

    if not (np.isfinite(x_filt).all() and np.isfinite(y_filt).all()):
        raise ValueError(f"1X 필터 출력에 NaN/Inf 포함 (f1x={f1x:.2f}Hz)")

    return x_filt, y_filt, f1x


def filter_2x_bandpass(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    fs: int,
    rpm_min: float = 300,
    rpm_max: float = 24000,
    bw_ratio: float = 0.15,
) -> tuple:
    """
    2X(2배 회전 주파수) 성분만 추출하는 밴드패스 필터.

    1X를 탐지한 후 2×f1x 대역에 Butterworth 4차 band-pass를 적용한다.

    Returns:
        (x_filt, y_filt, f2x)
    """
    from scipy.signal import butter, filtfilt

    f1x = detect_1x_freq(x_mil, fs, rpm_min=rpm_min, rpm_max=rpm_max)
    f2x = f1x * 2.0
    bw  = max(f2x * bw_ratio, 1.0)
    low  = max(f2x - bw, 0.5)
    high = min(f2x + bw, fs / 2.0 - 0.5)

    if low >= high:
        raise ValueError(
            f"2X 필터 대역 무효: low={low:.2f} >= high={high:.2f}Hz "
            f"(f2x={f2x:.2f}Hz, fs={fs}Hz) — 샘플레이트가 너무 낮습니다"
        )

    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    x_filt = filtfilt(b, a, x_mil)
    y_filt = filtfilt(b, a, y_mil)

    if not (np.isfinite(x_filt).all() and np.isfinite(y_filt).all()):
        raise ValueError(f"2X 필터 출력에 NaN/Inf 포함 (f2x={f2x:.2f}Hz)")

    return x_filt, y_filt, f2x


def filter_broadband(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    fs: int,
    hp_hz: float = 0.5,
) -> tuple:
    """
    브로드밴드 필터 — DC 드리프트만 제거하고 전체 주파수 성분을 유지.

    hp_hz 이상의 고역통과(Butterworth 2차) 필터를 적용한다.

    Returns:
        (x_filt, y_filt)
    """
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    b, a = butter(2, hp_hz / nyq, btype="high")
    x_filt = filtfilt(b, a, x_mil)
    y_filt = filtfilt(b, a, y_mil)

    if not (np.isfinite(x_filt).all() and np.isfinite(y_filt).all()):
        raise ValueError("브로드밴드 필터 출력에 NaN/Inf 포함")

    return x_filt, y_filt


def make_orbit_display_image(x_mil, y_mil, axis_lim=3.0, img_size=256):
    """
    Display 전용 line-trace orbit 이미지.
    ADC 양자화로 인해 히스토그램 bins가 희소한 경우에도
    연속선으로 실제 궤도 경로를 표현함.
    모델 추론에는 사용하지 말 것 — 추론에는 make_orbit_image_v2 사용.
    반환: (img_size, img_size) uint8
    """
    from PIL import ImageDraw as _IDraw, Image as _PILImg
    scale = (img_size - 1) / (2.0 * axis_lim)
    center = (img_size - 1) / 2.0

    def _to_px(xm, ym):
        px = int(round(center + xm * scale))
        # center + ym*scale: 양의 ym → py 증가 → 이미지 하단
        # → make_orbit_image_v2 와 동일한 y-flipped 규약
        # → render_with_axes 의 FLIP_TOP_BOTTOM 이 올바르게 복원함
        py = int(round(center + ym * scale))
        return (
            max(0, min(img_size - 1, px)),
            max(0, min(img_size - 1, py)),
        )

    canvas = _PILImg.new("L", (img_size, img_size), 0)
    draw = _IDraw.Draw(canvas)

    # 연속선 렌더링: 각 인접 시간 샘플을 선분으로 연결
    n = len(x_mil)
    step = max(1, n // 10000)  # 최대 10000 선분으로 제한
    pts = [_to_px(x_mil[i], y_mil[i]) for i in range(0, n, step)]
    if len(pts) >= 2:
        draw.line(pts, fill=220, width=1)

    arr = np.array(canvas, dtype=np.float32)
    arr = gaussian_filter(arr, sigma=1.5)
    arr = arr / (arr.max() + 1e-8)
    return (arr * 255).astype(np.uint8)


def make_multiscale_orbit(x_mil, y_mil, img_size=256, dynamic=True, hybrid=False):
    """
    3채널 멀티스케일 orbit 이미지 생성.
    채널 0 (R): fine  — 중심 밀도 확대
    채널 1 (G): mid   — 중간 시야
    채널 2 (B): wide  — 전체 orbit 형태

    hybrid=True (권장):
        Ch0 fine : dynamic (신호 진폭 기준 1/6 zoom — 형상 디테일)
        Ch1 mid  : dynamic (신호 진폭 기준 1/2 zoom — 형상 전체)
        Ch2 wide : 고정 HYBRID_WIDE_LIM (3.0 mil) — 절대 진폭 인코딩
        → 형상(Ch0/1) + 절대 진폭(Ch2)을 동시에 학습.
        → 경미/심각 고장을 진폭으로 구분 가능.
        ※ HYBRID_WIDE_LIM 변경 시 반드시 재학습 필요.

    dynamic=True (레거시 기본값):
        세 채널 모두 동적 스케일 (진폭 정보 소거됨).

    dynamic=False (레거시):
        고정 MULTISCALE_AXIS_LIMS = (1.0, 3.0, 6.0) 사용.

    반환: (img_size, img_size, 3) uint8 — PIL RGB Image로 바로 변환 가능
    """
    if hybrid or dynamic:
        wide_lim = compute_dynamic_axis_lim(x_mil, y_mil)
        mid_lim  = max(wide_lim / 2.0, 0.3)
        fine_lim = max(wide_lim / 6.0, 0.1)
    else:
        fine_lim, mid_lim, wide_lim = MULTISCALE_AXIS_LIMS

    ch_fine = make_orbit_image_v2(x_mil, y_mil, axis_lim=fine_lim, img_size=img_size)
    ch_mid  = make_orbit_image_v2(x_mil, y_mil, axis_lim=mid_lim,  img_size=img_size)
    # hybrid 모드: wide 채널에 고정 스케일 사용 (절대 진폭 인코딩)
    ch_wide = make_orbit_image_v2(
        x_mil, y_mil,
        axis_lim=HYBRID_WIDE_LIM if hybrid else wide_lim,
        img_size=img_size,
    )

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


def compute_dataset_scale(xy_pairs: list, percentile: float = 99.5) -> float:
    """
    학습 데이터셋 전체에서 고정 스케일을 계산합니다.

    각 샘플의 99.5퍼센타일 진폭을 수집한 뒤, 그 중앙값의 2배를 반환합니다.
    - 중앙값 사용: 극단적 이상치에 강건
    - ×2 마진: 정상 진폭을 [0, ~0.5] 범위에 위치시켜 고장 진폭과 명확히 구분

    Args:
        xy_pairs: [(x_mil, y_mil), ...] 정상 학습 샘플 목록
        percentile: 각 샘플 내 진폭 통계 퍼센타일

    Returns:
        scale_mil (float): 체크포인트에 저장할 고정 스케일 값
    """
    per_sample_maxes = []
    for x_mil, y_mil in xy_pairs:
        combined = np.concatenate([np.abs(x_mil), np.abs(y_mil)])
        per_sample_maxes.append(np.percentile(combined, percentile))
    if not per_sample_maxes:
        return FIXED_1D_SCALE_MIL
    return float(np.median(per_sample_maxes) * 2.0)


def prepare_1d_input_fixed(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    scale_mil: float = FIXED_1D_SCALE_MIL,
) -> np.ndarray:
    """
    고정 스케일 기반 1D CNN 입력 (X, Y 2채널).

    per-sample 정규화 대신 데이터셋 수준 고정 스케일을 사용하여
    절대 진폭 정보를 보존합니다.

    - 정상 (0.1–0.5 mil, scale=3.0): 출력 ≈ [0.03, 0.17]
    - 불평형 심각 (3.0 mil):          출력 ≈ 1.0
    - 오일 훨 (>3.0 mil):             출력 > 1.0 (클리핑 없음, 네트워크가 처리)

    Returns:
        np.ndarray (2, N) float32
    """
    return np.stack(
        [x_mil / scale_mil, y_mil / scale_mil], axis=0
    ).astype(np.float32)


def prepare_3ch_input_fixed(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    z_mil: np.ndarray | None,
    scale_mil: float = FIXED_1D_SCALE_MIL,
) -> np.ndarray:
    """
    고정 스케일 기반 3채널 입력 (X, Y, Z 축방향 포함).

    z_mil이 None(축방향 센서 미연결)이면 0으로 채워진 채널을 추가합니다.

    Returns:
        np.ndarray (3, N) float32
    """
    x_s = x_mil / scale_mil
    y_s = y_mil / scale_mil
    z_s = z_mil / scale_mil if z_mil is not None else np.zeros_like(x_mil)
    return np.stack([x_s, y_s, z_s], axis=0).astype(np.float32)


def make_spectrogram_4ch(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    scale_mil: float = FIXED_1D_SCALE_MIL,
    fs: int = 40_000,
    nperseg: int = SPEC_NPERSEG,
    noverlap: int = SPEC_NOVERLAP,
    f_max_hz: float = SPEC_F_MAX_HZ,
) -> np.ndarray:
    """
    고정 스케일 X/Y 신호 → 4채널 로그 스펙트로그램.

    채널 구성:
      Ch0: log(1 + |STFT_X|²)              — X 전력 스펙트럼
      Ch1: log(1 + |STFT_Y|²)              — Y 전력 스펙트럼
      Ch2: log(1 + |Re(Gxy)|)              — 교차 스펙트럼 실수부 (동위상 성분)
      Ch3: log(1 + |Im(Gxy)|)              — 교차 스펙트럼 허수부 (위상차 = 와류 방향)

    Ch3(허수부)의 부호가 순방향/역방향 와류를 인코딩합니다:
      +Im: X가 Y보다 90° 앞섬 → CCW 순방향 와류 (불평형, 오일 훨)
      -Im: X가 Y보다 90° 뒤짐 → CW 역방향 와류 (러빙)

    고정 스케일 입력을 사용하므로 절대 진폭이 스펙트럼 강도에 보존됩니다.
    (per-sample 정규화 시 진폭 소거 — 이 함수는 그렇게 하지 않음)

    Returns:
        np.ndarray (4, F_bins, T_frames) float32
        F_bins ≤ f_max_hz / (fs / nperseg) + 1
    """
    from scipy.signal import stft as _stft

    x_s = x_mil / scale_mil
    y_s = y_mil / scale_mil

    freqs, _, Zx = _stft(
        x_s, fs=fs, nperseg=nperseg, noverlap=noverlap,
        boundary="zeros", padded=True,
    )
    _, _, Zy = _stft(
        y_s, fs=fs, nperseg=nperseg, noverlap=noverlap,
        boundary="zeros", padded=True,
    )

    # 주파수 범위 자르기 (0 ~ f_max_hz)
    f_res = fs / nperseg                           # Hz per bin
    f_bin_max = int(np.ceil(f_max_hz / f_res)) + 1
    f_bin_max = min(f_bin_max, len(freqs))
    Zx = Zx[:f_bin_max]
    Zy = Zy[:f_bin_max]

    # 각 채널 계산
    Sx  = np.log1p(np.abs(Zx) ** 2)               # X 전력
    Sy  = np.log1p(np.abs(Zy) ** 2)               # Y 전력
    Gxy = Zx.conj() * Zy                           # 교차 스펙트럼 Gxy = X* Y
    Cre = np.log1p(np.abs(np.real(Gxy)))           # 동위상
    Cim = np.log1p(np.abs(np.imag(Gxy)))           # 위상차 (와류 방향 인코딩)

    return np.stack([Sx, Sy, Cre, Cim], axis=0).astype(np.float32)


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
        # ──────────────────────────────────────────────────────────────
        # 물리적으로 타당한 증강만 허용
        #
        # ❌ RandomRotation(360°): 전방위 회전은 궤도 장축 방향(베어링
        #    강성 비대칭 정보)을 소거한다. ±5° 이내만 허용.
        # ❌ RandomHorizontalFlip: X 프로브 부호 반전 → 와류 방향
        #    (순방향/역방향) 정보가 뒤집힌다. 오일 훨/러빙 판별 불가.
        # ✅ 소각도 회전(±5°): 센서 장착 오차 범위 모사
        # ✅ 미소 스케일 지터: 진폭 측정 잡음 모사 (형상 보존)
        # ──────────────────────────────────────────────────────────────
        ops += [
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, scale=(0.92, 1.08)),
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
