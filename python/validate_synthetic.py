"""
validate_synthetic.py
======================
합성 데이터 타당성(Validity) 검증 스크립트.

검증 항목:
  1. 1X 회전 주파수 자동 검출 (정상 신호 FFT 기반)
  2. FFT 주파수 분석: 클래스별 특징 주파수 진폭 비교
     - normal      : 특징 주파수 저진폭
     - unbalance   : 1X 지배적
     - misalignment: 2X 우세 (1X 혼재)
     - oil_whip    : 0.45X 서브동기 성분 지배
  3. 진폭 통계 비교: RMS / Kurtosis / Crest Factor
  4. 타당성 자동 판정: 이론 기준 대비 실제 지배 주파수 확인
  5. 시각화 저장 (out_dir):
     - orbit_grid.png    : 클래스별 멀티스케일 orbit 이미지 그리드
     - fft_spectrum.png  : 클래스별 평균 FFT 스펙트럼
     - pca_plot.png      : 특징 공간 PCA 2D 분포
  6. 종합 보고서 출력 + validation_report.json 저장

실행 예시:
  python validate_synthetic.py --data_dir ../data
  python validate_synthetic.py --data_dir ../data --n_files 5 --no_plot
"""

import os
import sys
import glob
import json
import argparse
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocess import (
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    make_multiscale_orbit,
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

FS = 40_000

# 검증 대상 클래스 (subdir, 파일 패턴, 표시명)
# RPM별로 분리하여 각 그룹의 주파수 특성을 독립 검증한다.
SOURCES = {
    "normal":            ("raw/normal",                      "*.BIN", "정상"),
    "unbalance_3600":    ("synthetic/3600rpm/unbalance",     "*.bin", "불균형(3600RPM)"),
    "unbalance_1200":    ("synthetic/1200rpm/unbalance",     "*.bin", "불균형(1200RPM)"),
    "misalignment_3600": ("synthetic/3600rpm/misalignment",  "*.bin", "정렬불량(3600RPM)"),
    "misalignment_1200": ("synthetic/1200rpm/misalignment",  "*.bin", "정렬불량(1200RPM)"),
    "oil_whip_3600":     ("synthetic/3600rpm/oil_whip",      "*.bin", "오일 휩(3600RPM)"),
    "oil_whip_1200":     ("synthetic/1200rpm/oil_whip",      "*.bin", "오일 휩(1200RPM)"),
    "abnormal":          ("raw/abnormal",                    "*.BIN", "비정상(합성)"),
}

# RPM 그룹별 기대 1X 주파수 (Hz)
# 3600 RPM → 60 Hz,  1200 RPM → 20 Hz
RPM_1X = {
    "3600": 60.0,
    "1200": 20.0,
}

# 클래스별 이론적 지배 주파수 배수 (None = 판정 기준 없음)
FAULT_DOMINANT = {
    "normal":            None,
    "unbalance_3600":    1.00,
    "unbalance_1200":    1.00,
    "misalignment_3600": 2.00,
    "misalignment_1200": 2.00,
    "oil_whip_3600":     0.45,
    "oil_whip_1200":     0.45,
    "abnormal":          None,
}


# ─────────────────────────────────────────────
# FFT 유틸리티
# ─────────────────────────────────────────────

def compute_fft(signal: np.ndarray, fs: int = FS):
    """1D 신호의 단측 FFT 주파수·진폭을 반환한다."""
    n     = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mags  = np.abs(np.fft.rfft(signal)) / n * 2
    return freqs, mags


def get_band_amplitude(freqs: np.ndarray, mags: np.ndarray,
                       center_hz: float, bw_hz: float = 2.0) -> float:
    """center_hz ± bw_hz 대역의 최대 진폭을 반환한다."""
    mask = (freqs >= center_hz - bw_hz) & (freqs <= center_hz + bw_hz)
    return float(mags[mask].max()) if mask.any() else 0.0


def detect_1x_freq(signals_xy, fs: int = FS,
                   search_range=(20.0, 70.0),
                   exclude_hz=None, exclude_bw: float = 1.5) -> float:
    """
    정상 신호 FFT 평균에서 1X 회전 주파수를 자동 검출한다.

    Args:
        search_range : 탐색 범위 Hz (기본 20~70 Hz = 1200~4200 RPM).
        exclude_hz   : 탐색에서 제외할 주파수 목록 (예: [60.0, 120.0] 전원 주파수).
        exclude_bw   : 제외 대역 폭 ±Hz.
    """
    sum_mags  = None
    ref_freqs = None
    count     = 0
    for x_mil, y_mil in signals_xy:
        for sig in (x_mil, y_mil):
            freqs, mags = compute_fft(sig, fs)
            if sum_mags is None:
                sum_mags  = mags.copy()
                ref_freqs = freqs
            else:
                sum_mags += mags
            count += 1

    if sum_mags is None:
        return 60.0  # 기본값: 3600 RPM

    avg_mags = sum_mags / count

    # 제외 대역 마스킹 (전원 주파수 간섭 제거)
    if exclude_hz:
        for hz in exclude_hz:
            excl = (ref_freqs >= hz - exclude_bw) & (ref_freqs <= hz + exclude_bw)
            avg_mags = avg_mags.copy()
            avg_mags[excl] = 0.0

    mask    = (ref_freqs >= search_range[0]) & (ref_freqs <= search_range[1])
    peak_hz = float(ref_freqs[mask][np.argmax(avg_mags[mask])])
    return peak_hz


# ─────────────────────────────────────────────
# 통계 유틸리티
# ─────────────────────────────────────────────

def compute_signal_stats(x_mil: np.ndarray, y_mil: np.ndarray) -> dict:
    """X/Y 신호의 진폭 통계 (RMS, Peak, Kurtosis, Crest Factor)를 계산한다."""
    both  = np.concatenate([x_mil, y_mil])
    rms   = float(np.sqrt(np.mean(both ** 2)))
    peak  = float(np.abs(both).max())
    mean  = float(both.mean())
    std   = float(both.std())
    kurt  = float(np.mean(((both - mean) / (std + 1e-10)) ** 4)) if std > 1e-10 else 0.0
    crest = peak / (rms + 1e-10)
    return {"rms": rms, "peak": peak, "kurtosis": kurt, "crest_factor": crest}


# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────

def load_class_data(data_dir: str, class_name: str,
                    n_files: int = 10, fs: int = FS):
    """
    클래스 데이터를 로딩한다.
    반환: List[Tuple[np.ndarray, np.ndarray]] — (x_mil_sec9, y_mil_sec9)
    """
    subdir, pattern, _ = SOURCES[class_name]
    class_path = os.path.join(data_dir, subdir)
    bin_files  = sorted(glob.glob(os.path.join(class_path, pattern)))[:n_files]

    if not bin_files:
        print(f"  [경고] {class_name}: 파일 없음 ({class_path})")
        return []

    signals = []
    for bin_path in bin_files:
        try:
            data     = parse_bin_legacy(bin_path, fs=fs)
            xy_pairs = extract_xy_pairs_legacy(data)
            for x, y in xy_pairs:
                x_mil, y_mil = volt_to_mil(x, y)
                s, e = 9 * fs, 10 * fs
                signals.append((x_mil[s:e].copy(), y_mil[s:e].copy()))
        except Exception as ex:
            print(f"    [오류] {os.path.basename(bin_path)}: {ex}")

    print(f"  {class_name:<15}: {len(bin_files)} 파일 → {len(signals)} 신호 쌍")
    return signals


# ─────────────────────────────────────────────
# 분석
# ─────────────────────────────────────────────

def _class_freq_1x(class_name: str, freq_1x_map: dict, default: float) -> float:
    """클래스명에서 RPM 키를 추출해 해당 1X 주파수를 반환한다."""
    if class_name in freq_1x_map:
        return freq_1x_map[class_name]
    # 클래스명에 RPM 그룹 키("3600"/"1200")가 포함된 경우 자동 매핑
    for rpm_key, hz in RPM_1X.items():
        if rpm_key in class_name:
            return hz
    return default


def run_fft_analysis(signals_by_class: dict, freq_1x: float,
                     freq_1x_map: dict = None) -> dict:
    """
    클래스별 특징 주파수 진폭(mean ± std)을 계산한다.
    freq_1x_map: {class_name: Hz} — RPM 그룹별 1X 오버라이드.
                 미지정 시 전체에 freq_1x 적용.
    """
    freq_1x_map = freq_1x_map or {}
    results = {}
    for class_name, signals in signals_by_class.items():
        f1x = _class_freq_1x(class_name, freq_1x_map, freq_1x)
        bands = {
            "0.45x": f1x * 0.45,
            "1x":    f1x * 1.00,
            "2x":    f1x * 2.00,
            "3x":    f1x * 3.00,
        }
        acc = {k: [] for k in bands}
        for x_mil, y_mil in signals:
            for sig in (x_mil, y_mil):
                freqs, mags = compute_fft(sig)
                for band, center in bands.items():
                    acc[band].append(get_band_amplitude(freqs, mags, center))
        results[class_name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in acc.items()
        }
        results[class_name]["_freq_1x_used"] = f1x
    return results


def run_stats_analysis(signals_by_class: dict) -> dict:
    """클래스별 신호 통계(mean ± std)를 계산한다."""
    results = {}
    for class_name, signals in signals_by_class.items():
        rows = [compute_signal_stats(x, y) for x, y in signals]
        results[class_name] = {
            k: {"mean": float(np.mean([r[k] for r in rows])),
                "std":  float(np.std( [r[k] for r in rows]))}
            for k in ["rms", "peak", "kurtosis", "crest_factor"]
        }
    return results


def check_validity(fft_results: dict) -> dict:
    """
    각 클래스가 이론적 지배 주파수를 실제로 나타내는지 판정한다.
    판정 기준: 해당 클래스의 기대 주파수 대역이 모든 분석 대역 중 최대인가.
    fft_results의 "_freq_1x_used" 키를 참조하여 실제 사용 주파수를 표시한다.
    """
    validity = {}
    for class_name, metrics in fft_results.items():
        dominant_mult = FAULT_DOMINANT.get(class_name)
        f1x_used      = metrics.get("_freq_1x_used", 0.0)

        if dominant_mult is None:
            validity[class_name] = {"pass": None, "reason": "판정 기준 없음 (수동 확인)"}
            continue

        amps = {
            0.45: metrics["0.45x"]["mean"],
            1.00: metrics["1x"]["mean"],
            2.00: metrics["2x"]["mean"],
            3.00: metrics["3x"]["mean"],
        }
        actual_max_mult = max(amps, key=amps.get)
        expected_amp    = amps[dominant_mult]
        actual_max_amp  = amps[actual_max_mult]
        passed          = abs(actual_max_mult - dominant_mult) < 0.1

        ref = f"(1X={f1x_used:.1f}Hz)"
        if passed:
            validity[class_name] = {
                "pass":   True,
                "reason": f"{dominant_mult}X={dominant_mult*f1x_used:.1f}Hz = {expected_amp:.5f}mil ✓ {ref}"
            }
        else:
            validity[class_name] = {
                "pass":   False,
                "reason": (
                    f"예상 {dominant_mult}X={dominant_mult*f1x_used:.1f}Hz = {expected_amp:.5f}mil, "
                    f"실제 최대 {actual_max_mult}X={actual_max_mult*f1x_used:.1f}Hz = {actual_max_amp:.5f}mil {ref}"
                )
            }
    return validity


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────

def save_orbit_grid(signals_by_class: dict, out_dir: str,
                    n_samples: int = 3, img_size: int = 128):
    """클래스별 멀티스케일 Orbit 이미지 그리드를 저장한다."""
    import matplotlib.pyplot as plt

    classes      = list(signals_by_class.keys())
    scale_labels = ["Fine\n(1.0mil)", "Mid\n(3.0mil)", "Wide\n(6.0mil)"]
    n_rows       = len(classes)
    n_cols       = n_samples * 3

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.7),
                             squeeze=False)

    for row, class_name in enumerate(classes):
        signals = signals_by_class[class_name]
        for col in range(n_samples):
            for ch in range(3):
                ax = axes[row][col * 3 + ch]
                if col >= len(signals):
                    ax.axis("off")
                    continue
                x_mil, y_mil = signals[col]
                rgb = make_multiscale_orbit(x_mil, y_mil, img_size=img_size)
                ax.imshow(rgb[:, :, ch], cmap="hot", vmin=0, vmax=255)
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"#{col + 1}\n{scale_labels[ch]}", fontsize=6)
        axes[row][0].set_ylabel(class_name, fontsize=8)

    fig.suptitle("멀티스케일 Orbit 이미지 검증 그리드", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "orbit_grid.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def save_fft_spectrum(signals_by_class: dict, freq_1x: float, out_dir: str):
    """클래스별 평균 FFT 스펙트럼을 저장한다."""
    import matplotlib.pyplot as plt

    classes  = list(signals_by_class.keys())
    COLORS   = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:green"]
    max_hz   = freq_1x * 5.5

    fig, axes = plt.subplots(len(classes), 1,
                             figsize=(10, 2.4 * len(classes)),
                             sharex=True, squeeze=False)

    vline_specs = [
        (0.45, "0.45X", "--"),
        (1.00, "1X",    "-"),
        (2.00, "2X",    "--"),
        (3.00, "3X",    ":"),
    ]

    for row, class_name in enumerate(classes):
        signals = signals_by_class[class_name]
        sum_mags, ref_freqs, count = None, None, 0
        for x_mil, y_mil in signals:
            for sig in (x_mil, y_mil):
                freqs, mags = compute_fft(sig)
                if sum_mags is None:
                    sum_mags, ref_freqs = mags.copy(), freqs
                else:
                    sum_mags += mags
                count += 1

        ax   = axes[row][0]
        mask = ref_freqs <= max_hz
        ax.plot(ref_freqs[mask], (sum_mags / count)[mask],
                color=COLORS[row % len(COLORS)], linewidth=0.8, label=class_name)
        ax.set_ylabel(class_name, fontsize=8)
        ax.grid(True, alpha=0.3)

        # 특징 주파수 수직선 + 레이블
        for mult, label, ls in vline_specs:
            hz = freq_1x * mult
            if hz <= max_hz:
                ax.axvline(hz, color="gray", linestyle=ls, linewidth=0.8, alpha=0.7)
                ax.text(hz, 0.88, label, fontsize=6, ha="center",
                        color="dimgray", transform=ax.get_xaxis_transform())

    axes[-1][0].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"클래스별 평균 FFT 스펙트럼  (1X = {freq_1x:.1f} Hz / {freq_1x * 60:.0f} RPM)",
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fft_spectrum.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def save_pca_plot(signals_by_class: dict, freq_1x: float, out_dir: str) -> float:
    """
    [0.45X, 1X, 2X, 3X 진폭 × 2채널] + [RMS, Kurtosis] 특징 벡터로
    PCA 2D 분포를 시각화한다.
    반환: PC1+PC2 누적 설명 분산
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    classes  = list(signals_by_class.keys())
    COLORS   = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:green"]
    MARKERS  = ["o", "s", "^", "D", "v"]
    features, labels = [], []

    for lbl_idx, class_name in enumerate(classes):
        for x_mil, y_mil in signals_by_class[class_name]:
            row = []
            for sig in (x_mil, y_mil):
                freqs, mags = compute_fft(sig)
                for mult in (0.45, 1.0, 2.0, 3.0):
                    row.append(get_band_amplitude(freqs, mags, freq_1x * mult))
            both = np.concatenate([x_mil, y_mil])
            rms  = float(np.sqrt(np.mean(both ** 2)))
            std  = both.std()
            kurt = float(np.mean(((both - both.mean()) / (std + 1e-10)) ** 4)) if std > 1e-10 else 0.0
            row += [rms, kurt]
            features.append(row)
            labels.append(lbl_idx)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    pca  = PCA(n_components=2)
    X2d  = pca.fit_transform(X)
    ev   = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, class_name in enumerate(classes):
        mask = y == i
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=COLORS[i % len(COLORS)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=class_name, alpha=0.65, s=28, edgecolors="none")

    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)", fontsize=9)
    ax.set_title(
        f"클래스 특징 공간 분포 (PCA)\n"
        f"특징: 0.45X/1X/2X/3X 진폭(×2ch) + RMS + Kurtosis  "
        f"| 누적 설명 분산 {(ev[0]+ev[1])*100:.1f}%",
        fontsize=9
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pca_plot.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return float(ev[0] + ev[1])


# ─────────────────────────────────────────────
# 심층 진단: 클래스별 실제 피크 주파수
# ─────────────────────────────────────────────

def compute_class_avg_fft(signals, fs: int = FS):
    """신호 목록의 평균 FFT (freqs, avg_mags)를 반환한다."""
    sum_mags, ref_freqs, count = None, None, 0
    for x_mil, y_mil in signals:
        for sig in (x_mil, y_mil):
            freqs, mags = compute_fft(sig, fs)
            if sum_mags is None:
                sum_mags, ref_freqs = mags.copy(), freqs
            else:
                sum_mags += mags
            count += 1
    return ref_freqs, (sum_mags / count)


def find_top_peaks(freqs, avg_mags, n_peaks: int = 5,
                   freq_lo: float = 5.0, freq_hi: float = 250.0,
                   min_dist_hz: float = 3.0):
    """
    평균 FFT에서 상위 n_peaks개 피크를 찾는다.
    반환: [(freq_hz, amplitude), ...]
    """
    from scipy.signal import find_peaks as _find_peaks
    mask      = (freqs >= freq_lo) & (freqs <= freq_hi)
    sub_f     = freqs[mask]
    sub_m     = avg_mags[mask]
    df        = float(sub_f[1] - sub_f[0]) if len(sub_f) > 1 else 1.0
    dist_bins = max(1, int(min_dist_hz / df))
    peak_idx, _ = _find_peaks(sub_m, distance=dist_bins)
    if len(peak_idx) == 0:
        return []
    sorted_idx = peak_idx[np.argsort(sub_m[peak_idx])[::-1]][:n_peaks]
    return [(float(sub_f[i]), float(sub_m[i])) for i in sorted_idx]


def run_per_class_diagnosis(signals_by_class: dict, freq_1x: float,
                            n_peaks: int = 5) -> dict:
    """
    클래스별 실제 상위 피크를 찾아 1X 기준 배수와 함께 출력한다.
    반환: {class_name: [(freq_hz, amp, ratio_to_1x), ...]}
    """
    results = {}
    print(f"\n  참조 1X = {freq_1x:.2f} Hz. 배수는 이에 대한 상대 값.\n")
    for class_name, signals in signals_by_class.items():
        freqs, avg_mags = compute_class_avg_fft(signals)
        peaks = find_top_peaks(freqs, avg_mags, n_peaks=n_peaks)
        results[class_name] = peaks
        peak_str = "  ".join(
            f"{hz:.1f}Hz({hz / freq_1x:.2f}X)={amp:.5f}mil"
            for hz, amp in peaks
        )
        print(f"  {class_name:<15}: {peak_str}")
    return results


def save_ratio_spectrum(signals_by_class: dict, freq_1x: float,
                        out_dir: str, max_hz: float = None):
    """
    고장 클래스 FFT / 정상 FFT 비율 스펙트럼을 저장한다.
    비율 >> 1 인 주파수가 실제 고장 특징 주파수이다.
    """
    import matplotlib.pyplot as plt

    if "normal" not in signals_by_class:
        return

    max_hz    = max_hz or (freq_1x * 7)
    fault_cls = [c for c in signals_by_class if c != "normal"]
    COLORS    = ["tab:orange", "tab:red", "tab:purple", "tab:green"]

    norm_freqs, norm_avg = compute_class_avg_fft(signals_by_class["normal"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, class_name in enumerate(fault_cls):
        _, fault_avg = compute_class_avg_fft(signals_by_class[class_name])
        ratio = fault_avg / (norm_avg + 1e-9)
        mask  = (norm_freqs >= 5.0) & (norm_freqs <= max_hz)
        ax.semilogy(norm_freqs[mask], ratio[mask],
                    color=COLORS[i % len(COLORS)],
                    linewidth=0.9, label=class_name)

    # 분석 기준 주파수 마킹
    for mult, label, ls in [(0.45, "0.45X", "--"), (1.0, "1X", "-"),
                             (2.0, "2X", "--"), (3.0, "3X", ":")]:
        hz = freq_1x * mult
        if hz <= max_hz:
            ax.axvline(hz, color="gray", linestyle=ls, linewidth=0.8, alpha=0.6)
            ax.text(hz, 0.92, label, fontsize=6.5, ha="center",
                    color="dimgray", transform=ax.get_xaxis_transform())

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="ratio=1 (정상과 동일)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fault / Normal (log scale)")
    ax.set_title(
        f"고장 클래스 / 정상 FFT 진폭 비율  (1X = {freq_1x:.1f} Hz / {freq_1x * 60:.0f} RPM)\n"
        f"비율 > 1 구간 = 정상 대비 에너지가 높은 주파수 → 실제 고장 주입 위치",
        fontsize=9
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fault_ratio_spectrum.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────
# 보고서 출력
# ─────────────────────────────────────────────

def print_and_save_report(fft_results: dict, stats_results: dict,
                          validity: dict, freq_1x: float,
                          pca_variance: float, out_dir: str,
                          sample_counts: dict):
    """종합 보고서를 터미널에 출력하고 JSON으로 저장한다."""
    SEP = "=" * 68

    print(f"\n{SEP}")
    print("  합성 데이터 타당성 검증 보고서")
    print(SEP)
    print(f"\n  검출된 1X : {freq_1x:.2f} Hz ({freq_1x * 60:.0f} RPM)")
    print(f"  분석 신호 : {sum(sample_counts.values())} 쌍 합계  "
          f"({', '.join(f'{k}:{v}' for k,v in sample_counts.items())})")

    # ── FFT 진폭 테이블
    print(f"\n  {'─'*64}")
    print("  [1] FFT 주파수 진폭 (단위: mil)")
    print(f"  {'클래스':<15} {'0.45X':>10} {'1X':>10} {'2X':>10} {'3X':>10}")
    print(f"  {'─'*55}")
    for cn, m in fft_results.items():
        print(
            f"  {cn:<15}"
            f"  {m['0.45x']['mean']:>8.5f}"
            f"  {m['1x']['mean']:>8.5f}"
            f"  {m['2x']['mean']:>8.5f}"
            f"  {m['3x']['mean']:>8.5f}"
        )

    # ── 통계 테이블
    print(f"\n  {'─'*64}")
    print("  [2] 진폭 통계 (mean ± std)")
    print(f"  {'클래스':<15} {'RMS(mil)':>14} {'Kurtosis':>10} {'Crest Factor':>14}")
    print(f"  {'─'*57}")
    for cn, st in stats_results.items():
        print(
            f"  {cn:<15}"
            f"  {st['rms']['mean']:>6.4f}±{st['rms']['std']:>.4f}"
            f"  {st['kurtosis']['mean']:>9.3f}"
            f"  {st['crest_factor']['mean']:>12.3f}"
        )

    # ── 타당성 판정
    print(f"\n  {'─'*64}")
    print("  [3] 타당성 자동 판정")
    pass_n, total_n = 0, 0
    for cn, v in validity.items():
        if v["pass"] is True:
            sym = "✓ PASS"
            pass_n  += 1
            total_n += 1
        elif v["pass"] is False:
            sym = "✗ FAIL"
            total_n += 1
        else:
            sym = "? N/A "
        print(f"  [{sym}] {cn:<15} {v['reason']}")

    # ── PCA 분리도
    print(f"\n  {'─'*64}")
    print("  [4] PCA 클래스 분리 가능성")
    if pca_variance is not None:
        pct = pca_variance * 100
        grade = "높음 ✓" if pct >= 80 else ("중간 ⚠" if pct >= 60 else "낮음 ✗")
        print(f"  PC1+PC2 누적 설명 분산 = {pct:.1f}%  → 분리 가능성 {grade}")

    # ── 종합 결론
    print(f"\n  {'─'*64}")
    print("  [결론]")
    if total_n > 0 and pass_n == total_n:
        print("  → 모든 합성 클래스가 이론적 주파수 특성을 만족합니다. (타당성 확인)")
    elif total_n > 0 and pass_n == 0:
        print("  → 합성 클래스 전체가 이론 기준을 미충족합니다. 생성 파라미터를 재검토하십시오.")
    else:
        failed = [cn for cn, v in validity.items() if v["pass"] is False]
        print(f"  → 일부 클래스({', '.join(failed)})가 기준 미충족 — 해당 합성 데이터 재검토 필요.")
    print(SEP)

    # ── JSON 저장
    report = {
        "freq_1x_hz": freq_1x,
        "freq_1x_rpm": round(freq_1x * 60),
        "sample_counts": sample_counts,
        "fft_results":   fft_results,
        "stats_results": stats_results,
        "validity":      validity,
        "pca_variance_pc1pc2": pca_variance,
    }
    json_path = os.path.join(out_dir, "validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON 저장: {json_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== 합성 데이터 타당성 검증 시작 ===")
    print(f"  data_dir : {args.data_dir}")
    print(f"  n_files  : {args.n_files} (클래스당 최대)")
    print(f"  out_dir  : {args.out_dir}")

    # [1] 데이터 로딩
    print("\n[1] 데이터 로딩")
    signals_by_class = {}
    for class_name in SOURCES:
        sigs = load_class_data(args.data_dir, class_name, n_files=args.n_files)
        if sigs:
            signals_by_class[class_name] = sigs

    if "normal" not in signals_by_class:
        print("[오류] normal 데이터 없음 — 1X 주파수 검출 불가.")
        return

    sample_counts = {cn: len(sigs) for cn, sigs in signals_by_class.items()}

    # [2] 1X 주파수 검출 (수동 지정 또는 자동)
    print("\n[2] 1X 회전 주파수 결정")
    if args.freq_1x is not None:
        freq_1x = args.freq_1x
        print(f"  → 1X = {freq_1x:.2f} Hz  ({freq_1x * 60:.0f} RPM)  [수동 지정]")
    else:
        exclude = [60.0, 120.0, 180.0] if args.exclude_powerline else None
        note    = " (60/120/180 Hz 전원 주파수 제외)" if args.exclude_powerline else ""
        freq_1x = detect_1x_freq(signals_by_class["normal"], exclude_hz=exclude)
        print(f"  → 1X = {freq_1x:.2f} Hz  ({freq_1x * 60:.0f} RPM)  [자동 검출{note}]")
        if not args.exclude_powerline and abs(freq_1x - 60.0) < 1.0:
            print("  [경고] 검출된 1X가 60 Hz 전원 주파수와 일치합니다.")
            print("         RCP 실제 회전수가 3600 RPM이 아니라면 --exclude_powerline")
            print("         또는 --freq_1x <실제값> 옵션을 사용하십시오.")

    # [3] FFT 분석
    print("\n[3] FFT 주파수 분석")
    fft_results = run_fft_analysis(signals_by_class, freq_1x)

    # [4] 통계 분석
    print("\n[4] 진폭 통계 분석")
    stats_results = run_stats_analysis(signals_by_class)

    # [5] 타당성 판정
    print("\n[5] 타당성 자동 판정")
    validity = check_validity(fft_results)
    for cn, v in validity.items():
        sym = "✓" if v["pass"] is True else ("✗" if v["pass"] is False else "?")
        print(f"  [{sym}] {cn:<15} {v['reason']}")

    # [6] 클래스별 실제 피크 주파수 심층 진단 (항상 실행)
    any_failed = any(v["pass"] is False for v in validity.values())
    print("\n[6] 클래스별 실제 주파수 피크 진단")
    run_per_class_diagnosis(signals_by_class, freq_1x)

    # [7~10] 시각화
    pca_variance = None
    if not args.no_plot:
        print("\n[7] Orbit 이미지 그리드 저장")
        save_orbit_grid(signals_by_class, args.out_dir,
                        n_samples=args.n_orbit_samples)

        print("\n[8] FFT 스펙트럼 저장")
        save_fft_spectrum(signals_by_class, freq_1x, args.out_dir)

        print("\n[9] 고장/정상 FFT 비율 스펙트럼 저장")
        save_ratio_spectrum(signals_by_class, freq_1x, args.out_dir)

        print("\n[10] PCA 분리 가능성 시각화")
        pca_variance = save_pca_plot(signals_by_class, freq_1x, args.out_dir)
    else:
        print("\n[7~10] --no_plot 지정: 시각화 건너뜀")

    # [11] 종합 보고서
    print_and_save_report(
        fft_results, stats_results, validity,
        freq_1x, pca_variance, args.out_dir, sample_counts
    )

    # 검증 실패 시 재실행 안내
    if any_failed and args.freq_1x is None and not args.exclude_powerline:
        print("\n  [재검토 권장]")
        print("  타당성 실패가 검출되었습니다. 아래 순서로 재실행하여 원인을 확인하십시오:")
        print("  1) fault_ratio_spectrum.png 에서 각 클래스의 실제 고장 주파수를 시각 확인")
        print("  2) 전원 주파수(60Hz) 간섭 의심 시:")
        print("     python validate_synthetic.py --data_dir ../data --exclude_powerline")
        print("  3) 실제 회전수를 알고 있다면:")
        print("     python validate_synthetic.py --data_dir ../data --freq_1x <Hz값>")

    print(f"\n  총 소요 시간: {time.time() - t_start:.1f}s")


def _parse_args():
    p = argparse.ArgumentParser(description="합성 데이터 타당성 검증")
    p.add_argument(
        "--data_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data"),
        help="data/raw, data/synthetic 가 위치한 상위 디렉토리"
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(SCRIPT_DIR, "..", "data", "validation"),
        help="결과 이미지 및 JSON 저장 디렉토리"
    )
    p.add_argument(
        "--n_files", type=int, default=10,
        help="클래스당 로딩할 최대 파일 수 (기본: 10)"
    )
    p.add_argument(
        "--n_orbit_samples", type=int, default=3,
        help="Orbit 그리드에 표시할 샘플 수 (기본: 3)"
    )
    p.add_argument(
        "--no_plot", action="store_true",
        help="시각화 생략 (matplotlib 없는 환경)"
    )
    p.add_argument(
        "--freq_1x", type=float, default=None,
        help="1X 회전 주파수 수동 지정 (Hz). 미지정 시 정상 신호 FFT에서 자동 검출"
    )
    p.add_argument(
        "--exclude_powerline", action="store_true",
        help="1X 자동 검출 시 60/120/180 Hz 전원 주파수 대역을 제외"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
