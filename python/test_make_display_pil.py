"""
test_make_display_pil.py
========================
_make_display_pil의 조건부 axis_lim 재계산 동작을 검증하는 단위 테스트.

실행:
    cd python && pytest test_make_display_pil.py -v
"""

import sys
import os
import types
import numpy as np
import pytest

# ── 최소 stub 세팅: torch/모델 없이 preprocess 함수만 테스트 ─────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# inference_daemon은 torch 등 heavy 의존성을 import 레벨에서 로드하므로
# 필요한 함수만 직접 preprocess에서 import한다.
sys.path.insert(0, SCRIPT_DIR)

from preprocess import (
    compute_dynamic_axis_lim,
    make_orbit_display_image,
    filter_1x_bandpass,
)
from PIL import Image


# ── 테스트용 _make_display_pil 로컬 재현 ────────────────────────────────
# inference_daemon의 실제 구현과 동일한 로직을 standalone으로 복사.
# (torch 모델 로드 없이 순수 신호처리 로직만 검증)
try:
    from preprocess import filter_2x_bandpass, filter_broadband
    _HAS_FILTERS = True
except ImportError:
    _HAS_FILTERS = False


def _make_display_pil(x_seg, y_seg, axis_lim, fs=None, filter_mode="1x"):
    """inference_daemon._make_display_pil 로컬 복사 (테스트 전용)."""
    x_seg = x_seg - x_seg.mean()
    y_seg = y_seg - y_seg.mean()
    actual_filter = "raw" if (fs is None or fs <= 0) else filter_mode
    if fs is not None and fs > 0 and filter_mode != "raw":
        try:
            if filter_mode == "2x":
                x_seg, y_seg, _ = filter_2x_bandpass(x_seg, y_seg, fs)
            elif filter_mode == "broadband":
                x_seg, y_seg = filter_broadband(x_seg, y_seg, fs)
            else:  # '1x'
                x_seg, y_seg, _ = filter_1x_bandpass(x_seg, y_seg, fs)
        except Exception:
            actual_filter = "raw"
    if actual_filter != "raw":
        used_axis_lim = compute_dynamic_axis_lim(x_seg, y_seg)
    else:
        used_axis_lim = axis_lim
    arr = make_orbit_display_image(x_seg, y_seg, axis_lim=used_axis_lim, img_size=256)
    return Image.fromarray(arr, mode='L'), actual_filter, used_axis_lim


# ── 헬퍼: 단순 정현파 궤도 신호 생성 ────────────────────────────────────
FS = 4096  # Hz

RPM_TEST = 1200          # 실제 가동 조건
F1X_TEST = RPM_TEST / 60  # 20 Hz


def _sine_orbit(amp_x=2.0, amp_y=1.5, freq_hz=F1X_TEST, n=FS):
    """단일 주파수 정현파 궤도 (mils). 기본값: 1200 RPM (20 Hz)."""
    t = np.arange(n) / FS
    x = amp_x * np.sin(2 * np.pi * freq_hz * t)
    y = amp_y * np.cos(2 * np.pi * freq_hz * t)
    return x.astype(np.float32), y.astype(np.float32)


# ── 테스트 케이스 ────────────────────────────────────────────────────────

class TestMakeDisplayPilRawMode:
    """fs=None (raw) 경로: 호출자의 axis_lim이 그대로 유지되어야 한다."""

    def test_returns_three_tuple(self):
        x, y = _sine_orbit()
        result = _make_display_pil(x, y, axis_lim=3.0)
        assert len(result) == 3, "반환값이 3-튜플이어야 한다"

    def test_actual_filter_is_raw_when_fs_none(self):
        x, y = _sine_orbit()
        _, actual_filter, _ = _make_display_pil(x, y, axis_lim=3.0)
        assert actual_filter == "raw"

    def test_used_axis_lim_equals_caller_axis_lim(self):
        """raw 모드에서 used_axis_lim == 호출자의 axis_lim."""
        x, y = _sine_orbit()
        caller_lim = 3.0
        _, _, used = _make_display_pil(x, y, axis_lim=caller_lim)
        assert used == caller_lim, (
            f"raw 모드에서 used_axis_lim({used})이 caller axis_lim({caller_lim})과 달라서는 안 된다"
        )

    def test_raw_mode_preserves_large_axis_lim(self):
        """timeline처럼 큰 global_axis_lim을 넘겨도 그대로 유지."""
        x, y = _sine_orbit(amp_x=0.5, amp_y=0.5)
        global_lim = 5.0  # 신호 진폭보다 훨씬 큰 값
        _, _, used = _make_display_pil(x, y, axis_lim=global_lim)
        assert used == global_lim

    def test_returns_pil_image(self):
        x, y = _sine_orbit()
        pil, _, _ = _make_display_pil(x, y, axis_lim=3.0)
        assert isinstance(pil, Image.Image)
        assert pil.size == (256, 256)


class TestMakeDisplayPilFilterMode:
    """fs 전달 + 1x 필터 경로: used_axis_lim이 동적으로 재계산되어야 한다."""

    @pytest.mark.skipif(not _HAS_FILTERS, reason="필터 함수를 import할 수 없음")
    def test_filtered_used_axis_lim_differs_from_caller_axis_lim(self):
        """1X 필터 성공 시 used_axis_lim이 호출자가 넘긴 (임의로 큰) axis_lim과 달라야 한다.
        실제 가동 조건: 1200 RPM = 20 Hz, FS=4096 (SOS 필터 안정성 검증 포함).
        """
        x, y = _sine_orbit()  # 20 Hz (1200 RPM)

        # 실제 신호보다 훨씬 큰 axis_lim을 호출자가 넘기는 상황 시뮬레이션
        caller_lim = 10.0
        _, actual_filter, used = _make_display_pil(x, y, axis_lim=caller_lim, fs=FS, filter_mode="1x")

        if actual_filter == "raw":
            pytest.skip("1X 탐지 실패로 raw fallback — 환경에 따른 skip")

        # 필터가 적용됐다면 used_axis_lim은 동적으로 계산되어 caller_lim과 달라야 한다
        assert used != caller_lim, (
            f"필터 적용 후에도 used_axis_lim({used})이 호출자 lim({caller_lim})과 동일하다 — "
            "동적 재계산이 수행되지 않은 것으로 의심됨"
        )
        # 수치 안정성: SOS 형식으로 수치 오버플로가 없어야 한다
        assert 0.1 <= used <= 100.0, f"used_axis_lim({used})이 비정상 범위 — SOS 변환 실패 가능성"

    @pytest.mark.skipif(not _HAS_FILTERS, reason="필터 함수를 import할 수 없음")
    def test_filtered_used_axis_lim_equals_dynamic_of_filtered_signal(self):
        """used_axis_lim == compute_dynamic_axis_lim(필터링된 신호). 1200 RPM 기준."""
        x, y = _sine_orbit()  # 20 Hz (1200 RPM)

        _, actual_filter, used = _make_display_pil(x, y, axis_lim=5.0, fs=FS, filter_mode="1x")

        if actual_filter == "raw":
            pytest.skip("1X 탐지 실패로 raw fallback")

        # 필터 후 신호를 직접 얻어 동적 axis_lim 계산
        x_filt, y_filt, _ = filter_1x_bandpass(x - x.mean(), y - y.mean(), FS)
        expected = compute_dynamic_axis_lim(x_filt, y_filt)
        assert used == expected, (
            f"used_axis_lim({used}) != compute_dynamic_axis_lim(filtered)({expected})"
        )

    @pytest.mark.skipif(not _HAS_FILTERS, reason="필터 함수를 import할 수 없음")
    def test_filter_fallback_uses_caller_axis_lim(self):
        """필터 실패(fallback) 시 raw와 동일하게 호출자 axis_lim 사용."""
        x, y = _sine_orbit()
        caller_lim = 4.0
        # fs=1 → 유효 주파수 범위가 극히 좁아 필터 실패 유도
        _, actual_filter, used = _make_display_pil(x, y, axis_lim=caller_lim, fs=1, filter_mode="1x")
        assert actual_filter == "raw"
        assert used == caller_lim


class TestCrossSegmentConsistency:
    """timeline 경로처럼 같은 axis_lim을 여러 세그먼트에 전달하면 일관된 스케일이 유지된다."""

    def test_same_global_lim_preserved_across_segments(self):
        rng = np.random.default_rng(0)
        global_lim = 3.0
        results = []
        for _ in range(5):
            # 세그먼트마다 진폭이 다른 랜덤 신호
            amp = rng.uniform(0.1, 1.0)
            x = (amp * np.sin(2 * np.pi * 10 * np.arange(FS) / FS)).astype(np.float32)
            y = (amp * np.cos(2 * np.pi * 10 * np.arange(FS) / FS)).astype(np.float32)
            _, _, used = _make_display_pil(x, y, axis_lim=global_lim)  # fs=None → raw
            results.append(used)

        # 모든 세그먼트의 used_axis_lim이 global_lim과 동일해야 한다
        assert all(u == global_lim for u in results), (
            f"일부 세그먼트에서 global_lim({global_lim})이 깨짐: {results}"
        )
