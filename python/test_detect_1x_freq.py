import numpy as np

from preprocess import detect_1x_freq


FS = 4096
F1X = 20.0


def _signal(a1=1.0, a2=0.0, a3=0.0, seconds=2.0):
    t = np.arange(int(FS * seconds), dtype=np.float64) / FS
    return (
        a1 * np.sin(2 * np.pi * F1X * t)
        + a2 * np.sin(2 * np.pi * 2 * F1X * t)
        + a3 * np.sin(2 * np.pi * 3 * F1X * t)
    )


def test_detect_1x_with_1x_dominant_signal():
    x = _signal(a1=1.0, a2=0.25, a3=0.1)

    detected = detect_1x_freq(x, FS, rpm_min=300, rpm_max=6000)

    assert np.isclose(detected, F1X)


def test_detect_1x_with_2x_dominant_signal():
    x = _signal(a1=0.2, a2=1.0, a3=0.1)

    detected = detect_1x_freq(x, FS, rpm_min=300, rpm_max=6000)

    assert np.isclose(detected, F1X)


def test_detect_1x_with_3x_dominant_signal():
    x = _signal(a1=0.3, a2=0.0, a3=1.0)

    detected = detect_1x_freq(x, FS, rpm_min=300, rpm_max=6000)

    assert np.isclose(detected, F1X)
