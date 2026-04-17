#!/usr/bin/env python3
"""
label_orbits.py
===============
지정 폴더 내의 모든 BIN 파일에 대해 궤도 이미지를 생성하고,
궤도 지름이 threshold(기본 3.0 mils) 이상이면 'abnormal',
미만이면 'normal'로 분류합니다.

궤도 지름 정의:
    diameter = max(peak-to-peak of X, peak-to-peak of Y) [mils]
    (회전기계 진단 현장 표준 — 단측 진폭이 아닌 전체 편위)

출력 구조:
    <output_dir>/
        normal/
            <stem>_<pos>_w<nn>.png
        abnormal/
            <stem>_<pos>_w<nn>.png
        labels.csv  — (filename, position, window, diameter_mils, label)

사용법:
    python label_orbits.py <bin_folder> [옵션]

옵션:
    --output DIR        결과 저장 폴더 (기본: <bin_folder>/orbit_labeled)
    --threshold FLOAT   비정상 판정 임계값, mils 단위 (기본: 3.0)
    --window-sec FLOAT  윈도우 길이, 초 단위 (기본: 1.0)
    --img-size INT      이미지 크기 px (기본: 256)
    --percentile FLOAT  peak-to-peak 산정 퍼센타일 (기본: 99.5 — 이상치 억제)
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── 프로젝트 내 공유 모듈 경로 설정 ─────────────────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from preprocess import (
    make_orbit_image_v2,
    parse_bin_legacy,
    extract_xy_pairs_legacy,
    volt_to_mil,
    filter_1x_bandpass,
)
from rcpvms_parser import RcpvmsParser

RCP_NAMES = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]


# ──────────────────────────────────────────────────────────────────────────
# 궤도 지름 계산
# ──────────────────────────────────────────────────────────────────────────

def orbit_diameter_mils(x_mil: np.ndarray, y_mil: np.ndarray, percentile: float = 99.5) -> float:
    """
    궤도 지름을 mils 단위로 반환합니다.

    peak-to-peak 진폭: 상위·하위 percentile 차이로 산정해 충격성 이상치를 억제합니다.
        pp_x = percentile_high(x) - percentile_low(x)
        pp_y = percentile_high(y) - percentile_low(y)
        diameter = max(pp_x, pp_y)
    """
    lo = 100.0 - percentile
    pp_x = float(np.percentile(x_mil, percentile) - np.percentile(x_mil, lo))
    pp_y = float(np.percentile(y_mil, percentile) - np.percentile(y_mil, lo))
    return max(pp_x, pp_y)


# ──────────────────────────────────────────────────────────────────────────
# 단일 BIN 파일 처리
# ──────────────────────────────────────────────────────────────────────────

def process_bin_file(
    bin_path: Path,
    output_dir: Path,
    threshold: float,
    window_sec: float,
    img_size: int,
    percentile: float,
) -> list[dict]:
    """
    BIN 파일 한 개를 처리합니다.

    Returns:
        list of dict  — 각 (위치, 윈도우)에 대한 결과 레코드
    """
    records = []

    # ── 포맷 감지 및 orbit 데이터 획득 ──────────────────────────────────
    fs = 40_000  # 기본값; 신규 파서에서 덮어씀
    try:
        info = RcpvmsParser.read_info(str(bin_path))
        orbit_map = RcpvmsParser.resolve_orbit_channels(info)

        if not orbit_map:
            print(f"  [skip] 변위 채널 없음: {bin_path.name}", file=sys.stderr)
            return records

        fs = info.sampling_rate
        orbit_result = RcpvmsParser.read_orbit_data(info, orbit_map, window_sec=window_sec)
        positions = orbit_result["positions"]
        n_windows = orbit_result["n_windows"]
        data = orbit_result["data"]   # {pos: [{"x": ndarray, "y": ndarray}, ...]}

    except Exception as exc:
        # 신규 파서 실패 → 구형 포맷 시도
        print(f"  [fallback] 신규 파서 실패 ({exc}), legacy 파서 시도: {bin_path.name}",
              file=sys.stderr)
        try:
            raw = parse_bin_legacy(str(bin_path))
            xy_pairs = extract_xy_pairs_legacy(raw)

            fs = 40_000
            window_samples = int(fs * window_sec)
            total_samples = raw.shape[1]
            n_windows = total_samples // window_samples

            # volt → mil 변환 후 윈도우 분할
            data = {}
            positions = []
            for i, (x_v, y_v) in enumerate(xy_pairs):
                if i >= len(RCP_NAMES):
                    break
                pos = RCP_NAMES[i]
                x_mil, y_mil = volt_to_mil(x_v, y_v, mil_per_volt=10.0)
                windows = []
                for wi in range(n_windows):
                    s = wi * window_samples
                    e = s + window_samples
                    x_w = x_mil[s:e].copy()
                    y_w = y_mil[s:e].copy()
                    # DC offset 제거 (RcpvmsParser.read_orbit_data와 동일)
                    x_w -= x_w.mean()
                    y_w -= y_w.mean()
                    windows.append({"x": x_w, "y": y_w})
                data[pos] = windows
                positions.append(pos)

        except Exception as exc2:
            print(f"  [error] legacy 파서도 실패, 건너뜀: {bin_path.name} — {exc2}",
                  file=sys.stderr)
            return records

    stem = bin_path.stem

    # ── 윈도우별 처리 ────────────────────────────────────────────────────
    for pos in positions:
        windows = data[pos]
        for wi, win in enumerate(windows):
            x_mil = win["x"]
            y_mil = win["y"]

            # 1. 1X 밴드패스 필터 적용 (동기 성분 추출)
            try:
                x_mil, y_mil, _ = filter_1x_bandpass(x_mil, y_mil, fs)
            except Exception:
                pass  # 필터 실패 시 원신호 유지

            # 2. 궤도 지름 계산
            diameter = orbit_diameter_mils(x_mil, y_mil, percentile=percentile)

            # NaN/Inf 신호는 건너뜀 (필터 fallback 후에도 발생 가능)
            if not np.isfinite(diameter) or diameter <= 0:
                print(f"  [skip] {bin_path.name} {pos} w{wi}: 지름 비정상 ({diameter})",
                      file=sys.stderr)
                continue

            # 3. 라벨 결정
            label = "abnormal" if diameter >= threshold else "normal"

            # 4. 이미지 생성 (axis_lim은 동적 결정 — 절대 진폭 보존)
            #    make_orbit_image_v2 의 axis_lim은 diameter/2 보다 약간 크게 설정
            axis_lim = max(diameter / 2.0 * 1.2, 0.5)
            img_arr = make_orbit_image_v2(x_mil, y_mil, axis_lim=axis_lim, img_size=img_size)
            img = Image.fromarray(img_arr, mode="L")

            # 4. 저장
            save_dir = output_dir / label
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{stem}_{pos}_w{wi:03d}.png"
            img.save(save_dir / fname)

            records.append({
                "filename": bin_path.name,
                "position": pos,
                "window": wi,
                "diameter_mils": round(diameter, 4),
                "label": label,
            })

    return records


# ──────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BIN 파일 폴더에서 궤도 이미지를 생성하고 normal/abnormal 라벨링",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bin_folder", help="BIN 파일이 있는 폴더 경로")
    parser.add_argument("--output", default=None,
                        help="결과 저장 폴더 (기본: <bin_folder>/orbit_labeled)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="비정상 판정 임계값 (mils)")
    parser.add_argument("--window-sec", type=float, default=1.0,
                        help="분석 윈도우 길이 (초)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="출력 이미지 크기 (px)")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="peak-to-peak 산정 퍼센타일 (이상치 억제)")
    parser.add_argument("--move", action="store_true",
                        help="원본 BIN 파일을 복사 대신 이동 (기본: 복사)")
    parser.add_argument("--file-label", choices=["any", "majority"], default="any",
                        help="파일 단위 라벨 결정 정책 — "
                             "any: 윈도우 하나라도 비정상이면 abnormal (기본), "
                             "majority: 과반수 기준")
    args = parser.parse_args()

    bin_folder = Path(args.bin_folder).resolve()
    if not bin_folder.is_dir():
        print(f"[error] 폴더가 존재하지 않습니다: {bin_folder}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).resolve() if args.output else bin_folder / "orbit_labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(bin_folder.glob("*.bin")) + sorted(bin_folder.glob("*.BIN"))
    if not bin_files:
        print(f"[error] BIN 파일을 찾을 수 없습니다: {bin_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"BIN 파일 {len(bin_files)}개 처리 시작")
    print(f"  임계값: {args.threshold} mils  |  윈도우: {args.window_sec}s  "
          f"|  이미지: {args.img_size}px")
    print(f"  출력 폴더: {output_dir}\n")

    file_op = "이동" if args.move else "복사"
    all_records: list[dict] = []
    for i, bp in enumerate(bin_files, 1):
        print(f"[{i:3d}/{len(bin_files)}] {bp.name}")
        records = process_bin_file(
            bp, output_dir,
            threshold=args.threshold,
            window_sec=args.window_sec,
            img_size=args.img_size,
            percentile=args.percentile,
        )
        all_records.extend(records)
        if records:
            n_abn = sum(1 for r in records if r["label"] == "abnormal")
            diameters = [r["diameter_mils"] for r in records]
            print(f"         {len(records)}개 윈도우, 비정상 {n_abn}개, "
                  f"지름 {min(diameters):.2f}~{max(diameters):.2f} mils")

            # ── 파일 단위 라벨 결정 ───────────────────────────────────────
            if args.file_label == "any":
                file_label = "abnormal" if n_abn > 0 else "normal"
            else:  # majority
                file_label = "abnormal" if n_abn >= len(records) / 2 else "normal"

            dest_dir = output_dir / file_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / bp.name
            try:
                if args.move:
                    shutil.move(str(bp), dest)
                else:
                    shutil.copy2(str(bp), dest)
                print(f"         [{file_label}] BIN {file_op}: {dest.relative_to(output_dir)}")
            except Exception as e:
                print(f"  [warn] BIN {file_op} 실패: {bp.name} — {e}", file=sys.stderr)
        else:
            print(f"         (처리 가능한 윈도우 없음, BIN 파일 {file_op} 생략)")

    # ── CSV 저장 ─────────────────────────────────────────────────────────
    csv_path = output_dir / "labels.csv"
    fieldnames = ["filename", "position", "window", "diameter_mils", "label"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # ── 요약 ─────────────────────────────────────────────────────────────
    n_total = len(all_records)
    n_abn = sum(1 for r in all_records if r["label"] == "abnormal")
    n_norm = n_total - n_abn
    print(f"\n{'='*55}")
    print(f"완료: 총 {n_total}개 윈도우")
    print(f"  normal   : {n_norm}개")
    print(f"  abnormal : {n_abn}개  (지름 >= {args.threshold} mils)")
    print(f"  labels.csv 저장: {csv_path}")


if __name__ == "__main__":
    main()
