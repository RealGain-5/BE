"""
rcpvms_parser.py
================
RCPVMS BIN 파일 파서.

지원 포맷:
  - 신규 포맷: file_version == "1.00" (DMD 변환 출력), 채널 info 블록 + 채널명 기반 매핑
  - 구형 포맷: file_version == "\x00\x00\x00\x00" (HANUL 계열 24채널), 인덱스 기반 매핑
"""

import sys
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List

POSITION_ORDER = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]
SUPPORTED_VERSIONS = {b"1.00", b"\x00\x00\x00\x00"}


@dataclass
class RcpvmsChannel:
    index: int
    ch_no: int
    ch_name: str
    ch_type: int  # 0=가속, 1=변위, 2=keyphasor


@dataclass
class RcpvmsFileInfo:
    filepath: str
    site_id: str
    total_ch: int
    sampling_rate: int
    event_duration_ms: int
    event_date: str
    g_per_v: float
    mils_per_v: float
    data_offset: int
    samples_per_ch: int
    is_legacy: bool  # True → 구형 포맷 (인덱스 기반 매핑)
    channels: List[RcpvmsChannel] = field(default_factory=list)


class RcpvmsParser:

    @staticmethod
    def read_info(filepath: str) -> RcpvmsFileInfo:
        """헤더 + 채널 info 블록 파싱 (raw data 미읽음).

        신규 포맷(file_version='1.00')과 구형 포맷(file_version='\x00\x00\x00\x00')
        모두 지원합니다.
        """
        # 헤더(512B) + 채널 블록을 단일 open으로 읽음.
        # total_ch를 헤더에서 읽은 직후 채널 블록도 같은 핸들로 이어서 읽는다.
        with open(filepath, "rb") as f:
            buf = f.read(512)
            if len(buf) < 512:
                raise ValueError("파일이 512B 미만입니다.")

            file_version = buf[0x2C:0x30]
            if file_version not in SUPPORTED_VERSIONS:
                raise ValueError(
                    f"지원하지 않는 파일 포맷 (file_version={file_version!r}). "
                    f"지원 버전: {SUPPORTED_VERSIONS}"
                )

            is_legacy = (file_version == b"\x00\x00\x00\x00")

            site_id = buf[0x00:0x08].rstrip(b"\x00").decode("utf-8", errors="replace")
            total_ch = struct.unpack_from("<H", buf, 0x0C)[0]
            event_date = buf[0x10:0x28].rstrip(b"\x00").decode("ascii", errors="replace")
            sampling_rate = struct.unpack_from("<I", buf, 0x30)[0]
            event_duration_ms = struct.unpack_from("<I", buf, 0x38)[0]
            g_per_v = struct.unpack_from("<f", buf, 0x40)[0]
            mils_per_v = struct.unpack_from("<f", buf, 0x44)[0]
            data_offset_raw = struct.unpack_from("<I", buf, 0x48)[0]
            # 구형 포맷(legacy)은 채널 info 블록이 없으므로 data_offset = 512 고정.
            # 헤더 기록값이 잘못된 경우가 있어 무시한다.
            if is_legacy:
                data_offset = 512
            else:
                data_offset = data_offset_raw if data_offset_raw >= 512 else 512

            if sampling_rate == 0:
                raise ValueError("sampling_rate가 0입니다.")
            samples_per_ch = sampling_rate * event_duration_ms // 1000

            # 채널 info 블록 파싱 — 신규 포맷만 존재
            channels: List[RcpvmsChannel] = []
            ch_block_size = total_ch * 20
            if (not is_legacy
                    and data_offset > 512
                    and data_offset >= 512 + ch_block_size
                    and total_ch > 0):
                # 같은 핸들로 채널 블록을 이어서 읽음 (현재 파일 포지션 = 512)
                ch_buf = f.read(ch_block_size)
                for i in range(total_ch):
                    o = i * 20
                    ch_no = struct.unpack_from("<H", ch_buf, o)[0]
                    ch_name = ch_buf[o + 2: o + 18].rstrip(b"\x00").decode("utf-8", errors="replace")
                    ch_type = ch_buf[o + 18]
                    channels.append(
                        RcpvmsChannel(index=i, ch_no=ch_no, ch_name=ch_name, ch_type=ch_type)
                    )
            else:
                # 구형 포맷 또는 채널 블록 없음 → 인덱스 기반 이름 부여
                for i in range(total_ch):
                    channels.append(
                        RcpvmsChannel(index=i, ch_no=i, ch_name=f"CH{i}", ch_type=0)
                    )

        return RcpvmsFileInfo(
            filepath=filepath,
            site_id=site_id,
            total_ch=total_ch,
            sampling_rate=sampling_rate,
            event_duration_ms=event_duration_ms,
            event_date=event_date,
            g_per_v=g_per_v,
            mils_per_v=mils_per_v,
            data_offset=data_offset,
            samples_per_ch=samples_per_ch,
            is_legacy=is_legacy,
            channels=channels,
        )

    @staticmethod
    def resolve_orbit_channels(info: RcpvmsFileInfo) -> dict:
        """파일 헤더·채널 정보를 직접 파싱해 orbit X/Y 채널 매핑을 동적으로 결정.

        신규 포맷 (file_version=1.00):
            ch_type==1(변위) 채널을 ch_no 기준 정렬 후 2개씩 (X, Y) 쌍으로 묶어
            POSITION_ORDER 순서대로 할당. 최대 4쌍(8채널)까지 사용.

        구형 포맷 (legacy, HANUL 계열 24채널):
            타입 정보 없음 → 6채널 블록 구조(블록 내 인덱스 4·5가 변위)로 fallback.
            블록 시작 오프셋: 0, 6, 12, 18 → RCP1A, RCP1B, RCP2A, RCP2B.
        """
        if info.is_legacy:
            # HANUL 계열 24채널 표준 레이아웃 fallback
            # 블록 구조: [acc_x, acc_y, acc_z, ?, disp_x, disp_y] × 4 RCP
            BLOCK_STARTS = [0, 6, 12, 18]
            result = {}
            for pos, blk_start in zip(POSITION_ORDER, BLOCK_STARTS):
                x_idx = blk_start + 4
                y_idx = blk_start + 5
                if x_idx < info.total_ch and y_idx < info.total_ch:
                    result[pos] = {
                        "x": x_idx,
                        "y": y_idx,
                        "x_name": f"CH{x_idx}",
                        "y_name": f"CH{y_idx}",
                    }
            return result

        # 신규 포맷: ch_type==1(변위) 채널 동적 탐지
        disp_channels = sorted(
            [ch for ch in info.channels if ch.ch_type == 1],
            key=lambda ch: (ch.ch_no, ch.index),
        )

        # 변위 채널이 없으면 빈 매핑 반환
        if not disp_channels:
            return {}

        # 2개씩 (X, Y) 쌍으로 묶어 최대 4쌍 사용 (홀수 채널은 마지막 1개 무시)
        pairs = [
            (disp_channels[i], disp_channels[i + 1])
            for i in range(0, len(disp_channels) - 1, 2)
        ][:4]

        result = {}
        for pos, (x_ch, y_ch) in zip(POSITION_ORDER, pairs):
            result[pos] = {
                "x": x_ch.index,
                "y": y_ch.index,
                "x_name": x_ch.ch_name,
                "y_name": y_ch.ch_name,
            }
        return result

    @staticmethod
    def read_orbit_data(
        info: RcpvmsFileInfo,
        orbit_map: dict,
        window_sec: float = 1.0,
    ) -> dict:
        """
        BIN 파일에서 궤도 데이터를 읽어 window_sec 단위 윈도우로 분할.

        Returns:
            {
                "positions": [str],
                "n_windows": int,
                "window_sec": float,
                "mils_per_v": float,
                "data": {
                    "RCP1A": [{"x": ndarray, "y": ndarray}, ...],
                    ...
                }
            }
        """
        positions = [p for p in POSITION_ORDER if p in orbit_map]
        if not positions:
            return {
                "positions": [],
                "n_windows": 0,
                "window_sec": window_sec,
                "mils_per_v": info.mils_per_v,
                "data": {},
            }

        if window_sec <= 0:
            raise ValueError(f"window_sec은 0보다 커야 합니다 (받은 값: {window_sec}).")
        window_samples = int(info.sampling_rate * window_sec)
        n_windows = info.samples_per_ch // window_samples
        if n_windows == 0:
            raise ValueError(
                f"데이터가 너무 짧습니다 "
                f"(samples_per_ch={info.samples_per_ch}, window_samples={window_samples})."
            )

        ch_bytes = info.samples_per_ch * 4  # float32 per sample

        # 필요한 채널 인덱스를 수집해 파일을 1회만 열고 모두 읽음
        needed_indices = list({
            idx
            for pos in positions
            for key in ("x", "y")
            for idx in [orbit_map[pos][key]]
        })

        def _parse_raw(raw: bytes, expected: int) -> np.ndarray:
            """raw bytes → float64 ndarray (zero-pad / truncate warning + NaN 제거)."""
            usable = len(raw) - (len(raw) % 4)
            arr = np.frombuffer(raw[:usable], dtype=np.float32).astype(np.float64)
            if len(arr) < expected:
                print(
                    f"[rcpvms_parser] warning: channel data shorter than expected "
                    f"({len(arr)} < {expected} samples); zero-padding applied",
                    file=sys.stderr,
                )
                padded = np.zeros(expected, dtype=np.float64)
                padded[:len(arr)] = arr
                return padded
            if len(arr) > expected:
                print(
                    f"[rcpvms_parser] warning: channel data longer than expected "
                    f"({len(arr)} > {expected} samples); truncated",
                    file=sys.stderr,
                )
            return arr

        ch_cache: dict = {}
        with open(info.filepath, "rb") as f:
            for ch_idx in needed_indices:
                f.seek(info.data_offset + ch_idx * ch_bytes)
                raw = f.read(ch_bytes)
                ch_cache[ch_idx] = np.nan_to_num(_parse_raw(raw, info.samples_per_ch) * info.mils_per_v)

        data = {}
        for pos in positions:
            x_idx = orbit_map[pos]["x"]
            y_idx = orbit_map[pos]["y"]
            x_full = ch_cache[x_idx]
            y_full = ch_cache[y_idx]

            windows = []
            for wi in range(n_windows):
                s = wi * window_samples
                e = s + window_samples
                x_win = x_full[s:e].copy()
                y_win = y_full[s:e].copy()
                # DC offset 제거
                x_win -= x_win.mean()
                y_win -= y_win.mean()
                windows.append({"x": x_win, "y": y_win})
            data[pos] = windows

        return {
            "positions": positions,
            "n_windows": n_windows,
            "window_sec": window_sec,
            "mils_per_v": info.mils_per_v,
            "data": data,
        }

    @staticmethod
    def read_orbit_window(
        info: RcpvmsFileInfo,
        orbit_map: dict,
        pos: str,
        wi: int,
        window_sec: float = 1.0,
    ) -> dict:
        """Read a single orbit window by seeking only the required X/Y channels."""
        if pos not in orbit_map:
            raise ValueError(f"position '{pos}' not in orbit_map")
        if window_sec <= 0:
            raise ValueError(f"window_sec must be positive (got {window_sec})")

        window_samples = int(info.sampling_rate * window_sec)
        if window_samples <= 0:
            raise ValueError(f"window_samples must be positive (got {window_samples})")
        n_windows = info.samples_per_ch // window_samples
        if wi < 0 or wi >= n_windows:
            raise IndexError(f"window index {wi} out of range (0~{n_windows - 1})")

        ch_bytes = info.samples_per_ch * 4
        s = wi * window_samples
        byte_count = window_samples * 4

        def _read_channel_window(f, ch_idx: int) -> np.ndarray:
            f.seek(info.data_offset + ch_idx * ch_bytes + s * 4)
            raw = f.read(byte_count)
            usable = len(raw) - (len(raw) % 4)
            arr = np.frombuffer(raw[:usable], dtype=np.float32).astype(np.float64)
            if len(arr) < window_samples:
                padded = np.zeros(window_samples, dtype=np.float64)
                padded[:len(arr)] = arr
                arr = padded
            elif len(arr) > window_samples:
                arr = arr[:window_samples]
            arr = np.nan_to_num(arr * info.mils_per_v)
            arr -= arr.mean()
            return arr

        x_idx = orbit_map[pos]["x"]
        y_idx = orbit_map[pos]["y"]
        with open(info.filepath, "rb") as f:
            x_win = _read_channel_window(f, x_idx)
            y_win = _read_channel_window(f, y_idx)

        return {"x": x_win, "y": y_win}
