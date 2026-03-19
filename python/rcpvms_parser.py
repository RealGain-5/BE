"""
rcpvms_parser.py
================
RCPVMS BIN 파일 파서.

지원 포맷:
  - 신규 포맷: file_version == "1.00" (DMD 변환 출력), 채널 info 블록 + 채널명 기반 매핑
  - 구형 포맷: file_version == "\x00\x00\x00\x00" (HANUL 계열 24채널), 인덱스 기반 매핑
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ─── 신규 포맷 채널 매핑 (DMD 변환 결과, 채널명 기반) ───────────────────────────
# [실측] AI 4/x = X방향 프로브, AI 5/x = Y방향 프로브
ORBIT_CHANNEL_MAP = {
    "RCPA1": {"x": "AI 4/1", "y": "AI 5/1"},
    "RCPA2": {"x": "AI 4/3", "y": "AI 5/3"},
    "RCPB1": {"x": "AI 4/5", "y": "AI 5/5"},
}

# ─── 구형 포맷 채널 매핑 (HANUL 계열 24채널, 인덱스 기반) ───────────────────────
# preprocess.py extract_xy_pairs_legacy() 와 동일한 배열 가정:
#   RCP별 블록: [acc_x, acc_y, acc_z, ?, disp_x, disp_y, ...]
#   채널 4,5 → RCPA1 X/Y, 10,11 → RCPA2, 16,17 → RCPB1, 22,23 → RCPB2
ORBIT_INDEX_MAP = {
    "RCPA1": {"x": 4,  "y": 5},
    "RCPA2": {"x": 10, "y": 11},
    "RCPB1": {"x": 16, "y": 17},
    "RCPB2": {"x": 22, "y": 23},
}

POSITION_ORDER = ["RCPA1", "RCPA2", "RCPB1", "RCPB2"]
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
        data_offset = data_offset_raw if data_offset_raw >= 512 else 512

        if sampling_rate == 0:
            raise ValueError("sampling_rate가 0입니다.")
        samples_per_ch = sampling_rate * event_duration_ms // 1000

        # 채널 info 블록 파싱 (신규 포맷만 존재)
        channels: List[RcpvmsChannel] = []
        ch_block_size = total_ch * 20
        if (not is_legacy
                and data_offset > 512
                and data_offset >= 512 + ch_block_size
                and total_ch > 0):
            with open(filepath, "rb") as f:
                f.seek(512)
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
        """채널 이름 or 인덱스 기반으로 orbit X/Y 채널 매핑 반환.

        신규 포맷: ORBIT_CHANNEL_MAP (채널명 기반)
        구형 포맷: ORBIT_INDEX_MAP (인덱스 기반, total_ch >= 24 필요)
        """
        if info.is_legacy:
            result = {}
            for pos in POSITION_ORDER:
                mapping = ORBIT_INDEX_MAP.get(pos)
                if mapping is None:
                    continue
                x_idx = mapping["x"]
                y_idx = mapping["y"]
                if x_idx < info.total_ch and y_idx < info.total_ch:
                    result[pos] = {
                        "x": x_idx,
                        "y": y_idx,
                        "x_name": f"CH{x_idx}",
                        "y_name": f"CH{y_idx}",
                    }
            return result
        else:
            name_to_idx = {ch.ch_name: ch.index for ch in info.channels}
            result = {}
            for pos in POSITION_ORDER:
                mapping = ORBIT_CHANNEL_MAP.get(pos)
                if mapping is None:
                    continue
                x_idx = name_to_idx.get(mapping["x"])
                y_idx = name_to_idx.get(mapping["y"])
                if x_idx is not None and y_idx is not None:
                    result[pos] = {
                        "x": x_idx,
                        "y": y_idx,
                        "x_name": mapping["x"],
                        "y_name": mapping["y"],
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
                    "RCPA1": [{"x": ndarray, "y": ndarray}, ...],
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

        window_samples = int(info.sampling_rate * window_sec)
        n_windows = info.samples_per_ch // window_samples if window_samples > 0 else 0
        if n_windows == 0:
            raise ValueError("데이터가 너무 짧습니다.")

        mm = np.memmap(
            info.filepath,
            dtype="float32",
            mode="r",
            offset=info.data_offset,
            shape=(info.total_ch, info.samples_per_ch),
        )

        data = {}
        for pos in positions:
            x_idx = orbit_map[pos]["x"]
            y_idx = orbit_map[pos]["y"]
            x_full = np.array(mm[x_idx], dtype=np.float64) * info.mils_per_v
            y_full = np.array(mm[y_idx], dtype=np.float64) * info.mils_per_v

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
