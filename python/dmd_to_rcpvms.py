"""
dmd_to_rcpvms.py
================
DMD 바이너리 파일 → RCPVMS BIN 파일 변환기.

10초 고정 윈도우, 전체 채널 포함, 스트리밍 방식 (1회 파일 스캔).
출력 포맷: RCPVMS 확장 BIN (512B 헤더 + 채널 info 블록 + float32 channel-major)
"""

import os
import re
import struct
from datetime import datetime, timedelta

import numpy as np

from dmd_parser import (
    DmdParser,
    DmdFileInfo,
    DMDF_MAGIC,
    DMDH_MAGIC,
    DMDH_OUTER_HDR,
    DMDH_INNER_HDR,
    FIRST_BLOCK_OFF,
    DATA_BLOCK_TYPE,
    DATA_LEVEL_RAW,
    _read_uint32_le,
    _parse_samples,
    _adc_max,
)

# ─────────────────────────────────────────────
# RCPVMS BIN 포맷 상수
# ─────────────────────────────────────────────
RCPVMS_HEADER_SIZE = 512
RCPVMS_CH_INFO_SIZE = 20       # per channel
RCPVMS_SYSTEM_ID = 2
RCPVMS_FILE_VERSION = b"1.00"

# 채널 타입 결정 맵
# RCP1A(X=AI 4/1, Y=AI 4/2), RCP1B(X=AI 4/4, Y=AI 4/5)
# RCP2A(X=AI 5/1, Y=AI 5/2), RCP2B(X=AI 5/4, Y=AI 5/5)
_DISP_CHANNELS = frozenset({
    "AI 4/1", "AI 4/2", "AI 4/4", "AI 4/5",
    "AI 5/1", "AI 5/2", "AI 5/4", "AI 5/5",
})


def _ch_type(name: str) -> int:
    """0=가속(또는 미연결), 1=변위, 2=keyphasor"""
    if name in _DISP_CHANNELS:
        return 1
    return 0


# ─────────────────────────────────────────────
# 날짜 유틸
# ─────────────────────────────────────────────
def _extract_date_from_path(dmd_path: str) -> str:
    """DMD 파일명에서 날짜 추출. SAEUL2_20240823_171733_001.dmd → 2024-08-23 17:17:33.000"""
    name = os.path.basename(dmd_path)
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", name)
    if m:
        return (
            f"{m.group(1)}-{m.group(2)}-{m.group(3)} "
            f"{m.group(4)}:{m.group(5)}:{m.group(6)}.000"
        )
    return "1970-01-01 00:00:00.000"


def _offset_date(base_date: str, offset_sec: int) -> str:
    """base_date + offset_sec 초 → RCPVMS event_date 문자열."""
    try:
        dt = datetime.strptime(base_date[:23], "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            dt = datetime.strptime(base_date[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime(1970, 1, 1)
    dt += timedelta(seconds=offset_sec)
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"


# ─────────────────────────────────────────────
# BIN 파일 쓰기 헬퍼
# ─────────────────────────────────────────────
def _write_header(f, total_ch, sampling_rate, event_duration_ms,
                  event_date, site_id, mils_per_v, g_per_v, data_offset):
    hdr = bytearray(RCPVMS_HEADER_SIZE)
    # site_id (8B)
    sb = site_id.encode("utf-8")[:8]
    hdr[0x00:0x00 + len(sb)] = sb
    # system_id = 2
    struct.pack_into("<H", hdr, 0x08, RCPVMS_SYSTEM_ID)
    # total_ch
    struct.pack_into("<H", hdr, 0x0C, total_ch)
    # event_date (24B)
    db = event_date.encode("ascii", errors="replace")[:24]
    hdr[0x10:0x10 + len(db)] = db
    # file_version
    hdr[0x2C:0x30] = RCPVMS_FILE_VERSION
    # sampling_rate
    struct.pack_into("<I", hdr, 0x30, sampling_rate)
    # event_duration_ms
    struct.pack_into("<I", hdr, 0x38, event_duration_ms)
    # g_per_v, mils_per_v
    struct.pack_into("<f", hdr, 0x40, g_per_v)
    struct.pack_into("<f", hdr, 0x44, mils_per_v)
    # data_offset
    struct.pack_into("<I", hdr, 0x48, data_offset)
    f.write(hdr)


def _write_ch_info(f, channels):
    for i, ch in enumerate(channels):
        blk = bytearray(RCPVMS_CH_INFO_SIZE)
        struct.pack_into("<H", blk, 0, i)
        nb = ch.name.encode("utf-8")[:16]
        blk[2:2 + len(nb)] = nb
        blk[18] = _ch_type(ch.name)
        f.write(blk)


def _flush_window(output_dir, base_name, window_idx, channels, window_data,
                  sampling_rate, window_sec, event_date,
                  site_id, mils_per_v, g_per_v, data_offset, window_samples):
    """버퍼된 윈도우 데이터를 BIN 파일로 기록."""
    out_name = f"{base_name}_{window_idx:04d}.bin"
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, "wb") as wf:
        _write_header(
            wf, len(channels), sampling_rate,
            window_sec * 1000, event_date,
            site_id, mils_per_v, g_per_v, data_offset,
        )
        _write_ch_info(wf, channels)
        for ci in range(len(channels)):
            arr = window_data[ci]
            if len(arr) < window_samples:
                padded = np.zeros(window_samples, dtype=np.float32)
                padded[: len(arr)] = arr
                arr = padded
            wf.write(arr.tobytes())
    return out_path


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────
class DmdToRcpvmsConverter:
    """DMD → RCPVMS BIN 스트리밍 변환기."""

    @staticmethod
    def convert(
        dmd_path: str,
        output_dir: str,
        window_sec: int = 10,
        mils_per_v: float = 10.0,
        g_per_v: float = 1.0,
        site_id: str = "",
        base_name: str = "",
        progress_callback=None,
    ) -> dict:
        """
        DMD 파일을 10초 단위 RCPVMS BIN 파일로 변환.

        Returns:
            {
                "output_dir": str,
                "files": [str, ...],
                "n_windows": int,
                "total_ch": int,
                "sampling_rate": int,
                "samples_per_window": int,
            }
        """
        # 1. DMD 메타데이터
        info = DmdParser.read_info(dmd_path)
        channels = info.channels
        total_ch = len(channels)
        if total_ch == 0:
            raise ValueError("DMD 파일에 채널이 없습니다.")

        # 샘플레이트 (채널 중 최대값 사용)
        sr_list = [ch.sample_rate for ch in channels if ch.sample_rate > 0]
        if not sr_list:
            raise ValueError("DMD 파일의 샘플레이트를 결정할 수 없습니다.")
        sampling_rate = int(max(sr_list))
        window_samples = sampling_rate * window_sec
        if window_samples == 0:
            raise ValueError("window_sec 또는 sampling_rate가 0입니다.")

        # 출력 디렉토리
        os.makedirs(output_dir, exist_ok=True)
        if not base_name:
            base_name = os.path.splitext(os.path.basename(dmd_path))[0]

        data_offset = RCPVMS_HEADER_SIZE + total_ch * RCPVMS_CH_INFO_SIZE
        base_event_date = _extract_date_from_path(dmd_path)

        # 2. 세그먼트 → 채널 매핑
        seg_ch_map: dict = {}  # {seg_id: [(ch_idx, channel_id, DmdChannelInfo)]}
        for ch_idx, ch in enumerate(channels):
            seg_ch_map.setdefault(ch.segment_id, []).append(
                (ch_idx, ch.channel_id, ch)
            )

        # 3. 채널별 버퍼
        buffers: list[list] = [[] for _ in range(total_ch)]
        buf_lens = [0] * total_ch

        window_idx = 0
        output_files: list[str] = []

        # 4. DMDH 블록 체인 스트리밍 스캔
        with open(dmd_path, "rb") as f:
            magic = f.read(4)
            if magic != DMDF_MAGIC:
                raise ValueError(f"DMD 파일이 아닙니다 (magic={magic!r})")

            block_offset = FIRST_BLOCK_OFF

            while True:
                f.seek(block_offset)
                outer = f.read(DMDH_OUTER_HDR)
                if len(outer) < DMDH_OUTER_HDR or outer[:4] != DMDH_MAGIC:
                    break

                size1 = _read_uint32_le(outer, 4)
                size2 = _read_uint32_le(outer, 8)
                flags = _read_uint32_le(outer, 12)

                if size2 == 0:
                    break

                blk_type = (flags >> 24) & 0xFF
                blk_level = (flags >> 16) & 0xFF
                seg_id = flags & 0x0000FFFF

                if (
                    blk_type == DATA_BLOCK_TYPE
                    and blk_level == DATA_LEVEL_RAW
                    and seg_id in seg_ch_map
                ):
                    seg_info = info.segments.get(seg_id)
                    if seg_info is not None:
                        n_ch = seg_info.n_channels
                        bps = seg_info.bits_per_sample
                        Bps = bps // 8
                        frame = n_ch * Bps

                        if frame > 0:
                            payload_size = size1
                            actual_frames = payload_size // frame

                            if actual_frames > 0:
                                payload_off = block_offset + DMDH_OUTER_HDR + DMDH_INNER_HDR
                                f.seek(payload_off)
                                payload = f.read(actual_frames * frame)

                                pa = np.frombuffer(
                                    payload[: actual_frames * frame],
                                    dtype=np.uint8,
                                ).reshape(actual_frames, n_ch * Bps)

                                adc_mx = _adc_max(bps)

                                for ch_idx, ch_id, ch_info in seg_ch_map[seg_id]:
                                    if ch_id >= n_ch:
                                        continue
                                    ch_slice = pa[:, ch_id * Bps: (ch_id + 1) * Bps]
                                    samples = _parse_samples(bytes(ch_slice.ravel()), bps)
                                    volts = (samples * (ch_info.volt_range / adc_mx)).astype(
                                        np.float32
                                    )
                                    buffers[ch_idx].append(volts)
                                    buf_lens[ch_idx] += len(volts)

                    # 윈도우 충족 시 즉시 기록
                    while all(bl >= window_samples for bl in buf_lens):
                        window_data = []
                        for ci in range(total_ch):
                            arr = np.concatenate(buffers[ci])
                            window_data.append(arr[:window_samples])
                            remainder = arr[window_samples:]
                            buffers[ci] = [remainder] if len(remainder) > 0 else []
                            buf_lens[ci] = len(remainder)

                        evt_date = _offset_date(base_event_date, window_idx * window_sec)
                        out_path = _flush_window(
                            output_dir, base_name, window_idx, channels,
                            window_data, sampling_rate, window_sec,
                            evt_date, site_id, mils_per_v, g_per_v,
                            data_offset, window_samples,
                        )
                        output_files.append(out_path)
                        window_idx += 1

                        if progress_callback:
                            progress_callback(window_idx, -1)

                block_offset += size2

        # 5. 잔여 데이터 (1초 이상이면 zero-pad 포함 저장)
        if all(bl > 0 for bl in buf_lens):
            min_remaining = min(buf_lens)
            if min_remaining >= sampling_rate:  # 최소 1초
                window_data = []
                for ci in range(total_ch):
                    arr = np.concatenate(buffers[ci]) if buffers[ci] else np.zeros(0, dtype=np.float32)
                    padded = np.zeros(window_samples, dtype=np.float32)
                    take = min(len(arr), window_samples)
                    padded[:take] = arr[:take]
                    window_data.append(padded)

                evt_date = _offset_date(base_event_date, window_idx * window_sec)
                out_path = _flush_window(
                    output_dir, base_name, window_idx, channels,
                    window_data, sampling_rate, window_sec,
                    evt_date, site_id, mils_per_v, g_per_v,
                    data_offset, window_samples,
                )
                output_files.append(out_path)
                window_idx += 1

        return {
            "output_dir": output_dir,
            "files": [os.path.basename(f) for f in output_files],
            "n_windows": window_idx,
            "total_ch": total_ch,
            "sampling_rate": sampling_rate,
            "samples_per_window": window_samples,
        }
