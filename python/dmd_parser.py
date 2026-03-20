"""
dmd_parser.py
=============
DMD 바이너리 파일 직접 파싱 → RCP 궤도 numpy 배열 반환.

지원 포맷: DMD Library ver.2.x (DMDF 헤더 + DMDH 블록 체인)
의존성: numpy, xml.etree.ElementTree, zlib (표준 라이브러리)

사용:
    info = DmdParser.read_info(path)
    windows = DmdParser.read_orbit_windows(path, info, window_sec=10)

[수정 이력]
- FLAGS_XML: 0x09000009(이벤트 XML) → 첫 번째 DMDH 블록 직접 사용
- DMDH_INNER_HDR: 53 → 33  (실측: size2-size1-20 = 36053-36000-20 = 33)
- 채널명: dmd_cfg XML 대신 oxy_config(zlib 압축 MeasurementConfig) 파싱
- FLAGS_RAW_DATA: == 0x02000000 → (flags>>24)==0x02 and level==0 로 완화
- 블록 당 채널 수: 전체 채널수 → 해당 세그먼트 채널 수
- 샘플 포맷: sint24 고정 → 세그먼트별 sint16 / sint24 지원
- 샘플레이트: 없는 XML 요소 대신 time_base_frequency_hz 사용
"""

import struct
import zlib
import numpy as np
from xml.etree import ElementTree as ET

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
DMDF_MAGIC      = b"DMDF"
DMDH_MAGIC      = b"DMDH"
DMDH_OUTER_HDR  = 20    # DMDH 블록 외부 헤더 크기 (bytes)
DMDH_INNER_HDR  = 33    # Raw 데이터 블록 내부 헤더 크기 (bytes)
                         # 실측: size2 - size1 - OUTER = 36053 - 36000 - 20 = 33
FIRST_BLOCK_OFF = 0x1000

# 데이터 블록 판별: flags 상위 바이트 == 0x02, 바이트2(level) == 0x00 → level-0 raw data
DATA_BLOCK_TYPE = 0x02
DATA_LEVEL_RAW  = 0x00

# ADC full-scale
DEFAULT_VOLT_RANGE = 10.0   # ±10 V
ADC_INT24_MAX = 2 ** 23     # 8_388_608
ADC_INT16_MAX = 2 ** 15     # 32_768

# RCP 궤도 채널 매핑 (DMD 채널 ShortId → RCP 이름)
DMD_ORBIT_CHANNEL_MAP = {
    "RCP1A": {"x": "AI 4/1", "y": "AI 4/2"},
    "RCP1B": {"x": "AI 4/4", "y": "AI 4/5"},
    "RCP2A": {"x": "AI 5/1", "y": "AI 5/2"},
    "RCP2B": {"x": "AI 5/4", "y": "AI 5/5"},
}
RCP_ORDER = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────
def _read_uint32_le(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _read_uint64_le(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", buf, offset)[0]


def _parse_int24_le(data: bytes) -> np.ndarray:
    """24-bit signed little-endian → float64 ndarray."""
    n = len(data) // 3
    if n == 0:
        return np.empty(0, dtype=np.float64)
    padded = np.zeros(n * 4, dtype=np.uint8)
    src = np.frombuffer(data[:n * 3], dtype=np.uint8)
    padded[0::4] = src[0::3]
    padded[1::4] = src[1::3]
    padded[2::4] = src[2::3]
    uint32_vals = padded.view(np.uint32)
    sign_mask = np.uint32(0x800000)
    neg_mask  = np.uint32(0xFF000000)
    needs_sign = (uint32_vals & sign_mask).astype(bool)
    int32_vals = uint32_vals.copy().astype(np.int32)
    int32_vals[needs_sign] = (int32_vals[needs_sign] | neg_mask.astype(np.int32))
    return int32_vals.astype(np.float64)


def _parse_int16_le(data: bytes) -> np.ndarray:
    """16-bit signed little-endian → float64 ndarray."""
    n = len(data) // 2
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return np.frombuffer(data[:n * 2], dtype=np.int16).astype(np.float64)


def _parse_samples(data: bytes, bits_per_sample: int) -> np.ndarray:
    """bits_per_sample에 따라 int24 또는 int16 파싱."""
    if bits_per_sample == 24:
        return _parse_int24_le(data)
    elif bits_per_sample == 16:
        return _parse_int16_le(data)
    else:
        raise ValueError(f"지원하지 않는 bits_per_sample: {bits_per_sample}")


def _adc_max(bits_per_sample: int) -> float:
    if bits_per_sample == 24:
        return float(ADC_INT24_MAX)
    elif bits_per_sample == 16:
        return float(ADC_INT16_MAX)
    return float(ADC_INT24_MAX)


def _bytes_per_sample(bits_per_sample: int) -> int:
    return bits_per_sample // 8


# ─────────────────────────────────────────────
# oxy_config 파싱 (zlib 압축 MeasurementConfig XML)
# ─────────────────────────────────────────────
def _find_oxy_config_block(f) -> tuple:
    """
    DMDH 블록 체인을 스캔해 oxy_config 세그먼트 (flags=0x0900000A) 의
    압축 데이터를 반환. 못 찾으면 (None, 0) 반환.
    """
    OXY_CONFIG_FLAGS = 0x0900000A  # segment_id=10 = oxy_config
    block_offset = FIRST_BLOCK_OFF
    for _ in range(200):
        f.seek(block_offset)
        outer = f.read(DMDH_OUTER_HDR)
        if len(outer) < DMDH_OUTER_HDR or outer[:4] != DMDH_MAGIC:
            break
        size1 = _read_uint32_le(outer, 4)
        size2 = _read_uint32_le(outer, 8)
        flags = _read_uint32_le(outer, 12)
        if size2 == 0:
            break
        if flags == OXY_CONFIG_FLAGS:
            f.seek(block_offset + DMDH_OUTER_HDR)
            blob = f.read(size1)
            return blob, size1
        block_offset += size2
    return None, 0


def _parse_oxy_config(blob: bytes) -> dict:
    """
    oxy_config zlib 압축 블록을 풀어 채널 정보를 반환.

    반환: {
        short_id: {
            "segment_id": int,
            "channel_id": int,
            "bits_per_sample": int,  # 24 or 16
            "volt_range": float,
            "sample_rate": float,
        }, ...
    }
    헤더 8바이트(커스텀) + zlib 스트림 구조.
    """
    SKIP = 8  # 커스텀 8바이트 헤더 뒤에 zlib 스트림 시작
    try:
        xml_bytes = zlib.decompress(blob[SKIP:])
    except Exception:
        return {}

    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return {}

    ns = "http://xml.dewetron.com/oxygen/config"
    result = {}

    for ch in root.findall(f"{{{ns}}}Channels/{{{ns}}}Channel"):
        acq = ch.find(f".//{{{ns}}}AcqSource")
        if acq is None:
            continue

        short_id    = None
        sample_res  = "sint24"
        volt_range  = DEFAULT_VOLT_RANGE
        sample_rate = 0.0

        for prop in acq.findall(f"{{{ns}}}ChannelConfig/{{{ns}}}Property"):
            pname = prop.get("name", "")
            v = prop.find("./")
            vt = v.text.strip('"') if v is not None and v.text else ""
            if pname == "ShortId":
                short_id = vt
            elif pname == "SampleResolution":
                sample_res = vt   # "sint24" or "sint16"
            elif pname == "SampleRate":
                # ScalarValue → Value 하위 요소
                val_el = prop.find(f".//{{{ns}}}Value")
                if val_el is not None and val_el.text:
                    try:
                        sample_rate = float(val_el.text)
                    except ValueError:
                        pass
            elif pname == "Range":
                rng = prop.find(f".//{{{ns}}}RangeMax")
                if rng is not None and rng.text:
                    try:
                        volt_range = abs(float(rng.text))
                    except ValueError:
                        pass

        if short_id is None:
            continue

        # DMDStorageDetails: segment_id, channel_id, raw_stride (bits)
        storage = ch.find(f".//{{{ns}}}DMDStorageDetails")
        if storage is None:
            continue

        try:
            seg_id  = int(storage.get("segment_id", "-1"))
            ch_id   = int(storage.get("channel_id", "-1"))
            raw_stride_bits = int(storage.get("raw_stride", "0"))
        except (TypeError, ValueError):
            continue

        if seg_id < 0 or ch_id < 0:
            continue

        # raw_stride는 bits 단위 (24=sint24, 16=sint16)
        bits = raw_stride_bits if raw_stride_bits in (16, 24) else (
            24 if "24" in sample_res else 16
        )

        result[short_id] = {
            "segment_id":     seg_id,
            "channel_id":     ch_id,
            "bits_per_sample": bits,
            "volt_range":     volt_range,
            "sample_rate":    sample_rate,
        }

    return result


# ─────────────────────────────────────────────
# 채널 / 세그먼트 메타 데이터
# ─────────────────────────────────────────────
class DmdChannelInfo:
    __slots__ = ("index", "name", "unit", "sample_rate",
                 "volt_range", "segment_id", "channel_id",
                 "bits_per_sample", "data_type")

    def __init__(self, index, name, unit, sample_rate,
                 volt_range=DEFAULT_VOLT_RANGE,
                 segment_id=None, channel_id=0,
                 bits_per_sample=24, data_type=0):
        self.index           = index
        self.name            = name
        self.unit            = unit
        self.sample_rate     = sample_rate
        self.volt_range      = volt_range
        self.segment_id      = segment_id
        self.channel_id      = channel_id       # 세그먼트 내 채널 인덱스
        self.bits_per_sample = bits_per_sample  # 24 or 16
        self.data_type       = data_type        # 0=sample, 5=metadata


class DmdSegmentInfo:
    """dmd_cfg XML에서 파싱한 세그먼트 정보."""
    __slots__ = ("identifier", "data_type", "sample_rate",
                 "n_channels", "bits_per_sample")

    def __init__(self, identifier, data_type, sample_rate,
                 n_channels, bits_per_sample=24):
        self.identifier      = identifier
        self.data_type       = data_type
        self.sample_rate     = sample_rate
        self.n_channels      = n_channels
        self.bits_per_sample = bits_per_sample


class DmdFileInfo:
    def __init__(self):
        self.channels: list         = []         # DmdChannelInfo 목록
        self.n_channels: int        = 0
        self.segments: dict         = {}         # {segment_id: DmdSegmentInfo}
        self.orbit_channels: dict   = {}         # {rcp_name: {"x": ch_info, "y": ch_info}}
        self.has_orbit: bool        = False


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────
class DmdParser:

    @staticmethod
    def read_info(path: str) -> DmdFileInfo:
        """
        DMD 파일 헤더 + XML 메타데이터 파싱.
        1) FIRST_BLOCK_OFF(0x1000)의 첫 번째 DMDH 블록 = dmd_cfg XML
        2) oxy_config 세그먼트(zlib) 스캔 → 채널명, 세그먼트 매핑
        """
        with open(path, "rb") as f:
            # DMDF 전역 헤더 확인
            magic = f.read(4)
            if magic != DMDF_MAGIC:
                raise ValueError(
                    f"DMD 파일이 아닙니다 (magic={magic!r}). DMDF 헤더가 없습니다."
                )

            # ── (1) dmd_cfg XML: 첫 번째 DMDH 블록 직접 읽기 ────────────
            f.seek(FIRST_BLOCK_OFF)
            outer = f.read(DMDH_OUTER_HDR)
            if len(outer) < DMDH_OUTER_HDR or outer[:4] != DMDH_MAGIC:
                raise ValueError("0x1000에서 DMDH 블록을 찾을 수 없습니다.")
            size1_cfg = _read_uint32_le(outer, 4)

            f.seek(FIRST_BLOCK_OFF + DMDH_OUTER_HDR)
            xml_bytes = f.read(size1_cfg)
            xml_str = xml_bytes.decode("utf-8", errors="replace").rstrip("\x00")
            cfg_root = ET.fromstring(xml_str)

            # ── (2) oxy_config 블록 (zlib 압축) 스캔 ────────────────────
            oxy_blob, _ = _find_oxy_config_block(f)

        # dmd_cfg XML → 세그먼트 메타 파싱
        segments: dict[int, DmdSegmentInfo] = {}
        cfg = cfg_root.find(".//file_content_config") or cfg_root

        for seg_el in cfg.findall(".//segments/segment") or cfg.findall(".//segment"):
            seg_id_el   = seg_el.find("identifier")
            dtype_el    = seg_el.find("data_type")
            sr_el       = seg_el.find("time_base_frequency_hz")

            seg_id    = int(seg_id_el.text)   if seg_id_el  is not None else -1
            data_type = int(dtype_el.text)    if dtype_el   is not None else 0
            sr        = float(sr_el.text)     if sr_el      is not None else 0.0

            n_ch = len(seg_el.findall(".//samples/sample"))

            if seg_id < 0 or data_type == 5:
                continue  # 메타데이터 세그먼트 제외

            segments[seg_id] = DmdSegmentInfo(
                identifier=seg_id,
                data_type=data_type,
                sample_rate=sr,
                n_channels=n_ch,
                bits_per_sample=24,   # oxy_config로 나중에 덮어씀
            )

        # oxy_config → 채널명 + 세그먼트별 bits_per_sample 결정
        oxy_map: dict = {}
        if oxy_blob is not None:
            oxy_map = _parse_oxy_config(oxy_blob)
            # segments bits_per_sample 업데이트 (oxy_config 기준)
            for short_id, ch_meta in oxy_map.items():
                sid = ch_meta["segment_id"]
                if sid in segments:
                    segments[sid].bits_per_sample = ch_meta["bits_per_sample"]
                    if ch_meta["sample_rate"] > 0:
                        segments[sid].sample_rate = ch_meta["sample_rate"]

        # 채널 목록 구성 (oxy_map 기준 → dmd_cfg 보완)
        channels: list[DmdChannelInfo] = []
        ch_index = 0

        if oxy_map:
            for short_id, meta in oxy_map.items():
                seg = segments.get(meta["segment_id"])
                sr  = meta["sample_rate"] or (seg.sample_rate if seg else 0.0)
                channels.append(DmdChannelInfo(
                    index=ch_index,
                    name=short_id,
                    unit="V",
                    sample_rate=sr,
                    volt_range=meta["volt_range"],
                    segment_id=meta["segment_id"],
                    channel_id=meta["channel_id"],
                    bits_per_sample=meta["bits_per_sample"],
                    data_type=0,
                ))
                ch_index += 1
        else:
            # oxy_config 없음 → dmd_cfg XML만으로 폴백
            for seg_id, seg in sorted(segments.items()):
                for ch_pos in range(seg.n_channels):
                    channels.append(DmdChannelInfo(
                        index=ch_index,
                        name=f"SEG{seg_id}_CH{ch_pos}",
                        unit="",
                        sample_rate=seg.sample_rate,
                        volt_range=DEFAULT_VOLT_RANGE,
                        segment_id=seg_id,
                        channel_id=ch_pos,
                        bits_per_sample=seg.bits_per_sample,
                        data_type=0,
                    ))
                    ch_index += 1

        info = DmdFileInfo()
        info.channels   = channels
        info.n_channels = len(channels)
        info.segments   = segments

        # 궤도 채널 매핑: short_id → DmdChannelInfo
        name_to_ch = {ch.name: ch for ch in channels}
        orbit_channels = {}
        for rcp in RCP_ORDER:
            mapping = DMD_ORBIT_CHANNEL_MAP[rcp]
            xch = name_to_ch.get(mapping["x"])
            ych = name_to_ch.get(mapping["y"])
            if xch is not None and ych is not None:
                orbit_channels[rcp] = {
                    "x": xch, "y": ych,
                    "x_name": mapping["x"], "y_name": mapping["y"],
                }

        info.orbit_channels = orbit_channels
        info.has_orbit      = len(orbit_channels) > 0
        return info

    @staticmethod
    def read_orbit_windows(
        path: str,
        info: DmdFileInfo,
        window_sec: int = 10,
        mil_per_volt: float = 10.0,
    ) -> list:
        """
        DMD 파일에서 궤도 채널 데이터를 읽어 window_sec 단위 윈도우 목록 반환.

        반환: List[dict]
          {
            "window_idx": int,
            "start_sec":  float,
            "end_sec":    float,
            "rcp": {
                "RCP1A": {"x": np.ndarray (mil), "y": np.ndarray (mil)},
                ...
            }
          }
        """
        if not info.has_orbit:
            raise ValueError("궤도 채널이 없습니다. 파일의 채널 목록을 확인하세요.")

        # 필요한 (segment_id, channel_id) 쌍 수집
        # {segment_id: {channel_id: [sample_chunks]}}
        needed: dict[int, set[int]] = {}
        for rcp_data in info.orbit_channels.values():
            for key in ("x", "y"):
                ch = rcp_data[key]
                needed.setdefault(ch.segment_id, set()).add(ch.channel_id)

        raw_data: dict[int, dict[int, list]] = {
            seg_id: {ch_id: [] for ch_id in ch_ids}
            for seg_id, ch_ids in needed.items()
        }

        with open(path, "rb") as f:
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

                # level-0 raw 데이터 블록 판별
                blk_type  = (flags >> 24) & 0xFF
                blk_level = (flags >> 16) & 0xFF
                seg_id    = flags & 0x0000FFFF  # 하위 2바이트 = segment_id

                if blk_type == DATA_BLOCK_TYPE and blk_level == DATA_LEVEL_RAW:
                    if seg_id in needed:
                        seg_info = info.segments.get(seg_id)
                        if seg_info is None:
                            block_offset += size2
                            continue

                        n_ch  = seg_info.n_channels
                        bps   = seg_info.bits_per_sample   # bits per sample
                        Bps   = bps // 8                   # bytes per sample
                        frame = n_ch * Bps                 # bytes per time frame

                        if frame == 0:
                            block_offset += size2
                            continue

                        # 내부 헤더(33 bytes) 건너뜀 → 데이터 시작
                        payload_offset = block_offset + DMDH_OUTER_HDR + DMDH_INNER_HDR
                        payload_size   = size1
                        actual_frames  = payload_size // frame

                        if actual_frames == 0:
                            block_offset += size2
                            continue

                        f.seek(payload_offset)
                        payload = f.read(actual_frames * frame)

                        # numpy 스트라이드로 채널 역인터리브 (Python 루프 대비 ~100x 빠름)
                        pa = np.frombuffer(payload[:actual_frames * frame],
                                           dtype=np.uint8).reshape(actual_frames, n_ch * Bps)

                        for ch_id in needed[seg_id]:
                            if ch_id >= n_ch:
                                continue
                            ch_slice = pa[:, ch_id * Bps: (ch_id + 1) * Bps]
                            samples = _parse_samples(bytes(ch_slice.ravel()), bps)
                            raw_data[seg_id][ch_id].append(samples)

                block_offset += size2

        # 채널별 연결 + ADC 스케일 → volt → mil 변환
        def get_channel_array(ch_info: DmdChannelInfo) -> np.ndarray:
            chunks = raw_data.get(ch_info.segment_id, {}).get(ch_info.channel_id, [])
            if not chunks:
                return np.empty(0, dtype=np.float64)
            arr = np.concatenate(chunks)
            # ADC raw → volt
            vr     = ch_info.volt_range
            adc_mx = _adc_max(ch_info.bits_per_sample)
            volts  = arr * (vr / adc_mx)
            # volt → mil (DC offset 제거 후 변환)
            unit = (ch_info.unit or "").lower()
            if unit in ("v", "volt", "volts", ""):
                return (volts - volts.mean()) * mil_per_volt
            else:
                return volts - volts.mean()

        channel_mil: dict = {}
        for rcp_data in info.orbit_channels.values():
            for key in ("x", "y"):
                ch = rcp_data[key]
                uid = (ch.segment_id, ch.channel_id)
                if uid not in channel_mil:
                    channel_mil[uid] = get_channel_array(ch)

        # 샘플레이트 결정
        orbit_sr = 0.0
        for rcp_data in info.orbit_channels.values():
            xch = rcp_data["x"]
            if xch.sample_rate > 0:
                orbit_sr = max(orbit_sr, xch.sample_rate)

        if orbit_sr <= 0:
            # dmd_cfg segments에서 폴백
            for rcp_data in info.orbit_channels.values():
                seg = info.segments.get(rcp_data["x"].segment_id)
                if seg and seg.sample_rate > 0:
                    orbit_sr = max(orbit_sr, seg.sample_rate)

        if orbit_sr <= 0:
            orbit_sr = 20_000.0  # 최종 fallback

        samples_per_window = int(orbit_sr * window_sec)
        if samples_per_window == 0:
            raise ValueError("window_sec 또는 sample_rate가 0입니다.")

        min_len = min(
            len(channel_mil.get((rd["x"].segment_id, rd["x"].channel_id), np.empty(0)))
            for rd in info.orbit_channels.values()
        )
        n_windows = min_len // samples_per_window
        if n_windows == 0:
            raise ValueError(
                f"데이터가 너무 짧습니다 (samples={min_len}, "
                f"window_samples={samples_per_window}). "
                "최소 1개의 윈도우가 필요합니다."
            )

        windows = []
        for wi in range(n_windows):
            s = wi * samples_per_window
            e = s + samples_per_window

            rcp_dict = {}
            for rcp_name, rcp_data in info.orbit_channels.items():
                xuid = (rcp_data["x"].segment_id, rcp_data["x"].channel_id)
                yuid = (rcp_data["y"].segment_id, rcp_data["y"].channel_id)
                x_arr = channel_mil.get(xuid, np.empty(0))
                y_arr = channel_mil.get(yuid, np.empty(0))
                rcp_dict[rcp_name] = {
                    "x": x_arr[s:e] if len(x_arr) >= e else x_arr[s:],
                    "y": y_arr[s:e] if len(y_arr) >= e else y_arr[s:],
                }

            windows.append({
                "window_idx": wi,
                "start_sec":  float(wi * window_sec),
                "end_sec":    float((wi + 1) * window_sec),
                "rcp":        rcp_dict,
            })

        return windows

    @staticmethod
    def read_raw_channels(
        path: str,
        info: DmdFileInfo,
        channel_names: list,
        max_samples: int = None,
    ) -> dict:
        """
        지정한 채널의 raw 전압 시계열(volt)을 반환.
        모델 학습용 데이터 추출에 사용.

        반환: {channel_name: np.ndarray (volt, float32)}
        """
        name_to_ch = {ch.name: ch for ch in info.channels}
        target_chs = [name_to_ch[n] for n in channel_names if n in name_to_ch]
        if not target_chs:
            raise ValueError(f"요청 채널 없음: {channel_names}")

        needed: dict[int, set[int]] = {}
        for ch in target_chs:
            needed.setdefault(ch.segment_id, set()).add(ch.channel_id)

        raw_data: dict[int, dict[int, list]] = {
            seg_id: {ch_id: [] for ch_id in ch_ids}
            for seg_id, ch_ids in needed.items()
        }

        with open(path, "rb") as f:
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

                blk_type  = (flags >> 24) & 0xFF
                blk_level = (flags >> 16) & 0xFF
                seg_id    = flags & 0x0000FFFF

                if blk_type == DATA_BLOCK_TYPE and blk_level == DATA_LEVEL_RAW:
                    if seg_id in needed:
                        seg_info = info.segments.get(seg_id)
                        if seg_info is None:
                            block_offset += size2
                            continue

                        n_ch  = seg_info.n_channels
                        bps   = seg_info.bits_per_sample
                        Bps   = bps // 8
                        frame = n_ch * Bps

                        if frame == 0:
                            block_offset += size2
                            continue

                        payload_offset = block_offset + DMDH_OUTER_HDR + DMDH_INNER_HDR
                        f.seek(payload_offset)
                        payload = f.read(size1)
                        actual_frames = len(payload) // frame

                        if actual_frames == 0:
                            block_offset += size2
                            continue

                        pa = np.frombuffer(payload[:actual_frames * frame],
                                           dtype=np.uint8).reshape(actual_frames, n_ch * Bps)

                        for ch_id in needed[seg_id]:
                            if ch_id >= n_ch:
                                continue
                            # 조기 종료: max_samples 초과 시
                            collected = sum(
                                len(c) for c in raw_data[seg_id][ch_id]
                            )
                            if max_samples and collected >= max_samples:
                                continue
                            ch_slice = pa[:, ch_id * Bps: (ch_id + 1) * Bps]
                            raw_data[seg_id][ch_id].append(
                                _parse_samples(bytes(ch_slice.ravel()), bps)
                            )

                block_offset += size2

        result = {}
        for ch in target_chs:
            chunks = raw_data.get(ch.segment_id, {}).get(ch.channel_id, [])
            if not chunks:
                result[ch.name] = np.empty(0, dtype=np.float32)
                continue
            arr = np.concatenate(chunks)
            if max_samples:
                arr = arr[:max_samples]
            # ADC raw → volt
            adc_mx = _adc_max(ch.bits_per_sample)
            volt   = arr * (ch.volt_range / adc_mx)
            result[ch.name] = volt.astype(np.float32)

        return result
