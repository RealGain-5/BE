"""
_compat.py
==========
PyTorch / Python 환경 호환성 패치.

학습·추론 스크립트의 최상단에서 가장 먼저 import 해야 합니다.
(import torch 보다 앞서 실행되어야 함)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
증상
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  torch._dynamo → torch.distributed.tensor._collective_utils 임포트 체인에서
  @torch.library.register_fake 데코레이터가 inspect.findsource 를 호출할 때
  OSError: could not get source code 발생 후 크래시.

  트리거: nn.MultiheadAttention(batch_first=True) 첫 순전파
  환경 : PyTorch 2.0~2.9 + Windows + Python 3.11 + CPU-only 빌드
  참조 : PyTorch GitHub issue #94206

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
적용 패치
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1) inspect.findsource 안전 래퍼:
     frozen/compiled 모듈에서 OSError 발생 시 빈 결과 반환 (무시)
     → torch.library.register_fake 소스 위치 기록 실패를 조용히 처리

  2) TORCHDYNAMO_DISABLE=1 환경변수:
     torch._dynamo 지연 임포트 자체를 비활성화
     → MultiheadAttention fast-path가 dynamo를 우회하여 Python으로 실행
     → 학습 속도에 실질적 영향 없음 (GPU compile 사용 안 하는 환경)
"""

import inspect as _inspect
import os as _os

# ── 패치 1: inspect.findsource 안전 래퍼 ────────────────────────────────────
_orig_findsource = _inspect.findsource


def _safe_findsource(obj):
    """
    원본 findsource를 호출하되, 실패 시 빈 결과를 반환합니다.
    frozen/compiled 모듈, C 확장, .pyc 전용 파일에서 발생하는
    OSError / TypeError / AttributeError 를 무시합니다.
    """
    try:
        return _orig_findsource(obj)
    except (OSError, TypeError, AttributeError):
        return ([], 0)


_inspect.findsource = _safe_findsource

# ── 패치 2: torch._dynamo 비활성화 ──────────────────────────────────────────
# import torch 보다 먼저 설정해야 효과 있음
_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
