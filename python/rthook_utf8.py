"""
PyInstaller runtime hook — UTF-8 강제 설정
부트로더가 Python을 초기화한 직후, 메인 스크립트 실행 전에 실행됨.
Windows에서 시스템 코드페이지(CP949/CP1252)로 파이프가 열리는 문제를 방지.
"""
import os
import sys
import io

os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    if hasattr(sys.stdin, 'buffer'):
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace',
                                      line_buffering=True)
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace',
                                      line_buffering=True)
except Exception:
    pass
