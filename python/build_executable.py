"""
PyInstaller 빌드 스크립트
단일 실행 파일로 infer_resnet_None.py를 번들링합니다.
"""
import PyInstaller.__main__
import sys
import os

# UTF-8 출력 설정 (Windows 한글 깨짐 방지)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 현재 스크립트 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'model')
spec_file = os.path.join(script_dir, 'infer_resnet.spec')

# PyInstaller 옵션
pyinstaller_args = [
    'infer_resnet_None.py',
    '--name=infer_resnet',
    '--onefile',  # 단일 실행 파일
    '--console',  # 콘솔 앱 (stdout/stderr 필요)
    '--clean',    # 이전 빌드 정리
    f'--distpath={os.path.join(script_dir, "dist")}',
    f'--workpath={os.path.join(script_dir, "build")}',
    f'--specpath={script_dir}',
    # 모델 파일 포함
    f'--add-data={model_dir}{os.pathsep}model',
    # 숨겨진 import 명시
    '--hidden-import=PIL._tkinter_finder',
    '--hidden-import=scipy.special._ufuncs_cxx',
    '--hidden-import=scipy.linalg.cython_blas',
    '--hidden-import=scipy.linalg.cython_lapack',
    '--hidden-import=scipy.ndimage',
    # 최적화
    '--optimize=2',
]

print("=" * 60)
print("PyInstaller 빌드 시작")
print("=" * 60)
print(f"작업 디렉토리: {script_dir}")
print(f"모델 디렉토리: {model_dir}")
print(f"출력 디렉토리: {os.path.join(script_dir, 'dist')}")
print("=" * 60)

try:
    PyInstaller.__main__.run(pyinstaller_args)
    print("\n" + "=" * 60)
    print("✅ 빌드 성공!")
    print(f"실행 파일: {os.path.join(script_dir, 'dist', 'infer_resnet.exe')}")
    print("=" * 60)
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ 빌드 실패: {e}")
    print("=" * 60)
    sys.exit(1)
