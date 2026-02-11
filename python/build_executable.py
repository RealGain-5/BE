"""
PyInstaller 빌드 스크립트
inference_daemon.py와 infer_resnet_None.py를 단일 실행 파일로 번들링합니다.
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

# 공통 hidden imports
common_hidden_imports = [
    '--hidden-import=PIL._tkinter_finder',
    '--hidden-import=scipy.special._ufuncs_cxx',
    '--hidden-import=scipy.linalg.cython_blas',
    '--hidden-import=scipy.linalg.cython_lapack',
    '--hidden-import=scipy.ndimage',
]

def build_executable(entry_script, output_name, include_model=True):
    """단일 실행 파일 빌드"""
    print("\n" + "=" * 60)
    print(f"빌드 시작: {entry_script} -> {output_name}.exe")
    print("=" * 60)
    
    entry_path = os.path.join(script_dir, entry_script)
    
    pyinstaller_args = [
        entry_path,
        f'--name={output_name}',
        '--onefile',
        '--console',
        '--clean',
        f'--distpath={os.path.join(script_dir, "dist")}',
        f'--workpath={os.path.join(script_dir, "build", output_name)}',
        f'--specpath={script_dir}',
        '--optimize=2',
    ]
    
    # 모델 파일 포함 (필요한 경우)
    if include_model:
        pyinstaller_args.append(f'--add-data={model_dir}{os.pathsep}model')
    
    # hidden imports 추가
    pyinstaller_args.extend(common_hidden_imports)
    
    try:
        PyInstaller.__main__.run(pyinstaller_args)
        exe_path = os.path.join(script_dir, 'dist', f'{output_name}.exe')
        print(f"✅ 빌드 성공: {exe_path}")
        return True
    except Exception as e:
        print(f"❌ 빌드 실패: {e}")
        return False

def main():
    print("=" * 60)
    print("PyInstaller 빌드 스크립트")
    print("=" * 60)
    print(f"작업 디렉토리: {script_dir}")
    print(f"모델 디렉토리: {model_dir}")
    print(f"출력 디렉토리: {os.path.join(script_dir, 'dist')}")
    
    # 빌드할 파일 목록: (스크립트, 출력명, 모델포함여부)
    build_targets = [
        ('inference_daemon.py', 'inference_daemon', True),
        ('infer_resnet_None.py', 'infer_resnet', True),
    ]
    
    results = []
    for entry_script, output_name, include_model in build_targets:
        success = build_executable(entry_script, output_name, include_model)
        results.append((output_name, success))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("빌드 결과 요약")
    print("=" * 60)
    all_success = True
    for name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {name}.exe: {status}")
        if not success:
            all_success = False
    print("=" * 60)
    
    if not all_success:
        sys.exit(1)

if __name__ == "__main__":
    main()
