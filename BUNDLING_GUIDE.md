# Python 런타임 번들링 가이드

## 개요

이 프로젝트는 Python 기반 ResNet18 모델을 PyInstaller로 단일 실행 파일로 번들링하여 Electron 앱에 포함합니다.

## 번들링 완료 상태

✅ **번들링이 완료되었습니다!**

- 실행 파일: `python/dist/infer_resnet.exe` (약 300MB)
- 모든 의존성 포함 (PyTorch, NumPy, SciPy, Matplotlib 등)
- 사용자 PC에 Python 설치 불필요

---

## 디렉토리 구조

```
rcp_5th/
├── python/
│   ├── infer_resnet_None.py          # 원본 Python 스크립트
│   ├── model/
│   │   └── resnet18_orbit_v3_None.pth  # 모델 가중치 (포함됨)
│   ├── dist/
│   │   └── infer_resnet.exe           # 번들된 실행 파일 ✅
│   ├── build/                         # PyInstaller 빌드 캐시
│   ├── build_executable.py            # 빌드 스크립트
│   ├── build.bat                      # Windows 빌드 배치 파일
│   └── requirements.txt               # Python 의존성
│
├── src/main/utils/pythonRunner.ts    # 실행 파일 호출 로직 ✅
└── electron-builder.yml               # Electron 빌드 설정 ✅
```

---

## 개발 vs 프로덕션 모드

### 개발 모드 (`npm run dev`)
- Python 스크립트 직접 실행: `python infer_resnet_None.py --bin_path ... --json`
- 시스템 Python 환경 필요
- 빠른 개발 및 디버깅

### 프로덕션 모드 (앱 설치 후)
- 번들된 실행 파일 실행: `infer_resnet.exe --bin_path ... --json`
- Python 설치 불필요
- 독립 실행 가능

---

## 빌드 과정

### 1. Python 패키지 설치 ✅
```bash
cd python
pip install -r requirements.txt
```

### 2. PyInstaller 설치 ✅
```bash
pip install pyinstaller
```

### 3. 실행 파일 빌드 ✅
```bash
# 방법 1: 스크립트 사용
python build_executable.py

# 방법 2: 배치 파일 사용 (Windows)
build.bat
```

빌드 결과:
- 실행 파일: `python/dist/infer_resnet.exe`
- 크기: 약 300MB (PyTorch 포함)
- 빌드 시간: 약 5-10분

---

## Electron 앱 빌드

### 개발 모드 테스트
```bash
npm run dev
```

### 프로덕션 빌드
```bash
# TypeScript 컴파일 + Vite 빌드
npm run build

# Windows 설치 파일 생성
npm run build:win

# 결과물
dist/rcp-desktop-app-1.0.0-setup.exe
```

---

## 번들링 설정 상세

### `pythonRunner.ts` (src/main/utils/pythonRunner.ts)

```typescript
constructor() {
  this.isProduction = app.isPackaged

  // 개발: Python 스크립트 직접 실행
  if (!this.isProduction) {
    this.executablePath = '' // 'python' 명령어 사용
  }
  // 프로덕션: 번들된 실행 파일
  else {
    this.executablePath = path.join(
      process.resourcesPath,
      'python',
      'infer_resnet.exe'
    )
  }
}
```

### `electron-builder.yml`

```yaml
extraResources:
  - from: python/dist/
    to: python/
    filter:
      - infer_resnet.exe
```

이 설정은:
- `python/dist/infer_resnet.exe`를 
- 설치된 앱의 `resources/python/infer_resnet.exe`로 복사합니다.

---

## 번들 크기 최적화

현재 실행 파일 크기: **300MB**

### 포함된 주요 라이브러리
- PyTorch: ~200MB
- NumPy, SciPy: ~50MB
- Matplotlib: ~30MB
- 기타: ~20MB

### 최적화 옵션 (선택)

1. **PyTorch CPU 전용**
   - 이미 CPU 버전 사용 중 ✅

2. **불필요한 모듈 제외**
   ```python
   # build_executable.py에 추가
   '--exclude-module=tkinter',
   '--exclude-module=matplotlib.tests',
   ```

3. **UPX 압축** (선택)
   ```python
   '--upx-dir=/path/to/upx',
   ```
   - 주의: PyTorch는 UPX 압축 시 오류 가능

---

## 재빌드가 필요한 경우

다음의 경우 실행 파일을 다시 빌드해야 합니다:

1. **Python 스크립트 수정**
   - `infer_resnet_None.py` 변경 시

2. **모델 가중치 변경**
   - `model/resnet18_orbit_v3_None.pth` 교체 시

3. **Python 패키지 업데이트**
   - torch, numpy 등 버전 변경 시

재빌드 명령어:
```bash
cd python
python build_executable.py
```

---

## 트러블슈팅

### 문제 1: 실행 파일이 생성되지 않음
**해결**:
```bash
# 빌드 캐시 삭제
cd python
rmdir /s /q build dist
python build_executable.py
```

### 문제 2: 실행 시 "DLL not found" 오류
**원인**: PyTorch DLL 누락  
**해결**: PyInstaller에 hidden imports 추가
```python
'--hidden-import=torch',
'--collect-all=torch',
```

### 문제 3: 모델 파일을 찾을 수 없음
**원인**: 실행 파일 내부 경로 문제  
**해결**: 이미 `--add-data` 옵션으로 해결됨 ✅
```python
f'--add-data={model_dir}{os.pathsep}model'
```

### 문제 4: 실행 파일이 너무 느림
**원인**: PyInstaller 압축 해제 시간  
**해결**: `--onefile` 대신 `--onedir` 사용 (폴더 배포)

---

## CI/CD 통합 (선택)

### GitHub Actions 예시

```yaml
name: Build Python Bundle

on: [push]

jobs:
  build:
    runs-on: windows-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd python
          pip install -r requirements.txt
          pip install pyinstaller
      
      - name: Build executable
        run: |
          cd python
          python build_executable.py
      
      - name: Upload artifact
        uses: actions/upload-artifact@v2
        with:
          name: python-bundle
          path: python/dist/infer_resnet.exe
```

---

## 라이선스 고려사항

번들된 실행 파일에는 다음 라이브러리가 포함됩니다:
- PyTorch (BSD License)
- NumPy (BSD License)
- SciPy (BSD License)
- Matplotlib (PSF License)
- Pillow (HPND License)

모든 라이선스는 상업적 사용 가능합니다.

---

## 결론

✅ Python 런타임 번들링 완료  
✅ 사용자 PC에 Python 설치 불필요  
✅ 독립 실행 가능한 Electron 앱  
✅ One-click 설치 지원  

**다음 단계**:
1. `npm run dev`로 개발 모드 테스트
2. `npm run build:win`으로 설치 파일 생성
3. 실제 BIN 파일로 추론 테스트
4. 배포

---

**작성일**: 2025년  
**문의**: 프로젝트 README 참조
