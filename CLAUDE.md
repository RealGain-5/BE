# RCPVMS 프로젝트 — Claude 작업 가이드

## 프로젝트 개요

회전 기계 상태 감시 시스템 (RCPVMS) Electron 데스크톱 앱.
RCPVMS BIN 파일을 읽어 궤도(orbit) 이미지를 시각화하고, AI 모델로 이상 탐지를 수행한다.

**스택**: Electron (main/preload/renderer) + Python 데몬 (stdin/stdout JSON IPC)  
**Python 가상환경**: `venv/Scripts/python` (프로젝트 루트 하위, Windows)

---

## 코드 구조

```
rcp_5th/
├── src/
│   ├── main/
│   │   ├── index.ts                  # IPC 핸들러 등록, Electron 앱 진입점
│   │   ├── services/
│   │   │   └── pythonService.ts      # PythonDaemonPool 래퍼, 모든 Python 호출 집중
│   │   └── utils/
│   │       └── PythonDaemonPool.ts   # Python 워커 풀 관리 (sticky worker 포함)
│   ├── preload/
│   │   └── index.ts                  # contextBridge API 노출 (window.api)
│   └── renderer/src/
│       ├── App.jsx                   # 탭 라우팅 (TAB_CONFIG 기반)
│       ├── components/
│       │   ├── ModelInference.jsx    # 앙상블 추론 UI
│       │   ├── MAEAnalysis.jsx       # MAE 이상 탐지 UI
│       │   ├── RcpvmsOrbitViewer.jsx # RCPVMS BIN 궤도 뷰어 (단일/배치)
│       │   ├── DmdOrbitViewer.jsx    # DMD 파일 변환 + 궤도 뷰어 (서브탭 2개)
│       │   └── shared/
│       │       ├── OrbitGrid.jsx           # 궤도 그리드 (IntersectionObserver + 배치 API)
│       │       ├── AnalysisModeLayout.jsx  # 단일/배치 모드 공통 레이아웃 (ModelInference·MAE 공유)
│       │       ├── FileOperationFlow.jsx   # 파일선택→로딩→정보→파라미터→실행→결과 플로우
│       │       ├── BatchResultList.jsx     # 배치 결과 아코디언 목록
│       │       ├── BatchActionButtons.jsx  # 배치 실행/취소 버튼 묶음
│       │       ├── SingleFileMode.jsx      # 단일 파일 선택+실행 UI
│       │       ├── BatchFileList.jsx       # 배치 파일 목록
│       │       ├── BatchProgressBar.jsx    # 배치 진행 바
│       │       ├── ConcurrencySelector.jsx # 병렬 처리 수준 선택기
│       │       ├── SubTabNav.jsx           # 서브탭 네비게이션
│       │       ├── ScaleModeToggle.jsx     # Auto/Fixed/User 스케일 토글
│       │       ├── OrbitControls.jsx       # 필터 모드 토글 + window_sec 입력 + axis_lim 입력
│       │       ├── StatusCell.jsx          # 파일 처리 상태 아이콘
│       │       └── ErrorDisplay.jsx        # 공통 에러 표시
│       ├── hooks/
│       │   ├── useAnalysisController.js    # 공통 배치 상태/핸들러 훅 (ModelInference·MAE 공유)
│       │   ├── useConcurrencySelector.js   # 병렬 수준 state + window.api.setConcurrencyLevel 호출
│       │   └── useObjectUrlImage.js        # object URL 생명주기 관리 훅 (자동 revoke)
│       └── utils/
│           ├── tabRegistry.js        # TAB_CONFIG 배열 — 탭 추가 시 여기만 수정
│           ├── fileUtils.js          # getFileName(filePath) — 경로에서 파일명 추출
│           ├── imageSource.js        # imageBufferToObjectUrl / imagePayloadToSource
│           └── labelStrategies.jsx   # LABEL_STRATEGIES — ensemble/mae 판정 라벨 렌더링 분리
├── python/
│   ├── inference_daemon.py           # 데몬 메인 루프 (모든 명령 처리)
│   ├── preprocess.py                 # 신호 전처리, FFT, 필터링, 이미지 생성
│   ├── rcpvms_parser.py              # RCPVMS BIN 파서
│   ├── dmd_parser.py                 # DMD 바이너리 파서
│   ├── dmd_to_rcpvms.py              # DMD → RCPVMS BIN 변환기
│   ├── model_loader.py               # 체크포인트 로드
│   ├── model_mae.py                  # MAE (Masked Autoencoder) 모델 정의
│   ├── infer_resnet_None.py          # ResNet 추론 + GradCAM + render_with_axes
│   ├── train_multiscale.py           # 멀티스케일 ResNet 학습
│   ├── train_1d_cnn.py               # OrbitCNN1D 학습
│   ├── train_mae.py                  # MAE 학습
│   ├── evaluate_ensemble.py          # 앙상블 성능 평가
│   └── model/
│       ├── resnet18_orbit_multiscale.pth
│       ├── orbit_cnn1d.pth
│       └── orbit_mae.pth
```

---

## IPC 통신 구조

```
Renderer (window.api.xxx)
    ↓ ipcRenderer.invoke
Preload (index.ts — contextBridge)
    ↓ IPC channel
Main (index.ts — ipcMain.handle)
    ↓ pythonService.runXxx()
PythonService (pythonService.ts)
    ↓ pool.sendCommand('command', payload)
PythonDaemonPool (워커 풀)
    ↓ stdin JSON
inference_daemon.py (Python)
    ↑ stdout JSON
```

**준비 신호**: Python 데몬은 모델 로드 완료 후 stderr에 `"model loaded successfully"` 출력.  
`PythonDaemonPool`이 이 문자열을 감지해야 워커가 준비 완료로 처리됨.

---

## 핵심 설계 원칙

### 1. 동시성 제어 (2계층)
- **Layer 1**: `runParallelBatch(semaphore)` — AbortController로 루프 차단, 기본 동시성 2 (범위 1~4)
- **Layer 2**: `PythonDaemonPool` — 워커 풀 큐 관리
- 취소: `abortController.abort()` (루프 차단) + `pool.cancelPendingJobs()` (큐 제거)
- in-flight 작업은 Python 프로세스를 종료하지 않으면 중단 불가

### 2. Sticky Worker 라우팅 (PythonDaemonPool.ts)
`rcpvms_orbit*` 명령에 한해 동일 filepath 요청을 동일 워커로 우선 배정 → Python 프로세스 로컬 캐시 재사용.

```typescript
stickyWorkerByKey: Map<string, number>  // filepath → workerId
getStickyWorkerId(command, payload)     // 'rcpvms_orbit*' 명령에서 filepath 추출
```

**starvation 방지**: 모든 pending job에 `preferredWorkerId`가 있고 해당 워커가 busy일 때,
idle 워커가 있으면 첫 번째 job을 캐시 미스 감수하고 즉시 실행 (`jobIndex = 0` fallback).  
**크래시 복구**: 워커 크래시 시 `handleWorkerCrash`에서 해당 워커의 sticky 매핑 항목 삭제.

### 3. Python 측 캐시 (inference_daemon.py)
```python
_rcpvms_header_cache  # filepath → (mtime, info, orbit_map)  — 헤더 파싱 캐시
_rcpvms_orbit_cache   # (filepath, mtime, window_sec) → orbit_data — 데이터 캐시 (1개 유지)
_dmd_info_cache       # dmd_path → (mtime, DmdFileInfo)  — DMD 정보 캐시
```
- `mtime` 기반 무효화: 파일이 변경되면 자동으로 캐시 재갱신
- `_rcpvms_orbit_cache`는 항목 1개만 유지 (`.clear()` 후 저장) — 메모리 절감

### 4. 배치 이미지 API
`rcpvms_orbit_multi` 명령: N개 `rcpvms_orbit_single` IPC 왕복 → 1회로 단축.
- 썸네일(96px): `render_with_axes` 생략 + PIL 전용 `_draw_crosshair`로 중심 십자선만 추가
- 모달 상세(256px): `render_with_axes` 포함 (matplotlib 축 레이블 + 눈금)

### 5. IntersectionObserver (OrbitGrid.jsx)
그리드가 뷰포트에 진입할 때만 `runRcpvmsOrbitMulti` 호출.  
스크롤로 내려오기 전까지 오프스크린 그리드는 이미지 요청 없음.

### 6. 이미지 소스 통합 (imageSource.js)
`rcpvms_orbit_single`은 Node Buffer(`image_buffer`)로 반환, 그 외는 base64(`image_b64`).
`imagePayloadToSource(data)` 헬퍼가 두 형식 모두 처리 → `<img src>` 또는 object URL 반환.
`useObjectUrlImage` 훅이 object URL 생명주기 자동 관리 (언마운트 시 revoke).

---

## 프론트엔드 리팩토링 구조

### 탭 등록 (`tabRegistry.js`)
```js
export const TAB_CONFIG = [
  { id: 'ensemble', label: '앙상블 분석', Component: ModelInference },
  { id: 'mae',      label: 'MAE 분석',   Component: MAEAnalysis },
  { id: 'dmd',      label: 'DMD 분석',   Component: DmdOrbitViewer },
  { id: 'rcpvms',   label: 'RCPVMS 뷰어', Component: RcpvmsOrbitViewer },
]
```
탭 추가 = `TAB_CONFIG` 배열에 항목 1개 추가로 완결, `App.jsx` 수정 불필요.

### 공통 레이아웃 계층
```
AnalysisModeLayout        ← ModelInference / MAEAnalysis 공유
  ├── SingleFileMode       ← 단일 파일 선택+실행
  ├── BatchFileList        ← 배치 파일 목록 (pending/running)
  ├── BatchProgressBar     ← 진행 바
  ├── BatchActionButtons   ← 실행/취소 버튼
  ├── BatchResultList      ← 완료 결과 아코디언
  ├── ConcurrencySelector  ← 병렬 수준 선택기
  └── ErrorDisplay         ← 에러 표시

FileOperationFlow          ← RcpvmsOrbitViewer / DmdOrbitViewer 공유
  (파일선택→로딩→파일정보→파라미터→실행→결과 5단계 플로우)
```

### 도메인 고유 로직 분리
- **`labelStrategies.jsx`**: `ensemble`(final_label) / `mae`(final_verdict) 판정 라벨 렌더링 전략을 한 곳에 집중.
- **`useAnalysisController.js`**: 배치 상태(files, batchProgress, batchLoading) + 핸들러 — ModelInference·MAE 공유.
- **`useConcurrencySelector.js`**: 병렬 수준 state + `window.api.setConcurrencyLevel` 호출 + 레벨 4 확인 대화상자.

---

## RCPVMS BIN 파일 포맷

```
[헤더 512B] + [채널 info 20B × N] + [float32 channel-major 데이터]
data_offset = 512 + total_ch × 20
mils_per_v  = 10.0  (기본값)
```

- `ch_type`: 0=가속, 1=변위(orbit), 2=keyphasor
- RCP 명칭: `RCP1A / RCP1B / RCP2A / RCP2B` (NIMS 표준, 구형 RCPA1 아님)

---

## AI 모델 파이프라인

### 앙상블 추론 (analyze 명령)
```
BIN → extract_rcp_xy_from_bin
    → sec9 구간 (9~10초) 슬라이싱
    → 멀티스케일 ResNet18 + OrbitCNN1D 앙상블
    → OOD 판정 (TV Distance + max_conf)
    → GradCAM + IG 시각화
```

### OOD 판정 (복합 기준)
- **조건 A**: TV Distance(resnet_probs, cnn1d_probs) > tv_threshold → 모달리티 불일치
- **조건 B**: max(앙상블 확률) < ood_threshold → 저신뢰도
- A OR B 시 `"unknown_abnormal"` 반환

설정 파일: `python/ensemble_config.json`
```json
{
  "resnet_weight": 0.5,
  "cnn1d_weight": 0.5,
  "ood_threshold": 0.70,
  "tv_threshold": 0.30
}
```

### MAE 이상 탐지 (mae_analyze 명령)
```
Stage 1: 슬라이딩 윈도우 배치 스윕 → 최고 점수 윈도우 선정
Stage 2: 최고 점수 윈도우 → 재구성 오차 최종 판정 (n_eval=10)
```
- OR 로직: `(score_1d > threshold_1d) OR (score_spec > threshold_spec)` 시 이상
- 설정 파일: `python/mae_config.json`

---

## 신호 처리 핵심

### FrequencyEstimate (preprocess.py)
```python
@dataclass(frozen=True)
class FrequencyEstimate:
    freq_hz: float
    confidence: float       # [0, 1] — snr 45% + harmonic 35% + base 20% 혼합
    flags: tuple[str, ...]  # 'weak_fundamental' | 'no_harmonic_support' | 'subharmonic_present'
```

### estimate_1x_freq (preprocess.py)
```python
estimate_1x_freq(x_mil, fs, y_mil=None, rpm_min=300, rpm_max=24000) -> FrequencyEstimate
```
Hann 윈도잉 + 하모닉 family scoring으로 1X 주파수 탐지 → `FrequencyEstimate` 반환:
1. 상위 128개 강한 피크의 1/2, 1/3, 1/4 서브하모닉을 후보로 추가
2. 하모닉 일관성 점수: `P(f)×1.0 + P(2f)×0.8 + P(3f)×0.5 + P(4f)×0.25`
3. fundamental 에너지 미달 시 score × 0.5 감점 (가짜 fundamental 억제)
4. 강한 서브하모닉 존재 시 penalty (2X/3X 오탐 방지)

`y_mil`을 제공하면 X+Y 파워 스펙트럼 합산 → 한쪽 프로브가 약한 경우에도 탐지 정확도 향상.  
`detect_1x_freq`는 backward-compat wrapper → `.freq_hz` 반환.

### 필터 모드 (_make_display_pil)
```python
_make_display_pil(x_seg, y_seg, axis_lim, fs=None, filter_mode="1x", img_size=256, f1x_hint=None)
```
- `raw`: DC 제거만
- `1x`: 1X 밴드패스 + 정수 사이클 트리밍
- `2x`: 2X 밴드패스 + 정수 사이클 트리밍
- `broadband`: 고역통과 (DC 드리프트 제거)
- `overlay`: Raw/BB/2X/1X 4가지 색상 오버레이 (`_make_overlay_pil`, estimate_1x_freq 1회 공유)

**confidence 기반 fallback** (`_1X_CONF_THRESHOLD = 0.35`):  
`1x`/`2x` 모드에서 `confidence < 0.35` 또는 `no_harmonic_support` 플래그 시 `broadband`로 자동 downgrade.  
노이즈성 신호에서 무의미한 bandpass 적용 방지.

**`f1x_hint`**: 사전 계산된 `FrequencyEstimate`를 주입하면 per-cell FFT를 건너뜀 (배치 최적화).

### _trim_to_integer_cycles
- 엣지 사이클 제거 (filtfilt 과도 응답 억제) → 나선형 궤도 방지
- 최소 2사이클 잔존 보장 (적응형 엣지 사이클 조절)
- `_FILTER_EDGE_CYCLES = 5` (기본값)

### _draw_crosshair (inference_daemon.py)
PIL 전용 경량 십자선. matplotlib 없이 중심 가로/세로 선을 그림.
`rcpvms_orbit_multi` 썸네일에서 `render_with_axes` 대신 사용.

---

## DMD → RCPVMS 변환 파이프라인

```
DMD 파일 → DmdParser.read_info() → 채널 매핑
         → DmdToRcpvmsConverter.convert()
         → RCPVMS BIN (10초 고정 윈도우, channel-major float32)
```

**실측 채널 분석 결과** (SAEUL2_20240823 기준):
- AI 1~3: 미연결 노이즈 (가속도계 없음)
- AI 4: X방향 변위 프로브 (6채널 모두 동일 신호)
- AI 5: Y방향 변위 프로브 (6채널 모두 동일 신호)
- 유효 궤도: **RCP1A 단 1개** (X=AI 4/1, Y=AI 5/1)

---

## RCP 명칭 규칙

**항상 `RCP1A / RCP1B / RCP2A / RCP2B` 사용** (NIMS 표준).  
구형 `RCPA1 / RCPA2 / RCPB1 / RCPB2`는 제거됨.  
- `src/main/index.ts` line 13: `const RCP_NAMES = ['RCP1A', 'RCP1B', 'RCP2A', 'RCP2B']`
- `inference_daemon.py` line 89: `RCP_NAMES = ["RCP1A", "RCP1B", "RCP2A", "RCP2B"]`

---

## 새 Python 명령 추가 절차

1. `inference_daemon.py` → `main()` 루프에 `elif command == "xxx":` 블록 추가
2. `pythonService.ts` → `async runXxx(...)` 메서드 추가
3. `src/main/index.ts` → `ipcMain.handle('xxx', ...)` 핸들러 추가
4. `src/preload/index.ts` → `api` 객체에 `xxx: (...) => ipcRenderer.invoke('xxx', ...)` 추가
5. 필요 시 `makeProgressChannel`로 진행 이벤트 채널 추가

## 새 탭 추가 절차

1. `src/renderer/src/components/` 에 컴포넌트 파일 생성
2. `src/renderer/src/utils/tabRegistry.js` → `TAB_CONFIG` 배열에 항목 추가
3. `App.jsx` 수정 불필요

---

## 미완료 작업

1. **앙상블 모델 재학습** — OE(Outlier Exposure) + 복합 지표(`val_acc × max(ood_rate, 0.01)`) + `--patience 10` 조기 종료 코드 적용 완료, 실제 재학습 필요
2. **일반화 평가** — 다른 날짜/조건 BIN 파일로 테스트
3. **DMD 변환 검증** — 실제 5.1GB DMD 파일 변환 테스트
4. **변환 BIN → CNN 학습** — 변환된 BIN 파일을 기존 학습 파이프라인에 투입

---

## 주의사항

- Python 데몬은 **싱글 스레드**: 동시 실행 안 됨. 병렬성은 PythonDaemonPool의 다중 워커로 확보.
- `_rcpvms_orbit_cache`는 1개 항목만 유지 → window_sec 변경 시 항상 재읽기 발생.
- `render_with_axes` 호출은 비용이 큼 (matplotlib 렌더링): 썸네일에서는 `_draw_crosshair`로 대체, 모달에서만 사용.
- 이미지는 base64로 전송: 대량 전송 시 IPC 부하 주의. `thumb_size=96` 고정 권장.
- `rcpvms_orbit_single` 응답은 `image_b64` 대신 `image_buffer`(Node Buffer)로 변환 후 전달됨. 렌더러는 `imagePayloadToSource`로 처리.
- Sticky worker는 `rcpvms_orbit*` 명령에만 적용. 다른 명령에는 일반 idle-first 배정.
