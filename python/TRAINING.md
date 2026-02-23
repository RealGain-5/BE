# 학습 및 평가 명령어

> 가상환경: `rcp_5th/venv`
> 실행 위치: 프로젝트 루트 (`C:\Users\yunha\Desktop\rcp_5th`)

---

## 모델 학습

### 1. OrbitCNN1D (1D CNN)

- 입력: Raw XY 진동 신호 (2 × 40,000 samples)
- 출력: `python/model/orbit_cnn1d.pth`
- 파라미터: ~3M

```
venv/Scripts/python.exe python/train_1d_cnn.py
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `../data` | data/raw, data/synthetic 상위 디렉토리 |
| `--epochs` | `50` | 학습 에폭 수 |
| `--batch_size` | `32` | 배치 크기 |
| `--lr` | `5e-4` | 초기 학습률 |

---

### 2. 멀티스케일 ResNet18

- 입력: 3채널 멀티스케일 Orbit 이미지 (256 × 256, dynamic axis_lim)
- 출력: `python/model/resnet18_orbit_multiscale.pth`
- 파라미터: ~11M

```
venv\Scripts\python.exe python\train_multiscale.py
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `../data` | data/raw, data/synthetic 상위 디렉토리 |
| `--epochs` | `50` | 학습 에폭 수 |
| `--batch_size` | `32` | 배치 크기 |
| `--lr` | `1e-4` | 초기 학습률 |

---

## 앙상블 성능 평가

두 모델(`orbit_cnn1d.pth`, `resnet18_orbit_multiscale.pth`)이 모두 존재해야 합니다.

```
venv\Scripts\python.exe python\evaluate_ensemble.py
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `../data` | data/raw, data/synthetic 상위 디렉토리 |

---

## 합성 데이터 유효성 검증

```
venv\Scripts\python.exe python\validate_synthetic.py
```

---

## 권장 실행 순서

```
1. venv/Scripts/python.exe python/train_1d_cnn.py
2. venv/Scripts/python.exe python/train_multiscale.py
3. venv/Scripts/python.exe python/evaluate_ensemble.py
```

---

## 성능 목표

| 지표 | 최소 합격선 | 권장 목표 |
|------|------------|----------|
| val_acc (4-class 합성) | 0.90 | 0.93 ~ 0.96 안정 |
| real_abnormal OOD 탐지율 | 0.60 | 0.75 이상 |
| OOD 오탐율 (정상·고장 → OOD 오분류) | — | 0.10 미만 |

> OOD 임계값: `python/ensemble_config.json` → `ood_threshold` (기본값 0.70)
