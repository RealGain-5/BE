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

## 앙상블 가중치 최적화 (신규)

두 모델 학습 완료 후, val split 예측값으로 최적 가중치 및 OOD 임계값을 탐색합니다.
결과는 `python/ensemble_config.json`에 자동 저장됩니다.

```
venv\Scripts\python.exe python\train_ensemble.py
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `../data` | data/raw 상위 디렉토리 |

**최적화 목표**: `score = val_acc × (1 − ood_fp_rate)`
- `alpha` (ResNet 가중치): 0.10 ~ 0.90, step=0.05
- `ood_threshold`: 0.45 ~ 0.85, step=0.025
- `tv_threshold`: 0.15 ~ 0.55, step=0.05

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

## 데이터 구성 (현행)

> **3600rpm 데이터 배제** (데이터 품질 문제 확인됨)
>
> | 소스 | 사용 여부 |
> |------|---------|
> | `data/raw/normal/` | ✅ 사용 |
> | `data/raw/normal_1200rpm/` | ✅ 사용 |
> | `data/raw/normal_3600rpm/` | ❌ **배제** |
> | `data/raw/abnormal/` | ✅ 사용 (OE) |
> | `data/synthetic/1200rpm/` | ✅ 사용 (validate) |
> | `data/synthetic/3600rpm/` | ❌ **배제** |

---

## 권장 실행 순서

```
1. venv/Scripts/python.exe python/train_1d_cnn.py        ← OrbitCNN1D 학습
2. venv/Scripts/python.exe python/train_multiscale.py     ← ResNet18 학습
3. venv/Scripts/python.exe python/train_ensemble.py       ← 앙상블 가중치 최적화 (신규)
4. venv/Scripts/python.exe python/evaluate_ensemble.py    ← 최종 성능 검증
```

---

## 성능 목표

| 지표 | 최소 합격선 | 권장 목표 |
|------|------------|----------|
| val_acc (4-class 합성) | 0.90 | 0.93 ~ 0.96 안정 |
| real_abnormal OOD 탐지율 | 0.60 | 0.75 이상 |
| OOD 오탐율 (정상·고장 → OOD 오분류) | — | 0.10 미만 |

> OOD 임계값: `python/ensemble_config.json` → `ood_threshold` (기본값 0.70)
