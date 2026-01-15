# RCP Inference Performance Benchmark

Python Daemon 병렬 처리 성능을 측정하는 벤치마크 도구입니다.

## 요구사항

- Python 3.8+
- PyTorch
- 프로젝트 의존성 설치 완료 (`python/requirements.txt`)

## 사용법

### 기본 사용

```bash
# 프로젝트 루트에서 실행
python benchmarks/benchmark_inference.py --bin-dir <BIN_파일_디렉토리>
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--bin-dir` | BIN 파일이 있는 디렉토리 경로 | (필수) |
| `--bin-file` | 단일 BIN 파일 경로 (반복 테스트용) | - |
| `--num-files` | 테스트할 파일 개수 | 8 |
| `--repeat` | 단일 파일 반복 횟수 (`--bin-file`과 함께 사용) | 1 |
| `--workers` | 테스트할 워커 수 (쉼표 구분) | 1,2,4 |
| `--skip-sequential` | 순차 처리 벤치마크 건너뛰기 | False |
| `--output` | 결과 JSON 파일 경로 | - |

### 예시

```bash
# 디렉토리의 BIN 파일 8개로 벤치마크 (워커 1,2,4 테스트)
python benchmarks/benchmark_inference.py --bin-dir D:/data/rcp_samples --num-files 8

# 단일 파일 10회 반복 테스트
python benchmarks/benchmark_inference.py --bin-file D:/data/sample.bin --repeat 10

# 2, 4 워커만 테스트 (순차 처리 건너뛰기)
python benchmarks/benchmark_inference.py --bin-dir ./test_data --workers 2,4 --skip-sequential

# 결과를 JSON으로 저장
python benchmarks/benchmark_inference.py --bin-dir ./test_data --output results.json
```

## 출력 예시

```
======================================================================
BENCHMARK SUMMARY
======================================================================

Workers    Total Time   Avg/File     Throughput      Speedup   
------------------------------------------------------------
1          120.45s      15.06s       0.066 files/s   1.00x     
2          62.33s       7.79s        0.128 files/s   1.93x     
4          35.12s       4.39s        0.228 files/s   3.43x     

✓ Best configuration: 4 workers (3.43x speedup)
```

## 측정 항목

1. **Total Time**: 전체 배치 처리 시간
2. **Avg/File**: 파일당 평균 처리 시간
3. **Throughput**: 초당 처리 파일 수
4. **Speedup**: 순차 처리 대비 속도 향상 배율

## 주의사항

- GPU 메모리가 제한적인 경우, 워커 수를 줄이세요
- 각 워커는 별도의 모델 인스턴스를 로드합니다 (~250MB/워커)
- 첫 실행 시 모델 로드 시간이 포함됩니다
