#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCP Inference Performance Benchmark

Python Daemon 병렬 처리 성능을 측정하는 벤치마크 스크립트입니다.

사용법:
    python benchmarks/benchmark_inference.py --bin-dir <BIN_FILES_DIR> [--num-files N] [--workers 1,2,4]

예시:
    python benchmarks/benchmark_inference.py --bin-dir D:/data/rcp_samples --num-files 8 --workers 1,2,4
"""

import sys
import os
import time
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json

# 프로젝트 python 폴더를 경로에 추가
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
PYTHON_DIR = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

# Python 모듈 임포트
try:
    import torch
    from model_loader import load_trained_model
    from infer_resnet_None import (
        make_orbit_pils_sec9_from_bin,
        predict_rcp_single,
        generate_gradcam_images,
    )
    from utils import image_to_base64
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print("Please run this script from the project root directory.")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """단일 추론 결과"""
    file_path: str
    elapsed_time: float  # seconds
    success: bool
    error: str = ""
    final_label: str = ""


@dataclass
class BatchResult:
    """배치 벤치마크 결과"""
    num_workers: int
    num_files: int
    total_time: float
    avg_time_per_file: float
    throughput: float  # files per second
    speedup: float  # vs sequential
    results: List[BenchmarkResult]


def load_model_once():
    """모델을 한 번 로드하고 반환"""
    # 워커 프로세스에서도 동작하도록 동적 import
    from model_loader import load_trained_model as _load_trained_model
    
    model_path = PYTHON_DIR / "model" / "resnet18_orbit_v3_None.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = _load_trained_model(str(model_path))
    model.to(device)
    model.eval()
    
    return model, class_names, device


def run_single_inference(bin_path: str, model, class_names) -> BenchmarkResult:
    """단일 BIN 파일에 대해 추론 수행 (GradCAM 포함)"""
    start_time = time.perf_counter()
    
    try:
        # 1. BIN 파일에서 orbit 이미지 생성
        rcp_to_pil = make_orbit_pils_sec9_from_bin(bin_path)
        
        results = {}
        images_b64 = {}
        
        # 2. 각 RCP에 대해 추론 + GradCAM 수행
        for rcp, pil_img in rcp_to_pil.items():
            # 추론
            pred_class, prob = predict_rcp_single(model, class_names, pil_img)
            results[rcp] = {
                "prediction": pred_class,
                "probabilities": {
                    name: float(p) for name, p in zip(class_names, prob)
                },
            }
            
            # GradCAM 생성
            gradcam_imgs = generate_gradcam_images(model, class_names, pil_img)
            
            # Base64 인코딩
            images_b64[rcp] = {
                "orbit": image_to_base64(pil_img),
                "overlay": image_to_base64(gradcam_imgs["overlay"]),
            }
        
        # 3. 최종 라벨 결정
        final_label = (
            "abnormal"
            if any(r["prediction"] == "abnormal" for r in results.values())
            else "normal"
        )
        
        elapsed = time.perf_counter() - start_time
        
        return BenchmarkResult(
            file_path=bin_path,
            elapsed_time=elapsed,
            success=True,
            final_label=final_label
        )
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return BenchmarkResult(
            file_path=bin_path,
            elapsed_time=elapsed,
            success=False,
            error=str(e)
        )


# 워커 프로세스용 전역 변수
_worker_model = None
_worker_class_names = None
_worker_initialized = False


def init_worker(python_dir: str):
    """워커 프로세스 초기화 (경로 설정 + 모델 로드)"""
    global _worker_model, _worker_class_names, _worker_initialized
    
    if _worker_initialized:
        return
    
    # 워커 프로세스에서 sys.path 설정
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    
    # 워커 프로세스에서 모듈 재import
    global make_orbit_pils_sec9_from_bin, predict_rcp_single, generate_gradcam_images, image_to_base64
    from infer_resnet_None import (
        make_orbit_pils_sec9_from_bin,
        predict_rcp_single,
        generate_gradcam_images,
    )
    from utils import image_to_base64
    
    # 모델 로드
    _worker_model, _worker_class_names, _ = load_model_once()
    _worker_initialized = True
    print(f"[Worker {os.getpid()}] Initialized and model loaded")


def worker_inference(args: Tuple[str, str]) -> BenchmarkResult:
    """워커 프로세스에서 실행되는 추론 함수"""
    bin_path, python_dir = args
    
    # 초기화 확인 (첫 호출 시 초기화)
    init_worker(python_dir)
    
    global _worker_model, _worker_class_names
    return run_single_inference(bin_path, _worker_model, _worker_class_names)


def benchmark_sequential(bin_files: List[str], model, class_names) -> BatchResult:
    """순차 처리 벤치마크"""
    print(f"\n{'='*60}")
    print(f"[Sequential] Processing {len(bin_files)} files...")
    print(f"{'='*60}")
    
    results = []
    start_time = time.perf_counter()
    
    for i, bin_path in enumerate(bin_files):
        print(f"  [{i+1}/{len(bin_files)}] {Path(bin_path).name}...", end=" ", flush=True)
        result = run_single_inference(bin_path, model, class_names)
        results.append(result)
        
        status = "✓" if result.success else "✗"
        print(f"{status} {result.elapsed_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    
    return BatchResult(
        num_workers=1,
        num_files=len(bin_files),
        total_time=total_time,
        avg_time_per_file=total_time / len(bin_files),
        throughput=len(bin_files) / total_time,
        speedup=1.0,
        results=results
    )


def benchmark_parallel(bin_files: List[str], num_workers: int, sequential_time: float) -> BatchResult:
    """병렬 처리 벤치마크"""
    print(f"\n{'='*60}")
    print(f"[Parallel: {num_workers} workers] Processing {len(bin_files)} files...")
    print(f"{'='*60}")
    
    results = []
    start_time = time.perf_counter()
    python_dir_str = str(PYTHON_DIR)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 모든 작업 제출 (python_dir 경로도 함께 전달)
        future_to_path = {
            executor.submit(worker_inference, (bin_path, python_dir_str)): bin_path 
            for bin_path in bin_files
        }
        
        # 완료된 순서대로 결과 수집
        completed = 0
        for future in as_completed(future_to_path):
            completed += 1
            bin_path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result.success else "✗"
                print(f"  [{completed}/{len(bin_files)}] {Path(bin_path).name} {status} {result.elapsed_time:.2f}s")
            except Exception as e:
                print(f"  [{completed}/{len(bin_files)}] {Path(bin_path).name} ✗ Error: {e}")
                results.append(BenchmarkResult(
                    file_path=bin_path,
                    elapsed_time=0,
                    success=False,
                    error=str(e)
                ))
    
    total_time = time.perf_counter() - start_time
    speedup = sequential_time / total_time if total_time > 0 else 0
    
    return BatchResult(
        num_workers=num_workers,
        num_files=len(bin_files),
        total_time=total_time,
        avg_time_per_file=total_time / len(bin_files),
        throughput=len(bin_files) / total_time,
        speedup=speedup,
        results=results
    )


def print_summary(results: List[BatchResult]):
    """벤치마크 결과 요약 출력"""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    # 테이블 헤더
    print(f"\n{'Workers':<10} {'Total Time':<12} {'Avg/File':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for r in results:
        throughput_str = f"{r.throughput:.3f} files/s"
        speedup_str = f"{r.speedup:.2f}x"
        print(f"{r.num_workers:<10} {r.total_time:<12.2f}s {r.avg_time_per_file:<12.2f}s {throughput_str:<15} {speedup_str:<10}")
    
    # 최적 설정 추천
    if len(results) > 1:
        best = max(results, key=lambda x: x.speedup)
        print(f"\n✓ Best configuration: {best.num_workers} workers ({best.speedup:.2f}x speedup)")
    
    # 성공/실패 통계
    print(f"\n{'='*70}")
    print("DETAILED STATISTICS")
    print(f"{'='*70}")
    
    for r in results:
        success_count = sum(1 for res in r.results if res.success)
        fail_count = len(r.results) - success_count
        
        if r.results:
            times = [res.elapsed_time for res in r.results if res.success]
            if times:
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                min_time = min(times)
                max_time = max(times)
                
                print(f"\n[{r.num_workers} Worker(s)]")
                print(f"  Success: {success_count}, Failed: {fail_count}")
                print(f"  Per-file time: avg={avg_time:.2f}s, std={std_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")


def find_bin_files(directory: str, num_files: int = None) -> List[str]:
    """디렉토리에서 BIN 파일 찾기"""
    bin_dir = Path(directory)
    if not bin_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    bin_files = list(bin_dir.glob("*.bin")) + list(bin_dir.glob("*.BIN"))
    bin_files = [str(f) for f in bin_files]
    
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in: {directory}")
    
    if num_files:
        bin_files = bin_files[:num_files]
    
    return bin_files


def main():
    parser = argparse.ArgumentParser(
        description="RCP Inference Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/benchmark_inference.py --bin-dir D:/data/samples --num-files 8
  python benchmarks/benchmark_inference.py --bin-dir ./test_data --workers 1,2,4
  python benchmarks/benchmark_inference.py --bin-file sample.bin --repeat 5
        """
    )
    
    parser.add_argument(
        "--bin-dir", 
        type=str, 
        help="Directory containing .bin files"
    )
    parser.add_argument(
        "--bin-file",
        type=str,
        help="Single .bin file to benchmark (will repeat N times)"
    )
    parser.add_argument(
        "--num-files", 
        type=int, 
        default=8,
        help="Number of files to process (default: 8)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat single file N times for benchmark (with --bin-file)"
    )
    parser.add_argument(
        "--workers", 
        type=str, 
        default="1,2,4",
        help="Comma-separated list of worker counts to test (default: 1,2,4)"
    )
    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        help="Skip sequential benchmark (use first parallel result as baseline)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # 입력 검증
    if not args.bin_dir and not args.bin_file:
        parser.error("Either --bin-dir or --bin-file is required")
    
    # BIN 파일 목록 생성
    if args.bin_file:
        bin_file = Path(args.bin_file)
        if not bin_file.exists():
            print(f"[ERROR] File not found: {args.bin_file}")
            sys.exit(1)
        bin_files = [str(bin_file)] * args.repeat
        print(f"[INFO] Single file mode: {bin_file.name} x {args.repeat}")
    else:
        bin_files = find_bin_files(args.bin_dir, args.num_files)
    
    print(f"\n{'#'*70}")
    print("RCP INFERENCE PERFORMANCE BENCHMARK")
    print(f"{'#'*70}")
    print(f"\nFiles to process: {len(bin_files)}")
    print(f"Worker configurations: {args.workers}")
    
    # 환경 정보
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 워커 수 파싱
    worker_counts = [int(w.strip()) for w in args.workers.split(",")]
    
    # 모델 로드 (순차 벤치마크용)
    print("\n[INFO] Loading model for sequential benchmark...")
    model, class_names, _ = load_model_once()
    print("[INFO] Model loaded successfully")
    
    # 벤치마크 실행
    all_results = []
    
    # 순차 벤치마크
    if not args.skip_sequential and 1 not in worker_counts:
        worker_counts = [1] + worker_counts
    
    sequential_time = None
    
    for num_workers in worker_counts:
        if num_workers == 1 and not args.skip_sequential:
            # 순차 처리
            result = benchmark_sequential(bin_files, model, class_names)
            sequential_time = result.total_time
        else:
            # 병렬 처리
            if sequential_time is None:
                # 순차 벤치마크 건너뛴 경우, 첫 병렬 결과를 기준으로 사용
                sequential_time = float('inf')
            result = benchmark_parallel(bin_files, num_workers, sequential_time)
            
            # 첫 병렬 결과를 기준으로 설정 (skip-sequential인 경우)
            if sequential_time == float('inf'):
                sequential_time = result.total_time
                result.speedup = 1.0
        
        all_results.append(result)
    
    # 결과 요약
    print_summary(all_results)
    
    # JSON 출력
    if args.output:
        output_data = {
            "config": {
                "num_files": len(bin_files),
                "device": device,
                "workers_tested": worker_counts
            },
            "results": [
                {
                    "num_workers": r.num_workers,
                    "total_time": r.total_time,
                    "avg_time_per_file": r.avg_time_per_file,
                    "throughput": r.throughput,
                    "speedup": r.speedup,
                    "success_count": sum(1 for res in r.results if res.success),
                    "fail_count": sum(1 for res in r.results if not res.success)
                }
                for r in all_results
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[INFO] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
