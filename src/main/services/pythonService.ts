import { PythonRunner, InferenceResult } from '../utils/pythonRunner'
import fs from 'fs'
import path from 'path'

// BatchProgress 타입 정의
export interface BatchProgress {
  total: number
  completed: number
  failed: number
  current: string | null
  running?: string[]        // 현재 실행 중인 파일 경로 배열
  runningCount?: number     // 실행 중인 파일 개수
  currentResult?: InferenceResult  // 방금 완료된 파일의 결과
  currentError?: string    // 방금 실패한 파일의 에러
}

class PythonService {
  private runner: PythonRunner
  private isInitialized: boolean = false
  private tempDirs: string[] = [] // 임시 디렉토리 추적
  private abortController: AbortController | null = null // 배치 취소용
  private maxConcurrent: number = 2  // 🆕 병렬 처리 수준 (기본값: 2)

  constructor() {
    this.runner = new PythonRunner()
  }

  /**
   * 🆕 병렬 처리 수준 설정 (1-4)
   */
  setMaxConcurrent(value: number): void {
    this.maxConcurrent = Math.max(1, Math.min(value, 4))
    console.log(`[PythonService] Concurrency level set to ${this.maxConcurrent}`)
  }

  /**
   * 🆕 현재 병렬 처리 수준 반환
   */
  getMaxConcurrent(): number {
    return this.maxConcurrent
  }

  /**
   * 서비스 초기화
   */
  async init(): Promise<void> {
    console.log('[PythonService] Initializing...')
    this.isInitialized = true
    console.log('[PythonService] Ready')
  }

  /**
   * BIN 파일에 대해 모델 추론 실행
   * @param binPath BIN 파일 경로
   * @returns 추론 결과
   */
  async runInference(binPath: string): Promise<InferenceResult> {
    if (!this.isInitialized) {
      await this.init()
    }

    // 새로운 분석 시작 전 이전 임시 파일 정리
    if (this.tempDirs.length > 0) {
      console.log('[PythonService] Cleaning up previous temp files before new inference...')
      this.cleanup()
    }

    // 파일 존재 확인
    if (!fs.existsSync(binPath)) {
      throw new Error(`BIN file not found: ${binPath}`)
    }

    // 확장자 검증
    const ext = path.extname(binPath).toLowerCase()
    if (ext !== '.bin') {
      throw new Error(`Invalid file type: ${ext}. Only .BIN files are supported.`)
    }

    // 파일 크기 확인 (너무 큰 파일 방지)
    const stats = fs.statSync(binPath)
    const fileSizeMB = stats.size / (1024 * 1024)
    console.log(`[PythonService] File size: ${fileSizeMB.toFixed(2)} MB`)

    if (fileSizeMB > 500) {
      throw new Error(`File too large: ${fileSizeMB.toFixed(2)} MB (max 500 MB)`)
    }

    console.log(`[PythonService] Running inference for: ${binPath}`)

    try {
      const result = await this.runner.runInference(binPath)
      console.log(`[PythonService] Inference completed: ${result.final_label}`)

      // 임시 디렉토리 추적
      if (result.temp_dir) {
        this.tempDirs.push(result.temp_dir)
        console.log(`[PythonService] Tracking temp dir: ${result.temp_dir}`)
      }

      // 결과 로깅
      console.log('[PythonService] Results:')
      for (const [rcp, data] of Object.entries(result.results)) {
        console.log(
          `  ${rcp}: ${data.prediction} (normal: ${(data.probabilities.normal * 100).toFixed(1)}%, abnormal: ${(data.probabilities.abnormal * 100).toFixed(1)}%)`
        )
      }

      return result
    } catch (error: any) {
      console.error('[PythonService] Inference failed:', error)
      throw error
    }
  }

  /**
   * 배치 추론 취소
   */
  cancelBatchInference(): void {
    console.log('[PythonService] Cancelling batch inference...')
    
    // AbortSignal 발동
    if (this.abortController) {
      this.abortController.abort()
    }
    
    // 🔧 모든 프로세스 강제 종료
    this.runner.cancelAllInferences()
    
    console.log(`[PythonService] Cancelled ${this.runner.getRunningCount()} running processes`)
  }

  /**
   * 배치 추론 실행 (여러 BIN 파일 순차 처리)
   * @param binPaths BIN 파일 경로 배열
   * @param onProgress 진행 상황 콜백
   * @returns 각 파일의 결과 또는 에러를 담은 Map
   */
  async runBatchInference(
    binPaths: string[],
    onProgress?: (progress: {
      total: number
      completed: number
      failed: number
      current: string | null
    }) => void
  ): Promise<Map<string, InferenceResult | Error>> {
    console.log(`[PythonService] Starting batch inference for ${binPaths.length} files`)

    // 새로운 분석 시작 전 이전 임시 파일 정리
    if (this.tempDirs.length > 0) {
      console.log('[PythonService] Cleaning up previous temp files before new batch inference...')
      this.cleanup()
    }

    // 새로운 AbortController 생성
    this.abortController = new AbortController()
    const signal = this.abortController.signal

    const results = new Map<string, InferenceResult | Error>()
    let failedCount = 0

    for (let i = 0; i < binPaths.length; i++) {
      const binPath = binPaths[i]

      // 취소 확인
      if (signal.aborted) {
        console.log('[PythonService] Batch inference was cancelled')
        results.set(binPath, new Error('Cancelled by user'))
        failedCount++
        continue
      }

      // 진행 상황 콜백
      if (onProgress) {
        onProgress({
          total: binPaths.length,
          completed: i,
          failed: failedCount,
          current: binPath
        })
      }

      console.log(`[PythonService] Processing ${i + 1}/${binPaths.length}: ${binPath}`)

      try {
        // signal을 전달하여 취소 가능하도록
        const result = await this.runner.runInference(binPath, signal)
        results.set(binPath, result)
        console.log(`[PythonService] ✓ Success: ${binPath} → ${result.final_label}`)
        
        // 임시 디렉토리 추적
        if (result.temp_dir) {
          this.tempDirs.push(result.temp_dir)
          console.log(`[PythonService] Tracking temp dir: ${result.temp_dir}`)
        }
      } catch (error: any) {
        console.error(`[PythonService] ✗ Failed: ${binPath}`, error.message)
        results.set(binPath, error)
        failedCount++
        
        // 취소된 경우 더 이상 진행하지 않음
        if (signal.aborted) {
          break
        }
        // 다른 에러는 계속 진행
      }
    }

    // 최종 진행 상황
    if (onProgress) {
      onProgress({
        total: binPaths.length,
        completed: binPaths.length,
        failed: failedCount,
        current: null
      })
    }

    console.log(
      `[PythonService] Batch inference completed: ${binPaths.length - failedCount} success, ${failedCount} failed`
    )

    // AbortController 정리
    this.abortController = null

    return results
  }

  /**
   * 🆕 Pool 패턴 기반 병렬 배치 추론
   * @param binPaths BIN 파일 경로 배열
   * @param onProgress 진행 상황 콜백 (Incremental Update)
   * @returns 각 파일의 상태를 담은 Map (경량화)
   */
  async runBatchInferenceParallel(
    binPaths: string[],
    onProgress?: (progress: BatchProgress) => void
  ): Promise<Map<string, { success: boolean; error?: string }>> {
    
    console.log(`[PythonService] Starting parallel batch: ${binPaths.length} files, concurrency: ${this.maxConcurrent}`)
    
    // 1. 이전 임시 파일 정리
    if (this.tempDirs.length > 0) {
      console.log('[PythonService] Cleaning up previous temp files...')
      this.cleanup()
    }
    
    // 2. AbortController 초기화
    this.abortController = new AbortController()
    const signal = this.abortController.signal
    
    // 3. 상태 초기화
    const results = new Map<string, { success: boolean; error?: string }>()
    const queue = [...binPaths]  // 처리할 파일 큐
    const runningPromises = new Set<Promise<void>>()  // 실행 중인 Promise들
    const runningPaths = new Set<string>()  // 현재 실행 중인 파일 경로
    
    let completedCount = 0
    let failedCount = 0
    
    // 4. Worker 함수 (개별 파일 처리)
    const worker = async (binPath: string): Promise<void> => {
      runningPaths.add(binPath)
      
      // 진행 상황 업데이트 (시작)
      onProgress?.({
        total: binPaths.length,
        completed: completedCount,
        failed: failedCount,
        current: binPath,
        running: Array.from(runningPaths),
        runningCount: runningPaths.size
      })
      
      try {
        const result = await this.runner.runInference(binPath, signal)
        
        // 성공 처리 (메모리 절약: 상태만 저장)
        results.set(binPath, { success: true })
        
        // temp_dir 추적
        if (result.temp_dir) {
          this.tempDirs.push(result.temp_dir)
          console.log(`[PythonService] Tracking temp dir: ${result.temp_dir}`)
        }
        
        completedCount++
        console.log(`[PythonService] ✓ Success (${completedCount}/${binPaths.length}): ${binPath} → ${result.final_label}`)
        
        // 🆕 핵심: 결과를 즉시 progress로 전송 (IPC 분산)
        onProgress?.({
          total: binPaths.length,
          completed: completedCount,
          failed: failedCount,
          current: binPath,
          running: Array.from(runningPaths),
          runningCount: runningPaths.size,
          currentResult: result  // 👈 결과 즉시 전송!
        })
        
      } catch (error: any) {
        // 실패 처리
        if (!signal.aborted) {
          results.set(binPath, { success: false, error: error.message })
          failedCount++
          console.error(`[PythonService] ✗ Failed (${failedCount}): ${binPath}`, error.message)
          
          // 🆕 핵심: 에러도 즉시 progress로 전송
          onProgress?.({
            total: binPaths.length,
            completed: completedCount,
            failed: failedCount,
            current: binPath,
            running: Array.from(runningPaths),
            runningCount: runningPaths.size,
            currentError: error.message  // 👈 에러 즉시 전송!
          })
        }
        
      } finally {
        // 실행 목록에서 제거
        runningPaths.delete(binPath)
      }
    }
    
    // 5. 메인 루프: Pool 패턴 (Semaphore)
    while (queue.length > 0 || runningPromises.size > 0) {
      
      // 취소 확인
      if (signal.aborted) {
        console.log('[PythonService] Batch inference aborted')
        break
      }
      
      // 슬롯 채우기: 빈 슬롯이 있고 파일이 남았으면 투입
      while (runningPromises.size < this.maxConcurrent && queue.length > 0) {
        const binPath = queue.shift()!
        
        // Promise 생성 및 Set에 추가
        const promise = worker(binPath).finally(() => {
          // 완료되면 Set에서 자동 제거 (핵심!)
          runningPromises.delete(promise)
        })
        
        runningPromises.add(promise)
      }
      
      // 종료 조건: 큐도 비고 실행 중인 것도 없음
      if (runningPromises.size === 0 && queue.length === 0) {
        break
      }
      
      // 가장 먼저 끝나는 작업 하나를 기다림 (슬롯 확보)
      if (runningPromises.size > 0) {
        await Promise.race(runningPromises)
      }
    }
    
    // 6. 완료 로그
    console.log(`[PythonService] Batch completed: ${completedCount} success, ${failedCount} failed`)
    
    // 7. 정리
    this.abortController = null
    
    return results
  }

  /**
   * 임시 이미지 파일 정리
   */
  cleanup(): void {
    console.log(`[PythonService] Cleaning up ${this.tempDirs.length} temp directories...`)

    let successCount = 0
    let failCount = 0

    for (const dir of this.tempDirs) {
      try {
        if (fs.existsSync(dir)) {
          fs.rmSync(dir, { recursive: true, force: true })
          console.log(`[PythonService] ✓ Removed: ${dir}`)
          successCount++
        } else {
          console.log(`[PythonService] ⊘ Already removed: ${dir}`)
        }
      } catch (error: any) {
        console.error(`[PythonService] ✗ Failed to remove ${dir}:`, error.message)
        failCount++
      }
    }

    console.log(
      `[PythonService] Cleanup completed: ${successCount} success, ${failCount} failed`
    )

    // 배열 초기화
    this.tempDirs = []
  }

  /**
   * 서비스 종료
   */
  shutdown(): void {
    console.log('[PythonService] Shutting down')

    // 종료 전 임시 파일 정리
    this.cleanup()

    this.isInitialized = false
  }
}

// Singleton 인스턴스
export const pythonService = new PythonService()
