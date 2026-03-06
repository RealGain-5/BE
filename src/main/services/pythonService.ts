import { InferenceResult } from '../utils/pythonRunner'
import { PythonDaemonPool } from '../utils/PythonDaemonPool'
import fs from 'fs'
import path from 'path'
import os from 'os'

// BatchProgress 타입 정의
export interface BatchProgress {
  total: number
  completed: number
  failed: number
  current: string | null
  running?: string[]        // 현재 실행 중인 파일 경로 배열
  runningCount?: number     // 실행 중인 파일 개수
  currentResult?: any              // 방금 완료된 파일의 결과
  currentError?: string    // 방금 실패한 파일의 에러
}

class PythonService {
  private pool: PythonDaemonPool
  private isInitialized: boolean = false
  private tempDirs: string[] = [] // 임시 디렉토리 추적
  private abortController: AbortController | null = null // 배치 취소용
  private maxConcurrent: number = 2  // 병렬 처리 수준 (기본값: 2)

  constructor() {
    this.pool = new PythonDaemonPool()
  }

  /**
   * 병렬 처리 수준 설정 (1-4) - 풀 크기 동적 조정
   */
  setMaxConcurrent(value: number): void {
    const newValue = Math.max(1, Math.min(value, 4))
    console.log(`[PythonService] Setting concurrency level to ${newValue}`)
    this.maxConcurrent = newValue

    // 이미 초기화된 경우 풀 크기 동적 조정
    if (this.isInitialized) {
      this.pool.resize(newValue).catch((err) => {
        console.error('[PythonService] Failed to resize pool:', err)
      })
    }
  }

  /**
   * 현재 병렬 처리 수준 반환
   */
  getMaxConcurrent(): number {
    return this.maxConcurrent
  }

  /**
   * 서비스 초기화 (데몬 풀이 준비될 때까지 대기)
   */
  async init(): Promise<void> {
    if (this.isInitialized) {
      console.log('[PythonService] Already initialized')
      return
    }

    console.log(`[PythonService] Initializing daemon pool with ${this.maxConcurrent} workers...`)

    // 데몬 풀 초기화 (모든 워커의 모델 로드 완료 대기)
    await this.pool.init(this.maxConcurrent)

    this.isInitialized = true
    console.log(`[PythonService] ✓ Ready (${this.maxConcurrent} daemon workers initialized)`)
  }

  /**
   * BIN 파일 경로 유효성 검증 (존재 여부 + 확장자)
   */
  private validateBinFile(binPath: string): void {
    if (!fs.existsSync(binPath)) {
      throw new Error(`BIN file not found: ${binPath}`)
    }
    const ext = path.extname(binPath).toLowerCase()
    if (ext !== '.bin') {
      throw new Error(`Invalid file type: ${ext}. Only .BIN files are supported.`)
    }
  }

  /**
   * Base64 이미지를 임시 파일로 저장하는 헬퍼 함수
   */
  private saveBase64Image(base64Data: string, filename: string, tempDir: string): string {
    // Base64 디코딩
    const base64Image = base64Data.replace(/^data:image\/\w+;base64,/, '')
    const imageBuffer = Buffer.from(base64Image, 'base64')

    // 파일 저장
    const filePath = path.join(tempDir, filename)
    fs.writeFileSync(filePath, imageBuffer)

    return filePath
  }

  /**
   * BIN 파일에 대해 모델 추론 실행 (PythonDaemonPool 사용)
   * @param binPath BIN 파일 경로
   * @returns 추론 결과
   */
  async runInference(binPath: string): Promise<InferenceResult> {
    if (!this.isInitialized) {
      await this.init()
    }

    this.validateBinFile(binPath)

    // 파일 크기 확인 (너무 큰 파일 방지)
    const stats = fs.statSync(binPath)
    const fileSizeMB = stats.size / (1024 * 1024)

    if (fileSizeMB > 500) {
      throw new Error(`File too large: ${fileSizeMB.toFixed(2)} MB (max 500 MB)`)
    }

    console.log(`[PythonService] Running inference via DaemonPool for: ${binPath}`)

    try {
      // DaemonPool에 analyze 명령 전송 (유휴 워커에 자동 분배)
      const response = await this.pool.sendCommand('analyze', { bin_path: binPath })

      console.log(`[PythonService] Inference completed: ${response.final_label}`)

      // 임시 디렉토리 생성 (Base64 이미지 저장용)
      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rcp-inference-'))
      this.tempDirs.push(tempDir)

      // Base64 이미지를 파일로 저장하고 visualization 구조 생성
      const visualization: any = {}

      for (const [rcp, imgData] of Object.entries(response.images)) {
        const orbitPath = this.saveBase64Image((imgData as any).orbit, `${rcp}_orbit.png`, tempDir)
        const heatmapPath = this.saveBase64Image((imgData as any).heatmap, `${rcp}_heatmap.png`, tempDir)
        const overlayPath = this.saveBase64Image((imgData as any).overlay, `${rcp}_overlay.png`, tempDir)

        visualization[rcp] = {
          orbit: orbitPath,
          gradcam: {
            original: orbitPath,
            heatmap: heatmapPath,
            overlay: overlayPath
          },
          temporal: []
        }

        // IG 이미지 (있을 때만 저장)
        const igData: any = {}
        if ((imgData as any).ig_resnet_heatmap) {
          igData.resnet_heatmap = this.saveBase64Image(
            (imgData as any).ig_resnet_heatmap, `${rcp}_ig_resnet_heatmap.png`, tempDir
          )
          igData.resnet_overlay = this.saveBase64Image(
            (imgData as any).ig_resnet_overlay, `${rcp}_ig_resnet_overlay.png`, tempDir
          )
        }
        if (Object.keys(igData).length > 0) {
          visualization[rcp].ig = igData
        }
      }

      // Timeline 이미지 생성 및 저장
      try {
        const timelineResponse = await this.pool.sendCommand('timeline', { bin_path: binPath })

        for (const [rcp, imgList] of Object.entries(timelineResponse)) {
          const temporalPaths = (imgList as string[]).map((b64, i) =>
            this.saveBase64Image(b64, `${rcp}_temporal_${i}.png`, tempDir)
          )
          if (visualization[rcp]) {
            visualization[rcp].temporal = temporalPaths
          }
        }
      } catch (timelineError: any) {
        console.error('[PythonService] Timeline generation failed:', timelineError.message)
      }

      // InferenceResult 포맷으로 변환
      const result: InferenceResult = {
        bin_path: binPath,
        model_path: binPath,   // 분석 대상 파일 경로
        model_info: response.model_info ?? 'ensemble (multiscale + 1d_cnn)',
        final_label: response.final_label as 'normal' | 'abnormal',
        results: response.results,
        visualization,
        temp_dir: tempDir
      }

      return result
    } catch (error: any) {
      console.error('[PythonService] Inference failed:', error)
      throw error
    }
  }

  /**
   * MAE 이상 탐지 실행
   * @param binPath BIN 파일 경로
   * @returns MAE 분석 결과 (data 필드 그대로 반환)
   */
  async runMAEAnalysis(binPath: string): Promise<any> {
    if (!this.isInitialized) {
      await this.init()
    }

    this.validateBinFile(binPath)

    console.log(`[PythonService] Running MAE analysis for: ${binPath}`)

    const response = await this.pool.sendCommand('mae_analyze', { bin_path: binPath })
    console.log(`[PythonService] MAE analysis completed: ${response.final_verdict}`)
    return response
  }

  /**
   * MAE FP 배치 평가 (이미지 없는 경량 분석, 정상 파일의 오탐율 측정)
   */
  async runMAEBatchFP(
    binPaths: string[],
    onProgress?: (p: { current: string; completed: number; total: number; fp: number }) => void
  ): Promise<{ results: any[]; fpCount: number; fpRate: number; total: number }> {
    if (!this.isInitialized) await this.init()
    this.fpAbortController = new AbortController()
    const signal = this.fpAbortController.signal

    const results: any[] = []
    let fpCount = 0

    for (let i = 0; i < binPaths.length; i++) {
      if (signal.aborted) break
      const binPath = binPaths[i]
      try {
        const response = await this.pool.sendCommand('mae_fp_check', { bin_path: binPath })
        const data = response.data
        const isFP = data.final_verdict === 'anomaly'
        if (isFP) fpCount++
        results.push({
          path: binPath,
          fileName: path.basename(binPath),
          verdict: data.final_verdict,
          maxNorm: data.max_normalized_score,
          fpRcps: isFP
            ? Object.entries(data.results as Record<string, any>)
                .filter(([, v]) => v.is_anomaly)
                .map(([k]) => k)
            : [],
          rcpScores: Object.fromEntries(
            Object.entries(data.results as Record<string, any>).map(([k, v]) => [k, v.normalized_score])
          ),
        })
      } catch (err: any) {
        results.push({ path: binPath, fileName: path.basename(binPath), verdict: 'error', maxNorm: 0, fpRcps: [], error: err.message })
      }
      onProgress?.({ current: path.basename(binPath), completed: i + 1, total: binPaths.length, fp: fpCount })
    }

    const validCount = results.filter(r => r.verdict !== 'error').length
    return { results, fpCount, fpRate: validCount > 0 ? fpCount / validCount : 0, total: results.length }
  }


  /** FP 배치 평가 취소 */
  private fpAbortController: AbortController | null = null
  cancelFPBatch(): void {
    this.fpAbortController?.abort()
  }

  /**
   * MAE 배치 분석 (단일 이미지 포함 전체 분석, orbit 배치와 동일 방식)
   */
  private maeAbortController: AbortController | null = null

  async runMAEBatch(
    binPaths: string[],
    onProgress?: (p: BatchProgress) => void
  ): Promise<void> {
    if (!this.isInitialized) await this.init()
    this.maeAbortController = new AbortController()
    const result = await this.runParallelBatch(
      binPaths,
      (p) => this.runMAEAnalysis(p),
      this.maeAbortController.signal,
      'MAE batch',
      onProgress
    )
    this.maeAbortController = null
    console.log(`[PythonService] MAE batch completed: ${result.completed} success, ${result.failed} failed`)
  }

  cancelMAEBatch(): void {
    // 새 작업 투입을 중단하고, 아직 워커에 전달되지 않은 대기 중인 작업을 취소
    this.maeAbortController?.abort()
    this.pool.cancelPendingJobs()
    // 주의: 이미 Python 워커에서 실행 중인 작업은 완료될 때까지 중단되지 않음
    // (Python 프로세스를 종료하지 않고는 in-flight 작업을 중단할 수 없음)
  }


  /**
   * 배치 추론 취소
   */
  cancelBatchInference(): void {
    console.log('[PythonService] Cancelling batch inference...')
    // 새 작업 투입을 중단하고, 아직 워커에 전달되지 않은 대기 중인 작업을 취소
    if (this.abortController) {
      this.abortController.abort()
    }
    this.pool.cancelPendingJobs()
    // 주의: 이미 Python 워커에서 실행 중인 작업은 완료될 때까지 중단되지 않음
    console.log('[PythonService] Batch inference cancelled')
  }

  /**
   * 공통 병렬 배치 실행 (Semaphore 패턴)
   * runBatchInferenceParallel / runMAEBatch 공유 핵심 로직
   */
  private async runParallelBatch(
    binPaths: string[],
    runner: (binPath: string) => Promise<any>,
    abortSignal: AbortSignal,
    logLabel: string,
    onProgress?: (p: BatchProgress) => void
  ): Promise<{ completed: number; failed: number }> {
    let completed = 0
    let failed = 0
    const runningJobs = new Map<string, Promise<void>>()

    console.log(`[PythonService] Starting PARALLEL ${logLabel}: ${binPaths.length} files, ${this.maxConcurrent} workers`)

    for (let i = 0; i < binPaths.length; i++) {
      if (abortSignal.aborted) break
      const binPath = binPaths[i]

      while (runningJobs.size >= this.maxConcurrent) {
        await Promise.race(runningJobs.values())
      }

      if (abortSignal.aborted) break

      const jobPromise = (async () => {
        try {
          onProgress?.({
            total: binPaths.length, completed, failed,
            current: binPath,
            running: Array.from(runningJobs.keys()),
            runningCount: runningJobs.size
          })
          const result = await runner(binPath)
          completed++
          onProgress?.({
            total: binPaths.length, completed, failed,
            current: binPath,
            running: Array.from(runningJobs.keys()).filter(k => k !== binPath),
            runningCount: runningJobs.size - 1,
            currentResult: result
          })
        } catch (err: any) {
          failed++
          onProgress?.({
            total: binPaths.length, completed, failed,
            current: binPath,
            running: Array.from(runningJobs.keys()).filter(k => k !== binPath),
            runningCount: runningJobs.size - 1,
            currentError: err.message
          })
        } finally {
          runningJobs.delete(binPath)
        }
      })()

      runningJobs.set(binPath, jobPromise)
    }

    if (runningJobs.size > 0) {
      await Promise.all(runningJobs.values())
    }

    return { completed, failed }
  }

  /**
   * 배치 추론 실행 (여러 BIN 파일 순차 처리) - 레거시 호환
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
        // runInference 호출 (데몬 방식)
        const result = await this.runInference(binPath)
        results.set(binPath, result)
        console.log(`[PythonService] ✓ Success: ${binPath} → ${result.final_label}`)
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
   * 배치 추론 (병렬 처리 - PythonDaemonPool 사용)
   */
  async runBatchInferenceParallel(
    binPaths: string[],
    onProgress?: (progress: BatchProgress) => void
  ): Promise<{ completed: number; failed: number }> {
    if (!this.isInitialized) await this.init()

    if (this.tempDirs.length > 0) {
      console.log('[PythonService] Cleaning up previous temp files...')
      this.cleanup()
    }

    this.abortController = new AbortController()
    const result = await this.runParallelBatch(
      binPaths,
      (p) => this.runInference(p),
      this.abortController.signal,
      'batch inference',
      onProgress
    )
    this.abortController = null
    console.log(`[PythonService] Batch completed: ${result.completed} success, ${result.failed} failed`)
    return result
  }

  /**
   * 풀 상태 조회
   */
  getPoolStatus(): {
    totalWorkers: number
    idleWorkers: number
    busyWorkers: number
    pendingJobs: number
  } {
    return this.pool.getStatus()
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
          successCount++
        }
      } catch (error: any) {
        console.error(`[PythonService] ✗ Failed to remove ${dir}:`, error.message)
        failCount++
      }
    }

    if (successCount > 0 || failCount > 0) {
      console.log(`[PythonService] Cleanup: ${successCount} removed, ${failCount} failed`)
    }

    // 배열 초기화
    this.tempDirs = []
  }

  /**
   * 서비스 종료 (PythonDaemonPool 종료)
   */
  shutdown(): void {
    console.log('[PythonService] Shutting down')

    // DaemonPool 종료
    this.pool.shutdown()

    // 종료 전 임시 파일 정리
    this.cleanup()

    this.isInitialized = false
  }
}

// Singleton 인스턴스
export const pythonService = new PythonService()
