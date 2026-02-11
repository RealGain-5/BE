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
  currentResult?: InferenceResult  // 방금 완료된 파일의 결과
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
        model_path: 'model/resnet18_orbit_v3_None.pth',
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
   * 배치 추론 취소
   */
  cancelBatchInference(): void {
    console.log('[PythonService] Cancelling batch inference...')

    // AbortSignal 발동
    if (this.abortController) {
      this.abortController.abort()
    }

    console.log('[PythonService] Batch inference cancelled')
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
   * 배치 추론 (진정한 병렬 처리 - PythonDaemonPool 사용)
   * @param binPaths BIN 파일 경로 배열
   * @param onProgress 진행 상황 콜백 (Incremental Update)
   * @returns 각 파일의 상태를 담은 Map (경량화)
   */
  async runBatchInferenceParallel(
    binPaths: string[],
    onProgress?: (progress: BatchProgress) => void
  ): Promise<Map<string, { success: boolean; error?: string }>> {

    console.log(`[PythonService] Starting PARALLEL batch inference: ${binPaths.length} files, ${this.maxConcurrent} workers`)

    // 1. 초기화 확인
    if (!this.isInitialized) {
      await this.init()
    }

    // 2. 이전 임시 파일 정리
    if (this.tempDirs.length > 0) {
      console.log('[PythonService] Cleaning up previous temp files...')
      this.cleanup()
    }

    // 3. AbortController 초기화
    this.abortController = new AbortController()
    const signal = this.abortController.signal

    // 4. 상태 초기화
    const results = new Map<string, { success: boolean; error?: string }>()
    let completedCount = 0
    let failedCount = 0
    const runningJobs = new Map<string, Promise<void>>()

    // 5. 병렬 처리 (Semaphore 패턴)
    for (let i = 0; i < binPaths.length; i++) {
      const binPath = binPaths[i]

      // 취소 확인
      if (signal.aborted) {
        console.log('[PythonService] Batch inference was cancelled')
        break
      }

      // 동시 실행 제한: maxConcurrent만큼 실행 중이면 하나가 완료될 때까지 대기
      if (runningJobs.size >= this.maxConcurrent) {
        await Promise.race(runningJobs.values())
      }

      // 취소 재확인 (대기 후)
      if (signal.aborted) {
        break
      }

      // 새 작업 시작
      const jobPromise = (async () => {
        try {
          // 진행 상황 업데이트 (시작)
          onProgress?.({
            total: binPaths.length,
            completed: completedCount,
            failed: failedCount,
            current: binPath,
            running: Array.from(runningJobs.keys()),
            runningCount: runningJobs.size
          })

          console.log(`[PythonService] [${runningJobs.size}/${this.maxConcurrent}] Starting: ${binPath}`)

          // 추론 실행 (풀의 유휴 워커에 자동 분배)
          const result = await this.runInference(binPath)

          // 성공 처리
          results.set(binPath, { success: true })
          completedCount++
          console.log(`[PythonService] ✓ Completed (${completedCount}/${binPaths.length}): ${binPath} → ${result.final_label}`)

          // 결과를 즉시 progress로 전송
          onProgress?.({
            total: binPaths.length,
            completed: completedCount,
            failed: failedCount,
            current: binPath,
            running: Array.from(runningJobs.keys()).filter(k => k !== binPath),
            runningCount: runningJobs.size - 1,
            currentResult: result
          })

        } catch (error: any) {
          // 실패 처리
          results.set(binPath, { success: false, error: error.message })
          failedCount++
          console.error(`[PythonService] ✗ Failed (${failedCount}): ${binPath}`, error.message)

          // 에러도 즉시 progress로 전송
          onProgress?.({
            total: binPaths.length,
            completed: completedCount,
            failed: failedCount,
            current: binPath,
            running: Array.from(runningJobs.keys()).filter(k => k !== binPath),
            runningCount: runningJobs.size - 1,
            currentError: error.message
          })
        } finally {
          runningJobs.delete(binPath)
        }
      })()

      runningJobs.set(binPath, jobPromise)
    }

    // 6. 남은 작업 완료 대기
    if (runningJobs.size > 0) {
      console.log(`[PythonService] Waiting for ${runningJobs.size} remaining jobs...`)
      await Promise.all(runningJobs.values())
    }

    // 7. 완료 로그
    console.log(`[PythonService] Batch completed: ${completedCount} success, ${failedCount} failed`)

    // 8. 정리
    this.abortController = null

    return results
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
