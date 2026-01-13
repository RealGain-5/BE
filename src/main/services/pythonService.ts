import { InferenceResult } from '../utils/pythonRunner'
import { PersistentPython } from '../utils/PersistentPython'
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
  private runner: PersistentPython
  private isInitialized: boolean = false
  private tempDirs: string[] = [] // 임시 디렉토리 추적
  private abortController: AbortController | null = null // 배치 취소용
  private maxConcurrent: number = 2  // 🆕 병렬 처리 수준 (기본값: 2)

  constructor() {
    this.runner = new PersistentPython()
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
   * 서비스 초기화 (데몬이 준비될 때까지 대기)
   */
  async init(): Promise<void> {
    if (this.isInitialized) {
      console.log('[PythonService] Already initialized')
      return
    }

    console.log('[PythonService] Initializing... waiting for Python daemon to be ready')

    // 🆕 데몬이 모델 로드를 완료할 때까지 대기
    await this.runner.waitUntilReady()

    this.isInitialized = true
    console.log('[PythonService] ✓ Ready (daemon initialized and model loaded)')
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
   * BIN 파일에 대해 모델 추론 실행 (PersistentPython 데몬 사용)
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

    console.log(`[PythonService] Running inference via PersistentPython daemon for: ${binPath}`)

    try {
      // PersistentPython 데몬에 analyze 명령 전송
      const response = await this.runner.sendCommand('analyze', { bin_path: binPath })

      console.log(`[PythonService] Inference completed: ${response.final_label}`)

      // 임시 디렉토리 생성 (Base64 이미지 저장용)
      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rcp-inference-'))
      this.tempDirs.push(tempDir)

      // Base64 이미지를 파일로 저장하고 visualization 구조 생성
      const visualization: any = {}

      for (const [rcp, imgData] of Object.entries(response.images)) {
        const orbitPath = this.saveBase64Image((imgData as any).orbit, `${rcp}_orbit.png`, tempDir)
        const overlayPath = this.saveBase64Image((imgData as any).overlay, `${rcp}_overlay.png`, tempDir)

        visualization[rcp] = {
          orbit: orbitPath,
          gradcam: {
            original: orbitPath,
            heatmap: overlayPath,
            overlay: overlayPath
          },
          temporal: []
        }
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

    console.log('[PythonService] Batch inference cancelled (daemon will continue running)')
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
   * 🆕 배치 추론 (순차 처리 - PersistentPython 데몬 사용)
   * @param binPaths BIN 파일 경로 배열
   * @param onProgress 진행 상황 콜백 (Incremental Update)
   * @returns 각 파일의 상태를 담은 Map (경량화)
   */
  async runBatchInferenceParallel(
    binPaths: string[],
    onProgress?: (progress: BatchProgress) => void
  ): Promise<Map<string, { success: boolean; error?: string }>> {

    console.log(`[PythonService] Starting batch inference with PersistentPython daemon: ${binPaths.length} files`)

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
    let completedCount = 0
    let failedCount = 0

    // 4. 순차 처리 (PersistentPython은 단일 프로세스 큐 방식)
    for (let i = 0; i < binPaths.length; i++) {
      const binPath = binPaths[i]

      // 취소 확인
      if (signal.aborted) {
        console.log('[PythonService] Batch inference was cancelled')
        results.set(binPath, { success: false, error: 'Cancelled by user' })
        failedCount++
        continue
      }

      // 진행 상황 업데이트 (시작)
      onProgress?.({
        total: binPaths.length,
        completed: completedCount,
        failed: failedCount,
        current: binPath,
        running: [binPath],
        runningCount: 1
      })

      console.log(`[PythonService] Processing ${i + 1}/${binPaths.length}: ${binPath}`)

      try {
        // runInference 호출 (데몬 방식)
        const result = await this.runInference(binPath)

        // 성공 처리
        results.set(binPath, { success: true })
        completedCount++
        console.log(`[PythonService] ✓ Success (${completedCount}/${binPaths.length}): ${binPath} → ${result.final_label}`)

        // 결과를 즉시 progress로 전송
        onProgress?.({
          total: binPaths.length,
          completed: completedCount,
          failed: failedCount,
          current: binPath,
          running: [],
          runningCount: 0,
          currentResult: result
        })

      } catch (error: any) {
        // 실패 처리
        if (!signal.aborted) {
          results.set(binPath, { success: false, error: error.message })
          failedCount++
          console.error(`[PythonService] ✗ Failed (${failedCount}): ${binPath}`, error.message)

          // 에러도 즉시 progress로 전송
          onProgress?.({
            total: binPaths.length,
            completed: completedCount,
            failed: failedCount,
            current: binPath,
            running: [],
            runningCount: 0,
            currentError: error.message
          })
        }

        // 취소된 경우 더 이상 진행하지 않음
        if (signal.aborted) {
          break
        }
        // 다른 에러는 계속 진행
      }
    }

    // 5. 완료 로그
    console.log(`[PythonService] Batch completed: ${completedCount} success, ${failedCount} failed`)

    // 6. 정리
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
   * 서비스 종료 (PersistentPython 데몬 프로세스 종료)
   */
  shutdown(): void {
    console.log('[PythonService] Shutting down')

    // PersistentPython 데몬 프로세스 종료
    this.runner.kill()

    // 종료 전 임시 파일 정리
    this.cleanup()

    this.isInitialized = false
  }
}

// Singleton 인스턴스
export const pythonService = new PythonService()
