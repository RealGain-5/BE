import { spawn, ChildProcess } from 'child_process'
import path from 'path'
import { app } from 'electron'

export interface VisualizationData {
  orbit: string
  gradcam: {
    original: string
    heatmap: string
    overlay: string
  }
  temporal: string[]
}

export interface InferenceResult {
  bin_path: string
  model_path: string
  final_label: 'normal' | 'abnormal'
  results: {
    [rcp: string]: {
      prediction: string
      probabilities: { [className: string]: number }
    }
  }
  visualization?: {
    [rcp: string]: VisualizationData
  }
  temp_dir?: string
}

export class PythonRunner {
  private executablePath: string
  private isProduction: boolean
  private runningProcesses: Set<ChildProcess> = new Set() // 🆕 다중 프로세스 관리

  constructor() {
    this.isProduction = app.isPackaged

    // 개발 환경: Python 스크립트 직접 실행
    if (!this.isProduction) {
      this.executablePath = '' // spawn에서 'python' 명령어 사용
    }
    // 프로덕션 환경: 번들된 실행 파일 사용
    else {
      this.executablePath = path.join(process.resourcesPath, 'python', 'infer_resnet.exe')
    }

    console.log('[PythonRunner] Initialized')
    console.log('  Mode:', this.isProduction ? 'Production' : 'Development')
    console.log('  Executable:', this.executablePath || 'python (system)')
  }

  /**
   * 모든 실행 중인 프로세스 강제 종료
   */
  cancelAllInferences(): void {
    console.log(`[PythonRunner] Killing ${this.runningProcesses.size} running processes`)

    for (const process of this.runningProcesses) {
      if (!process.killed) {
        process.kill('SIGTERM')
      }
    }

    this.runningProcesses.clear()
  }

  /**
   * 현재 실행 중인 프로세스 개수
   */
  getRunningCount(): number {
    return this.runningProcesses.size
  }

  /**
   * 단일 파일 추론 실행
   */
  async runInference(binPath: string, signal?: AbortSignal): Promise<InferenceResult> {
    // 이미 취소된 경우
    if (signal?.aborted) {
      return Promise.reject(new Error('Inference was cancelled before starting'))
    }

    let command: string
    let args: string[]
    let cwd: string

    // 개발 환경: Python 스크립트 직접 실행
    if (!this.isProduction) {
      command = 'python'
      args = [
        path.join(process.cwd(), 'python', 'infer_resnet_None.py'),
        '--bin_path',
        binPath,
        '--device',
        'cpu',
        '--json',
        '--with-images' // 시각화 이미지 생성
      ]
      cwd = path.join(process.cwd(), 'python')
    }
    // 프로덕션 환경: 번들된 실행 파일 사용
    else {
      command = this.executablePath
      args = ['--bin_path', binPath, '--device', 'cpu', '--json', '--with-images']
      cwd = path.dirname(this.executablePath)
    }

    console.log(`[PythonRunner] Spawning: ${command} ${args.join(' ')}`)
    console.log(`[PythonRunner] Working directory: ${cwd}`)

    const pythonProcess: ChildProcess = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1', // Python 출력 버퍼링 비활성화
        PYTHONIOENCODING: 'utf-8' // UTF-8 인코딩 강제 (한글 깨짐 방지)
      }
    })

    // 🆕 Set에 프로세스 추가
    this.runningProcesses.add(pythonProcess)

    // 🆕 AbortSignal 리스너 (취소 처리)
    const abortHandler = () => {
      console.log(`[PythonRunner] Aborting process for: ${binPath}`)
      if (!pythonProcess.killed) {
        pythonProcess.kill('SIGTERM')
      }
      this.runningProcesses.delete(pythonProcess)
    }

    signal?.addEventListener('abort', abortHandler)

    try {
      // 결과 대기
      const result = await this.waitForResult(pythonProcess, signal, binPath)
      return result
    } catch (error) {
      if (signal?.aborted) {
        throw new Error('Cancelled by user')
      }
      throw error
    } finally {
      // 🆕 정리
      signal?.removeEventListener('abort', abortHandler)
      this.runningProcesses.delete(pythonProcess)

      // 프로세스가 아직 살아있으면 종료
      if (!pythonProcess.killed) {
        pythonProcess.kill('SIGTERM')
      }
    }
  }

  /**
   * 프로세스 결과 대기 (내부 메서드)
   */
  private async waitForResult(
    pythonProcess: ChildProcess,
    signal: AbortSignal | undefined,
    binPath: string
  ): Promise<InferenceResult> {
    return new Promise((resolve, reject) => {
      let stdout = ''
      let stderr = ''

      pythonProcess.stdout?.on('data', (data) => {
        const output = data.toString()
        stdout += output
        console.log('[PythonRunner] stdout:', output.trim())
      })

      pythonProcess.stderr?.on('data', (data) => {
        const output = data.toString()
        stderr += output
        console.error('[PythonRunner] stderr:', output.trim())
      })

      pythonProcess.on('close', (code) => {
        // timer 해제 로직 추가
        clearTimeout(timeoutId)

        console.log(`[PythonRunner] Process exited with code ${code}`)

        if (signal?.aborted) {
          reject(new Error('Cancelled'))
          return
        }

        if (code !== 0) {
          reject(
            new Error(
              `Python process exited with code ${code} for file: ${binPath}\n` +
                `Stderr: ${stderr}\n` +
                `Stdout: ${stdout}`
            )
          )
          return
        }

        try {
          // JSON 출력만 추출 (마지막 줄)
          const lines = stdout.trim().split('\n')
          const jsonLine = lines[lines.length - 1]
          console.log('[PythonRunner] Parsing JSON:', jsonLine)

          const result: InferenceResult = JSON.parse(jsonLine)
          resolve(result)
        } catch (err: any) {
          reject(new Error(`Failed to parse JSON for file: ${binPath}: ${err.message}\nOutput: ${stdout}`))
        }
      })

      pythonProcess.on('error', (err) => {
        reject(new Error(`Failed to start Python process: ${err.message}`))
      })

      // 타임아웃 설정 (60초)
      const timeoutId = setTimeout(() => {
        if (!pythonProcess.killed) {
          console.warn('[PythonRunner] Timeout reached, killing process')
          pythonProcess.kill('SIGTERM')
          reject(new Error('Inference timeout (60s)'))
        }
      }, 60000)
    })
  }
}
