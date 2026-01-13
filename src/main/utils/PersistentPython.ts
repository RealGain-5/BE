import { spawn, ChildProcess } from 'child_process'
import path from 'path'
import { app } from 'electron'

interface PythonRequest {
  command: 'analyze' | 'timeline' // command to Python Daemon
  payload: any
  resolve: (value: any) => void // success
  reject: (reason?: any) => void // fail
}

export class PersistentPython {
  private process: ChildProcess | null = null
  private queue: PythonRequest[] = [] // request queue
  private isProcessing = false
  private readyPromise: Promise<void> | null = null // 🆕 모델 로드 완료 대기
  private readyResolve: (() => void) | null = null
  private stdoutBuffer: string = '' // 🆕 stdout 버퍼 (청크 합치기)

  constructor() {
    this.readyPromise = new Promise((resolve) => {
      this.readyResolve = resolve
    })
    this.startProcess()
  }

  /**
   * 🆕 데몬이 준비될 때까지 기다림 (모델 로드 완료)
   */
  async waitUntilReady(): Promise<void> {
    if (this.readyPromise) {
      await this.readyPromise
    }
  }

  private startProcess() {
    const scriptPath = app.isPackaged
      ? path.join(process.resourcesPath, 'python', 'inference_daemon.py')
      : path.join(process.cwd(), 'python', 'inference_daemon.py')

    console.log(`[PersistentPython] Starting Daemon at: ${scriptPath}`)

    const cwd = app.isPackaged
      ? path.join(process.resourcesPath, 'python')
      : path.join(process.cwd(), 'python')

    console.log(`[PersistentPython] Working directory: ${cwd}`)

    // stdio <-> pipe
    this.process = spawn('python', ['-u', scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONIOENCODING: 'utf-8'
      }
    })

    // stdout listener (python -> electron)
    this.process.stdout?.on('data', (data) => {
      const output = data.toString()

      // 버퍼에 청크 추가
      this.stdoutBuffer += output

      // 개행 문자로 완전한 라인 분리
      const lines = this.stdoutBuffer.split('\n')

      // 마지막 요소는 불완전할 수 있으므로 버퍼에 남김
      this.stdoutBuffer = lines.pop() || ''

      // 완전한 라인들만 처리
      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const result = JSON.parse(line)
          console.log(`[PersistentPython] ✓ Successfully parsed JSON from stdout`)
          this.completeRequest(result)
        } catch (e) {
          console.log(`[python stdout (non-JSON)]: ${line.substring(0, 100)}...`)
          console.error(`[PersistentPython] Failed to parse JSON:`, e)
        }
      }
    })

    // stderr listener (python log/err)
    this.process.stderr?.on('data', (data) => {
      const output = data.toString()
      console.error(`[python stderr]: ${output}`)

      // 🆕 모델 로드 완료 감지
      if (output.includes('model loaded successfully') && this.readyResolve) {
        console.log('[PersistentPython] ✓ Model loaded, daemon is ready!')
        this.readyResolve()
        this.readyResolve = null
        this.readyPromise = null
      }
    })

    // terminate process
    this.process.on('close', (code) => {
      console.log(`[PersistentPython] Process exited with code ${code}`)
      this.process = null
      this.isProcessing = false
    })
  }

  // Service send request
  public sendCommand(command: 'analyze' | 'timeline', payload: any): Promise<any> {
    return new Promise((resolve, reject) => {
      console.log(`[PersistentPython] sendCommand called: ${command}`)
      this.queue.push({ command, payload, resolve, reject })
      console.log(`[PersistentPython] Queue length after push: ${this.queue.length}`)
      console.log(`[PersistentPython] isProcessing: ${this.isProcessing}, hasProcess: ${!!this.process}`)
      // start process
      this.processQueue()
    })
  }

  // pop queue => send request to python
  private processQueue() {
    if (this.isProcessing || this.queue.length === 0 || !this.process) {
      console.log(`[PersistentPython] processQueue skipped: isProcessing=${this.isProcessing}, queueLen=${this.queue.length}, hasProcess=${!!this.process}`)
      return
    }

    this.isProcessing = true
    const request = this.queue[0] // top

    try {
      const message =
        JSON.stringify({
          command: request.command,
          payload: request.payload
        }) + '\n'

      console.log(`[PersistentPython] Sending command to daemon: ${request.command}`)
      console.log(`[PersistentPython] Payload:`, request.payload)

      this.process.stdin?.write(message, (err) => {
        if (err) {
          console.error(`[PersistentPython] Failed to write to stdin:`, err)
          this.isProcessing = false
          request.reject(err)
          this.queue.shift()
          this.processQueue() // 다음 큐 처리
        } else {
          console.log(`[PersistentPython] ✓ Command sent successfully`)
        }
      })
    } catch (err) {
      console.error(`[PersistentPython] Exception in processQueue:`, err)
      this.isProcessing = false
      request.reject(err)
      this.queue.shift()
      this.processQueue() // 다음 큐 처리
    }
  }

  // resolve promise
  private completeRequest(result: any) {
    console.log(`[PersistentPython] Received response from daemon:`, result)

    const request = this.queue.shift()
    this.isProcessing = false

    if (request) {
      if (result.status === 'ok') {
        console.log(`[PersistentPython] ✓ Request completed successfully`)
        request.resolve(result.data)
      } else {
        console.error(`[PersistentPython] ✗ Request failed:`, result.message)
        request.reject(new Error(result.message || 'Unknown Python Error'))
      }
    } else {
      console.warn(`[PersistentPython] ⚠ Received response but no request in queue`)
    }

    // 다음 큐 처리
    this.processQueue()
  }

  // cleanup process
  public kill() {
    if (this.process) {
      this.process.kill()
      this.process = null
    }
  }
}
