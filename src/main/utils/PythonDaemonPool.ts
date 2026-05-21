import { spawn, ChildProcess } from 'child_process'
import path from 'path'
import { app } from 'electron'
import { EventEmitter } from 'events'

interface PythonRequest {
  command: string
  payload: any
  resolve: (value: any) => void
  reject: (reason?: any) => void
}

interface WorkerState {
  id: number
  process: ChildProcess | null
  status: 'idle' | 'busy' | 'starting' | 'error'
  currentJob: string | null
  queue: PythonRequest[]
  stdoutBuffer: string
  readyPromise: Promise<void> | null
  readyResolve: (() => void) | null
  readyReject: ((err: Error) => void) | null
}

interface PendingJob {
  command: string
  payload: any
  resolve: (value: any) => void
  reject: (reason: any) => void
  preferredWorkerId?: number
}

export class PythonDaemonPool extends EventEmitter {
  private workers: WorkerState[] = []
  private maxWorkers: number = 2
  private pendingJobs: PendingJob[] = []
  private isShuttingDown: boolean = false
  private stickyWorkerByKey: Map<string, number> = new Map()

  constructor() {
    super()
  }

  /**
   * Initialize pool with specified number of workers
   */
  async init(count: number): Promise<void> {
    this.maxWorkers = Math.max(1, Math.min(count, 4))
    console.log(`[DaemonPool] Initializing pool with ${this.maxWorkers} workers...`)

    const initPromises: Promise<void>[] = []

    for (let i = 0; i < this.maxWorkers; i++) {
      initPromises.push(this.createWorker(i))
    }

    // Wait for all workers to be ready (model loaded)
    await Promise.all(initPromises)
    console.log(`[DaemonPool] ✓ All ${this.maxWorkers} workers ready`)
  }

  /**
   * Create a single worker
   */
  private async createWorker(id: number): Promise<void> {
    console.log(`[DaemonPool] Creating worker ${id}...`)

    const workerState: WorkerState = {
      id,
      process: null,
      status: 'starting',
      currentJob: null,
      queue: [],
      stdoutBuffer: '',
      readyPromise: null,
      readyResolve: null,
      readyReject: null
    }

    // Create ready promise (resolve on success, reject on startup failure)
    workerState.readyPromise = new Promise((resolve, reject) => {
      workerState.readyResolve = resolve
      workerState.readyReject = reject
    })

    this.workers[id] = workerState
    this.startWorkerProcess(id)

    // Wait for model to load
    await workerState.readyPromise
    workerState.status = 'idle'
    console.log(`[DaemonPool] ✓ Worker ${id} ready`)
  }

  /**
   * Start the Python process for a worker
   */
  private startWorkerProcess(workerId: number): void {
    const worker = this.workers[workerId]
    if (!worker) return

    // 패키징 여부에 따라 실행 경로 결정
    const isPackaged = app.isPackaged

    const exePath = isPackaged
      ? path.join(process.resourcesPath, 'python', 'inference_daemon.exe')
      : null

    const scriptPath = isPackaged
      ? null
      : path.join(process.cwd(), 'python', 'inference_daemon.py')

    const cwd = isPackaged
      ? path.join(process.resourcesPath, 'python')
      : path.join(process.cwd(), 'python')

    console.log(`[DaemonPool] Worker ${workerId} starting process...`)
    console.log(`[DaemonPool] isPackaged: ${isPackaged}`)
    console.log(`[DaemonPool] cwd: ${cwd}`)

    // 패키징된 앱: exe 직접 실행 / 개발 모드: python 인터프리터로 스크립트 실행
    if (isPackaged && exePath) {
      console.log(`[DaemonPool] Spawning exe: ${exePath}`)
      worker.process = spawn(exePath, [], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8'
        }
      })
    } else if (scriptPath) {
      console.log(`[DaemonPool] Spawning python script: ${scriptPath}`)
      worker.process = spawn('python', ['-u', scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8'
        }
      })
    }

    const proc = worker.process
    if (!proc) {
      console.error(`[DaemonPool] Failed to spawn process for worker ${workerId}`)
      return
    }

    // stdout listener
    proc.stdout?.on('data', (data) => {
      this.handleWorkerStdout(workerId, data.toString())
    })

    // stderr listener
    proc.stderr?.on('data', (data) => {
      const output = data.toString()
      console.error(`[Worker ${workerId} stderr]: ${output}`)

      // Model load complete detection
      if (output.includes('model loaded successfully') && worker.readyResolve) {
        console.log(`[DaemonPool] Worker ${workerId} model loaded`)
        worker.readyResolve()
        worker.readyResolve = null
        worker.readyReject = null
        worker.readyPromise = null
      }
    })

    // Process exit handler
    proc.on('close', (code) => {
      console.log(`[DaemonPool] Worker ${workerId} process exited with code ${code}`)
      worker.process = null

      // 모델 로드 완료 전에 프로세스가 종료된 경우 (ex: 모델 파일 없음, sys.exit(1))
      // readyResolve가 아직 남아 있으면 readyPromise를 reject하여 createWorker hang 방지
      if (worker.readyReject) {
        worker.readyReject(new Error(`Worker ${workerId} exited before model loaded (code ${code})`))
        worker.readyReject = null
        worker.readyResolve = null
        worker.readyPromise = null
      }

      if (!this.isShuttingDown) {
        this.handleWorkerCrash(workerId)
      }
    })
  }

  /**
   * Handle stdout data from a worker
   */
  private handleWorkerStdout(workerId: number, data: string): void {
    const worker = this.workers[workerId]
    if (!worker) return

    worker.stdoutBuffer += data
    const lines = worker.stdoutBuffer.split('\n')
    worker.stdoutBuffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const result = JSON.parse(line)
        console.log(`[DaemonPool] Worker ${workerId} completed request`)
        this.completeWorkerRequest(workerId, result)
      } catch (e) {
        console.log(`[Worker ${workerId} stdout (non-JSON)]: ${line.substring(0, 100)}`)
      }
    }
  }

  /**
   * Complete a request for a worker
   */
  private completeWorkerRequest(workerId: number, result: any): void {
    const worker = this.workers[workerId]
    if (!worker) return

    const request = worker.queue.shift()
    worker.status = 'idle'
    worker.currentJob = null

    if (request) {
      if (result.status === 'ok') {
        request.resolve(result.data)
      } else {
        request.reject(new Error(result.message || 'Unknown Python Error'))
      }
    }

    // Process next pending job
    this.processNextJob()
  }

  /**
   * Handle worker crash with auto-restart
   */
  private handleWorkerCrash(workerId: number): void {
    console.log(`[DaemonPool] Worker ${workerId} crashed, restarting...`)

    const worker = this.workers[workerId]
    if (!worker) return

    // Reject current job if any
    for (const request of worker.queue) {
      request.reject(new Error(`Worker ${workerId} crashed`))
    }
    worker.queue = []

    // Remove sticky mapping for this worker so subsequent requests are re-routed
    // to a live worker instead of waiting for the restarting process to warm up.
    for (const [key, id] of this.stickyWorkerByKey) {
      if (id === workerId) this.stickyWorkerByKey.delete(key)
    }

    // Restart worker
    worker.status = 'starting'
    worker.readyPromise = new Promise((resolve, reject) => {
      worker.readyResolve = resolve
      worker.readyReject = reject
    })

    this.startWorkerProcess(workerId)

    worker.readyPromise.then(() => {
      worker.status = 'idle'
      console.log(`[DaemonPool] ✓ Worker ${workerId} restarted`)
      this.processNextJob()
    })
  }

  /**
   * Find an idle worker
   */
  private getIdleWorker(preferredWorkerId?: number): WorkerState | null {
    if (preferredWorkerId !== undefined) {
      const preferred = this.workers[preferredWorkerId]
      if (preferred && preferred.status === 'idle' && preferred.process) {
        return preferred
      }
    }

    for (const worker of this.workers) {
      if (worker && worker.status === 'idle' && worker.process) {
        return worker
      }
    }
    return null
  }

  /**
   * Process the next pending job if a worker is available
   */
  private processNextJob(): void {
    if (this.pendingJobs.length === 0) return

    let jobIndex = -1
    let worker: WorkerState | null = null
    for (let i = 0; i < this.pendingJobs.length; i++) {
      const candidate = this.pendingJobs[i]
      worker = this.getIdleWorker(candidate.preferredWorkerId)
      if (worker && (candidate.preferredWorkerId === undefined || worker.id === candidate.preferredWorkerId)) {
        jobIndex = i
        break
      }
    }

    if (jobIndex === -1) {
      worker = this.getIdleWorker()
      if (!worker) return
      jobIndex = this.pendingJobs.findIndex((job) => job.preferredWorkerId === undefined)
      // All pending jobs have a preferred worker but none of them are idle.
      // Dispatch the first job to the available worker and accept the cache miss,
      // otherwise preferred-only queues starve indefinitely while workers sit idle.
      if (jobIndex === -1) jobIndex = 0
    }

    if (!worker) return

    const [job] = this.pendingJobs.splice(jobIndex, 1)
    this.sendToWorker(worker.id, job)
  }

  private getStickyWorkerId(command: string, payload: any): number | undefined {
    if (!command.startsWith('rcpvms_orbit')) return undefined
    const filepath = payload?.filepath
    if (!filepath || this.workers.length === 0) return undefined

    const existing = this.stickyWorkerByKey.get(filepath)
    if (existing !== undefined && this.workers[existing]) return existing

    const next = this.stickyWorkerByKey.size % this.workers.length
    this.stickyWorkerByKey.set(filepath, next)
    return next
  }

  /**
   * Send a command to a specific worker
   */
  private sendToWorker(workerId: number, job: PendingJob): void {
    const worker = this.workers[workerId]
    if (!worker || !worker.process) {
      job.reject(new Error(`Worker ${workerId} not available`))
      return
    }

    worker.status = 'busy'
    worker.currentJob = job.payload?.bin_path || 'unknown'
    worker.queue.push({
      command: job.command,
      payload: job.payload,
      resolve: job.resolve,
      reject: job.reject
    })

    const message = JSON.stringify({
      command: job.command,
      payload: job.payload
    }) + '\n'

    console.log(`[DaemonPool] Worker ${workerId} processing: ${worker.currentJob}`)

    worker.process.stdin?.write(message, 'utf8', (err) => {
      if (err) {
        console.error(`[DaemonPool] Worker ${workerId} stdin write error:`, err)
        worker.status = 'error'
        const request = worker.queue.shift()
        if (request) {
          request.reject(err)
        }
        this.handleWorkerCrash(workerId)
      }
    })
  }

  /**
   * Send a command to the pool - dispatches to an idle worker
   */
  public sendCommand(command: string, payload: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const preferredWorkerId = this.getStickyWorkerId(command, payload)
      const job: PendingJob = { command, payload, resolve, reject, preferredWorkerId }

      const worker = this.getIdleWorker(preferredWorkerId)
      if (worker) {
        this.sendToWorker(worker.id, job)
      } else {
        // All workers busy, queue the job
        console.log(`[DaemonPool] All workers busy, queuing job. Queue size: ${this.pendingJobs.length + 1}`)
        this.pendingJobs.push(job)
      }
    })
  }

  /**
   * Resize the pool (add or remove workers)
   */
  async resize(newCount: number): Promise<void> {
    newCount = Math.max(1, Math.min(newCount, 4))

    if (newCount === this.maxWorkers) {
      console.log(`[DaemonPool] Pool size unchanged (${newCount})`)
      return
    }

    console.log(`[DaemonPool] Resizing pool from ${this.maxWorkers} to ${newCount} workers`)

    if (newCount > this.maxWorkers) {
      // Add workers
      const addPromises: Promise<void>[] = []
      for (let i = this.maxWorkers; i < newCount; i++) {
        addPromises.push(this.createWorker(i))
      }
      await Promise.all(addPromises)
    } else {
      // Remove workers — in-flight 작업이 완료될 때까지 기다린 후 종료
      const removalPromises: Promise<void>[] = []
      for (let i = newCount; i < this.maxWorkers; i++) {
        const worker = this.workers[i]
        if (!worker) continue

        const waitAndKill = async () => {
          // 실행 중인 작업이 있으면 완료 시점까지 대기
          if (worker.status === 'busy' && worker.queue.length > 0) {
            await new Promise<void>((resolve) => {
              const currentReq = worker.queue[0]
              const origResolve = currentReq.resolve
              const origReject = currentReq.reject
              currentReq.resolve = (v) => { origResolve(v); resolve() }
              currentReq.reject  = (e) => { origReject(e);  resolve() }
            })
          }
          // 대기 중 나머지 작업(있을 경우) reject 후 프로세스 종료
          for (const request of worker.queue) {
            request.reject(new Error(`Worker ${i} removed by pool resize`))
          }
          worker.queue = []
          worker.process?.kill()
          worker.process = null
          worker.status = 'error'
        }

        removalPromises.push(waitAndKill())
      }
      await Promise.all(removalPromises)
      this.workers.length = newCount
    }

    this.maxWorkers = newCount
    console.log(`[DaemonPool] ✓ Pool resized to ${newCount} workers`)
  }

  /**
   * Get pool status
   */
  getStatus(): {
    totalWorkers: number
    idleWorkers: number
    busyWorkers: number
    pendingJobs: number
    workerDetails: Array<{ id: number; status: string; currentJob: string | null }>
  } {
    const workerDetails = this.workers.map((w) => ({
      id: w.id,
      status: w.status,
      currentJob: w.currentJob
    }))

    const { idleWorkers, busyWorkers } = this.workers.reduce(
      (acc, w) => {
        if (w.status === 'idle') acc.idleWorkers++
        else if (w.status === 'busy') acc.busyWorkers++
        return acc
      },
      { idleWorkers: 0, busyWorkers: 0 }
    )

    return {
      totalWorkers: this.workers.length,
      idleWorkers,
      busyWorkers,
      pendingJobs: this.pendingJobs.length,
      workerDetails
    }
  }

  /**
   * Get list of currently running jobs
   */
  getRunningJobs(): string[] {
    return this.workers
      .filter((w) => w.status === 'busy' && w.currentJob)
      .map((w) => w.currentJob!)
  }

  /**
   * Cancel all queued (not yet dispatched) jobs.
   * In-flight jobs already sent to a Python worker cannot be interrupted
   * without killing the process — they will complete normally.
   */
  cancelPendingJobs(): void {
    const count = this.pendingJobs.length
    for (const job of this.pendingJobs) {
      job.reject(new Error('Batch cancelled'))
    }
    this.pendingJobs = []
    if (count > 0) {
      console.log(`[DaemonPool] Cancelled ${count} pending jobs`)
    }
  }

  /**
   * Shutdown the pool
   */
  shutdown(): void {
    console.log(`[DaemonPool] Shutting down pool...`)
    this.isShuttingDown = true

    // Reject all pending jobs
    for (const job of this.pendingJobs) {
      job.reject(new Error('Pool shutting down'))
    }
    this.pendingJobs = []

    // Kill all workers
    for (const worker of this.workers) {
      if (worker?.process) {
        worker.process.kill()
        worker.process = null
      }
    }
    this.workers = []

    console.log(`[DaemonPool] ✓ Pool shutdown complete`)
  }

  /**
   * Wait until at least one worker is ready
   */
  async waitUntilReady(): Promise<void> {
    const readyPromises = this.workers
      .filter((w) => w.readyPromise)
      .map((w) => w.readyPromise!)

    if (readyPromises.length > 0) {
      await Promise.race(readyPromises)
    }
  }
}
