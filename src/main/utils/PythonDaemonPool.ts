import { spawn, ChildProcess } from 'child_process'
import path from 'path'
import { app } from 'electron'
import { EventEmitter } from 'events'

interface PythonRequest {
  command: 'analyze' | 'timeline' | 'svdd_analyze' | 'mae_analyze'
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
}

interface PendingJob {
  command: 'analyze' | 'timeline' | 'svdd_analyze' | 'mae_analyze'
  payload: any
  resolve: (value: any) => void
  reject: (reason: any) => void
}

export class PythonDaemonPool extends EventEmitter {
  private workers: WorkerState[] = []
  private maxWorkers: number = 2
  private pendingJobs: PendingJob[] = []
  private isShuttingDown: boolean = false

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
      readyResolve: null
    }

    // Create ready promise
    workerState.readyPromise = new Promise((resolve) => {
      workerState.readyResolve = resolve
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
        worker.readyPromise = null
      }
    })

    // Process exit handler
    proc.on('close', (code) => {
      console.log(`[DaemonPool] Worker ${workerId} process exited with code ${code}`)
      worker.process = null

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

    // Restart worker
    worker.status = 'starting'
    worker.readyPromise = new Promise((resolve) => {
      worker.readyResolve = resolve
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
  private getIdleWorker(): WorkerState | null {
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

    const worker = this.getIdleWorker()
    if (!worker) return

    const job = this.pendingJobs.shift()!
    this.sendToWorker(worker.id, job)
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

    worker.process.stdin?.write(message, (err) => {
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
  public sendCommand(command: 'analyze' | 'timeline' | 'svdd_analyze' | 'mae_analyze', payload: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const job: PendingJob = { command, payload, resolve, reject }

      const worker = this.getIdleWorker()
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
      // Remove workers — reject in-flight jobs before killing
      for (let i = newCount; i < this.maxWorkers; i++) {
        const worker = this.workers[i]
        if (worker) {
          // Reject any queued/in-flight jobs so their promises resolve immediately
          for (const request of worker.queue) {
            request.reject(new Error(`Worker ${i} removed by pool resize`))
          }
          worker.queue = []
          worker.process?.kill()
          worker.process = null
          worker.status = 'error'
        }
      }
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

    return {
      totalWorkers: this.workers.length,
      idleWorkers: this.workers.filter((w) => w.status === 'idle').length,
      busyWorkers: this.workers.filter((w) => w.status === 'busy').length,
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
