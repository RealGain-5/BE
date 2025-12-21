import { ElectronAPI } from '@electron-toolkit/preload'

interface BatchProgress {
  total: number
  completed: number
  failed: number
  current: string | null
  running?: string[]         // 🆕 현재 실행 중인 파일 경로 배열
  runningCount?: number      // 🆕 실행 중인 파일 개수
  result?: any               // 🆕 방금 완료된 파일의 결과
  error?: string             // 🆕 방금 발생한 에러
}

interface InferenceAPI {
  saveLog: (action: string, details: string) => Promise<any>
  getLogs: () => Promise<any>
  login: (id: string, pw: string) => Promise<any>
  register: (id: string, pw: string) => Promise<any>
  logout: () => Promise<any>
  checkSession: () => Promise<any>
  selectBinFile: () => Promise<string | null>
  runInference: (binPath: string) => Promise<any>
  selectBinFiles: () => Promise<string[] | null>
  setConcurrencyLevel: (level: number) => Promise<any>  // 🆕
  runBatchInference: (binPaths: string[]) => Promise<any>
  cancelBatchInference: () => Promise<any>
  onBatchProgress: (callback: (progress: BatchProgress) => void) => void
  offBatchProgress: () => void
  exportResultsJson: (data: any) => Promise<any>
  exportResultsCsv: (data: any[]) => Promise<any>
  exportResultsExcel: (data: any[]) => Promise<any>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: InferenceAPI
  }
}
