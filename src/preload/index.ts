import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  // 로그 저장 요청 함수
  saveLog: (action, details) => ipcRenderer.invoke('db-insert-log', { action, details }),
  // 로그 조회 요청 함수
  getLogs: () => ipcRenderer.invoke('db-get-logs'),

  // 로그인
  login: (id, pw) => ipcRenderer.invoke('auth-login', { id, pw }),

  // 회원가입 API
  register: (id, pw) => ipcRenderer.invoke('auth-register', { id, pw }),

  // 로그아웃
  logout: () => ipcRenderer.invoke('auth-logout'),

  // 세션 체크
  checkSession: () => ipcRenderer.invoke('auth-check'),

  // Python 모델 추론 (단일 파일)
  selectBinFile: () => ipcRenderer.invoke('select-bin-file'),
  runInference: (binPath: string) => ipcRenderer.invoke('model-inference', binPath),

  // Python 모델 추론 (배치)
  selectBinFiles: () => ipcRenderer.invoke('select-bin-files'),
  setConcurrencyLevel: (level: number) => ipcRenderer.invoke('set-concurrency-level', level),  // 🆕
  runBatchInference: (binPaths: string[]) => ipcRenderer.invoke('model-batch-inference', binPaths),
  cancelBatchInference: () => ipcRenderer.invoke('model-batch-cancel'),
  onBatchProgress: (callback: (progress: any) => void) => {
    ipcRenderer.on('batch-inference-progress', (_, progress) => callback(progress))
  },
  offBatchProgress: () => {
    ipcRenderer.removeAllListeners('batch-inference-progress')
  },

  // 결과 내보내기
  exportResultsJson: (data: any) => ipcRenderer.invoke('export-results-json', data),
  exportResultsCsv: (data: any[]) => ipcRenderer.invoke('export-results-csv', data),
  exportResultsExcel: (data: any[]) => ipcRenderer.invoke('export-results-excel', data)
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api) // api 노출
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
}
