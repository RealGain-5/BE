import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

/** 중복 리스너 누적 방지 + 제거를 묶어 주는 팩토리 */
function makeProgressChannel(channel: string) {
  return {
    on: (cb: (p: any) => void) => {
      ipcRenderer.removeAllListeners(channel)
      ipcRenderer.on(channel, (_, p) => cb(p))
    },
    off: () => ipcRenderer.removeAllListeners(channel),
  }
}

const _batchProgress  = makeProgressChannel('batch-inference-progress')
const _maeBatchProgress = makeProgressChannel('mae-batch-progress')
const _maeFPProgress  = makeProgressChannel('mae-fp-progress')
const _rcpvmsOrbitBatchProgress = makeProgressChannel('rcpvms-orbit-batch-progress')

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

  // DMD 파일 분석
  selectDmdFile: () => ipcRenderer.invoke('select-dmd-file'),
  runDmdInfo: (dmdPath: string) => ipcRenderer.invoke('dmd-info', dmdPath),

  // DMD → RCPVMS 변환
  selectOutputDir: () => ipcRenderer.invoke('select-output-dir'),
  runDmdConvertToRcpvms: (dmdPath: string, outputDir: string, options: any) =>
    ipcRenderer.invoke('dmd-convert-to-rcpvms', dmdPath, outputDir, options),

  // RCPVMS BIN 분석 (단일 파일)
  runRcpvmsInfo: (filepath: string) => ipcRenderer.invoke('rcpvms-info', filepath),
  runRcpvmsOrbit: (filepath: string, windowSec: number, userAxisLim?: number) =>
    ipcRenderer.invoke('rcpvms-orbit', filepath, windowSec, userAxisLim),

  // RCPVMS BIN 배치 궤도 분석 (다중 파일 병렬)
  runRcpvmsOrbitBatch: (binPaths: string[], windowSec: number, userAxisLim?: number) =>
    ipcRenderer.invoke('rcpvms-orbit-batch', binPaths, windowSec, userAxisLim),
  cancelRcpvmsOrbitBatch: () => ipcRenderer.invoke('rcpvms-orbit-batch-cancel'),
  onRcpvmsOrbitBatchProgress: _rcpvmsOrbitBatchProgress.on,
  offRcpvmsOrbitBatchProgress: _rcpvmsOrbitBatchProgress.off,

  // BIN 폴더 선택 → 내부 .BIN 파일 목록 반환
  selectBinFolder: () => ipcRenderer.invoke('select-bin-folder'),

  // Python 모델 추론 (단일 파일)
  selectBinFile: () => ipcRenderer.invoke('select-bin-file'),
  runInference: (binPath: string) => ipcRenderer.invoke('model-inference', binPath),

  // Python 모델 추론 (배치)
  selectBinFiles: () => ipcRenderer.invoke('select-bin-files'),
  setConcurrencyLevel: (level: number) => ipcRenderer.invoke('set-concurrency-level', level),
  runBatchInference: (binPaths: string[]) => ipcRenderer.invoke('model-batch-inference', binPaths),
  cancelBatchInference: () => ipcRenderer.invoke('model-batch-cancel'),
  onBatchProgress:  _batchProgress.on,
  offBatchProgress: _batchProgress.off,

  // MAE 이상 탐지
  runMAEAnalysis: (binPath: string) => ipcRenderer.invoke('mae-analyze', binPath),

  // MAE 배치 분석
  runMAEBatch: (binPaths: string[]) => ipcRenderer.invoke('mae-batch', binPaths),
  cancelMAEBatch: () => ipcRenderer.invoke('mae-batch-cancel'),
  onMAEBatchProgress:  _maeBatchProgress.on,
  offMAEBatchProgress: _maeBatchProgress.off,

  // MAE FP 배치 평가
  runMAEBatchFP: (binPaths: string[]) => ipcRenderer.invoke('mae-batch-fp', binPaths),
  cancelFPBatch: () => ipcRenderer.invoke('fp-batch-cancel'),
  onMAEFPProgress: _maeFPProgress.on,
  offFPProgress:  _maeFPProgress.off,

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
