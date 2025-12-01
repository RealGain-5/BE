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
  checkSession: () => ipcRenderer.invoke('auth-check')
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
