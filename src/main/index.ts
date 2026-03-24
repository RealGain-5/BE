import { app, shell, BrowserWindow, ipcMain, dialog, protocol, net } from 'electron'
import { join } from 'path'
import fs from 'fs'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { pathToFileURL } from 'url' // url 변환을 위해 필요

// import db module
import { initDB, insertLog, getRecentLogs } from './database/db'

// ─── 내보내기 공통 헬퍼 ───────────────────────────────────────────
const RCP_NAMES = ['RCP1A', 'RCP1B', 'RCP2A', 'RCP2B'] as const

function todayDateString(): string {
  return new Date().toISOString().slice(0, 10)
}

function fileBaseName(filePath: string): string {
  return filePath.split(/[/\\]/).pop() ?? filePath
}

function statusKorean(status: string): string {
  if (status === 'completed') return '완료'
  if (status === 'failed') return '실패'
  return '대기'
}

function formatRcpText(rcpData: any): string {
  if (!rcpData) return 'N/A'
  const prob = (rcpData.probabilities.abnormal * 100).toFixed(1)
  return `${rcpData.prediction}(${prob}%)`
}

async function showExportDialog(
  title: string,
  ext: string,
  extLabel: string
): Promise<Electron.SaveDialogReturnValue> {
  return dialog.showSaveDialog({
    title,
    defaultPath: `분석결과_${todayDateString()}.${ext}`,
    filters: [{ name: `${extLabel} Files`, extensions: [ext] }]
  })
}
import { loginUser, logoutUser, checkAuth, registerUser } from './services/auth'
import { pythonService } from './services/pythonService'

function createWindow(): void {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    title: `rcpvms-ver${app.getVersion()}`,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // custom media protocol 등록
  // 프론트엔드의 <img src = ...> => 요청 처리를 실제 로컬 파일로 연결
  protocol.handle('media', (req) => {
    // 1. 'media://' 접두어 제거
    // 3개의 슬래시(///)가 오거나 2개(//)가 올 경우 모두 대응하기 위해 replace 사용
    let pathToServe = req.url.replace(/^media:\/\//, '')

    // 2. URL 디코딩
    pathToServe = decodeURIComponent(pathToServe)

    // [중요 수정] "C/Users/..." 처럼 드라이브 문자 뒤에 콜론(:)이 없는 경우 복구
    // 정규식: "알파벳 한 글자 + 슬래시"로 시작하는 경우 (예: "C/")
    if (/^[a-zA-Z]\//.test(pathToServe)) {
      pathToServe = pathToServe.charAt(0) + ':' + pathToServe.slice(1)
    }

    // [추가 보정] 혹시 "/C:/Users" 처럼 앞에 슬래시가 붙어있는 경우 제거
    if (pathToServe.startsWith('/') && /^[a-zA-Z]:/.test(pathToServe.slice(1))) {
      pathToServe = pathToServe.slice(1)
    }

    // 3. 절대 경로로 변환하여 로드
    const fileUrl = pathToFileURL(pathToServe).toString()

    console.log(`[Media Fix] Input: ${req.url}`)
    console.log(`[Media Fix] Path:  ${pathToServe}`)
    console.log(`[Media Fix] URL:   ${fileUrl}`)

    return net.fetch(fileUrl)
  })

  // DB initialized
  initDB()
  insertLog('APP_START', 'Application has started successfully.')

  // 🆕 Python 데몬 미리 시작 (Cold Start 제거)
  console.log('[App] Pre-starting Python daemon...')
  pythonService.init().catch((err) => {
    console.error('[App] Failed to start Python daemon:', err)
  })

  // IPC 핸들러 등록
  // 프론트엔드에서 요청 받을 준비
  ipcMain.handle('db-insert-log', (_, { action, details }) => {
    return insertLog(action, details)
  })

  // 로그 조회 요청 처리
  ipcMain.handle('db-get-logs', () => {
    return getRecentLogs()
  })

  // 로그인
  ipcMain.handle('auth-login', async (_, { id, pw }) => {
    return await loginUser(id, pw)
  })

  // 회원가입
  ipcMain.handle('auth-register', async (_, { id, pw }) => {
    return await registerUser(id, pw)
  })

  // 로그아웃
  ipcMain.handle('auth-logout', async () => {
    return logoutUser()
  })

  // 세션 체크
  ipcMain.handle('auth-check', async () => {
    return checkAuth()
  })

  // DMD 파일 선택 다이얼로그
  ipcMain.handle('select-dmd-file', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'DMD Files', extensions: ['dmd', 'DMD'] }]
    })
    if (result.canceled) return null
    return result.filePaths[0]
  })

  // DMD 파일 정보 조회
  ipcMain.handle('dmd-info', async (_, dmdPath: string) => {
    try {
      const data = await pythonService.runDmdInfo(dmdPath)
      return { success: true, data }
    } catch (error: any) {
      console.error('[IPC] dmd-info error:', error)
      return { success: false, error: error.message }
    }
  })

  // DMD → RCPVMS BIN 변환
  ipcMain.handle('dmd-convert-to-rcpvms', async (_, dmdPath: string, outputDir: string, options: any) => {
    try {
      const data = await pythonService.runDmdConvertToRcpvms(dmdPath, outputDir, options ?? {})
      return { success: true, data }
    } catch (error: any) {
      console.error('[IPC] dmd-convert-to-rcpvms error:', error)
      return { success: false, error: error.message }
    }
  })

  // 출력 디렉토리 선택 다이얼로그
  ipcMain.handle('select-output-dir', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory', 'createDirectory']
    })
    if (result.canceled) return null
    return result.filePaths[0]
  })

  // RCPVMS BIN 파일 정보 조회
  ipcMain.handle('rcpvms-info', async (_, filepath: string) => {
    try {
      const data = await pythonService.runRcpvmsInfo(filepath)
      return { success: true, data }
    } catch (error: any) {
      console.error('[IPC] rcpvms-info error:', error)
      return { success: false, error: error.message }
    }
  })

  // RCPVMS BIN 궤도 이미지 생성
  ipcMain.handle('rcpvms-orbit', async (_, filepath: string, windowSec: number, scaleMode: string = 'auto') => {
    try {
      const data = await pythonService.runRcpvmsOrbit(filepath, windowSec, scaleMode)
      return { success: true, data }
    } catch (error: any) {
      console.error('[IPC] rcpvms-orbit error:', error)
      return { success: false, error: error.message }
    }
  })

  // RCPVMS BIN 배치 궤도 이미지 생성 (병렬)
  ipcMain.handle('rcpvms-orbit-batch', async (event, binPaths: string[], windowSec: number, scaleMode: string = 'auto') => {
    try {
      await pythonService.runRcpvmsOrbitBatch(binPaths, windowSec, scaleMode, (p) => {
        event.sender.send('rcpvms-orbit-batch-progress', p)
      })
      return { success: true }
    } catch (error: any) {
      console.error('[IPC] rcpvms-orbit-batch error:', error)
      return { success: false, error: error.message }
    }
  })

  // RCPVMS 배치 궤도 취소
  ipcMain.handle('rcpvms-orbit-batch-cancel', async () => {
    pythonService.cancelRcpvmsOrbitBatch()
    return { success: true }
  })

  // BIN 폴더 선택 → 내부 .BIN 파일 목록 반환
  ipcMain.handle('select-bin-folder', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory']
    })
    if (result.canceled) return null
    const folderPath = result.filePaths[0]
    try {
      const entries = fs.readdirSync(folderPath)
      const binFiles = entries
        .filter((f) => /\.bin$/i.test(f))
        .map((f) => join(folderPath, f))
        .sort()
      return binFiles
    } catch (err) {
      console.error('[select-bin-folder] readdirSync 실패:', err)
      return []
    }
  })

  // BIN 파일 선택 다이얼로그 (단일 파일)
  ipcMain.handle('select-bin-file', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'BIN Files', extensions: ['bin', 'BIN'] }]
    })

    if (result.canceled) {
      return null
    }
    return result.filePaths[0]
  })

  // BIN 파일 선택 다이얼로그 (다중 파일)
  ipcMain.handle('select-bin-files', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'BIN Files', extensions: ['bin', 'BIN'] }]
    })

    if (result.canceled) {
      return null
    }
    return result.filePaths // 배열 반환
  })

  // 모델 추론 실행 (단일 파일)
  ipcMain.handle('model-inference', async (_, binPath: string) => {
    try {
      const result = await pythonService.runInference(binPath)
      return { success: true, data: result }
    } catch (error: any) {
      console.error('[IPC] model-inference error:', error)
      return { success: false, error: error.message }
    }
  })

  // 🆕 병렬 처리 수준 설정
  ipcMain.handle('set-concurrency-level', async (_, level: number) => {
    try {
      pythonService.setMaxConcurrent(level)
      console.log(`[IPC] Concurrency level set to ${level}`)
      return { success: true }
    } catch (error: any) {
      console.error('[IPC] set-concurrency-level error:', error)
      return { success: false, error: error.message }
    }
  })

  // 배치 모델 추론 실행 (병렬 처리 적용)
  ipcMain.handle('model-batch-inference', async (event, binPaths: string[]) => {
    try {
      console.log(`[IPC] Starting batch inference for ${binPaths.length} files`)

      // 🔧 병렬 처리로 변경 (Incremental Update)
      const results = await pythonService.runBatchInferenceParallel(binPaths, (progress) => {
        event.sender.send('batch-inference-progress', progress)
      })

      console.log('[IPC] Batch inference completed')
      return {
        success: true,
        summary: {
          total: binPaths.length,
          completed: results.completed,
          failed: results.failed
        }
      }
      
    } catch (error: any) {
      console.error('[IPC] model-batch-inference error:', error)
      return { success: false, error: error.message }
    }
  })

  // MAE 이상 탐지 실행
  ipcMain.handle('mae-analyze', async (_, binPath: string) => {
    try {
      const result = await pythonService.runMAEAnalysis(binPath)
      return { success: true, data: result }
    } catch (error: any) {
      console.error('[IPC] mae-analyze error:', error)
      return { success: false, error: error.message }
    }
  })

  // MAE 배치 분석
  ipcMain.handle('mae-batch', async (event, binPaths: string[]) => {
    try {
      await pythonService.runMAEBatch(binPaths, (p) => {
        event.sender.send('mae-batch-progress', p)
      })
      return { success: true }
    } catch (error: any) {
      return { success: false, error: error.message }
    }
  })

  // MAE 배치 취소
  ipcMain.handle('mae-batch-cancel', async () => {
    pythonService.cancelMAEBatch()
    return { success: true }
  })

  // MAE FP 배치 평가
  ipcMain.handle('mae-batch-fp', async (event, binPaths: string[]) => {
    try {
      const result = await pythonService.runMAEBatchFP(binPaths, (p) => {
        event.sender.send('mae-fp-progress', p)
      })
      return { success: true, data: result }
    } catch (error: any) {
      return { success: false, error: error.message }
    }
  })

  // FP 배치 평가 취소
  ipcMain.handle('fp-batch-cancel', async () => {
    pythonService.cancelFPBatch()
    return { success: true }
  })

  // 배치 추론 취소
  ipcMain.handle('model-batch-cancel', async () => {
    try {
      pythonService.cancelBatchInference()
      console.log('[IPC] Batch inference cancelled')
      return { success: true }
    } catch (error: any) {
      console.error('[IPC] model-batch-cancel error:', error)
      return { success: false, error: error.message }
    }
  })

  // 결과 내보내기 (JSON)
  ipcMain.handle('export-results-json', async (_, data: any) => {
    try {
      const dlg = await showExportDialog('분석 결과 저장', 'json', 'JSON')
      if (dlg.canceled || !dlg.filePath) return { success: false, cancelled: true }

      fs.writeFileSync(dlg.filePath, JSON.stringify(data, null, 2), 'utf-8')
      console.log('[IPC] Results exported to JSON:', dlg.filePath)
      return { success: true, filePath: dlg.filePath }
    } catch (error: any) {
      console.error('[IPC] export-results-json error:', error)
      return { success: false, error: error.message }
    }
  })

  // 결과 내보내기 (CSV)
  ipcMain.handle('export-results-csv', async (_, data: any[]) => {
    try {
      const dlg = await showExportDialog('분석 결과 저장 (CSV)', 'csv', 'CSV')
      if (dlg.canceled || !dlg.filePath) return { success: false, cancelled: true }

      const csvLines = [
        '파일명,최종판정,상태,RCP1A,RCP1B,RCP2A,RCP2B',
        ...data.map((item) => {
          const rcpCols = RCP_NAMES.map((rcp) => formatRcpText(item.result?.results?.[rcp]))
          return `"${fileBaseName(item.path)}",${item.result?.final_label?.toUpperCase() || 'N/A'},${statusKorean(item.status)},${rcpCols.join(',')}`
        })
      ]

      fs.writeFileSync(dlg.filePath, '\ufeff' + csvLines.join('\n'), 'utf-8') // BOM for Excel
      console.log('[IPC] Results exported to CSV:', dlg.filePath)
      return { success: true, filePath: dlg.filePath }
    } catch (error: any) {
      console.error('[IPC] export-results-csv error:', error)
      return { success: false, error: error.message }
    }
  })

  // 결과 내보내기 (Excel with Images)
  ipcMain.handle('export-results-excel', async (_, data: any[]) => {
    try {
      const ExcelJS = require('exceljs')
      const dlg = await showExportDialog('분석 결과 저장 (Excel)', 'xlsx', 'Excel')
      if (dlg.canceled || !dlg.filePath) return { success: false, cancelled: true }

      const workbook = new ExcelJS.Workbook()
      const worksheet = workbook.addWorksheet('분석 결과')

      worksheet.columns = [
        { header: '파일명', key: 'filename', width: 25 },
        { header: '최종판정', key: 'label', width: 12 },
        { header: '상태', key: 'status', width: 10 },
        { header: 'RCP1A', key: 'rcp1a', width: 20 },
        { header: 'RCP1A 이미지', key: 'rcp1a_img', width: 25 },
        { header: 'RCP1B', key: 'rcp1b', width: 20 },
        { header: 'RCP1B 이미지', key: 'rcp1b_img', width: 25 },
        { header: 'RCP2A', key: 'rcp2a', width: 20 },
        { header: 'RCP2A 이미지', key: 'rcp2a_img', width: 25 },
        { header: 'RCP2B', key: 'rcp2b', width: 20 },
        { header: 'RCP2B 이미지', key: 'rcp2b_img', width: 25 }
      ]
      worksheet.getRow(1).font = { bold: true }
      worksheet.getRow(1).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE0E0E0' } }

      // RCP 이미지 열 인덱스 (0-based)
      const RCP_IMAGE_COLS: Record<string, number> = { RCP1A: 4, RCP1B: 6, RCP2A: 8, RCP2B: 10 }

      for (let i = 0; i < data.length; i++) {
        const item = data[i]
        const rcpResults: Record<string, string> = {}
        RCP_NAMES.forEach((rcp) => { rcpResults[rcp.toLowerCase()] = formatRcpText(item.result?.results?.[rcp]) })

        const row = worksheet.addRow({
          filename: fileBaseName(item.path),
          label: item.result?.final_label?.toUpperCase() || 'N/A',
          status: statusKorean(item.status),
          rcp1a: rcpResults.rcp1a, rcp1b: rcpResults.rcp1b,
          rcp2a: rcpResults.rcp2a, rcp2b: rcpResults.rcp2b
        })

        if (item.result?.visualization) {
          for (const rcp of RCP_NAMES) {
            const overlayPath = item.result.visualization[rcp]?.gradcam?.overlay
            if (!overlayPath) continue
            try {
              if (fs.existsSync(overlayPath)) {
                const imageId = workbook.addImage({ buffer: fs.readFileSync(overlayPath), extension: 'png' })
                worksheet.addImage(imageId, {
                  tl: { col: RCP_IMAGE_COLS[rcp], row: i + 1 },
                  ext: { width: 150, height: 150 }
                })
              }
            } catch (imgError) {
              console.error(`[Excel] Failed to add image for ${rcp}:`, imgError)
            }
          }
          row.height = 120
        }
      }

      await workbook.xlsx.writeFile(dlg.filePath)
      console.log('[IPC] Results exported to Excel:', dlg.filePath)
      return { success: true, filePath: dlg.filePath }
    } catch (error: any) {
      console.error('[IPC] export-results-excel error:', error)
      return { success: false, error: error.message }
    }
  })

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => console.log('pong'))

  createWindow()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// 앱 종료 전 정리 작업
app.on('before-quit', () => {
  console.log('[App] Before quit - cleaning up resources...')
  pythonService.shutdown()
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
