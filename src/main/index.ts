import { app, shell, BrowserWindow, ipcMain, dialog, protocol, net } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { pathToFileURL } from 'url' // url 변환을 위해 필요

// import db module
import { initDB, insertLog, getRecentLogs } from './database/db'
import { loginUser, logoutUser, checkAuth, registerUser } from './services/auth'
import { pythonService } from './services/pythonService'

function createWindow(): void {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
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
        // 실시간 진행 상황 + 결과 전송
        event.sender.send('batch-inference-progress', {
          total: progress.total,
          completed: progress.completed,
          failed: progress.failed,
          current: progress.current,
          running: progress.running,         // 🆕 실행 중인 파일들
          runningCount: progress.runningCount, // 🆕 실행 중인 개수
          result: progress.currentResult,    // 🆕 방금 완료된 결과
          error: progress.currentError       // 🆕 방금 발생한 에러
        })
      })

      // 🔧 최종 반환값은 가벼운 요약만 (메모리 절약)
      console.log('[IPC] Batch inference completed')
      return { 
        success: true, 
        summary: {
          total: binPaths.length,
          completed: Array.from(results.values()).filter(r => r.success).length,
          failed: Array.from(results.values()).filter(r => !r.success).length
        }
      }
      
    } catch (error: any) {
      console.error('[IPC] model-batch-inference error:', error)
      return { success: false, error: error.message }
    }
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
      const result = await dialog.showSaveDialog({
        title: '분석 결과 저장',
        defaultPath: `분석결과_${new Date().toISOString().slice(0, 10)}.json`,
        filters: [{ name: 'JSON Files', extensions: ['json'] }]
      })

      if (result.canceled || !result.filePath) {
        return { success: false, cancelled: true }
      }

      const fs = require('fs')
      fs.writeFileSync(result.filePath, JSON.stringify(data, null, 2), 'utf-8')
      console.log('[IPC] Results exported to JSON:', result.filePath)

      return { success: true, filePath: result.filePath }
    } catch (error: any) {
      console.error('[IPC] export-results-json error:', error)
      return { success: false, error: error.message }
    }
  })

  // 결과 내보내기 (CSV)
  ipcMain.handle('export-results-csv', async (_, data: any[]) => {
    try {
      const result = await dialog.showSaveDialog({
        title: '분석 결과 저장 (CSV)',
        defaultPath: `분석결과_${new Date().toISOString().slice(0, 10)}.csv`,
        filters: [{ name: 'CSV Files', extensions: ['csv'] }]
      })

      if (result.canceled || !result.filePath) {
        return { success: false, cancelled: true }
      }

      // CSV 생성
      const csvLines = [
        '파일명,최종판정,상태,RCP1A,RCP1B,RCP2A,RCP2B', // 헤더
        ...data.map((item) => {
          const filename = item.path.split(/[/\\]/).pop()
          const status = item.status === 'completed' ? '완료' : item.status === 'failed' ? '실패' : '대기'
          const label = item.result?.final_label?.toUpperCase() || 'N/A'
          
          // RCP별 결과
          const rcps = ['RCP1A', 'RCP1B', 'RCP2A', 'RCP2B']
          const rcpResults = rcps.map((rcp) => {
            const rcpData = item.result?.results?.[rcp]
            if (!rcpData) return 'N/A'
            const prob = (rcpData.probabilities.abnormal * 100).toFixed(1)
            return `${rcpData.prediction}(${prob}%)`
          })

          return `"${filename}",${label},${status},${rcpResults.join(',')}`
        })
      ]

      const fs = require('fs')
      fs.writeFileSync(result.filePath, '\ufeff' + csvLines.join('\n'), 'utf-8') // BOM for Excel
      console.log('[IPC] Results exported to CSV:', result.filePath)

      return { success: true, filePath: result.filePath }
    } catch (error: any) {
      console.error('[IPC] export-results-csv error:', error)
      return { success: false, error: error.message }
    }
  })

  // 결과 내보내기 (Excel with Images)
  ipcMain.handle('export-results-excel', async (_, data: any[]) => {
    try {
      const ExcelJS = require('exceljs')
      const fs = require('fs')

      const result = await dialog.showSaveDialog({
        title: '분석 결과 저장 (Excel)',
        defaultPath: `분석결과_${new Date().toISOString().slice(0, 10)}.xlsx`,
        filters: [{ name: 'Excel Files', extensions: ['xlsx'] }]
      })

      if (result.canceled || !result.filePath) {
        return { success: false, cancelled: true }
      }

      const workbook = new ExcelJS.Workbook()
      const worksheet = workbook.addWorksheet('분석 결과')

      // 헤더 설정
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

      // 헤더 스타일
      worksheet.getRow(1).font = { bold: true }
      worksheet.getRow(1).fill = {
        type: 'pattern',
        pattern: 'solid',
        fgColor: { argb: 'FFE0E0E0' }
      }

      const rcps = ['RCP1A', 'RCP1B', 'RCP2A', 'RCP2B']
      
      // 데이터 행 추가
      for (let i = 0; i < data.length; i++) {
        const item = data[i]
        const filename = item.path.split(/[/\\]/).pop()
        const status = item.status === 'completed' ? '완료' : item.status === 'failed' ? '실패' : '대기'
        const label = item.result?.final_label?.toUpperCase() || 'N/A'

        // RCP별 결과
        const rcpResults: any = {}
        rcps.forEach((rcp) => {
          const rcpData = item.result?.results?.[rcp]
          if (rcpData) {
            const prob = (rcpData.probabilities.abnormal * 100).toFixed(1)
            rcpResults[rcp.toLowerCase()] = `${rcpData.prediction}(${prob}%)`
          } else {
            rcpResults[rcp.toLowerCase()] = 'N/A'
          }
        })

        // 행 추가
        const row = worksheet.addRow({
          filename,
          label,
          status,
          rcp1a: rcpResults.rcp1a,
          rcp1b: rcpResults.rcp1b,
          rcp2a: rcpResults.rcp2a,
          rcp2b: rcpResults.rcp2b
        })

        // 오버레이 이미지 추가
        if (item.result?.visualization) {
          // 각 RCP의 이미지를 해당 컬럼에 삽입
          const rcpImageColumns = {
            'RCP1A': 4,  // E열 (0-based index)
            'RCP1B': 6,  // G열
            'RCP2A': 8,  // I열
            'RCP2B': 10  // K열
          }
          
          for (const rcp of rcps) {
            const vizData = item.result.visualization[rcp]
            if (vizData && vizData.gradcam && vizData.gradcam.overlay) {
              const overlayPath = vizData.gradcam.overlay
              
              try {
                if (fs.existsSync(overlayPath)) {
                  const imageBuffer = fs.readFileSync(overlayPath)
                  const imageId = workbook.addImage({
                    buffer: imageBuffer,
                    extension: 'png'
                  })

                  // 이미지를 정확한 셀에 삽입
                  const imageCol = rcpImageColumns[rcp]
                  worksheet.addImage(imageId, {
                    tl: { col: imageCol, row: i + 1 },  // top-left 정확히 셀 시작점
                    ext: { width: 150, height: 150 }
                  })
                }
              } catch (imgError) {
                console.error(`[Excel] Failed to add image for ${rcp}:`, imgError)
              }
            }
          }

          // 행 높이 조정 (이미지 크기에 맞춤)
          row.height = 120
        }
      }

      // 파일 저장
      await workbook.xlsx.writeFile(result.filePath)
      console.log('[IPC] Results exported to Excel:', result.filePath)

      return { success: true, filePath: result.filePath }
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
