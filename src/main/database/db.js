import Database from 'better-sqlite3'
import { app } from 'electron'
import { join } from 'path'

// db 파일 저장 위치 설정
// app.getPath('userData') => 이 앱이 데이터를 저장해도 되는 공식적인 폴더를 자동으로 찾아줌
const dbPath = join(app.getPath('userData'), 'logs.db')

// db 연결 => 파일이 없을 경우 자동 생성
const db = new Database(dbPath, { verbose: console.log })

// table 초기화
export function initDB() {
  // 기본적인 로그 테이블 system_logs 생성(id, 작업내용, created_at)
  db.exec(`
    CREATE TABLE IF NOT EXISTS system_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      action TEXT NOT NULL,
      details TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `)

  // User 테이블 -> 비밀번호는 해시값으로 저장하기
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `)
  console.log('Database initialized at: ', dbPath)
}

// 외부에서 호출 가능한 로그 저장 함수
export function insertLog(action, details = '') {
  try {
    const stmt = db.prepare('INSERT INTO system_logs (action, details) VALUES (?, ?)')
    stmt.run(action, details)
    console.log(`[LOG SAVED] ${action}: ${details}`)
    return true
  } catch (error) {
    console.error('Failed to save log:', error)
    return false
  }
}

// 테스트용 최근 로그 조회 함수
export function getRecentLogs(limit = 10) {
  const stmt = db.prepare('SELECT * FROM system_logs ORDER BY id DESC LIMIT ?')
  return stmt.all(limit)
}

// ID로 유저 찾기
export function findUserByUsername(username) {
  try {
    const stmt = db.prepare('SELECT * FROM users WHERE username = ?')
    return stmt.get(username) // 없으면 undefined 반환됨 (정상)
  } catch (error) {
    console.error('DB Find User Error:', error)
    return undefined
  }
}

// 유저 생성 <- 회원가입
export function createUser(username, hashedPassword) {
  try {
    const stmt = db.prepare('INSERT INTO users (username, password) VALUES (?, ?)')
    const info = stmt.run(username, hashedPassword)
    return info.changes > 0 // 성공 시 true 반환
  } catch (error) {
    console.error('User creation failed:', error)
    return false // 중복 ID 등 에러 발생 시
  }
}
