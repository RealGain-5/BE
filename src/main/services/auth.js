import Store from 'electron-store'
import bcrypt from 'bcryptjs' // <- 암호화 라이브러리
import { findUserByUsername, createUser } from '../database/db'

// 세션 저장소 초기화 => 추후 암호화 옵션 추가하기
const store = new Store({ name: 'user-session' })

// 회원가입 처리
export async function registerUser(id, pw) {
  // 이미 존재하는 ID 인지 확인
  const existingUser = findUserByUsername(id)
  // 존재하는 아이디 에러 처리
  if (existingUser) {
    return { success: false, message: '이미 존재하는 아이디입니다.' }
  }

  // 비밀번호 암호화
  const hashedPassword = bcrypt.hashSync(pw, 10)

  // DB에 저장
  const isCreated = createUser(id, hashedPassword)

  if (isCreated) {
    return { success: true, message: '회원가입 성공! 로그인해주세요.' }
  } else {
    return { success: false, message: '회원가입 중 오류가 발생했습니다.' }
  }
}

// 로그인 처리
export async function loginUser(id, pw) {
  try {
    const user = findUserByUsername(id)

    // 유저가 없거나 비밀번호 불일치
    if (!user || !bcrypt.compareSync(pw, user.password)) {
      return { success: false, message: '아이디 또는 비밀번호가 일치하지 않습니다.' }
    }

    // 로그인 성공: 세션 저장
    const sessionToken = `local_token_${Date.now()}`

    // 저장하기 전에 객체가 확실한지 확인
    const userInfo = { username: user.username, id: user.id }

    store.set('auth_token', sessionToken)
    store.set('user_info', userInfo)

    return { success: true, username: user.username }
  } catch (err) {
    console.error('Login Error:', err)
    return { success: false, message: '로그인 처리 중 오류 발생' }
  }
}

// 로그아웃 처리
export function logoutUser() {
  store.delete('auth_token')
  store.delete('user_info')
  return { success: true }
}

// 세션 체크
export function checkAuth() {
  try {
    const token = store.get('auth_token')
    const user = store.get('user_info')

    // user가 undefined일 때 접근하지 않도록 방어 코드 추가
    if (token && user && user.username) {
      return { isLoggedIn: true, user }
    }
    return { isLoggedIn: false }
  } catch (err) {
    console.error('Session Check Error:', err)
    return { isLoggedIn: false }
  }
}
