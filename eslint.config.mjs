import { defineConfig } from 'eslint/config'
import tseslint from '@electron-toolkit/eslint-config-ts'
import eslintConfigPrettier from '@electron-toolkit/eslint-config-prettier'

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()]
  },
  preload: {
    plugins: [externalizeDepsPlugin()]
  },
  renderer: {
    // 서브모듈 안의 index.html를 가리킴
    resolve: {
      alias: {
        '@renderer': resolve('src/renderer/src')
      }
    },
    plugins: [react()]
  }
})
