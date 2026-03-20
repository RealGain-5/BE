@echo off
echo ================================================
echo Python Executable Build Script
echo ================================================
echo.

:: PyInstaller 설치 확인 및 설치
echo [1/3] Checking PyInstaller...
..\venv\Scripts\pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    ..\venv\Scripts\pip install pyinstaller
) else (
    echo PyInstaller is already installed.
)
echo.

:: 빌드 실행
echo [2/3] Building executable...
..\venv\Scripts\python build_executable.py
if errorlevel 1 (
    echo.
    echo ================================================
    echo Build failed! Please check the error messages.
    echo ================================================
    pause
    exit /b 1
)
echo.

:: 결과 확인
echo [3/3] Verifying build...
if exist "dist\inference_daemon.exe" (
    echo.
    echo ================================================
    echo SUCCESS! Executable created at:
    echo %cd%\dist\inference_daemon.exe
    echo ================================================
    echo.
    echo File size:
    dir "dist\inference_daemon.exe" | find "inference_daemon.exe"
) else (
    echo.
    echo ================================================
    echo ERROR: Executable not found!
    echo ================================================
)
echo.
pause
