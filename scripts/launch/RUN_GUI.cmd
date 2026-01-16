@echo off
REM ============================================================================
REM GPU Benchmark Suite - GUI Launcher
REM ============================================================================

cd /d "%~dp0"

echo ========================================
echo   GPU Benchmark Suite - GUI
echo ========================================
echo.
echo Checking for running instances...

REM Kill any existing instances
taskkill /F /IM "GPU-Benchmark-GUI.exe" >nul 2>&1
ping 127.0.0.1 -n 2 >nul

if not exist "build\Release\GPU-Benchmark-GUI.exe" (
    echo ERROR: GPU-Benchmark-GUI.exe not found!
    echo.
    echo Please build the project first:
    echo   1. Run BUILD.cmd
    echo   2. Or build manually with CMake
    echo.
    pause
    exit /b 1
)

echo Starting GUI application...
echo.
echo PLEASE WAIT: The window may take 2-3 seconds to appear
echo              while detecting your GPU hardware...
echo.

start "" "build\Release\GPU-Benchmark-GUI.exe"

echo ========================================
echo   GUI Application Launched!
echo ========================================
echo.
echo Look for the "GPU Benchmark Suite" window
echo.
echo If no window appears:
echo   - Check Task Manager (it might be running)
echo   - Try running: build\Release\GPU-Benchmark-GUI.exe
echo   - Check GPU drivers are installed
echo.
echo Press any key to close this window...

pause >nul
