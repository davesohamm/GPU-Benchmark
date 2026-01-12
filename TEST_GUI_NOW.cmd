@echo off
echo ========================================
echo   GPU Benchmark GUI - READY TO TEST!
echo ========================================
echo.
echo IMPORTANT INSTRUCTIONS:
echo.
echo 1. A window will appear in 2-3 seconds
echo 2. Wait for GPU detection to complete
echo 3. You'll see:
echo    - Your GPU name
echo    - Three green checkmarks (CUDA, OpenCL, DirectCompute)
echo.
echo 4. To benchmark:
echo    - Select Backend (CUDA recommended)
echo    - Select Suite (Quick for fast test)
echo    - Click "Start Benchmark"
echo.
echo 5. Results will appear in the table!
echo.
echo Launching GUI now...
echo.

cd /d "%~dp0"
start "" "build\Release\GPU-Benchmark-GUI.exe"

echo.
echo Window should appear in 2-3 seconds!
echo.
echo If nothing happens:
echo  - Check Task Manager for GPU-Benchmark-GUI.exe
echo  - Try: build\Release\GPU-Benchmark-GUI.exe
echo.
pause
