@echo off
echo ================================================================================
echo   GPU Benchmark GUI - Test All Backends
echo ================================================================================
echo.
echo This will test all three backends (CUDA, OpenCL, DirectCompute) in the GUI.
echo.
echo FIXED ISSUES:
echo  - Added proper error handling (no more sudden crashes)
echo  - GUI now uses actual benchmark classes (same as CLI)
echo  - Exceptions are caught and displayed
echo.
echo INSTRUCTIONS:
echo.
echo 1. GUI window will open
echo 2. Test each backend:
echo    A) CUDA + Quick (should work - 15 sec)
echo    B) DirectCompute + Quick (should work - 20 sec)
echo    C) OpenCL + Quick (FIXED - should work now! 20 sec)
echo.
echo 3. If OpenCL still crashes:
echo    - The error will be displayed in the GUI
echo    - Window will NOT close suddenly
echo.
echo Press any key to launch GUI...
pause >nul

cd /d "%~dp0"
start "" "build\Release\GPU-Benchmark-GUI.exe"

echo.
echo GUI launched!
echo.
echo TEST PROCEDURE:
echo ----------------
echo [Test 1] CUDA + Quick:
echo   - Select Backend: CUDA
echo   - Select Suite: Quick  
echo   - Click "Start Benchmark"
echo   - Wait for results (15 seconds)
echo   - Should show: VectorAdd result with ~170 GB/s
echo.
echo [Test 2] DirectCompute + Quick:
echo   - Select Backend: DirectCompute
echo   - Select Suite: Quick
echo   - Click "Start Benchmark"  
echo   - Wait for results (20 seconds)
echo   - Should show: VectorAdd result with ~145-160 GB/s
echo.
echo [Test 3] OpenCL + Quick (THE FIX!):
echo   - Select Backend: OpenCL
echo   - Select Suite: Quick
echo   - Click "Start Benchmark"
echo   - Wait for results (20 seconds - first run compiles kernels)
echo   - Should show: VectorAdd result with ~155-170 GB/s
echo   - If error: Will display in GUI (no crash!)
echo.
echo If OpenCL works = WE'RE READY TO DISTRIBUTE! 
echo.
pause
