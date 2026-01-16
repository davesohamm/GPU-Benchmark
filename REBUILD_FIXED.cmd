@echo off
echo ========================================================================
echo   REBUILDING FIXED APPLICATIONS
echo ========================================================================
echo.
echo WHAT'S BEING FIXED:
echo  1. CLI app (GPU-Benchmark.exe) now uses proper backend methods
echo  2. Each backend (CUDA/OpenCL/DirectCompute) uses its own kernels
echo  3. No more crashes from calling CUDA functions on non-CUDA backends
echo.
echo REBUILDING...
echo.

cd /d "%~dp0"

cmake --build build --config Release --target GPU-Benchmark 2>&1 | findstr /C:"error " /C:"GPU-Benchmark.exe"

if %ERRORLEVEL% == 0 (
    echo.
    echo ========================================================================
    echo   BUILD SUCCESSFUL!
    echo ========================================================================
    echo.
    echo WHAT TO TEST:
    echo.
    echo [Test 1] CLI Application:
    echo   Command: build\Release\GPU-Benchmark.exe
    echo   Expected: Tests all 3 backends (CUDA, OpenCL, DirectCompute)
    echo   Time: 30-60 seconds
    echo.
    echo [Test 2] Check Results:
    echo   File: benchmark_results_working.csv
    echo   Should show results for all available backends
    echo.
    echo ========================================================================
    echo   RUNNING CLI TEST NOW...
    echo ========================================================================
    echo.
    
    build\Release\GPU-Benchmark.exe
    
) else (
    echo.
    echo ========================================================================
    echo   BUILD FAILED!
    echo ========================================================================
    echo.
    echo Check the error messages above.
    echo Common issues:
    echo  - Missing header files
    echo  - Syntax errors
    echo  - Linker errors
    echo.
)

pause
