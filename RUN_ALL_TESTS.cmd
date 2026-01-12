@echo off
REM Automated test script for GPU Benchmark suite
REM Run this after BUILD.cmd to test all kernels

echo ========================================
echo   GPU Benchmark - Complete Test Suite
echo ========================================
echo.

cd /d Y:\GPU-Benchmark\build\Release

if not exist test_logger.exe (
    echo ERROR: Executables not found!
    echo Please run BUILD.cmd first
    pause
    exit /b 1
)

echo [1/6] Testing Logger...
echo ----------------------------------------
call test_logger.exe
if errorlevel 1 (
    echo FAILED: test_logger
    pause
    exit /b 1
)
echo.

echo [2/6] Testing Simple CUDA...
echo ----------------------------------------
call test_cuda_simple.exe
if errorlevel 1 (
    echo FAILED: test_cuda_simple
    pause
    exit /b 1
)
echo.

echo [3/6] Testing CUDA Backend...
echo ----------------------------------------
call test_cuda_backend.exe
if errorlevel 1 (
    echo FAILED: test_cuda_backend
    pause
    exit /b 1
)
echo.

echo [4/6] Testing Matrix Multiplication...
echo ----------------------------------------
call test_matmul.exe
if errorlevel 1 (
    echo FAILED: test_matmul
    pause
    exit /b 1
)
echo.

echo [5/6] Testing 2D Convolution...
echo ----------------------------------------
call test_convolution.exe
if errorlevel 1 (
    echo FAILED: test_convolution
    pause
    exit /b 1
)
echo.

echo [6/6] Testing Parallel Reduction...
echo ----------------------------------------
call test_reduction.exe
if errorlevel 1 (
    echo FAILED: test_reduction
    pause
    exit /b 1
)
echo.

echo ========================================
echo   ALL TESTS PASSED SUCCESSFULLY!
echo ========================================
echo.
echo Test Suite Complete:
echo   - Logger: OK
echo   - Simple CUDA: OK
echo   - CUDA Backend: OK
echo   - Matrix Multiplication: OK
echo   - 2D Convolution: OK
echo   - Parallel Reduction: OK
echo.
echo Your RTX 3050 is CRUSHING IT! 
echo ========================================
echo.
pause
