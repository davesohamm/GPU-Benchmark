@echo off
REM ============================================================================
REM  GPU-Benchmark Main Application Runner
REM ============================================================================
REM
REM  PURPOSE:
REM    Runs the main GPU-Benchmark.exe application with various options.
REM
REM  USAGE:
REM    RUN_MAIN_APP.cmd                  Run with default (standard suite)
REM    RUN_MAIN_APP.cmd --quick          Run quick benchmark suite
REM    RUN_MAIN_APP.cmd --standard       Run standard benchmark suite
REM    RUN_MAIN_APP.cmd --full           Run full benchmark suite
REM    RUN_MAIN_APP.cmd --help           Show help message
REM
REM ============================================================================

echo.
echo ========================================
echo   GPU Benchmark - Main Application
echo ========================================
echo.

REM Check if build directory exists
if not exist "build\Release\GPU-Benchmark.exe" (
    echo ERROR: GPU-Benchmark.exe not found!
    echo.
    echo Please build the project first:
    echo   BUILD.cmd
    echo.
    pause
    exit /b 1
)

REM Change to build directory
cd build\Release

REM Run with passed arguments (or default)
if "%1"=="" (
    echo Running with default settings standard suite...
    echo.
    GPU-Benchmark.exe
) else (
    echo Running with arguments: %*
    echo.
    GPU-Benchmark.exe %*
)

REM Return to project root
cd ..\..

echo.
echo ========================================
echo   Execution Complete
echo ========================================
echo.
echo Results saved to: benchmark_results.csv
echo.

pause
