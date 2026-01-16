@echo off
echo ========================================================================
echo   TESTING OPENCL FIX - Work Group Size Issue
echo ========================================================================
echo.
echo WHAT WAS FIXED:
echo  - OpenCL now queries device for maximum work group size
echo  - Automatically uses safe work group size (64, 128, or 256)
echo  - Global work size properly calculated
echo.
echo PREVIOUS RESULTS:
echo  CUDA: 175.011 GB/s - PASS
echo  OpenCL: inf GB/s - FAIL (Invalid work group size)
echo  DirectCompute: 177.181 GB/s - PASS
echo.
echo EXPECTED NOW:
echo  All 3 backends should PASS!
echo.
echo ========================================================================
pause

cd /d "%~dp0"

echo.
echo Running GPU-Benchmark.exe...
echo.
build\Release\GPU-Benchmark.exe

echo.
echo ========================================================================
echo   CHECK RESULTS
echo ========================================================================
echo.
echo Look for:
echo  - CUDA VectorAdd: ~175 GB/s [PASS]
echo  - OpenCL VectorAdd: ~165-175 GB/s [PASS]  ^<-- Should work now!
echo  - DirectCompute VectorAdd: ~177 GB/s [PASS]
echo.
echo CSV file: benchmark_results_working.csv
echo.
pause
