@echo off
echo ========================================================================
echo   TESTING FIXED CLI APPLICATION
echo ========================================================================
echo.
echo WHAT WAS FIXED:
echo  - Each backend now uses its OWN kernel execution method
echo  - CUDA uses CUDA launchers
echo  - OpenCL uses OpenCL kernel compilation/execution
echo  - DirectCompute uses HLSL shader compilation/dispatch
echo.
echo WHAT TO EXPECT:
echo  - All 3 backends tested (CUDA, OpenCL, DirectCompute)
echo  - ~30-60 seconds total (includes kernel compilation)
echo  - Results for each backend
echo  - CSV file exported
echo  - NO CRASHES!
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
echo   TEST COMPLETE!
echo ========================================================================
echo.
echo CHECK:
echo  1. Did all 3 backends complete? (CUDA, OpenCL, DirectCompute)
echo  2. Did you see GB/s values for each?
echo  3. Did all show [PASS] status?
echo  4. Was benchmark_results_working.csv created?
echo.
echo IF ALL YES:
echo  - Application is WORKING!
echo  - All backends functional!
echo  - Ready to fix GUI next!
echo.
echo IF ANY NO:
echo  - Tell me which backend failed
echo  - Copy any error messages
echo  - Check if test programs work:
echo    * test_opencl_backend.exe
echo    * test_directcompute_backend.exe
echo.
pause
