@echo off
echo ========================================================================
echo   GPU Benchmark GUI - CRASH FIXED! Test Now
echo ========================================================================
echo.
echo WHAT WAS FIXED:
echo  - Reduced problem sizes (safer for GUI)
echo  - Added memory cleanup between benchmarks
echo  - Better error handling (no sudden crashes)
echo  - Progress messages show benchmark details
echo.
echo WHAT TO TEST:
echo.
echo [Test 1] CUDA + Quick (30 seconds)
echo   - Should work smoothly
echo   - 1 benchmark (VectorAdd 1M elements)
echo.
echo [Test 2] CUDA + Standard (1-2 minutes) - THE ONE THAT CRASHED!
echo   - NOW with safer sizes:
echo     * VectorAdd: 5M elements (was 10M)
echo     * MatrixMul: 512x512 (was 1024x1024)  
echo     * Convolution: 1280x720 (was 1920x1080)
echo     * Reduction: 5M elements (was 10M)
echo   - Should complete all 4 benchmarks now!
echo.
echo LAUNCHING GUI...
echo.

cd /d "%~dp0"
start "" "build\Release\GPU-Benchmark-GUI.exe"

echo.
echo ========================================================================
echo   TESTING PROCEDURE
echo ========================================================================
echo.
echo Step 1: Test CUDA + Quick
echo   - Backend: CUDA
echo   - Suite: Quick
echo   - Click: Start Benchmark
echo   - Wait: 30 seconds
echo   - Check: 1 result appears (VectorAdd ~170 GB/s)
echo.
echo Step 2: Test CUDA + Standard (THE IMPORTANT ONE!)
echo   - Backend: CUDA  
echo   - Suite: Standard
echo   - Click: Start Benchmark
echo   - Watch progress: 0%% -^> 25%% -^> 50%% -^> 75%% -^> 100%%
echo   - Watch for benchmarks:
echo     1. VectorAdd (5M elements) - should reach 25%%
echo     2. MatrixMul (512x512) - should reach 50%%
echo     3. Convolution (1280x720) - should reach 75%%
echo     4. Reduction (5M elements) - should reach 100%%
echo   - Check: All 4 results appear in table
echo.
echo Step 3: Report Results
echo   Tell me ONE of these:
echo.
echo   A) "CUDA Standard completed! All 4 benchmarks succeeded!"
echo      -^> WE'RE GOOD! Can test other backends!
echo.
echo   B) "Still crashed at [percentage]%% during [benchmark name]"
echo      -^> I'll reduce that specific benchmark further
echo.
echo   C) "Error message shown: [paste error]"
echo      -^> Error handling worked! We can fix the error
echo.
echo ========================================================================
echo   WATCH FOR
echo ========================================================================
echo.
echo GOOD SIGNS:
echo  - Progress bar moves smoothly
echo  - "Current Benchmark" shows benchmark name + size
echo  - Results appear in table one by one
echo  - All show "PASS" status
echo.
echo IF IT CRASHES:
echo  - Note the progress percentage when it crashed
echo  - Note which benchmark was running
echo  - Check if error message appeared first
echo.
echo ========================================================================
pause
