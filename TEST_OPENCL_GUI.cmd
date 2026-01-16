@echo off
echo.
echo Testing OpenCL in GUI...
echo.
echo IMPORTANT: Before running the GUI,let me check if OpenCL works in CLI first.
echo.

echo Running CLI OpenCL test...
cd /d "%~dp0"
build\Release\GPU-Benchmark.exe > opencl_cli_test.txt 2>&1

echo.
echo Checking CLI results...
type opencl_cli_test.txt | findstr /C:"OpenCL"

echo.
echo.
echo If CLI OpenCL worked above, then the GUI should work too.
echo Press any key to launch GUI and test OpenCL there...
pause

start "" "build\Release\GPU-Benchmark-GUI.exe"

echo.
echo GUI launched!
echo.
echo Instructions:
echo 1. Select Backend: OpenCL
echo 2. Select Suite: Standard
echo 3. Click "Start Benchmark"
echo 4. Watch for crash or success
echo.
echo If it crashes, we'll add better error handling.
echo.
pause
