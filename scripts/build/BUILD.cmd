@echo off
REM Build script for GPU Benchmark project
REM Run this from Developer Command Prompt for VS 2022

echo =============================================
echo   GPU Benchmark - Build Script
echo =============================================
echo.

REM Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo ERROR: CMakeLists.txt not found!
    echo Please run this script from Y:\GPU-Benchmark
    exit /b 1
)

echo [1/4] Creating build directory...
if exist build rmdir /s /q build
mkdir build
cd build

echo.
echo [2/4] Configuring with CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..
if errorlevel 1 (
    echo ERROR: CMake configuration failed!
    cd ..
    exit /b 1
)

echo.
echo [3/4] Building Release version...
cmake --build . --config Release
if errorlevel 1 (
    echo ERROR: Build failed!
    cd ..
    exit /b 1
)

cd ..

echo.
echo [4/4] Build complete!
echo.
echo =============================================
echo   Executables created:
echo =============================================
echo   build\Release\test_logger.exe
echo   build\Release\test_cuda_simple.exe
echo   build\Release\test_cuda_backend.exe
echo =============================================
echo.
echo To run tests:
echo   .\build\Release\test_logger.exe
echo   .\build\Release\test_cuda_simple.exe
echo   .\build\Release\test_cuda_backend.exe
echo.
echo =============================================
