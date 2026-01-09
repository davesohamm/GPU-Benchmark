# Simple setup checker for GPU Benchmark project

Write-Host ""
Write-Host "=== GPU BENCHMARK - SETUP CHECK ===" -ForegroundColor Cyan
Write-Host ""

# Check Visual Studio
Write-Host "1. Visual Studio C++ Compiler..." -ForegroundColor Yellow
if (Get-Command cl -ErrorAction SilentlyContinue) {
    Write-Host "   FOUND" -ForegroundColor Green
} else {
    Write-Host "   NOT FOUND - Install Visual Studio 2022" -ForegroundColor Red
}

# Check CUDA
Write-Host "2. CUDA Toolkit..." -ForegroundColor Yellow
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    Write-Host "   FOUND" -ForegroundColor Green
    nvcc --version | Select-String "release"
} else {
    Write-Host "   NOT FOUND - Install from developer.nvidia.com/cuda-downloads" -ForegroundColor Red
}

# Check CMake
Write-Host "3. CMake..." -ForegroundColor Yellow
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "   FOUND" -ForegroundColor Green
    cmake --version | Select-Object -First 1
} else {
    Write-Host "   NOT FOUND - Install from cmake.org" -ForegroundColor Red
}

# Check NVIDIA GPU
Write-Host "4. NVIDIA GPU..." -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "   FOUND" -ForegroundColor Green
    nvidia-smi --query-gpu=name --format=csv,noheader
} else {
    Write-Host "   nvidia-smi not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "Read STATUS_AND_SETUP.md for installation instructions" -ForegroundColor White
Write-Host ""
