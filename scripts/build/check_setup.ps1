################################################################################
# check_setup.ps1
#
# PURPOSE: Check if development environment is ready for GPU Benchmark project
#
# USAGE: 
#   Open PowerShell in this directory and run:
#   .\check_setup.ps1
################################################################################

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║        GPU BENCHMARK - DEVELOPMENT ENVIRONMENT CHECK          ║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: Visual Studio
Write-Host "1. Checking Visual Studio C++ Compiler..." -ForegroundColor Yellow
try {
    $clPath = where.exe cl 2>$null
    if ($clPath) {
        Write-Host "   ✓ FOUND: C++ Compiler (cl.exe)" -ForegroundColor Green
        $clVersion = & cl 2>&1 | Select-String "Version"
        Write-Host "     $clVersion" -ForegroundColor Gray
    } else {
        Write-Host "   ✗ NOT FOUND: Visual Studio C++ Compiler" -ForegroundColor Red
        Write-Host "     Install: Visual Studio 2022 with C++ Desktop Development" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "   ✗ NOT FOUND: Visual Studio C++ Compiler" -ForegroundColor Red
    Write-Host "     Install: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check 2: CUDA Toolkit
Write-Host "2. Checking CUDA Toolkit..." -ForegroundColor Yellow
try {
    $nvccPath = where.exe nvcc 2>$null
    if ($nvccPath) {
        Write-Host "   ✓ FOUND: CUDA Compiler (nvcc)" -ForegroundColor Green
        $cudaVersion = & nvcc --version 2>&1 | Select-String "release"
        Write-Host "     $cudaVersion" -ForegroundColor Gray
    } else {
        Write-Host "   ✗ NOT FOUND: CUDA Toolkit" -ForegroundColor Red
        Write-Host "     Install: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "   ✗ NOT FOUND: CUDA Toolkit" -ForegroundColor Red
    Write-Host "     Required for CUDA backend development" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check 3: CMake
Write-Host "3. Checking CMake..." -ForegroundColor Yellow
try {
    $cmakePath = where.exe cmake 2>$null
    if ($cmakePath) {
        Write-Host "   ✓ FOUND: CMake" -ForegroundColor Green
        $cmakeVersion = & cmake --version 2>&1 | Select-String "version"
        Write-Host "     $cmakeVersion" -ForegroundColor Gray
    } else {
        Write-Host "   ✗ NOT FOUND: CMake" -ForegroundColor Red
        Write-Host "     Install: https://cmake.org/download/" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "   ✗ NOT FOUND: CMake" -ForegroundColor Red
    Write-Host "     Required for building the project" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check 4: NVIDIA GPU
Write-Host "4. Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaSmiPath = where.exe nvidia-smi 2>$null
    if ($nvidiaSmiPath) {
        Write-Host "   ✓ FOUND: NVIDIA GPU Driver" -ForegroundColor Green
        $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
        Write-Host "     GPU: $gpuInfo" -ForegroundColor Gray
    } else {
        Write-Host "   ⚠ WARNING: nvidia-smi not found" -ForegroundColor Yellow
        Write-Host "     Update GPU drivers from: https://www.nvidia.com/drivers" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠ WARNING: Could not query NVIDIA GPU" -ForegroundColor Yellow
}
Write-Host ""

# Check 5: System Information
Write-Host "5. System Information..." -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor
$ram = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)

Write-Host "   OS: $($os.Caption) Build $($os.BuildNumber)" -ForegroundColor Gray
Write-Host "   CPU: $($cpu.Name)" -ForegroundColor Gray
Write-Host "   RAM: $ram GB" -ForegroundColor Gray
Write-Host ""

# Check 6: Project Files
Write-Host "6. Checking Project Files..." -ForegroundColor Yellow
$requiredFiles = @(
    "README.md",
    "CMakeLists.txt",
    "src\main.cpp",
    "src\core\Timer.h",
    "src\core\Timer.cpp",
    "src\core\DeviceDiscovery.h",
    "src\core\DeviceDiscovery.cpp"
)

$filesOk = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ MISSING: $file" -ForegroundColor Red
        $filesOk = $false
    }
}

if (!$filesOk) {
    Write-Host "   Some project files are missing!" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Summary
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✓ READY TO DEVELOP! All required tools are installed." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Green
    Write-Host "1. Read QUICKSTART.md for development guide" -ForegroundColor White
    Write-Host "2. Build the project:" -ForegroundColor White
    Write-Host "   mkdir build" -ForegroundColor Gray
    Write-Host "   cd build" -ForegroundColor Gray
    Write-Host "   cmake .. -G 'Visual Studio 17 2022' -A x64" -ForegroundColor Gray
    Write-Host "   cmake --build . --config Release" -ForegroundColor Gray
} else {
    Write-Host "✗ SETUP INCOMPLETE - Some tools are missing!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Read STATUS_AND_SETUP.md for installation instructions" -ForegroundColor White
    Write-Host "2. Install missing tools listed above" -ForegroundColor White
    Write-Host "3. Restart PowerShell and run this script again" -ForegroundColor White
}
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Pause so user can read
Read-Host "Press Enter to exit"
