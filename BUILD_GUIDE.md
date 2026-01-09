# 🔨 Build Guide - GPU Compute Benchmark Tool

This guide provides detailed, step-by-step instructions for building the GPU Compute Benchmark Tool from source on Windows 11.

---

## 📋 Prerequisites

### 1. Visual Studio 2019 or 2022

**Required Workloads**:
- Desktop development with C++
- Windows 10/11 SDK

**Installation**:
1. Download Visual Studio from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/)
2. Run the installer
3. Select "Desktop development with C++"
4. Ensure "Windows 11 SDK" is checked
5. Install (requires ~7GB disk space)

### 2. NVIDIA CUDA Toolkit

**Version**: 11.0 or higher (12.x recommended)

**Installation**:
1. Visit [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Select:
   - **Operating System**: Windows
   - **Architecture**: x86_64
   - **Version**: 11
   - **Installer Type**: exe (local)
3. Download (~3GB) and run installer
4. **Important**: Check "Visual Studio Integration" during installation
5. Verify installation:
   ```powershell
   nvcc --version
   ```
   Should output: `Cuda compilation tools, release 12.x`

### 3. Windows SDK

Usually included with Visual Studio. Verify it's installed:
- Open Visual Studio Installer
- Modify your installation
- Check "Windows 11 SDK (10.0.22000.0)" or later

### 4. Graphics Drivers

**NVIDIA GPU**:
- Download latest drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
- Your RTX 3050 needs driver version 450.0 or higher

---

## 🏗️ Build Process

### Method 1: Visual Studio GUI (Recommended for Beginners)

#### Step 1: Open the Project
1. Navigate to `y:\GPU-Benchmark\`
2. Double-click `GPU-Benchmark.sln`
3. Visual Studio will open

#### Step 2: Configure Build Settings
1. At the top, set build configuration:
   - **Configuration**: Release (for best performance)
   - **Platform**: x64
2. **Why Release?**: Debug builds are 10-50x slower for GPU code

#### Step 3: Build the Solution
1. Press **Ctrl+Shift+B** or
2. Menu: **Build → Build Solution**
3. Watch the Output window for progress
4. Build time: ~2-5 minutes (first build)

#### Step 4: Verify Build
Look for:
```
========== Build: 1 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========
```

Output location: `y:\GPU-Benchmark\build\Release\GPU-Benchmark.exe`

### Method 2: Command Line (Advanced)

```powershell
# Open Visual Studio Developer PowerShell
# (Search for "Developer PowerShell for VS 2022" in Start Menu)

# Navigate to project
cd y:\GPU-Benchmark

# Build using MSBuild
msbuild GPU-Benchmark.sln /p:Configuration=Release /p:Platform=x64

# Run
.\build\Release\GPU-Benchmark.exe
```

---

## 🔧 Project Configuration Details

### Solution Structure

The Visual Studio solution contains these projects:

1. **GPU-Benchmark** (Main executable)
   - Builds: `GPU-Benchmark.exe`
   - Contains: Main application, GUI, orchestration

2. **Core** (Static library)
   - Builds: `Core.lib`
   - Contains: Framework interfaces, timing, device discovery

3. **CUDA-Backend** (CUDA library)
   - Builds: `CUDABackend.lib`
   - Contains: CUDA kernels and backend implementation
   - **Special**: Uses NVCC compiler for .cu files

4. **OpenCL-Backend** (Static library)
   - Builds: `OpenCLBackend.lib`
   - Contains: OpenCL backend and kernel loaders

5. **DirectCompute-Backend** (Static library)
   - Builds: `DirectComputeBackend.lib`
   - Contains: DirectCompute backend and HLSL shaders

6. **Visualization** (Static library)
   - Builds: `Visualization.lib`
   - Contains: OpenGL renderer and GUI

### Include Directories

The project automatically configures these include paths:

```
$(ProjectDir)src\
$(ProjectDir)include\
$(CUDA_PATH)\include\
$(WindowsSdkDir)Include\
```

### Library Dependencies

**CUDA Backend**:
- `cudart_static.lib` - CUDA runtime
- `cuda.lib` - CUDA driver API

**OpenCL Backend**:
- `OpenCL.lib` - OpenCL loader

**DirectCompute Backend**:
- `d3d11.lib` - Direct3D 11
- `d3dcompiler.lib` - HLSL shader compiler
- `dxgi.lib` - DirectX Graphics Infrastructure

**Visualization**:
- `opengl32.lib` - OpenGL
- `glfw3.lib` - Windowing library (included)
- `glad.lib` - OpenGL loader (included)

---

## 🐛 Common Build Errors and Solutions

### Error: "Cannot open include file: 'cuda_runtime.h'"

**Cause**: CUDA Toolkit not installed or not found by Visual Studio

**Solution**:
1. Verify CUDA installation:
   ```powershell
   echo $env:CUDA_PATH
   ```
   Should output: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

2. If empty, reinstall CUDA Toolkit with Visual Studio integration

3. Restart Visual Studio after CUDA installation

### Error: "LINK : fatal error LNK1104: cannot open file 'cudart_static.lib'"

**Cause**: CUDA library path not configured

**Solution**:
1. Right-click project → Properties
2. Linker → General → Additional Library Directories
3. Add: `$(CUDA_PATH)\lib\x64`

### Error: "MSB8036: The Windows SDK version X.X was not found"

**Cause**: Required Windows SDK version not installed

**Solution**:
1. Open Visual Studio Installer
2. Modify installation
3. Individual Components → SDKs, libraries, and frameworks
4. Check "Windows 11 SDK (10.0.22621.0)" or latest
5. Install and restart Visual Studio

### Error: "CUDA_ARCH_BIN is not defined"

**Cause**: Project doesn't know your GPU architecture

**Solution**:
1. Open `CUDABackend` project properties
2. CUDA C/C++ → Device → Code Generation
3. Set to: `compute_86,sm_86` (for RTX 3050 - Ampere architecture)
   - RTX 3050/3060: `sm_86`
   - RTX 3070/3080/3090: `sm_86`
   - RTX 4000 series: `sm_89`

### Error: "OpenCL.lib not found"

**Cause**: OpenCL SDK not installed

**Solution**:
OpenCL.lib should be included in:
1. CUDA Toolkit: `$(CUDA_PATH)\lib\x64\OpenCL.lib` (NVIDIA)
2. Or update graphics driver (includes OpenCL runtime)

### Warning: "warning C4819: The file contains a character that cannot be represented"

**Cause**: File encoding issue

**Solution**:
1. File → Advanced Save Options
2. Encoding: UTF-8 without signature
3. Save

---

## ⚙️ Build Optimization

### Release vs Debug

| Setting | Debug | Release |
|---------|-------|---------|
| Optimization | `/Od` (disabled) | `/O2` (maximize speed) |
| Inline Functions | No | Yes |
| Runtime Checks | Enabled | Disabled |
| GPU Code Speed | Very slow | Fast |
| Recommended For | Development | Benchmarking |

**Always use Release mode for actual benchmarking!**

### CUDA Optimization Flags

In `CUDABackend` project properties → CUDA C/C++:

**Device → Code Generation**:
```
compute_86,sm_86
```
- `compute_86`: PTX intermediate code for Ampere
- `sm_86`: Native binary for Ampere

**Optimization**:
- Use fast math: `-use_fast_math`
- Max registers: `--maxrregcount=64` (tune this)

### Visual Studio Performance

**First Build Speed Tips**:
1. Close other applications (frees RAM)
2. Disable antivirus scanning of build folder temporarily
3. Use local drive (not network drive)
4. Enable multi-processor compilation:
   - Project Properties → C/C++ → General
   - Multi-processor Compilation: Yes (`/MP`)

---

## 📦 Output Files

After successful build:

```
build/
└── Release/
    ├── GPU-Benchmark.exe          # Main executable (~5MB)
    ├── Core.lib                    # Core library
    ├── CUDABackend.lib             # CUDA backend
    ├── OpenCLBackend.lib           # OpenCL backend
    ├── DirectComputeBackend.lib    # DirectCompute backend
    ├── Visualization.lib           # Renderer
    └── shaders/                    # Copied shader files
        ├── vector_add.hlsl
        ├── matrix_mul.hlsl
        ├── convolution.hlsl
        └── reduction.hlsl
```

---

## 🚀 Running the Built Application

### From Visual Studio
1. Press **F5** (Run with debugging) or **Ctrl+F5** (Run without debugging)
2. Application window should open

### From File Explorer
1. Navigate to `y:\GPU-Benchmark\build\Release\`
2. Double-click `GPU-Benchmark.exe`

### From Command Line
```powershell
cd y:\GPU-Benchmark\build\Release
.\GPU-Benchmark.exe

# Or with arguments
.\GPU-Benchmark.exe --all --output=results.csv
```

---

## 📊 Verifying Successful Build

When you run the application, you should see:

```
=== GPU Compute Benchmark Tool ===
Initializing...

Detecting Hardware...
  GPU: NVIDIA GeForce RTX 3050 Laptop GPU
  VRAM: 4096 MB
  Compute Capability: 8.6
  Driver Version: 546.12

Detecting Backends...
  ✓ CUDA Backend: Initialized
  ✓ OpenCL Backend: Initialized (Platform: NVIDIA CUDA)
  ✓ DirectCompute Backend: Initialized (Feature Level 11.0)

Ready to run benchmarks!
```

If you see all three backends initialized, your build is correct!

---

## 🔄 Incremental Builds

After modifying code:

**Small changes** (single file):
- Build time: 5-30 seconds
- Only modified compilation units rebuild

**Kernel changes** (.cu, .cl, .hlsl files):
- Build time: 10-60 seconds
- Kernel compilation is fast

**Header changes** (.h files):
- Build time: 30-120 seconds
- Files including the header rebuild

**Tip**: Use forward declarations to minimize header dependencies

---

## 🧹 Clean Build

If you encounter strange build errors:

**Visual Studio**:
1. Build → Clean Solution
2. Delete `build/` folder manually
3. Build → Rebuild Solution

**Command Line**:
```powershell
msbuild GPU-Benchmark.sln /t:Clean
Remove-Item -Recurse -Force build\
msbuild GPU-Benchmark.sln /p:Configuration=Release /p:Platform=x64
```

---

## 📚 Next Steps

After successful build:
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the codebase
2. Run the application and try different benchmarks
3. Read [RESULTS_INTERPRETATION.md](RESULTS_INTERPRETATION.md) to understand results
4. Explore the source code with extensive comments

---

## 🆘 Still Having Issues?

**Check these**:
1. Visual Studio version: 2019 or 2022
2. CUDA Toolkit installed: `nvcc --version`
3. Windows SDK: Visual Studio Installer → Modify
4. Platform: x64 (not Win32)
5. GPU drivers: Latest from NVIDIA

**System Requirements**:
- Windows 10/11 64-bit
- 10 GB free disk space (for tools)
- NVIDIA GPU (for CUDA backend)
- 4 GB RAM minimum for compilation

---

**Build Time Expectations**:
- First build: 2-5 minutes
- Incremental: 5-30 seconds
- Clean rebuild: 1-3 minutes

Happy Building! 🎉
