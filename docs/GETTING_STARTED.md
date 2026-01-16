# 🚀 Getting Started - Complete Setup Guide

## Quick Start (5 Minutes)

**For the impatient:** 

1. Download `GPU-Benchmark-GUI.exe` from `build/Release/`
2. Double-click to run
3. Select your GPU API (CUDA/OpenCL/DirectCompute)
4. Click "Run Benchmark"
5. Watch your GPU flex! 💪

---

## Complete Installation Guide

### System Requirements

#### Minimum Requirements
- **OS:** Windows 10 (64-bit) or Windows 11
- **GPU:** Any DirectX 11 compatible GPU
  - NVIDIA: GTX 600 series or newer
  - AMD: Radeon HD 7000 series or newer
  - Intel: HD Graphics 4000 or newer
- **RAM:** 4GB system memory
- **Storage:** 100MB free space

#### Recommended Requirements
- **OS:** Windows 11 (64-bit)
- **GPU:** NVIDIA RTX series (for full CUDA support)
  - RTX 2000/3000/4000 series recommended
- **RAM:** 8GB+ system memory
- **Storage:** 500MB free space (for source code)

#### For Development
- **Visual Studio:** 2019 or 2022 (Community/Professional/Enterprise)
- **CUDA Toolkit:** 11.0 or newer (tested with 12.x)
- **CMake:** 3.18 or newer
- **Windows SDK:** 10.0 or newer
- **Git:** For cloning the repository

---

## Option 1: Using Pre-Built Executable (Easiest)

### Step 1: Locate the Executable
```
GPU-Benchmark/
└── build/
    └── Release/
        └── GPU-Benchmark-GUI.exe  ← This is it!
```

### Step 2: Run It
- **Method 1:** Double-click `GPU-Benchmark-GUI.exe`
- **Method 2:** Use launch script:
  ```cmd
  scripts\launch\RUN_GUI.cmd
  ```

### Step 3: First Run
1. Application window opens with GPU Benchmark Suite branding
2. Check "System Capabilities" section:
   - ✅ Green = API available
   - ❌ Red = API not available (driver/hardware issue)
3. See your GPU name and available backends

### Step 4: Run Your First Benchmark
1. Select Backend: CUDA, OpenCL, or DirectCompute
2. Select Suite: Standard (recommended for first run)
3. Click "Run Benchmark" button
4. Wait ~30-60 seconds for completion
5. View results in graphs and table

---

## Option 2: Building from Source (For Developers)

### Prerequisites Installation

#### 1. Install Visual Studio 2022

**Download:** https://visualstudio.microsoft.com/downloads/

**Required Workloads:**
- "Desktop development with C++"
- "Windows SDK 10.0.xxxxx"

**Installation Steps:**
1. Run installer
2. Select "Desktop development with C++"
3. In "Individual Components" tab, ensure:
   - MSVC v143 C++ compiler
   - Windows 10/11 SDK
   - CMake tools for Windows
4. Install (takes ~15-20 minutes)

#### 2. Install CUDA Toolkit (For NVIDIA GPUs)

**Download:** https://developer.nvidia.com/cuda-downloads

**Version:** 12.x recommended (11.x also works)

**Installation Steps:**
1. Download installer (~3GB)
2. Run installer
3. Choose "Custom" installation
4. Select:
   - CUDA Compiler (nvcc)
   - CUDA Runtime
   - CUDA Documentation (optional)
   - Visual Studio Integration
5. Install (takes ~10 minutes)
6. Verify installation:
   ```cmd
   nvcc --version
   ```
   Should output: `Cuda compilation tools, release 12.x`

#### 3. Install CMake

**Download:** https://cmake.org/download/

**Version:** 3.18 or newer

**Installation Steps:**
1. Download Windows installer (x64)
2. Run installer
3. **Important:** Check "Add CMake to system PATH for all users"
4. Install
5. Verify:
   ```cmd
   cmake --version
   ```

#### 4. Install Git (Optional)

**Download:** https://git-scm.com/downloads

Only needed if cloning from repository.

---

### Building the Project

#### Step 1: Get the Source Code

**Option A: Clone from Git**
```cmd
git clone https://github.com/davesohamm/GPU-Benchmark.git
cd GPU-Benchmark
```

**Option B: Download ZIP**
1. Download ZIP from GitHub
2. Extract to `Y:\GPU-Benchmark\` (or any location)
3. Open terminal in that directory

#### Step 2: Open Developer Command Prompt

**Method 1: Via Start Menu**
1. Start Menu → Visual Studio 2022
2. Click "Developer Command Prompt for VS 2022"

**Method 2: Via Visual Studio**
1. Open Visual Studio 2022
2. Tools → Command Line → Developer Command Prompt

#### Step 3: Navigate to Project
```cmd
cd /d Y:\GPU-Benchmark
```
(Replace `Y:\` with your actual path)

#### Step 4: Download ImGui (GUI Framework)
```cmd
scripts\build\DOWNLOAD_IMGUI.cmd
```

This downloads ImGui library to `external/imgui/`

#### Step 5: Build the Project

**Option A: Use Build Script (Easiest)**
```cmd
scripts\build\BUILD.cmd
```

This script:
- Creates `build/` directory
- Runs CMake configuration
- Compiles in Release mode
- Takes ~2-3 minutes

**Option B: Manual CMake**
```cmd
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
cd ..
```

**What Gets Built:**
```
build/Release/
├── GPU-Benchmark-GUI.exe      ← Main GUI application
├── GPU-Benchmark.exe          ← CLI version (for testing)
├── test_cuda_backend.exe      ← Unit tests
├── test_opencl_backend.exe
└── ... (other test executables)
```

#### Step 6: Run the GUI
```cmd
scripts\launch\RUN_GUI.cmd
```

Or directly:
```cmd
build\Release\GPU-Benchmark-GUI.exe
```

---

## Troubleshooting

### Build Errors

#### Error: "CUDA not found"
**Solution:**
1. Install CUDA Toolkit from NVIDIA
2. Ensure `nvcc` is in PATH:
   ```cmd
   where nvcc
   ```
   Should output: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\nvcc.exe`
3. Restart terminal/Visual Studio

#### Error: "CMake not found"
**Solution:**
1. Install CMake
2. Add to PATH manually:
   - Control Panel → System → Advanced → Environment Variables
   - Add `C:\Program Files\CMake\bin` to PATH
3. Restart terminal

#### Error: "Cannot open compiler"
**Solution:**
1. Open **Developer Command Prompt** (not regular CMD)
2. Or run: `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`

#### Error: "LNK1104: cannot open file"
**Solution:**
1. Close all running instances of the exe
2. Run:
   ```cmd
   taskkill /F /IM GPU-Benchmark-GUI.exe
   ```
3. Rebuild

#### Error: "ImGui not found"
**Solution:**
```cmd
scripts\build\DOWNLOAD_IMGUI.cmd
```

Then rebuild.

### Runtime Errors

#### Error: "CUDA driver version mismatch"
**Solution:**
1. Update NVIDIA drivers: https://www.nvidia.com/download/index.aspx
2. Restart computer
3. Run again

#### Error: "OpenCL.dll not found"
**Solution:**
- OpenCL should come with GPU drivers
- Reinstall GPU drivers (NVIDIA/AMD/Intel)

#### Error: "d3d11.dll not found"
**Solution:**
- Update Windows
- Install DirectX End-User Runtime: https://www.microsoft.com/en-us/download/details.aspx?id=35

#### App crashes on startup
**Solution:**
1. Check Windows Event Viewer for details
2. Ensure GPU drivers are up to date
3. Try running as Administrator
4. Check if antivirus is blocking it

---

## Understanding the Output

### GUI Application

#### System Capabilities Section
Shows what's available on your system:
```
CUDA:          ✅ Available (NVIDIA RTX 3050)
OpenCL:        ✅ Available (v3.0)
DirectCompute: ✅ Available (DirectX 11.1)
```

#### Benchmark Selection
- **Backend:** CUDA / OpenCL / DirectCompute
- **Suite:** 
  - Quick (10M elements) - 10 seconds
  - Standard (50M elements) - 30 seconds
  - Comprehensive (100M elements) - 60 seconds

#### Results Display
- **Live Progress Bar:** Shows current benchmark progress
- **Performance Graphs:** Real-time line charts for each benchmark
- **History Tracking:** Stores up to 100 test results
- **Test Indexing:** "Test 1", "Test 2", etc. with timestamps

#### Metrics Explained
- **Bandwidth (GB/s):** Memory transfer speed (higher = better)
- **GFLOPS:** Compute performance (billions of FLOPs per second)
- **Time (ms):** How long the benchmark took

---

## First Benchmark Run - What to Expect

### CUDA (NVIDIA GPUs)

**Expected Performance on RTX 3050:**
```
VectorAdd:     ~180 GB/s (80% of peak bandwidth)
MatrixMul:     ~800-1000 GFLOPS
Convolution:   ~300 GB/s
Reduction:     ~150 GB/s
```

**Completion Time:** ~30 seconds (Standard suite)

### OpenCL (Cross-Vendor)

**Expected Performance:**
```
VectorAdd:     ~150-170 GB/s
MatrixMul:     ~700-900 GFLOPS
Convolution:   ~250 GB/s
Reduction:     ~130 GB/s
```

Slightly slower than CUDA due to driver overhead.

### DirectCompute (Windows Native)

**Expected Performance:**
```
VectorAdd:     ~140-160 GB/s
MatrixMul:     ~600-800 GFLOPS
Convolution:   ~200 GB/s
Reduction:     ~120 GB/s
```

Good performance, always available on Windows.

---

## CSV Export

### Exporting Results

1. Click "Export CSV" button
2. Choose save location (file dialog opens)
3. Enter filename (e.g., `my_benchmark_results.csv`)
4. Click "Save"

### CSV Format
```csv
Backend,Benchmark,Bandwidth(GB/s),GFLOPS,Time(ms),Timestamp
CUDA,VectorAdd,182.4,0.0,0.82,2026-01-09 14:30:45
CUDA,MatrixMul,245.6,1023.5,8.45,2026-01-09 14:30:54
...
```

### Analyzing in Excel/Python

**Excel:**
1. Open CSV in Excel
2. Create charts from data
3. Use PivotTables for analysis

**Python:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df.groupby('Backend')['Bandwidth(GB/s)'].mean().plot(kind='bar')
plt.show()
```

---

## Advanced Usage

### Running Specific Benchmarks

Currently, GUI runs all 4 benchmarks. For selective testing, use CLI version (in development).

### Comparing Multiple Runs

1. Run benchmark with Backend A
2. Run benchmark with Backend B
3. Compare graphs side-by-side
4. Export both to CSV for detailed analysis

### Custom Problem Sizes

Modify suite selection:
- **Quick:** 10M elements (~10 sec)
- **Standard:** 50M elements (~30 sec)
- **Comprehensive:** 100M elements (~60 sec)

Larger sizes = more accurate measurements (less noise).

---

## Command-Line Scripts

### Build Scripts (`scripts/build/`)

```cmd
BUILD.cmd                  # Full build (recommended)
REBUILD_FIXED.cmd          # Clean rebuild
check_setup.ps1            # Verify environment setup (PowerShell)
DOWNLOAD_IMGUI.cmd         # Download ImGui library
```

### Launch Scripts (`scripts/launch/`)

```cmd
RUN_GUI.cmd                # Launch GUI application
LAUNCH_GUI_SIMPLE.cmd      # Simple launcher
RUN_MAIN_APP.cmd           # Launch CLI version
```

### Test Scripts (`tests/test-scripts/`)

```cmd
RUN_ALL_TESTS.cmd          # Run all unit tests
TEST_COMPLETE_SUITE.cmd    # Comprehensive test suite
TEST_ALL_BACKENDS_GUI.cmd  # Test all backends in GUI
```

---

## Configuration

### Changing CUDA Compute Capability

If you have a different NVIDIA GPU:

**Edit:** `CMakeLists.txt`
```cmake
# Line 14:
set(CMAKE_CUDA_ARCHITECTURES 86)  # Change to your GPU's compute capability
```

**Compute Capabilities:**
- RTX 4000 series: 89
- RTX 3000 series: 86
- RTX 2000 series: 75
- GTX 1000 series: 61
- GTX 900 series: 52

**Reference:** https://developer.nvidia.com/cuda-gpus

### Changing Problem Sizes

**Edit:** `src/gui/main_gui_fixed.cpp`

Search for "Suite" definitions and modify:
```cpp
case 0: size = 10'000'000; break;   // Quick
case 1: size = 50'000'000; break;   // Standard
case 2: size = 100'000'000; break;  // Comprehensive
```

Rebuild after changes:
```cmd
scripts\build\BUILD.cmd
```

---

## Video Tutorial (Recommended!)

Coming soon: Step-by-step video guide on YouTube.

---

## Next Steps

### After Your First Benchmark:

1. **Explore Different Backends** - Compare CUDA vs OpenCL vs DirectCompute
2. **Try Different Suites** - See how performance scales with problem size
3. **Export to CSV** - Analyze results in Excel/Python
4. **Read Documentation** - Understand what each benchmark measures
5. **Check Source Code** - Learn GPU programming from working examples

### Learning Resources:

- [Why This Project?](docs/WHY_THIS_PROJECT.md) - Philosophy and motivation
- [Architecture](docs/ARCHITECTURE.md) - Deep technical dive
- [Internal Workings](docs/INTERNAL_WORKINGS.md) - How everything works
- [Build Setup Guide](docs/build-setup/BUILD_GUIDE.md) - Detailed build instructions

---

## Getting Help

### Documentation
- **Main README:** `README.md` (you are here)
- **Architecture:** `docs/ARCHITECTURE.md`
- **Build Guide:** `docs/build-setup/BUILD_GUIDE.md`
- **User Guides:** `docs/user-guides/`

### Common Issues
- Check [Troubleshooting](#troubleshooting) section above
- Read `docs/bug-fixes/` folder for known issues
- Check GitHub Issues (if public repo)

### Contact
- **Developer:** Soham Dave
- **GitHub:** https://github.com/davesohamm
- **Project:** GPU Benchmark Suite v1.0

---

## Success Checklist

After following this guide, you should have:

- [ ] GPU-Benchmark-GUI.exe running successfully
- [ ] Completed at least one benchmark run
- [ ] Seen performance graphs with your GPU's results
- [ ] Understanding of what each benchmark measures
- [ ] Exported results to CSV (optional)
- [ ] Explored different backends (CUDA/OpenCL/DirectCompute)

---

**Congratulations! You're now ready to benchmark your GPU like a pro!** 🎉🚀

**Next:** Read the [main README](../README.md) for complete project documentation.

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**For:** GPU Benchmark Suite
