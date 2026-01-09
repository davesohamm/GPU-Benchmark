# 🚀 Quick Start Guide

## Get Started in 5 Minutes!

This guide will help you understand and start working on the GPU Benchmark project **right now**.

---

## 📋 Prerequisites Check

Before you begin, verify you have:

- [ ] **Windows 11** (you have this!)
- [ ] **Visual Studio 2022** with C++ desktop development
- [ ] **NVIDIA RTX 3050** (you have this!)
- [ ] **NVIDIA Drivers** (latest version)
- [ ] **CUDA Toolkit 12.x** (Download: https://developer.nvidia.com/cuda-downloads)
- [ ] **CMake 3.18+** (Download: https://cmake.org/download/)

### Quick Verification Commands

Open PowerShell and run:

```powershell
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check CMake
cmake --version

# Check Visual Studio
code --version  # or check Start Menu for "Visual Studio 2022"
```

If any command fails, install the missing tool first!

---

## 📚 What You Have Right Now

### ✅ **Complete** (Ready to Use)
1. **All Documentation** - 5 comprehensive markdown files
2. **Core Framework** - Timer, DeviceDiscovery, Interface definitions
3. **Main Application** - Entry point with CLI mode
4. **Build System** - CMakeLists.txt
5. **Example CUDA Kernel** - vector_add.cu with detailed comments

### 🔨 **To Be Implemented** (Your Work)
1. **Logger.cpp** - CSV export and console output
2. **BenchmarkRunner** - Orchestrates benchmark execution
3. **CUDA Backend** - Implements IComputeBackend for CUDA
4. **Benchmarks** - VectorAddBenchmark, etc.
5. **OpenCL/DirectCompute** - Additional backends (optional)
6. **Visualization** - OpenGL rendering (bonus)

---

## 🎯 Your First Hour - Step by Step

### Step 1: Read the Main Documents (15 minutes)

**Read in this order:**

1. **README.md** (5 min) - Get the big picture
2. **PROJECT_SUMMARY.md** (10 min) - Understand what's done and what's next

**Skip for now:** ARCHITECTURE.md, BUILD_GUIDE.md (read when needed)

### Step 2: Explore the Code Structure (10 minutes)

Open these files and skim them (don't read every line yet):

```
src/
├── main.cpp                    # Start here - application flow
├── core/
│   ├── IComputeBackend.h      # The interface all backends implement
│   ├── Timer.h                # Timing utility
│   └── DeviceDiscovery.h      # GPU detection
└── backends/
    └── cuda/
        ├── README.md          # CUDA-specific guide
        └── kernels/
            └── vector_add.cu  # Example kernel (well commented!)
```

**Goal:** Get a feel for the organization.

### Step 3: Try to Build (10 minutes)

Even though it's not complete, let's see what happens:

```powershell
# Navigate to project
cd Y:\GPU-Benchmark

# Create build directory
mkdir build
cd build

# Generate Visual Studio solution
cmake .. -G "Visual Studio 17 2022" -A x64

# If successful, you'll see GPU-Benchmark.sln
```

**Expected Result:** CMake will generate the solution but may show warnings about missing files. That's OK!

### Step 4: Plan Your Implementation (15 minutes)

**Decision time: What to implement first?**

For your **first working demo**, implement **in this order**:

1. **Logger.cpp** (1-2 hours)
   - Start with just console output
   - Add CSV later

2. **Simple CUDA Backend** (2-3 hours)
   - Just enough to launch vector_add kernel
   - Don't implement all methods yet

3. **Simple VectorAddBenchmark** (1 hour)
   - One benchmark that uses the CUDA backend

4. **Connect everything in main.cpp** (30 minutes)
   - Make it run end-to-end

**Total time to first demo: ~5-7 hours**

### Step 5: Read the Detailed Implementation Guides (10 minutes)

When you're ready to code, read these:

1. **src/core/README.md** - Understand the core framework
2. **src/backends/cuda/README.md** - CUDA implementation guide
3. **src/backends/cuda/kernels/vector_add.cu** - Study the example kernel

---

## 🎬 Implementation Order (Recommended)

### Phase 1: Make Something Run (High Priority)

**Goal:** Get "Hello World" working - a minimal system that runs ONE benchmark on CUDA.

#### 1. Implement Logger.cpp (Start Here!)

**File:** `src/core/Logger.cpp`
**Time:** 1-2 hours

**What to implement:**
```cpp
// Minimum viable implementation:
void Logger::Info(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void Logger::Error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

// Leave CSV export for later
```

**Why start here?**
- Simplest component
- Needed by everything else
- Builds confidence
- Immediate visual feedback

#### 2. Create Minimal CUDA Backend

**File:** `src/backends/cuda/CUDABackend.h`

**Template:**
```cpp
#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "core/IComputeBackend.h"
#include <cuda_runtime.h>

namespace GPUBenchmark {

class CUDABackend : public IComputeBackend {
public:
    CUDABackend();
    ~CUDABackend();
    
    // IComputeBackend implementation
    bool Initialize() override;
    void Shutdown() override;
    
    void* AllocateMemory(size_t size) override;
    void FreeMemory(void* ptr) override;
    
    void CopyHostToDevice(void* dst, const void* src, size_t size) override;
    void CopyDeviceToHost(void* dst, const void* src, size_t size) override;
    
    bool ExecuteKernel(const std::string& name, const KernelParams& params) override;
    void Synchronize() override;
    
    DeviceInfo GetDeviceInfo() const override;
    BackendType GetBackendType() const override { return BackendType::CUDA; }
    std::string GetBackendName() const override { return "CUDA"; }
    
    void StartTimer() override;
    void StopTimer() override;
    double GetElapsedTime() override;
    
    bool IsAvailable() const override;
    std::string GetLastError() const override;

private:
    int m_deviceID;
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    std::string m_lastError;
};

} // namespace GPUBenchmark

#endif // CUDA_BACKEND_H
```

**File:** `src/backends/cuda/CUDABackend.cpp`

**Start with just these methods:**
```cpp
#include "CUDABackend.h"
#include <iostream>

namespace GPUBenchmark {

bool CUDABackend::Initialize() {
    // Check if CUDA available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return false;
    
    // Set device
    cudaSetDevice(0);
    
    // Create events for timing
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
    
    return true;
}

void* CUDABackend::AllocateMemory(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void CUDABackend::CopyHostToDevice(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

// ... implement other methods ...

} // namespace
```

#### 3. Create Simple Benchmark

**File:** `src/benchmarks/VectorAddBenchmark.h`

```cpp
#ifndef VECTOR_ADD_BENCHMARK_H
#define VECTOR_ADD_BENCHMARK_H

#include "core/IComputeBackend.h"
#include "core/Logger.h"
#include <vector>

class VectorAddBenchmark {
public:
    VectorAddBenchmark(size_t size = 1000000);  // Default: 1M elements
    
    void Run(GPUBenchmark::IComputeBackend* backend);
    bool Verify();
    
private:
    size_t m_size;
    std::vector<float> m_hostA, m_hostB, m_hostC;
    void* m_deviceA;
    void* m_deviceB;
    void* m_deviceC;
};

#endif
```

#### 4. Update main.cpp

Add code to actually run the benchmark:

```cpp
int RunCLIMode(const CommandLineArgs& args, const SystemCapabilities& caps) {
    // Initialize CUDA backend
    if (caps.cuda.available) {
        CUDABackend cuda;
        if (cuda.Initialize()) {
            // Run vector add benchmark
            VectorAddBenchmark benchmark(1000000);
            benchmark.Run(&cuda);
            
            std::cout << "Benchmark complete!" << std::endl;
        }
    }
    return 0;
}
```

---

## 🔨 Build and Run

### Step 1: Build
```powershell
cd Y:\GPU-Benchmark\build
cmake --build . --config Release
```

### Step 2: Run
```powershell
cd Release
.\GPU-Benchmark.exe --cli --all
```

### Step 3: Expected Output
```
╔═══════════════════════════════════════════════════════════════╗
║           GPU COMPUTE BENCHMARK & VISUALIZATION               ║
╚═══════════════════════════════════════════════════════════════╝

=== System Discovery ===
Detecting system capabilities...
  Found 1 GPU(s)
  Primary GPU: NVIDIA GeForce RTX 3050 Laptop GPU
  ✓ CUDA available: CUDA 12.0 (Compute Capability 8.6)
  
=== Running in CLI Mode ===
Running benchmarks...
  [CUDA] Vector Addition (1M elements): 0.234 ms | 51.3 GB/s ✓

Benchmark complete!
```

---

## 📖 When You Get Stuck

### Common Issues

**1. "CUDA not found" during CMake**
- Install CUDA Toolkit
- Set `CUDA_PATH` environment variable
- Restart PowerShell

**2. "Cannot open include file: 'cuda_runtime.h'"**
- CUDA Toolkit not installed correctly
- Check Visual Studio has CUDA integration

**3. "LNK2019: Unresolved external symbol"**
- Missing library in CMakeLists.txt
- Check `target_link_libraries()`

**4. Kernel doesn't run / no output**
- Check `cudaGetLastError()` after kernel launch
- Add `cudaDeviceSynchronize()` before reading results
- Use `printf()` in kernel for debugging

### Where to Find Answers

1. **In this project:**
   - `BUILD_GUIDE.md` - Build troubleshooting
   - `ARCHITECTURE.md` - Design questions
   - `src/core/README.md` - Framework questions

2. **External resources:**
   - NVIDIA CUDA Documentation (official)
   - StackOverflow (tag: cuda)
   - NVIDIA Developer Forums

---

## 🎓 Learning Path

### Week 1: Core + CUDA Backend
- Day 1-2: Read all documentation
- Day 3-4: Implement Logger, minimal CUDA backend
- Day 5-6: Implement vector_add benchmark
- Day 7: Test, debug, verify results

### Week 2: Complete Implementation
- Day 8-10: Implement all CUDA kernels
- Day 11-12: Implement all benchmarks
- Day 13-14: Testing, optimization, documentation

### Week 3: Polish + Optional Features
- Day 15-16: OpenCL backend (optional)
- Day 17-18: DirectCompute backend (optional)
- Day 19-20: Visualization (optional)
- Day 21: Final testing, prepare demo

---

## ✅ Checkpoint: "First Working Demo"

You'll know you've succeeded when you can:

1. Run: `GPU-Benchmark.exe --cli --benchmark=vector_add --backend=cuda`
2. See: System detects your RTX 3050
3. See: "CUDA available: CUDA 12.0"
4. See: "Vector Addition: X.XX ms | XX.X GB/s ✓"
5. Results saved to: `results/benchmark_results.csv`

**Celebrate!** 🎉 You've built a working GPU benchmarking tool!

---

## 💡 Pro Tips

### Development Workflow

1. **Compile often** - Don't write 100 lines before compiling
2. **Test incrementally** - Test each component before moving on
3. **Use printf/cout** - Print everything when debugging
4. **Check errors** - Every CUDA call can fail, check them all
5. **Verify correctness** - Fast wrong results are useless

### Code Organization

1. **One feature at a time** - Don't try to implement everything at once
2. **Comment as you go** - Future you will thank present you
3. **Commit frequently** - Use git to track progress
4. **Keep it simple** - Start with minimal viable version

### Interview Preparation

1. **Explain design decisions** - Why this architecture?
2. **Know the trade-offs** - CUDA performance vs OpenCL portability
3. **Understand performance** - Why is it memory-bound?
4. **Practice demo** - Show it working smoothly

---

## 🎯 Next Steps

1. ✅ Read this guide (you're doing it!)
2. 🔨 Implement Logger.cpp
3. 🔨 Implement CUDA Backend
4. 🔨 Create Vector Add Benchmark
5. 🔨 Make it run end-to-end
6. 🎉 Celebrate your working demo!

---

**Ready? Let's build something amazing! 🚀**

**Remember:** The project is already well-structured with excellent documentation. You're not starting from zero - you're completing a well-designed system!

---

**Questions?** Re-read PROJECT_SUMMARY.md for the big picture, or BUILD_GUIDE.md for specific build issues.

**Good luck!** 💪
