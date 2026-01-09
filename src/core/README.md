# Core Framework Documentation

## Overview

The **Core Framework** is the foundation of the GPU Compute Benchmark Tool. It provides API-agnostic abstractions that allow different compute backends (CUDA, OpenCL, DirectCompute) to be used interchangeably.

---

## 🎯 Purpose

The core framework serves three main purposes:

1. **Abstraction**: Define common interfaces for all GPU compute operations
2. **Orchestration**: Coordinate benchmark execution across multiple backends
3. **Measurement**: Provide accurate timing and performance metrics

---

## 📁 Files in This Directory

### 1. `IComputeBackend.h`
**What it is**: Abstract interface (pure virtual class) that all GPU backend implementations must follow.

**Why it exists**: Allows the application to work with CUDA, OpenCL, and DirectCompute through the same interface, without knowing which one is being used.

**Key concept**: Polymorphism - treat different backends uniformly.

**Analogy**: Like a USB port - doesn't matter if you plug in a mouse, keyboard, or flash drive, the interface is the same.

---

### 2. `Timer.h` and `Timer.cpp`
**What it is**: High-resolution timing utility for measuring GPU operations.

**Why it exists**: Measuring GPU performance requires special timing techniques because GPUs work asynchronously from the CPU.

**Two timing modes**:
- **CPU Timer**: Uses Windows `QueryPerformanceCounter` for host-side measurements
- **GPU Timer**: Uses backend-specific events (CUDA events, OpenCL events, D3D queries)

**Why both?**: 
- CPU timer measures total time including overhead
- GPU timer measures only actual GPU execution time

**Key learning**: CPU timestamp != GPU execution time!

---

### 3. `DeviceDiscovery.h` and `DeviceDiscovery.cpp`
**What it is**: System capability detection and hardware enumeration.

**Why it exists**: The application needs to know what GPU and APIs are available at runtime.

**What it detects**:
- GPU model (NVIDIA RTX 3050, AMD RX 6600, etc.)
- Available APIs (CUDA, OpenCL, DirectCompute)
- GPU memory size
- Driver version
- Compute capability

**Key concept**: Runtime detection, not compile-time configuration.

**Why important**: Same executable works on different computers!

---

### 4. `BenchmarkRunner.h` and `BenchmarkRunner.cpp`
**What it is**: The orchestrator that coordinates benchmark execution.

**Why it exists**: Running benchmarks correctly requires careful setup, timing, verification, and cleanup. This class handles all of that.

**Workflow**:
```
1. Initialize backend
2. For each benchmark:
   a. Setup memory
   b. Warmup runs (stabilize GPU clocks)
   c. Timed execution
   d. Result verification
   e. Cleanup
3. Aggregate and report results
```

**Key responsibility**: Ensure fair comparison between backends.

---

### 5. `Logger.h` and `Logger.cpp`
**What it is**: Result logging, CSV export, and console output.

**Why it exists**: Benchmark results need to be saved for later analysis and comparison.

**Features**:
- Console output with color coding
- CSV file export (Excel-compatible)
- Timestamped result files
- Error logging

**Output format**:
```csv
Timestamp,Backend,Benchmark,Size,ExecutionMS,TransferMS,Bandwidth_GB/s
2026-01-09 12:00:00,CUDA,VectorAdd,1000000,0.234,1.23,51.3
```

---

## 🔄 How Components Interact

```
┌──────────────────┐
│  Application     │  (main.cpp, GUI)
└────────┬─────────┘
         │
         ↓
┌────────────────────────────┐
│  BenchmarkRunner           │  ← Orchestrates everything
│  (Calls all other classes) │
└────────┬───────────────────┘
         │
         ├─→ DeviceDiscovery  (Detect GPUs and APIs)
         │
         ├─→ IComputeBackend  (Execute kernels)
         │     ↑
         │     └─ CUDABackend, OpenCLBackend, DirectComputeBackend
         │
         ├─→ Timer            (Measure performance)
         │
         └─→ Logger           (Save results)
```

---

## 🎓 Key Concepts Explained

### What is an "Interface" (Abstract Base Class)?

In C++, an interface is a class with only pure virtual functions (no implementation).

```cpp
// This is an interface
class IComputeBackend {
public:
    virtual bool Initialize() = 0;  // = 0 means "pure virtual"
    virtual void Shutdown() = 0;
    // ... more methods ...
};
```

**Why use interfaces?**
- Enforces consistency (all backends have same methods)
- Enables polymorphism (treat different backends the same way)
- Allows runtime selection (choose backend at program startup)

**Example usage**:
```cpp
IComputeBackend* backend;

if (cudaAvailable) {
    backend = new CUDABackend();
} else if (openclAvailable) {
    backend = new OpenCLBackend();
} else {
    backend = new DirectComputeBackend();
}

// Now we can use 'backend' without caring which one it is!
backend->Initialize();
backend->ExecuteKernel(...);
```

---

### Why Separate Interface from Implementation?

**Problem**: If each backend had different function names:
```cpp
cudaBackend->CudaLaunchKernel(...);
openclBackend->EnqueueNDRangeKernel(...);
directComputeBackend->DispatchComputeShader(...);
```

Now the calling code needs to know which backend is being used!

**Solution**: Use common interface:
```cpp
backend->ExecuteKernel(...);  // Works for all backends!
```

---

### What is "Polymorphism"?

Greek words: "poly" (many) + "morph" (forms)

**Meaning**: One interface, many implementations.

**Example from real life**:
- Interface: "Drawable shape"
- Implementations: Circle, Square, Triangle
- Common operation: `Draw()`
- Each shape draws itself differently, but you call the same method

**In our project**:
- Interface: `IComputeBackend`
- Implementations: CUDA, OpenCL, DirectCompute
- Common operation: `ExecuteKernel()`
- Each backend launches kernels differently, but same interface

---

### What is "RAII" (Resource Acquisition Is Initialization)?

**Principle**: Tie resource lifetime to object lifetime.

**Example**:
```cpp
class Timer {
    LARGE_INTEGER start;
    
public:
    Timer() {
        // Constructor acquires resource (starts timer)
        QueryPerformanceCounter(&start);
    }
    
    ~Timer() {
        // Destructor releases resource (stops timer)
        // Cleanup happens automatically!
    }
};

// Usage
{
    Timer t;  // Timer starts
    // ... do work ...
}  // Timer stops automatically when 't' goes out of scope
```

**Why good?**
- Can't forget to cleanup
- Exception-safe (cleanup happens even if error occurs)
- Clear ownership semantics

---

## 🔍 Design Patterns Used

### 1. Strategy Pattern
**Where**: IComputeBackend interface
**What**: Select algorithm (CUDA/OpenCL/DirectCompute) at runtime
**Why**: Same problem, multiple solutions, choose the best available

### 2. Template Method Pattern
**Where**: BenchmarkRunner
**What**: Define algorithm structure, let subclasses fill in details
**Why**: Ensures consistent benchmark methodology

### 3. Singleton Pattern
**Where**: Logger
**What**: Only one logger instance throughout application
**Why**: All components write to same log file

### 4. Facade Pattern
**Where**: DeviceDiscovery
**What**: Simple interface hiding complex detection logic
**Why**: Application doesn't need to know detection details

---

## 📊 Memory Management Strategy

### CPU Memory (Host)
- Allocated with `new` or `malloc`
- Freed with `delete` or `free`
- Standard C++ memory management

### GPU Memory (Device)
- Allocated through backend interface: `AllocateMemory()`
- Freed through backend interface: `FreeMemory()`
- Backend translates to API-specific calls:
  - CUDA: `cudaMalloc()` / `cudaFree()`
  - OpenCL: `clCreateBuffer()` / `clReleaseMemObject()`
  - DirectCompute: `ID3D11Device::CreateBuffer()` / `Release()`

### Transfer Between CPU and GPU
- Host → Device: `CopyHostToDevice()`
- Device → Host: `CopyDeviceToHost()`

**Visual representation**:
```
CPU Memory (RAM)          GPU Memory (VRAM)
┌──────────────┐          ┌──────────────┐
│  float* a    │ ────────→│  void* d_a   │  CopyHostToDevice
│  [1,2,3,4]   │          │  [1,2,3,4]   │
└──────────────┘          └──────────────┘
                                 │
                                 ↓
                          GPU Kernel Execution
                                 │
                                 ↓
CPU Memory (RAM)          GPU Memory (VRAM)
┌──────────────┐          ┌──────────────┐
│  float* result│←──────────│void* d_result│ CopyDeviceToHost
│  [2,4,6,8]   │          │  [2,4,6,8]   │
└──────────────┘          └──────────────┘
```

---

## ⏱️ Timing Methodology

### Why GPU Timing is Tricky

**Problem**: GPU executes asynchronously!

```cpp
// WRONG way to time GPU
auto start = std::chrono::steady_clock::now();
LaunchKernel();  // This returns immediately!
auto end = std::chrono::steady_clock::now();  // Measures ~0 ms!
```

**Why wrong?**
- `LaunchKernel()` submits work to GPU and returns immediately
- GPU executes asynchronously in the background
- CPU timer captures only submission overhead, not actual GPU execution

**Correct approach**:
```cpp
// RIGHT way to time GPU
GPUEvent start, end;
RecordEvent(start);  // Mark start on GPU timeline
LaunchKernel();
RecordEvent(end);    // Mark end on GPU timeline
Synchronize();       // Wait for GPU to finish
float milliseconds = GetElapsedTime(start, end);  // Query GPU timer
```

---

## 🎯 Error Handling Philosophy

### Three levels of errors:

1. **Expected (Informational)**
   - Example: CUDA unavailable on AMD GPU
   - Action: Disable backend, report to user, continue
   - Severity: INFO

2. **Unexpected (Warning)**
   - Example: Kernel compilation warning
   - Action: Log warning, try to continue
   - Severity: WARNING

3. **Fatal (Error)**
   - Example: Out of GPU memory
   - Action: Clean up, report, exit gracefully
   - Severity: ERROR

**Key principle**: Never fail silently!

---

## 🧪 Testing Strategy

Each core component can be tested independently:

### Timer Testing
```cpp
Timer t;
t.Start();
Sleep(100);  // 100ms
t.Stop();
assert(t.GetTimeMS() >= 100 && t.GetTimeMS() <= 110);
```

### DeviceDiscovery Testing
```cpp
auto capabilities = DeviceDiscovery::Discover();
assert(!capabilities.gpuName.empty());
assert(capabilities.gpuMemoryMB > 0);
```

### Backend Interface Testing
```cpp
IComputeBackend* backend = CreateBackend();
assert(backend->Initialize());
void* mem = backend->AllocateMemory(1024);
assert(mem != nullptr);
backend->FreeMemory(mem);
```

---

## 🔧 Extending the Core

### Adding a New Backend (e.g., Vulkan)

1. Create `VulkanBackend.h` and `VulkanBackend.cpp`
2. Inherit from `IComputeBackend`:
   ```cpp
   class VulkanBackend : public IComputeBackend {
       // Implement all pure virtual functions
   };
   ```
3. Add detection in `DeviceDiscovery.cpp`
4. Register in `BenchmarkRunner.cpp`

**That's it!** No changes needed to core framework.

---

### Adding a New Metric (e.g., Power Consumption)

1. Add field to result structure:
   ```cpp
   struct BenchmarkResult {
       double executionTimeMS;
       double transferTimeMS;
       double powerWatts;  // NEW
   };
   ```
2. Implement measurement in backends
3. Update Logger to export new metric
4. Update Renderer to visualize it

---

## 📚 Further Learning

### Recommended Reading
1. **"Design Patterns" by Gang of Four**: Classic patterns like Strategy, Template Method
2. **"Effective C++" by Scott Meyers**: Best practices for interfaces and polymorphism
3. **"CUDA C Programming Guide"**: Official NVIDIA documentation
4. **"Heterogeneous Computing with OpenCL 2.0"**: OpenCL deep dive

### Related Concepts
- **Virtual Function Table (vtable)**: How polymorphism works under the hood
- **RAII**: Resource management pattern
- **Smart Pointers**: Modern C++ memory management
- **Abstract Factory Pattern**: Creating families of related objects

---

## ❓ Common Questions

### Q: Why not use templates instead of virtual functions?

**A**: Templates require compile-time knowledge of types. We need runtime selection (user's GPU determines backend).

### Q: Why separate .h and .cpp files?

**A**: Faster compilation, cleaner interface, implementation hiding.

### Q: What's the overhead of virtual functions?

**A**: Minimal (~1 extra indirection). For GPU code, API overhead >> virtual function cost.

### Q: Could we use function pointers instead of classes?

**A**: Yes, but classes provide better encapsulation, state management, and lifetime control.

---

**Next**: Read individual backend READMEs to understand API-specific implementations.

- [CUDA Backend](../backends/cuda/README.md)
- [OpenCL Backend](../backends/opencl/README.md)
- [DirectCompute Backend](../backends/directcompute/README.md)
