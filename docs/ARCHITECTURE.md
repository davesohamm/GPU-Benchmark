# 🏛️ Architecture Documentation

This document provides a deep dive into the architectural design of the GPU Compute Benchmark Tool, explaining key design decisions, patterns, and implementation details.

---

## 🎯 Design Principles

### 1. **Separation of Concerns**
- Compute backends are completely isolated from visualization
- Each backend implements the same abstract interface
- Benchmarks are API-agnostic, backends provide implementations

### 2. **Hardware Agnostic Design**
- Runtime capability detection, not compile-time assumptions
- Same executable runs on different hardware
- Graceful degradation when features unavailable

### 3. **Fairness in Comparison**
- Identical algorithms across all backends
- Same workload sizes and memory patterns
- Separate timing for compute vs memory transfer

### 4. **Minimal CPU Overhead**
- GPU-side synchronization where possible
- Avoid readbacks during active benchmarking
- High-resolution timers without interrupts

### 5. **Extensibility**
- Easy to add new benchmarks
- Easy to add new backends
- Modular component replacement

---

## 🏗️ Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: APPLICATION                                        │
│  ├─ main.cpp            : Entry point, initialization       │
│  ├─ GUI.cpp             : User interface and event handling  │
│  └─ CLI.cpp             : Command-line argument parsing      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: BENCHMARK ORCHESTRATION                            │
│  ├─ BenchmarkRunner     : Coordinates execution              │
│  ├─ Logger              : Results collection and export      │
│  └─ SystemInfo          : Hardware capability queries        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: ABSTRACTION (Core Framework)                       │
│  ├─ IComputeBackend     : Abstract interface for all GPUs   │
│  ├─ IBenchmark          : Abstract benchmark definition      │
│  ├─ Timer               : High-resolution timing             │
│  └─ DeviceDiscovery     : Runtime GPU and API detection      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: IMPLEMENTATION (Backends)                          │
│  ├─ CUDABackend         : NVIDIA CUDA implementation         │
│  ├─ OpenCLBackend       : OpenCL implementation              │
│  ├─ DirectComputeBackend: DirectCompute implementation       │
│  └─ Renderer            : OpenGL visualization (separate)    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 0: HARDWARE                                           │
│  └─ GPU Driver → GPU Hardware                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### IComputeBackend Interface

**Purpose**: Defines the contract that all compute backends must implement.

**Key Methods**:
```cpp
class IComputeBackend {
public:
    virtual bool Initialize() = 0;
    virtual void Shutdown() = 0;
    
    virtual void* AllocateMemory(size_t size) = 0;
    virtual void FreeMemory(void* ptr) = 0;
    
    virtual void CopyHostToDevice(void* dst, const void* src, size_t size) = 0;
    virtual void CopyDeviceToHost(void* dst, const void* src, size_t size) = 0;
    
    virtual void ExecuteKernel(const std::string& kernelName, 
                               const KernelParams& params) = 0;
    
    virtual void Synchronize() = 0;
    
    virtual std::string GetDeviceName() = 0;
    virtual size_t GetDeviceMemory() = 0;
};
```

**Why This Design?**
- Polymorphism allows treating all backends uniformly
- BenchmarkRunner doesn't need to know which backend it's using
- Easy to add new backends (just implement the interface)

---

### BenchmarkRunner

**Purpose**: Orchestrates benchmark execution across multiple backends.

**Workflow**:
```
1. Discovery Phase
   ├─ Query available backends
   ├─ Initialize each backend
   └─ Report capabilities

2. Execution Phase
   For each benchmark:
      For each backend:
         ├─ Allocate memory
         ├─ Copy data to device
         ├─ START_TIMER
         ├─ Execute kernel
         ├─ Synchronize
         ├─ STOP_TIMER
         ├─ Copy results back
         └─ Verify correctness

3. Results Phase
   ├─ Aggregate timing data
   ├─ Calculate statistics
   └─ Export results
```

**Key Features**:
- Automatic warmup runs (GPU frequency scaling)
- Multiple iterations for statistical significance
- Result verification (ensures correctness)
- Timeout protection (prevents hangs)

---

### Timer Implementation

**Challenge**: Accurately measuring GPU operations

**Problem**: CPU timers don't account for GPU asynchrony

**Solution**: Two-level timing strategy

```cpp
class Timer {
    // CPU-side timing (for host overhead)
    LARGE_INTEGER cpuStart, cpuEnd, frequency;
    
    // GPU-side timing (API-specific)
    void* gpuStartEvent;
    void* gpuEndEvent;
    
public:
    void StartCPU();
    void StopCPU();
    double GetCPUTimeMS();
    
    void StartGPU();
    void StopGPU();
    double GetGPUTimeMS();
};
```

**Backend-Specific GPU Timing**:

**CUDA**:
```cpp
cudaEvent_t start, end;
cudaEventCreate(&start);
cudaEventCreate(&end);
cudaEventRecord(start);
// ... kernel execution ...
cudaEventRecord(end);
cudaEventSynchronize(end);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, end);
```

**OpenCL**:
```cpp
cl_event event;
clEnqueueNDRangeKernel(..., &event);
clWaitForEvents(1, &event);
cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
double milliseconds = (end - start) / 1e6;
```

**DirectCompute**:
```cpp
// Use D3D11 queries
ID3D11Query* startQuery, *endQuery;
D3D11_QUERY_DESC queryDesc = { D3D11_QUERY_TIMESTAMP, 0 };
device->CreateQuery(&queryDesc, &startQuery);
device->CreateQuery(&queryDesc, &endQuery);
context->End(startQuery);
// ... dispatch compute shader ...
context->End(endQuery);
UINT64 startTime, endTime;
context->GetData(startQuery, &startTime, sizeof(UINT64), 0);
context->GetData(endQuery, &endTime, sizeof(UINT64), 0);
```

---

## 🎨 Visualization Architecture

### OpenGL Renderer

**Design Decision**: Why separate from compute?

**Reason**: Mixing compute and rendering on same context can cause:
- Performance interference
- Driver state pollution
- Timing measurement corruption

**Architecture**:
```
┌─────────────────────┐
│  Benchmark Results  │  (CPU-side storage)
│  ├─ Timing data     │
│  ├─ Bandwidth stats │
│  └─ Error rates     │
└──────────┬──────────┘
           │
           ↓
    ┌──────────────┐
    │  Renderer    │
    │  (OpenGL)    │
    └──────┬───────┘
           │
           ↓ (Vertex data)
    ┌──────────────┐
    │  GPU         │
    │  (Rendering  │
    │   Pipeline)  │
    └──────────────┘
```

**Rendering Pipeline**:
1. **Data Preparation**: Format results as vertex buffers
2. **Vertex Processing**: Position bar graphs/lines
3. **Fragment Shading**: Color coding by performance
4. **Compositing**: Final display with UI overlay

**Shaders**:
- `vertex.glsl`: Transforms benchmark data to screen space
- `fragment.glsl`: Colors based on performance thresholds

---

## 🔀 Backend Implementation Details

### CUDA Backend

**File**: `src/backends/cuda/CUDABackend.cpp`

**Initialization**:
```cpp
bool CUDABackend::Initialize() {
    // 1. Check CUDA availability
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return false;
    
    // 2. Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 3. Check compute capability (need 3.0+)
    if (prop.major < 3) return false;
    
    // 4. Set device
    cudaSetDevice(0);
    
    return true;
}
```

**Memory Management**:
```cpp
void* CUDABackend::AllocateMemory(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;  // Returns device pointer
}
```

**Kernel Execution**:
```cpp
void CUDABackend::ExecuteKernel(const std::string& name, const KernelParams& params) {
    // Calculate grid/block dimensions
    dim3 block(256);  // 256 threads per block
    dim3 grid((params.numElements + block.x - 1) / block.x);
    
    // Launch appropriate kernel
    if (name == "vector_add") {
        vectorAddKernel<<<grid, block>>>(params.input1, params.input2, 
                                          params.output, params.numElements);
    }
    // ... other kernels ...
}
```

---

### OpenCL Backend

**File**: `src/backends/opencl/OpenCLBackend.cpp`

**Initialization** (More Complex):
```cpp
bool OpenCLBackend::Initialize() {
    // 1. Get platform (NVIDIA, AMD, Intel, etc.)
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (numPlatforms == 0) return false;
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    // 2. Get GPU device
    cl_uint numDevices;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (numDevices == 0) return false;
    
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    
    // 3. Create context
    context = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, nullptr);
    
    // 4. Create command queue with profiling enabled
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, nullptr);
    
    // 5. Load and compile kernels from source strings
    LoadAndCompileKernels();
    
    return true;
}
```

**Runtime Kernel Compilation**:
```cpp
void OpenCLBackend::LoadAndCompileKernels() {
    // Kernel source is embedded as string
    const char* vectorAddSource = R"(
        __kernel void vector_add(__global const float* a, 
                                 __global const float* b,
                                 __global float* c, 
                                 int n) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    )";
    
    // Create program from source
    cl_program program = clCreateProgramWithSource(context, 1, &vectorAddSource, nullptr, nullptr);
    
    // Compile
    clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    
    // Extract kernel
    kernels["vector_add"] = clCreateKernel(program, "vector_add", nullptr);
}
```

---

### DirectCompute Backend

**File**: `src/backends/directcompute/DirectComputeBackend.cpp`

**Initialization**:
```cpp
bool DirectComputeBackend::Initialize() {
    // 1. Create D3D11 device
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Hardware acceleration
        nullptr,
        0,
        nullptr, 0,
        D3D11_SDK_VERSION,
        &device,
        &featureLevel,
        &context
    );
    
    if (FAILED(hr)) return false;
    
    // 2. Check compute shader support (need 11.0+)
    if (featureLevel < D3D_FEATURE_LEVEL_11_0) return false;
    
    // 3. Load and compile HLSL shaders
    LoadShaders();
    
    return true;
}
```

**HLSL Shader Compilation**:
```cpp
void DirectComputeBackend::LoadShaders() {
    // Read shader from file
    std::ifstream shaderFile("shaders/vector_add.hlsl");
    std::string shaderCode((std::istreambuf_iterator<char>(shaderFile)),
                            std::istreambuf_iterator<char>());
    
    // Compile HLSL to bytecode
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompile(
        shaderCode.c_str(),
        shaderCode.length(),
        "vector_add.hlsl",
        nullptr,
        nullptr,
        "CSMain",                  // Entry point
        "cs_5_0",                  // Compute shader 5.0
        0, 0,
        &shaderBlob,
        &errorBlob
    );
    
    if (FAILED(hr)) {
        // Handle compilation error
        return;
    }
    
    // Create compute shader
    ID3D11ComputeShader* shader;
    device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &shader
    );
    
    computeShaders["vector_add"] = shader;
}
```

**Buffer Creation**:
```cpp
void* DirectComputeBackend::AllocateMemory(size_t size) {
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = size;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);
    
    ID3D11Buffer* buffer;
    device->CreateBuffer(&desc, nullptr, &buffer);
    return buffer;
}
```

---

## 📊 Benchmark Implementation Pattern

Each benchmark follows this structure:

```cpp
class VectorAddBenchmark : public IBenchmark {
public:
    void Setup(IComputeBackend* backend) override {
        // 1. Allocate host memory
        hostInputA = new float[size];
        hostInputB = new float[size];
        hostOutput = new float[size];
        
        // 2. Initialize data
        for (int i = 0; i < size; i++) {
            hostInputA[i] = static_cast<float>(i);
            hostInputB[i] = static_cast<float>(i * 2);
        }
        
        // 3. Allocate device memory
        deviceInputA = backend->AllocateMemory(size * sizeof(float));
        deviceInputB = backend->AllocateMemory(size * sizeof(float));
        deviceOutput = backend->AllocateMemory(size * sizeof(float));
        
        // 4. Copy data to device
        backend->CopyHostToDevice(deviceInputA, hostInputA, size * sizeof(float));
        backend->CopyHostToDevice(deviceInputB, hostInputB, size * sizeof(float));
    }
    
    BenchmarkResult Run(IComputeBackend* backend) override {
        BenchmarkResult result;
        
        // Warmup (stabilize GPU clocks)
        for (int i = 0; i < 3; i++) {
            backend->ExecuteKernel("vector_add", params);
            backend->Synchronize();
        }
        
        // Actual benchmark
        Timer timer;
        timer.StartGPU();
        
        for (int i = 0; i < iterations; i++) {
            backend->ExecuteKernel("vector_add", params);
        }
        
        backend->Synchronize();
        timer.StopGPU();
        
        result.executionTimeMS = timer.GetGPUTimeMS() / iterations;
        
        // Measure memory transfer
        timer.StartCPU();
        backend->CopyDeviceToHost(hostOutput, deviceOutput, size * sizeof(float));
        timer.StopCPU();
        
        result.transferTimeMS = timer.GetCPUTimeMS();
        
        // Verify results
        result.correct = VerifyResults();
        
        return result;
    }
    
    bool VerifyResults() override {
        for (int i = 0; i < size; i++) {
            float expected = hostInputA[i] + hostInputB[i];
            if (abs(hostOutput[i] - expected) > 0.001f) {
                return false;
            }
        }
        return true;
    }
};
```

---

## 🔍 Device Discovery

**File**: `src/core/DeviceDiscovery.cpp`

**Process**:
```cpp
struct SystemCapabilities {
    bool cudaAvailable;
    bool openclAvailable;
    bool directComputeAvailable;
    
    std::string gpuName;
    size_t gpuMemoryMB;
    std::string driverVersion;
};

SystemCapabilities DiscoverCapabilities() {
    SystemCapabilities caps;
    
    // 1. Try CUDA
    caps.cudaAvailable = TestCUDAAvailability();
    
    // 2. Try OpenCL
    caps.openclAvailable = TestOpenCLAvailability();
    
    // 3. Try DirectCompute
    caps.directComputeAvailable = TestDirectComputeAvailability();
    
    // 4. Query GPU info
    caps.gpuName = GetGPUName();
    caps.gpuMemoryMB = GetGPUMemoryMB();
    caps.driverVersion = GetDriverVersion();
    
    return caps;
}
```

**Why Runtime Detection?**
- Same .exe works on NVIDIA, AMD, and Intel GPUs
- Graceful degradation (if CUDA unavailable, use OpenCL)
- Professional error messages instead of crashes

---

## 📈 Performance Considerations

### Memory Coalescing

**Problem**: Uncoalesced memory access kills performance

**Solution**: Ensure stride-1 access patterns

**CUDA Example**:
```cuda
// BAD: Strided access
__global__ void badKernel(float* data, int stride) {
    int i = threadIdx.x * stride;  // Non-coalesced!
    data[i] = ...;
}

// GOOD: Coalesced access
__global__ void goodKernel(float* data) {
    int i = threadIdx.x;  // Adjacent threads access adjacent memory
    data[i] = ...;
}
```

### Bank Conflicts (Shared Memory)

**Problem**: Multiple threads accessing same bank causes serialization

**Solution**: Pad shared memory or use offset indexing

```cuda
__shared__ float sharedData[256 + 16];  // Padding avoids bank conflicts
```

### Occupancy

**Goal**: Keep GPU fully utilized

**Factors**:
- Registers per thread
- Shared memory per block
- Block size

**Tool**: CUDA Occupancy Calculator (included in CUDA Toolkit)

---

## 🎯 Design Patterns Used

### 1. **Strategy Pattern** (Backends)
Different algorithms (CUDA/OpenCL/DirectCompute) for same task, selected at runtime.

### 2. **Template Method Pattern** (Benchmarks)
Benchmark base class defines workflow; derived classes implement specifics.

### 3. **Facade Pattern** (BenchmarkRunner)
Simplified interface hiding complex backend interactions.

### 4. **Factory Pattern** (Backend creation)
```cpp
IComputeBackend* CreateBackend(BackendType type) {
    switch (type) {
        case CUDA: return new CUDABackend();
        case OpenCL: return new OpenCLBackend();
        case DirectCompute: return new DirectComputeBackend();
    }
}
```

### 5. **RAII Pattern** (Resource management)
Automatic cleanup in destructors prevents leaks.

---

## 🔐 Error Handling Strategy

### Levels of Error Handling

**1. Initialization Errors** (Expected):
- Backend unavailable → Disable, report to user
- Example: "CUDA not available - NVIDIA GPU required"

**2. Runtime Errors** (Unexpected):
- Out of memory → Report, skip benchmark
- Kernel compilation failed → Log error, continue with other backends

**3. Critical Errors** (Fatal):
- Driver crash → Terminate gracefully with diagnostic info

### Error Reporting

```cpp
enum class ErrorSeverity {
    INFO,        // Normal operation
    WARNING,     // Degraded functionality
    ERROR,       // Feature unavailable
    CRITICAL     // Application cannot continue
};

void LogError(ErrorSeverity severity, const std::string& message) {
    std::cout << "[" << SeverityToString(severity) << "] " << message << std::endl;
    
    if (severity == ErrorSeverity::CRITICAL) {
        // Save diagnostic info
        // Prompt user
        // Exit gracefully
    }
}
```

---

## 📊 Result Data Flow

```
Benchmark Execution
       ↓
  Raw Timing Data
       ↓
  Statistical Analysis (mean, median, std dev)
       ↓
  ┌─────────────┬──────────────┐
  ↓             ↓              ↓
Display     CSV Export    Visualization
(Console)   (File)        (OpenGL)
```

---

## 🧵 Threading Model

**Single-threaded design** (for simplicity and timing accuracy)

- Main thread handles UI and orchestration
- GPU executes asynchronously (but we synchronize for timing)
- No CPU parallelism (would complicate benchmarking)

**Future Enhancement**: Multi-threaded backend execution (run all backends in parallel)

---

## 🎓 Key Learnings

1. **GPU APIs are fundamentally similar**: Memory allocation, kernel launch, synchronization
2. **Timing is subtle**: Need GPU-side events, not CPU-side timers
3. **Warmup matters**: First run is slower due to GPU frequency scaling
4. **Verification is essential**: Easy to get incorrect results fast
5. **Abstraction has cost**: But enables clean architecture

---

**Next**: Read individual backend READMEs for API-specific details.

- [CUDA Backend](src/backends/cuda/README.md)
- [OpenCL Backend](src/backends/opencl/README.md)
- [DirectCompute Backend](src/backends/directcompute/README.md)
