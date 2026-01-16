# ⚙️ Internal Workings - Deep Technical Dive

## Table of Contents
- [Application Flow](#application-flow)
- [Backend Internals](#backend-internals)
- [Benchmark Execution](#benchmark-execution)
- [Memory Management](#memory-management)
- [Timing and Synchronization](#timing-and-synchronization)
- [GUI Threading Model](#gui-threading-model)
- [Data Structures](#data-structures)

---

## Application Flow

### Startup Sequence

```
main() / WinMain()
    ↓
Initialize ImGui + DirectX 11
    ↓
Detect System Capabilities
    ├─ Query CUDA availability
    ├─ Query OpenCL availability
    ├─ Query DirectCompute availability
    └─ Get GPU information (DXGI)
    ↓
Display Main Window
    ↓
User Selects Backend & Suite
    ↓
Launch Worker Thread
    ↓
Execute Benchmarks
    ↓
Update GUI with Progress
    ↓
Display Results & Graphs
```

### Detailed Startup Process

#### 1. Window Creation (Windows API)
```cpp
// Create Win32 window
HWND hwnd = CreateWindowW(
    L"GPU Benchmark Suite",
    WS_OVERLAPPEDWINDOW,
    ...
);

// Load and set application icon
HICON hIcon = LoadImage(hInst, MAKEINTRESOURCE(101), ...);
SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
```

#### 2. DirectX 11 Initialization
```cpp
// Create D3D11 device and swap chain
D3D11CreateDeviceAndSwapChain(
    nullptr,                    // Default adapter
    D3D_DRIVER_TYPE_HARDWARE,
    nullptr,
    createDeviceFlags,
    featureLevels,
    D3D11_SDK_VERSION,
    &sd,                        // Swap chain descriptor
    &g_pSwapChain,
    &g_pd3dDevice,
    &g_featureLevel,
    &g_pd3dDeviceContext
);
```

#### 3. ImGui Initialization
```cpp
IMGUI_CHECKVERSION();
ImGui::CreateContext();
ImGui::StyleColorsDark();

// Initialize platform and renderer backends
ImGui_ImplWin32_Init(hwnd);
ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);
```

#### 4. System Capability Detection
```cpp
SystemCapabilities DetectCapabilities() {
    SystemCapabilities caps;
    
    // CUDA detection
    try {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        caps.cuda.available = (err == cudaSuccess && deviceCount > 0);
        
        if (caps.cuda.available) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            caps.cuda.deviceName = prop.name;
            caps.cuda.computeCapability = prop.major * 10 + prop.minor;
            caps.cuda.totalMemoryMB = prop.totalGlobalMem / (1024 * 1024);
        }
    } catch (...) {
        caps.cuda.available = false;
    }
    
    // OpenCL detection
    try {
        cl_uint numPlatforms;
        clGetPlatformIDs(0, nullptr, &numPlatforms);
        caps.opencl.available = (numPlatforms > 0);
        
        if (caps.opencl.available) {
            // Get first GPU device
            cl_platform_id platform;
            clGetPlatformIDs(1, &platform, nullptr);
            
            cl_device_id device;
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            
            // Query device info
            char deviceName[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            caps.opencl.deviceName = deviceName;
        }
    } catch (...) {
        caps.opencl.available = false;
    }
    
    // DirectCompute is always available on Windows with DirectX 11
    caps.directCompute.available = true;
    
    // Get GPU info from DXGI
    IDXGIFactory* pFactory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
    
    IDXGIAdapter* pAdapter;
    pFactory->EnumAdapters(0, &pAdapter);
    
    DXGI_ADAPTER_DESC desc;
    pAdapter->GetDesc(&desc);
    
    // Convert wchar_t to char
    char gpuName[128];
    wcstombs(gpuName, desc.Description, sizeof(gpuName));
    caps.primaryGPU = gpuName;
    
    return caps;
}
```

---

## Backend Internals

### CUDA Backend Implementation

#### Memory Allocation Strategy
```cpp
// Device memory allocation
void* CUDABackend::AllocateMemory(size_t size) {
    void* devPtr = nullptr;
    cudaError_t err = cudaMalloc(&devPtr, size);
    
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA allocation failed");
    }
    
    // Track allocation for cleanup
    allocations.push_back(devPtr);
    
    return devPtr;
}

// Automatic cleanup (RAII)
CUDABackend::~CUDABackend() {
    for (void* ptr : allocations) {
        cudaFree(ptr);
    }
}
```

#### Kernel Launch Process
```cpp
void CUDABackend::ExecuteVectorAdd(float* d_a, float* d_b, float* d_c, int n) {
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;  // Optimal for most GPUs
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
```

#### GPU Timing with Events
```cpp
class CUDATimer {
    cudaEvent_t start, stop;
    
public:
    CUDATimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    void Start() {
        cudaEventRecord(start, 0);
    }
    
    void Stop() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);  // Wait for completion
    }
    
    float GetMilliseconds() {
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};
```

### OpenCL Backend Implementation

#### Context and Queue Setup
```cpp
bool OpenCLBackend::Initialize() {
    // 1. Get platform
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    // 2. Get device
    cl_device_id device;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    
    // 3. Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    
    // 4. Create command queue with profiling
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    queue = clCreateCommandQueue(context, device, props, nullptr);
    
    // 5. Compile kernels
    CompileKernels();
    
    return true;
}
```

#### Runtime Kernel Compilation
```cpp
void OpenCLBackend::CompileKernels() {
    // Kernel source as string (embedded in code)
    const char* vectorAddSource = R"(
        __kernel void vectorAdd(
            __global const float* a,
            __global const float* b,
            __global float* c,
            int n)
        {
            int gid = get_global_id(0);
            if (gid < n) {
                c[gid] = a[gid] + b[gid];
            }
        }
    )";
    
    // Create program from source
    cl_program program = clCreateProgramWithSource(
        context, 1, &vectorAddSource, nullptr, nullptr
    );
    
    // Compile program
    cl_int err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                              0, nullptr, &logSize);
        
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              logSize, log.data(), nullptr);
        
        throw std::runtime_error("Kernel compilation failed:\n" + 
                                 std::string(log.data()));
    }
    
    // Extract kernel
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", nullptr);
    kernels["vectorAdd"] = kernel;
}
```

#### Kernel Execution with Profiling
```cpp
double OpenCLBackend::ExecuteKernel(const std::string& name, 
                                     const KernelParams& params) {
    cl_kernel kernel = kernels[name];
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &params.input1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &params.input2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &params.output);
    clSetKernelArg(kernel, 3, sizeof(int), &params.size);
    
    // Calculate work sizes
    size_t globalWorkSize = params.size;
    size_t localWorkSize = 256;
    
    // Enqueue kernel with profiling event
    cl_event event;
    clEnqueueNDRangeKernel(
        queue, kernel,
        1,                      // 1D work
        nullptr,                // No offset
        &globalWorkSize,
        &localWorkSize,
        0, nullptr,             // No wait events
        &event                  // Profiling event
    );
    
    // Wait for completion
    clWaitForEvents(1, &event);
    
    // Get timing information
    cl_ulong timeStart, timeEnd;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &timeStart, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &timeEnd, nullptr);
    
    // Convert nanoseconds to milliseconds
    double ms = (timeEnd - timeStart) / 1e6;
    
    clReleaseEvent(event);
    
    return ms;
}
```

### DirectCompute Backend Implementation

#### Compute Shader Compilation
```cpp
bool DirectComputeBackend::CompileShader(const std::string& hlslSource) {
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    
    HRESULT hr = D3DCompile(
        hlslSource.c_str(),
        hlslSource.length(),
        "vector_add.hlsl",      // Virtual filename
        nullptr,                // No defines
        nullptr,                // No include handler
        "CSMain",               // Entry point
        "cs_5_0",               // Shader model 5.0
        0, 0,
        &shaderBlob,
        &errorBlob
    );
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::string errorMsg((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
            throw std::runtime_error("Shader compilation failed:\n" + errorMsg);
        }
        return false;
    }
    
    // Create compute shader
    ID3D11ComputeShader* computeShader;
    hr = device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &computeShader
    );
    
    shaderBlob->Release();
    
    if (FAILED(hr)) {
        return false;
    }
    
    computeShaders["vectorAdd"] = computeShader;
    return true;
}
```

#### Buffer Management
```cpp
ID3D11Buffer* DirectComputeBackend::CreateBuffer(size_t size) {
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = static_cast<UINT>(size);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);
    
    ID3D11Buffer* buffer;
    HRESULT hr = device->CreateBuffer(&desc, nullptr, &buffer);
    
    if (FAILED(hr)) {
        throw std::runtime_error("Buffer creation failed");
    }
    
    return buffer;
}
```

#### Compute Shader Dispatch
```cpp
void DirectComputeBackend::DispatchCompute(const std::string& shaderName,
                                            const KernelParams& params) {
    // Set compute shader
    ID3D11ComputeShader* shader = computeShaders[shaderName];
    context->CSSetShader(shader, nullptr, 0);
    
    // Create and set UAVs (Unordered Access Views)
    ID3D11UnorderedAccessView* uavs[3];
    CreateUAV(params.input1, &uavs[0]);
    CreateUAV(params.input2, &uavs[1]);
    CreateUAV(params.output, &uavs[2]);
    
    context->CSSetUnorderedAccessViews(0, 3, uavs, nullptr);
    
    // Set constant buffer (kernel parameters)
    struct Constants {
        UINT size;
        UINT padding[3];
    } constants = { params.size, {0,0,0} };
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    context->Map(constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &constants, sizeof(Constants));
    context->Unmap(constantBuffer, 0);
    
    context->CSSetConstantBuffers(0, 1, &constantBuffer);
    
    // Dispatch compute shader
    // 256 threads per group, calculate number of groups
    UINT numGroups = (params.size + 255) / 256;
    context->Dispatch(numGroups, 1, 1);
    
    // Unbind resources
    ID3D11UnorderedAccessView* nullUAVs[3] = { nullptr, nullptr, nullptr };
    context->CSSetUnorderedAccessViews(0, 3, nullUAVs, nullptr);
    
    // Cleanup
    for (int i = 0; i < 3; i++) {
        uavs[i]->Release();
    }
}
```

---

## Benchmark Execution

### Execution Pipeline

```
User Clicks "Run Benchmark"
    ↓
Create Worker Thread
    ↓
FOR EACH benchmark type:
    ↓
    Allocate Host Memory
    Initialize Test Data
        ↓
    Allocate Device Memory
    Copy Data to Device
        ↓
    Warmup Phase (3 iterations)
    ├─ Execute kernel
    ├─ Synchronize
    └─ (Stabilize GPU clocks)
        ↓
    Measurement Phase
    ├─ Start GPU timer
    ├─ Execute kernel (N iterations)
    ├─ Stop GPU timer
    └─ Record time
        ↓
    Copy Results Back
    Verify Correctness
        ↓
    Calculate Metrics
    ├─ Bandwidth (GB/s)
    ├─ Throughput (GFLOPS)
    └─ Efficiency (%)
        ↓
    Update GUI
    ├─ Progress bar
    ├─ Current result
    └─ Add to history
        ↓
    Free Device Memory
    Free Host Memory
    ↓
END LOOP
    ↓
Display Final Summary
Export to CSV (if requested)
```

### Detailed Benchmark Workflow

#### 1. Data Preparation
```cpp
void VectorAddBenchmark::Setup(size_t size) {
    this->size = size;
    
    // Allocate host memory
    hostA = new float[size];
    hostB = new float[size];
    hostC = new float[size];
    hostReference = new float[size];
    
    // Initialize with test data
    for (size_t i = 0; i < size; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    // Calculate reference (CPU)
    for (size_t i = 0; i < size; i++) {
        hostReference[i] = hostA[i] + hostB[i];
    }
}
```

#### 2. GPU Execution
```cpp
BenchmarkResult VectorAddBenchmark::Run(IComputeBackend* backend) {
    BenchmarkResult result;
    
    // Allocate device memory
    void* devA = backend->AllocateMemory(size * sizeof(float));
    void* devB = backend->AllocateMemory(size * sizeof(float));
    void* devC = backend->AllocateMemory(size * sizeof(float));
    
    // Transfer data to GPU
    auto t_start = std::chrono::high_resolution_clock::now();
    backend->CopyHostToDevice(devA, hostA, size * sizeof(float));
    backend->CopyHostToDevice(devB, hostB, size * sizeof(float));
    auto t_end = std::chrono::high_resolution_clock::now();
    
    result.transferTimeMS = 
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    // Warmup (stabilize GPU clocks)
    for (int i = 0; i < 3; i++) {
        backend->ExecuteKernel("vectorAdd", {devA, devB, devC, size});
        backend->Synchronize();
    }
    
    // Actual benchmark
    const int iterations = 10;
    backend->StartTimer();
    
    for (int i = 0; i < iterations; i++) {
        backend->ExecuteKernel("vectorAdd", {devA, devB, devC, size});
    }
    
    backend->Synchronize();
    result.kernelTimeMS = backend->GetElapsedTimeMS() / iterations;
    
    // Copy results back
    backend->CopyDeviceToHost(hostC, devC, size * sizeof(float));
    
    // Verify correctness
    result.correct = VerifyResults();
    
    // Calculate performance metrics
    size_t bytesProcessed = 3 * size * sizeof(float);  // Read A, B; Write C
    result.bandwidthGBps = (bytesProcessed / result.kernelTimeMS) / 1e6;
    
    // Cleanup
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}
```

#### 3. Result Verification
```cpp
bool VectorAddBenchmark::VerifyResults() {
    const float epsilon = 1e-5f;  // Floating-point tolerance
    int errors = 0;
    
    for (size_t i = 0; i < size && errors < 10; i++) {
        float diff = std::abs(hostC[i] - hostReference[i]);
        if (diff > epsilon) {
            std::cerr << "Error at index " << i 
                      << ": GPU=" << hostC[i]
                      << ", CPU=" << hostReference[i]
                      << ", diff=" << diff << std::endl;
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "✓ Results verified: All " << size 
                  << " elements correct!" << std::endl;
    } else {
        std::cout << "✗ Verification failed: " << errors 
                  << " errors found" << std::endl;
    }
    
    return errors == 0;
}
```

---

## Memory Management

### RAII Pattern for GPU Resources

```cpp
class GPUMemoryHandle {
    IComputeBackend* backend;
    void* devicePtr;
    
public:
    GPUMemoryHandle(IComputeBackend* backend, size_t size)
        : backend(backend)
    {
        devicePtr = backend->AllocateMemory(size);
    }
    
    void* Get() const { return devicePtr; }
    
    ~GPUMemoryHandle() {
        if (devicePtr) {
            backend->FreeMemory(devicePtr);
        }
    }
    
    // Prevent copying (move-only)
    GPUMemoryHandle(const GPUMemoryHandle&) = delete;
    GPUMemoryHandle& operator=(const GPUMemoryHandle&) = delete;
    
    // Allow moving
    GPUMemoryHandle(GPUMemoryHandle&& other) noexcept
        : backend(other.backend), devicePtr(other.devicePtr)
    {
        other.devicePtr = nullptr;
    }
};

// Usage:
void SomeBenchmark() {
    GPUMemoryHandle memA(backend, 1024 * sizeof(float));
    GPUMemoryHandle memB(backend, 1024 * sizeof(float));
    
    // Use memA.Get() and memB.Get()
    // ...
    
    // Automatic cleanup when going out of scope!
}
```

---

## Timing and Synchronization

### Why GPU Timing is Special

**Problem:** GPUs execute asynchronously from CPU.

```cpp
// WRONG: This measures CPU time, not GPU time!
auto start = std::chrono::high_resolution_clock::now();
launchKernel<<<...>>>();  // Returns immediately!
auto end = std::chrono::high_resolution_clock::now();
// This duration is ~microseconds, not the actual kernel time!
```

**Solution:** Use GPU-side timing APIs.

### Correct GPU Timing

#### CUDA
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<...>>>();
cudaEventRecord(stop);

cudaEventSynchronize(stop);  // Wait for kernel completion

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

#### OpenCL
```cpp
cl_event event;
clEnqueueNDRangeKernel(..., &event);
clWaitForEvents(1, &event);

cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, ...);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, ...);

double ms = (end - start) / 1e6;
```

#### DirectCompute
```cpp
ID3D11Query* queryStart, *queryEnd;
// Create timestamp queries
context->End(queryStart);
context->Dispatch(...);
context->End(queryEnd);

UINT64 startTime, endTime, freq;
context->GetData(queryStart, &startTime, sizeof(UINT64), 0);
context->GetData(queryEnd, &endTime, sizeof(UINT64), 0);

// Get GPU timestamp frequency
queryDisjoint->GetData(&disjointData, sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT), 0);
freq = disjointData.Frequency;

double ms = (endTime - startTime) * 1000.0 / freq;
```

---

## GUI Threading Model

### Main Thread vs Worker Thread

```
┌────────────────────────────────────────────────────┐
│ MAIN THREAD (GUI)                                  │
│                                                    │
│  while (running) {                                 │
│      Process Windows messages                      │
│      Update ImGui frame                            │
│      Render UI                                     │
│      Check worker thread status  ←────┐           │
│      Display progress bar              │           │
│  }                                     │           │
└────────────────────────────────────────┼───────────┘
                                         │
                                         │ Atomic variables
                                         │ (progress, status)
                                         │
┌────────────────────────────────────────┼───────────┐
│ WORKER THREAD (Benchmarks)             │           │
│                                        │           │
│  Initialize backend                    │           │
│  for each benchmark:                   │           │
│      Setup data                        │           │
│      Run benchmark                     │           │
│      Update progress  ─────────────────┘           │
│      Store results                                 │
│  Cleanup                                           │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Thread-Safe Communication

```cpp
struct AppState {
    // Atomic variables for thread-safe access
    std::atomic<bool> benchmarkRunning{false};
    std::atomic<float> currentProgress{0.0f};
    std::atomic<int> currentBenchmark{0};
    
    // Mutex-protected data
    std::mutex resultsMutex;
    std::vector<BenchmarkResult> results;
    
    // Worker thread handle
    std::thread workerThread;
};

// Main thread (GUI)
void UpdateGUI(AppState& state) {
    if (state.benchmarkRunning.load()) {
        float progress = state.currentProgress.load();
        ImGui::ProgressBar(progress);
    }
    
    // Lock mutex to read results
    {
        std::lock_guard<std::mutex> lock(state.resultsMutex);
        for (const auto& result : state.results) {
            DisplayResult(result);
        }
    }
}

// Worker thread (Benchmarks)
void WorkerThread(AppState& state) {
    state.benchmarkRunning.store(true);
    
    for (int i = 0; i < numBenchmarks; i++) {
        auto result = RunBenchmark(i);
        
        // Update progress
        float progress = (i + 1) / (float)numBenchmarks;
        state.currentProgress.store(progress);
        
        // Store result (mutex-protected)
        {
            std::lock_guard<std::mutex> lock(state.resultsMutex);
            state.results.push_back(result);
        }
    }
    
    state.benchmarkRunning.store(false);
}
```

---

## Data Structures

### System Capabilities
```cpp
struct SystemCapabilities {
    struct BackendInfo {
        bool available;
        std::string deviceName;
        size_t totalMemoryMB;
        std::string driverVersion;
    };
    
    struct GPUInfo {
        std::string name;
        std::string vendor;
        size_t dedicatedMemoryMB;
        size_t sharedMemoryMB;
    };
    
    std::vector<GPUInfo> gpus;
    int primaryGPUIndex;
    
    BackendInfo cuda;
    BackendInfo opencl;
    BackendInfo directCompute;
};
```

### Benchmark Result
```cpp
struct BenchmarkResult {
    std::string benchmarkName;  // "VectorAdd", "MatrixMul", etc.
    std::string backendName;    // "CUDA", "OpenCL", "DirectCompute"
    
    // Timing
    double kernelTimeMS;        // GPU execution time
    double transferTimeMS;      // CPU→GPU + GPU→CPU transfer time
    double totalTimeMS;         // Total wall-clock time
    
    // Performance Metrics
    double bandwidthGBps;       // Memory bandwidth (GB/s)
    double throughputGFLOPS;    // Compute throughput (GFLOPS)
    double efficiency;          // % of theoretical peak
    
    // Verification
    bool correct;               // Result verified against CPU?
    std::string errorMessage;   // If incorrect, what went wrong
    
    // Metadata
    std::string timestamp;      // When test was run
    size_t problemSize;         // Number of elements processed
    int iterations;             // Number of iterations averaged
};
```

### History Tracking
```cpp
struct BenchmarkHistory {
    struct TestResult {
        float bandwidth;        // Main metric for graph
        double timestamp;       // Unix timestamp for sorting
        std::string testID;     // "Test 1", "Test 2", etc.
        double gflops;          // Secondary metric
        double timeMS;          // Execution time
        std::string dateTime;   // Human-readable "Jan 9, 2026 14:30"
    };
    
    std::vector<TestResult> vectorAdd;
    std::vector<TestResult> matrixMul;
    std::vector<TestResult> convolution;
    std::vector<TestResult> reduction;
    
    int totalTests = 0;  // Incremented with each test
};

// Separate history for each backend
BenchmarkHistory cudaHistory;
BenchmarkHistory openclHistory;
BenchmarkHistory directcomputeHistory;
```

---

## Performance Considerations

### Memory Coalescing
```cpp
// BAD: Uncoalesced access (strided)
__global__ void badKernel(float* data, int stride) {
    int idx = threadIdx.x * stride;  // NOT coalesced!
    data[idx] = ...;
}
// Threads 0,1,2,... access addresses 0, stride, 2*stride, ...
// Result: Multiple memory transactions, low bandwidth

// GOOD: Coalesced access (sequential)
__global__ void goodKernel(float* data) {
    int idx = threadIdx.x;  // Coalesced!
    data[idx] = ...;
}
// Threads 0,1,2,... access addresses 0,1,2,...
// Result: Single memory transaction per warp, high bandwidth
```

### Occupancy vs. Performance
```
High Occupancy ≠ High Performance!

Occupancy = Active Warps / Maximum Warps

Too Low Occupancy (< 25%):
  - Not enough work to hide memory latency
  - GPU underutilized

Too High Occupancy (> 75%):
  - May limit registers per thread
  - May limit shared memory per block
  - Can reduce ILP (instruction-level parallelism)

Sweet Spot: 40-60% occupancy often optimal
```

---

**This is how the internal machinery works. Now you can explain it confidently!**

---

**Created by:** Soham Dave  
**Date:** January 2026  
**Purpose:** Deep technical understanding for interviews and development
