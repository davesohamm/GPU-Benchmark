# CUDA Backend Documentation

## Overview

The CUDA backend provides NVIDIA-specific GPU compute capabilities using the CUDA programming model. This is typically the **fastest** backend on NVIDIA GPUs because it has direct hardware access.

---

## Why CUDA?

### Advantages
✅ **Performance**: Direct hardware access, minimal overhead  
✅ **Maturity**: Extensive tooling, profiling, debugging  
✅ **Features**: Access to Tensor Cores, async operations, unified memory  
✅ **Optimization**: Highly optimized compiler, extensive optimization opportunities  
✅ **Documentation**: Comprehensive official documentation from NVIDIA

### Disadvantages
❌ **NVIDIA Only**: Requires NVIDIA GPU (won't work on AMD/Intel)  
❌ **Proprietary**: Not an open standard like OpenCL  
❌ **Driver Dependency**: Requires NVIDIA drivers installed  

---

## CUDA Programming Model

### Thread Hierarchy

```
Grid (all threads)
  └─ Block (group of threads)
      └─ Thread (individual execution unit)
```

**Example:**
```cuda
// Launch configuration: <<<gridSize, blockSize>>>
vectorAddKernel<<<256, 256>>>(...);
// This launches:
//   - 256 blocks
//   - 256 threads per block
//   - Total: 65,536 threads!
```

### Memory Hierarchy

```
Global Memory (slowest, largest)    ~4 GB on RTX 3050
  ├─ L2 Cache                       ~1 MB
  └─ L1 Cache / Shared Memory       ~100 KB per SM
      └─ Registers (fastest)        ~65K per SM
```

**Access times (approximate):**
- Registers: 1 cycle
- Shared Memory: ~30 cycles
- L1 Cache: ~30 cycles
- L2 Cache: ~200 cycles
- Global Memory: ~500 cycles

---

## Files in This Directory

### CUDABackend.h
**Purpose**: Header file declaring the CUDA backend class

**Key class**: `CUDABackend : public IComputeBackend`

**What it contains**:
- Class declaration
- Private member variables (device pointers, CUDA events)
- Public method declarations (inherited from IComputeBackend)

### CUDABackend.cpp
**Purpose**: Implementation of CUDA backend

**What it contains**:
- CUDA initialization (cudaSetDevice, query properties)
- Memory management (cudaMalloc, cudaFree, cudaMemcpy)
- Kernel launching logic
- Error checking and reporting
- GPU timing using CUDA events

### kernels/ Directory
**Purpose**: CUDA kernel implementations (.cu files)

**Files**:
1. **vector_add.cu**: Element-wise vector addition
2. **matrix_mul.cu**: Matrix multiplication with shared memory
3. **convolution.cu**: 2D image convolution
4. **reduction.cu**: Parallel sum reduction

---

## CUDA API Essentials

### Device Management

```cpp
// Query number of CUDA devices
int deviceCount;
cudaGetDeviceCount(&deviceCount);

// Select device (use device 0)
cudaSetDevice(0);

// Query device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Device name: prop.name
// Total memory: prop.totalGlobalMem
// Compute capability: prop.major, prop.minor
```

### Memory Management

```cpp
// Allocate GPU memory
float* d_data;
cudaMalloc(&d_data, 1024 * sizeof(float));

// Copy CPU → GPU
float h_data[1024];
cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);

// Copy GPU → CPU
cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

### Kernel Execution

```cpp
// Define kernel (in .cu file)
__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Launch kernel (in .cpp file)
int threadsPerBlock = 256;
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
myKernel<<<blocks, threadsPerBlock>>>(d_data, n);

// Wait for kernel to finish
cudaDeviceSynchronize();
```

### Timing

```cpp
// Create events
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record start time
cudaEventRecord(start);

// Launch kernel
myKernel<<<blocks, threads>>>(...);

// Record end time
cudaEventRecord(stop);

// Wait for events to complete
cudaEventSynchronize(stop);

// Calculate elapsed time
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

// Cleanup
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## Error Handling

Every CUDA call can fail! Always check errors:

```cpp
cudaError_t error = cudaMalloc(&d_data, size);
if (error != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    // Handle error...
}
```

**Common CUDA errors**:
- `cudaErrorMemoryAllocation`: Out of GPU memory
- `cudaErrorInvalidConfiguration`: Invalid kernel launch parameters
- `cudaErrorNoDevice`: No CUDA-capable device
- `cudaErrorInvalidValue`: Invalid function argument

---

## Performance Tips

### 1. Memory Coalescing

**Problem**: Threads accessing non-contiguous memory

**Bad** (each thread reads from different memory location):
```cuda
__global__ void badKernel(float* data) {
    int idx = threadIdx.x * STRIDE;  // Large stride = bad!
    data[idx] = ...;
}
```

**Good** (adjacent threads access adjacent memory):
```cuda
__global__ void goodKernel(float* data) {
    int idx = threadIdx.x;  // Contiguous = good!
    data[idx] = ...;
}
```

### 2. Shared Memory Usage

Shared memory is **100x faster** than global memory!

```cuda
__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory tile
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Load data into shared memory
    As[ty][tx] = A[...];
    Bs[ty][tx] = B[...];
    __syncthreads();  // Wait for all threads to load
    
    // Compute using shared memory (fast!)
    for (int k = 0; k < BLOCK_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
}
```

### 3. Occupancy Optimization

**Goal**: Keep all GPU cores busy

**Factors**:
- Threads per block (aim for 128-256)
- Registers per thread (less = more concurrent threads)
- Shared memory per block (less = more concurrent blocks)

**Tool**: CUDA Occupancy Calculator (included in CUDA Toolkit)

### 4. Avoid Divergence

**Problem**: Threads in a warp taking different code paths

**Bad** (half the threads idle):
```cuda
if (threadIdx.x < 16) {
    // Only first 16 threads execute
    // Other 16 threads wait!
}
```

**Better** (all threads active):
```cuda
// Design algorithm so all threads do useful work
```

---

## Debugging CUDA

### Compile-Time Checks

```bash
nvcc -lineinfo -G my_kernel.cu
```

**Flags**:
- `-lineinfo`: Add line number info
- `-G`: Generate debug info
- `-Xcompiler /W4`: Enable C++ warnings

### Runtime Checks

```cpp
// Check for errors after kernel launch
myKernel<<<blocks, threads>>>(...);
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
}
```

### CUDA Tools

- **Nsight Compute**: Kernel profiler
- **Nsight Systems**: System-wide profiler
- **cuda-memcheck**: Memory debugger (like Valgrind for GPU)
- **nvprof**: Command-line profiler

---

## RTX 3050 Specifications

Your GPU capabilities:

| Specification | Value |
|---------------|-------|
| Architecture | Ampere (GA107) |
| Compute Capability | 8.6 |
| CUDA Cores | 2048 |
| SM Count | 16 |
| Boost Clock | ~1740 MHz |
| Memory | 4 GB GDDR6 |
| Memory Bandwidth | ~192 GB/s |
| Memory Bus | 128-bit |
| TDP | 60W (laptop variant) |
| FP32 Performance | ~7.1 TFLOPS |

**What this means for benchmarking**:
- **Memory bandwidth** will be the bottleneck for most operations
- **4 GB VRAM** limits maximum problem size
- **2048 cores** is modest (for comparison, RTX 3090 has 10496)
- **Compute capability 8.6** supports all modern CUDA features

---

## Kernel Implementation Guide

### 1. Vector Addition (Simplest)

**Algorithm**: `C[i] = A[i] + B[i]`

**Characteristics**:
- Memory-bound (limited by bandwidth)
- No synchronization needed
- Perfect coalescing possible
- Good starter kernel

**Expected Performance**:
- 40-60% of memory bandwidth
- ~0.2-0.5 ms for 1M elements

### 2. Matrix Multiplication

**Algorithm**: `C = A × B`

**Characteristics**:
- Compute-intensive
- Benefits from shared memory
- Cache-friendly with tiling
- More complex

**Expected Performance**:
- Naive: ~5% of peak FLOPS
- With shared memory: ~30% of peak FLOPS
- cuBLAS library: ~70% of peak FLOPS

### 3. Convolution

**Algorithm**: 2D filter applied to image

**Characteristics**:
- Memory-bound with reuse
- Benefits from constant memory for filter
- Texture cache helps
- Real-world use case

**Expected Performance**:
- 30-50% of bandwidth
- ~2-5 ms for 1024×1024 image

### 4. Reduction

**Algorithm**: Sum all elements → single value

**Characteristics**:
- Requires synchronization
- Tree-based reduction
- Shared memory critical
- Bank conflicts possible

**Expected Performance**:
- 50-70% of bandwidth
- ~0.1-0.3 ms for 1M elements

---

## Building with CUDA

### CMakeLists.txt Integration

```cmake
find_package(CUDA REQUIRED)

cuda_add_executable(myapp
    main.cpp
    kernels/vector_add.cu
)

target_link_libraries(myapp ${CUDA_LIBRARIES})
```

### Compute Capability

**Your GPU (RTX 3050)**: Compute Capability 8.6

**CMake flag**:
```cmake
set(CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86")
```

**What this does**:
- Compile for Ampere architecture
- Use architecture-specific optimizations
- Enables all GPU features

---

## Common Mistakes

### 1. Forgetting to Synchronize

**Wrong**:
```cpp
kernel<<<...>>>(...);
cudaMemcpy(h_result, d_result, ...);  // Might copy before kernel finishes!
```

**Right**:
```cpp
kernel<<<...>>>(...);
cudaDeviceSynchronize();  // Wait for kernel
cudaMemcpy(h_result, d_result, ...);
```

### 2. Not Checking Errors

**Wrong**:
```cpp
cudaMalloc(&ptr, size);  // Might fail silently!
```

**Right**:
```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    // Handle error
}
```

### 3. Launching Too Few Threads

**Wrong**:
```cpp
myKernel<<<1, 256>>>(...);  // Only 256 threads! (underutilizes GPU)
```

**Right**:
```cpp
int blocks = (n + 255) / 256;
myKernel<<<blocks, 256>>>(...);  // Thousands of threads!
```

---

## Interview Talking Points

When discussing CUDA implementation:

1. **Memory Hierarchy**: "I used shared memory to reduce global memory accesses by 10x"

2. **Coalescing**: "I ensured adjacent threads access adjacent memory for maximum bandwidth"

3. **Occupancy**: "I optimized register usage to achieve 75% occupancy on the GPU"

4. **Synchronization**: "I used __syncthreads() to coordinate threads within a block"

5. **Error Handling**: "I check every CUDA call and provide meaningful error messages"

6. **Portability**: "While CUDA is NVIDIA-specific, my abstract interface allows easy backend swapping"

---

## References

- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices Guide**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **CUDA Samples**: Included with CUDA Toolkit
- **NVIDIA Developer Blog**: https://developer.nvidia.com/blog

---

**Next Steps**: Implement CUDABackend.cpp and your first kernel (vector_add.cu)!
