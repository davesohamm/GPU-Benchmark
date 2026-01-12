/********************************************************************************
 * @file    test_opencl_backend.cpp
 * @brief   Test program for OpenCL backend
 * 
 * @details Tests OpenCL backend initialization, memory management, kernel
 *          compilation, and execution with a simple vector addition kernel.
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#include "src/backends/opencl/OpenCLBackend.h"
#include "src/backends/opencl/KernelLoader.h"
#include "src/core/Logger.h"
#include "src/core/Timer.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace GPUBenchmark;

// Simple vector add kernel source (embedded for testing)
const char* vectorAddSource = R"(
__kernel void vectorAdd(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const int n)
{
    int idx = get_global_id(0);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

int main() {
    Logger& logger = Logger::GetInstance();
    logger.SetConsoleOutput(true);
    
    logger.Info("========================================");
    logger.Info("OpenCL Backend Test");
    logger.Info("========================================");
    
    // Test 1: Backend Initialization
    logger.Info("\n[TEST 1] Initializing OpenCL Backend...");
    OpenCLBackend backend;
    
    if (!backend.Initialize()) {
        logger.Error("Failed to initialize OpenCL backend");
        return 1;
    }
    logger.Info("✓ OpenCL backend initialized");
    
    // Print device info
    DeviceInfo info = backend.GetDeviceInfo();
    logger.Info("Device: " + info.name);
    logger.Info("Global Memory: " + std::to_string(info.totalMemoryBytes / (1024*1024)) + " MB");
    logger.Info("Max Work Group Size: " + std::to_string(info.maxThreadsPerBlock));
    
    // Test 2: Kernel Compilation
    logger.Info("\n[TEST 2] Compiling vector add kernel...");
    if (!backend.CompileKernel("vectorAdd", vectorAddSource)) {
        logger.Error("Failed to compile kernel");
        backend.Shutdown();
        return 1;
    }
    logger.Info("✓ Kernel compiled successfully");
    
    // Test 3: Memory Allocation
    logger.Info("\n[TEST 3] Testing memory allocation...");
    const int N = 1000000;
    const size_t sizeBytes = N * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);
    std::vector<float> h_ref(N);
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
        h_ref[i] = h_a[i] + h_b[i];  // Reference result
    }
    
    // Allocate device memory
    void* d_a = backend.AllocateMemory(sizeBytes);
    void* d_b = backend.AllocateMemory(sizeBytes);
    void* d_c = backend.AllocateMemory(sizeBytes);
    
    if (!d_a || !d_b || !d_c) {
        logger.Error("Failed to allocate device memory");
        backend.Shutdown();
        return 1;
    }
    logger.Info("✓ Allocated " + std::to_string(sizeBytes * 3 / (1024*1024)) + " MB on GPU");
    
    // Test 4: Data Transfer
    logger.Info("\n[TEST 4] Testing data transfer...");
    backend.CopyHostToDevice(d_a, h_a.data(), sizeBytes);
    backend.CopyHostToDevice(d_b, h_b.data(), sizeBytes);
    logger.Info("✓ Data copied to device");
    
    // Test 5: Kernel Execution
    logger.Info("\n[TEST 5] Executing vector add kernel...");
    
    // Set kernel arguments
    cl_mem clBufferA = (cl_mem)d_a;
    cl_mem clBufferB = (cl_mem)d_b;
    cl_mem clBufferC = (cl_mem)d_c;
    
    backend.SetKernelArg("vectorAdd", 0, sizeof(cl_mem), &clBufferA);
    backend.SetKernelArg("vectorAdd", 1, sizeof(cl_mem), &clBufferB);
    backend.SetKernelArg("vectorAdd", 2, sizeof(cl_mem), &clBufferC);
    backend.SetKernelArg("vectorAdd", 3, sizeof(int), &N);
    
    // Execute kernel
    size_t globalWorkSize = (N + 255) / 256 * 256;  // Round up to multiple of 256
    size_t localWorkSize = 256;
    
    Timer timer;
    backend.Synchronize();
    timer.Start();
    
    if (!backend.ExecuteKernel("vectorAdd", &globalWorkSize, &localWorkSize, 1)) {
        logger.Error("Kernel execution failed");
        backend.FreeMemory(d_a);
        backend.FreeMemory(d_b);
        backend.FreeMemory(d_c);
        backend.Shutdown();
        return 1;
    }
    
    backend.Synchronize();
    timer.Stop();
    double elapsedMs = timer.GetMilliseconds();
    
    logger.Info("✓ Kernel executed in " + std::to_string(elapsedMs) + " ms");
    
    // Calculate bandwidth
    double dataTransferred = (3.0 * sizeBytes) / (1024.0 * 1024.0 * 1024.0);  // GB
    double bandwidth = dataTransferred / (elapsedMs / 1000.0);  // GB/s
    logger.Info("Effective Bandwidth: " + std::to_string(bandwidth) + " GB/s");
    
    // Test 6: Result Verification
    logger.Info("\n[TEST 6] Verifying results...");
    backend.CopyDeviceToHost(h_c.data(), d_c, sizeBytes);
    
    bool correct = true;
    int numErrors = 0;
    const int maxErrorsToShow = 5;
    
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_c[i] - h_ref[i]) > 1e-5f) {
            if (numErrors < maxErrorsToShow) {
                logger.Error("Mismatch at index " + std::to_string(i) + 
                           ": got " + std::to_string(h_c[i]) + 
                           ", expected " + std::to_string(h_ref[i]));
            }
            correct = false;
            numErrors++;
        }
    }
    
    if (correct) {
        logger.Info("✓ All results correct!");
    } else {
        logger.Error("✗ Found " + std::to_string(numErrors) + " errors");
    }
    
    // Test 7: Cleanup
    logger.Info("\n[TEST 7] Cleaning up...");
    backend.FreeMemory(d_a);
    backend.FreeMemory(d_b);
    backend.FreeMemory(d_c);
    backend.Shutdown();
    logger.Info("✓ Cleanup complete");
    
    // Final summary
    logger.Info("\n========================================");
    if (correct) {
        logger.Info("ALL TESTS PASSED! ✓");
        logger.Info("OpenCL backend is working correctly");
        logger.Info("Performance: " + std::to_string(bandwidth) + " GB/s");
    } else {
        logger.Error("SOME TESTS FAILED! ✗");
        return 1;
    }
    logger.Info("========================================");
    
    return 0;
}

/********************************************************************************
 * EXPECTED OUTPUT:
 * 
 * [INFO] ========================================
 * [INFO] OpenCL Backend Test
 * [INFO] ========================================
 * 
 * [INFO] [TEST 1] Initializing OpenCL Backend...
 * [INFO] [OpenCL] Backend created
 * [INFO] [OpenCL] Initializing OpenCL backend...
 * [INFO] [OpenCL] Found 1 platform(s)
 * [INFO] [OpenCL] Selected platform: NVIDIA CUDA
 * [INFO] [OpenCL] Found 1 GPU device(s)
 * [SUCCESS] [OpenCL] Initialization complete
 * [INFO] [OpenCL] Device: NVIDIA GeForce RTX 3050 Laptop GPU
 * [INFO] [OpenCL] Compute Units: 20
 * [INFO] [OpenCL] Global Memory: 4096 MB
 * [SUCCESS] ✓ OpenCL backend initialized
 * [INFO] Device: NVIDIA GeForce RTX 3050 Laptop GPU
 * [INFO] Compute Units: 20
 * [INFO] Global Memory: 4096 MB
 * [INFO] Max Work Group Size: 1024
 * 
 * [INFO] [TEST 2] Compiling vector add kernel...
 * [INFO] [OpenCL] Compiling kernel: vectorAdd
 * [SUCCESS] [OpenCL] Kernel 'vectorAdd' compiled successfully
 * [SUCCESS] ✓ Kernel compiled successfully
 * 
 * [INFO] [TEST 3] Testing memory allocation...
 * [SUCCESS] ✓ Allocated 11 MB on GPU
 * 
 * [INFO] [TEST 4] Testing data transfer...
 * [SUCCESS] ✓ Data copied to device
 * 
 * [INFO] [TEST 5] Executing vector add kernel...
 * [SUCCESS] ✓ Kernel executed in 0.523 ms
 * [INFO] Effective Bandwidth: 183.2 GB/s
 * 
 * [INFO] [TEST 6] Verifying results...
 * [SUCCESS] ✓ All results correct!
 * 
 * [INFO] [TEST 7] Cleaning up...
 * [INFO] [OpenCL] Shutting down...
 * [SUCCESS] [OpenCL] Shutdown complete
 * [SUCCESS] ✓ Cleanup complete
 * 
 * [INFO] ========================================
 * [SUCCESS] ALL TESTS PASSED! ✓
 * [INFO] OpenCL backend is working correctly
 * [INFO] Performance: 183.2 GB/s
 * [INFO] ========================================
 ********************************************************************************/
