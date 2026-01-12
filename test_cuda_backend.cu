// Full CUDA Backend test with vector addition
#include "src/backends/cuda/CUDABackend.h"
#include "src/core/Logger.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace GPUBenchmark;

// Simple vector add kernel
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

bool test_vector_add(CUDABackend& cuda) {
    Logger& logger = Logger::GetInstance();
    logger.Info("\n=== Testing Vector Addition ===");
    
    const int N = 1000000;
    const size_t bytes = N * sizeof(float);
    
    logger.Info("Problem size: " + std::to_string(N) + " elements");
    logger.Info("Memory required: " + std::to_string(bytes / (1024 * 1024)) + " MB");
    
    // Allocate host memory
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    logger.Info("Allocating GPU memory...");
    float* d_a = static_cast<float*>(cuda.AllocateMemory(bytes));
    float* d_b = static_cast<float*>(cuda.AllocateMemory(bytes));
    float* d_c = static_cast<float*>(cuda.AllocateMemory(bytes));
    
    if (!d_a || !d_b || !d_c) {
        logger.Error("Failed to allocate GPU memory");
        if (d_a) cuda.FreeMemory(d_a);
        if (d_b) cuda.FreeMemory(d_b);
        if (d_c) cuda.FreeMemory(d_c);
        return false;
    }
    
    // Copy to device
    logger.Info("Copying data to GPU...");
    cuda.CopyHostToDevice(d_a, h_a.data(), bytes);
    cuda.CopyHostToDevice(d_b, h_b.data(), bytes);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    logger.Info("Launch config: " + std::to_string(gridSize) + " blocks x " + std::to_string(blockSize) + " threads");
    
    logger.Info("Launching kernel...");
    cuda.StartTimer();
    vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cuda.StopTimer();
    cuda.Synchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        logger.Error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
        cuda.FreeMemory(d_a);
        cuda.FreeMemory(d_b);
        cuda.FreeMemory(d_c);
        return false;
    }
    
    double kernel_ms = cuda.GetElapsedTime();
    logger.Info("Kernel execution time: " + std::to_string(kernel_ms) + " ms");
    
    // Copy result back
    logger.Info("Copying result from GPU...");
    cuda.CopyDeviceToHost(h_c.data(), d_c, bytes);
    
    // Verify
    logger.Info("Verifying result...");
    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5) {
            errors++;
            logger.Error("Error at " + std::to_string(i) + ": " + 
                        std::to_string(h_c[i]) + " != " + std::to_string(expected));
        }
    }
    
    // Cleanup
    cuda.FreeMemory(d_a);
    cuda.FreeMemory(d_b);
    cuda.FreeMemory(d_c);
    
    if (errors == 0) {
        logger.Info("✓ RESULT CORRECT! All " + std::to_string(N) + " elements match!");
        
        // Calculate bandwidth
        double totalBytes = 3.0 * bytes;
        double bandwidth = (totalBytes / (kernel_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        logger.Info("Kernel bandwidth: " + std::to_string(bandwidth) + " GB/s");
        
        return true;
    } else {
        logger.Error("✗ RESULT INCORRECT! Found " + std::to_string(errors) + " errors");
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU Benchmark - CUDA Backend Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    Logger& logger = Logger::GetInstance();
    
    // Initialize CUDA backend
    CUDABackend cuda;
    
    if (!cuda.Initialize()) {
        logger.Error("Failed to initialize CUDA backend");
        return 1;
    }
    
    // Get device info
    DeviceInfo info = cuda.GetDeviceInfo();
    
    std::cout << "\n=== Device Information ===" << std::endl;
    std::cout << "Name: " << info.name << std::endl;
    std::cout << "Compute Capability: " << info.computeCapabilityMajor << "." 
              << info.computeCapabilityMinor << std::endl;
    std::cout << "Total Memory: " << std::fixed << std::setprecision(2)
              << (info.totalMemoryBytes / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "Max Threads/Block: " << info.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;
    
    // Test vector addition
    bool success = test_vector_add(cuda);
    
    // Shutdown
    cuda.Shutdown();
    
    std::cout << "\n========================================" << std::endl;
    if (success) {
        std::cout << "  ✓ ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "  ✗ TESTS FAILED!" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return success ? 0 : 1;
}
