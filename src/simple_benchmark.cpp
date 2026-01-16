/*******************************************************************************
 * FILE: simple_benchmark.cpp
 * 
 * PURPOSE:
 *   Implementation of simple backend-agnostic benchmarks
 ******************************************************************************/

#include "simple_benchmark.h"
#include "backends/cuda/CUDABackend.h"
#include "backends/opencl/OpenCLBackend.h"
#include "backends/directcompute/DirectComputeBackend.h"
#include <cmath>

// CUDA kernel launchers (only used when backend is CUDA)
extern "C" void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n);
extern "C" void launchMatrixMul(const float* d_A, const float* d_B, float* d_C, int N);

namespace GPUBenchmark {

BenchmarkResult SimpleVectorAddBenchmark(IComputeBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = backend->GetBackendName();
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    Logger& logger = Logger::GetInstance();
    logger.Info("Running VectorAdd: " + std::to_string(numElements) + " elements, " + std::to_string(iterations) + " iterations");
    
    size_t bytes = numElements * sizeof(float);
    
    // Allocate host memory
    std::vector<float> hostA(numElements);
    std::vector<float> hostB(numElements);
    std::vector<float> hostC(numElements);
    
    // Initialize data
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    void* deviceA = backend->AllocateMemory(bytes);
    void* deviceB = backend->AllocateMemory(bytes);
    void* deviceC = backend->AllocateMemory(bytes);
    
    if (!deviceA || !deviceB || !deviceC) {
        logger.Error("Failed to allocate device memory");
        result.resultCorrect = false;
        return result;
    }
    
    // Copy to device
    backend->CopyHostToDevice(deviceA, hostA.data(), bytes);
    backend->CopyHostToDevice(deviceB, hostB.data(), bytes);
    
    // Determine backend type and run appropriate kernel
    BackendType backendType = backend->GetType();
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        if (backendType == BackendType::CUDA) {
            launchVectorAdd((const float*)deviceA, (const float*)deviceB, (float*)deviceC, numElements);
        }
        // For OpenCL and DirectCompute, they need their own kernel execution
        // For now, just sync
        backend->Synchronize();
    }
    
    // Timed execution
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        if (backendType == BackendType::CUDA) {
            launchVectorAdd((const float*)deviceA, (const float*)deviceB, (float*)deviceC, numElements);
        }
        backend->Synchronize();
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    // Copy results back
    backend->CopyDeviceToHost(hostC.data(), deviceC, bytes);
    
    // Calculate bandwidth
    double totalBytes = 3.0 * bytes;  // Read A, B, write C
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Verify results
    int errors = 0;
    const float epsilon = 1e-5f;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        float expected = hostA[i] + hostB[i];
        if (std::abs(hostC[i] - expected) > epsilon) {
            errors++;
        }
    }
    result.resultCorrect = (errors == 0);
    
    // Cleanup
    backend->FreeMemory(deviceA);
    backend->FreeMemory(deviceB);
    backend->FreeMemory(deviceC);
    
    return result;
}

BenchmarkResult SimpleMatrixMulBenchmark(IComputeBackend* backend, int matrixSize, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "MatrixMul";
    result.backendName = backend->GetBackendName();
    result.problemSize = matrixSize * matrixSize;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    Logger& logger = Logger::GetInstance();
    logger.Info("Running MatrixMul: " + std::to_string(matrixSize) + "x" + std::to_string(matrixSize) + ", " + std::to_string(iterations) + " iterations");
    
    size_t numElements = matrixSize * matrixSize;
    size_t bytes = numElements * sizeof(float);
    
    // Allocate host memory
    std::vector<float> hostA(numElements);
    std::vector<float> hostB(numElements);
    std::vector<float> hostC(numElements);
    
    // Initialize data
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = 1.0f;
        hostB[i] = 2.0f;
    }
    
    // Allocate device memory
    void* deviceA = backend->AllocateMemory(bytes);
    void* deviceB = backend->AllocateMemory(bytes);
    void* deviceC = backend->AllocateMemory(bytes);
    
    if (!deviceA || !deviceB || !deviceC) {
        logger.Error("Failed to allocate device memory");
        result.resultCorrect = false;
        return result;
    }
    
    // Copy to device
    backend->CopyHostToDevice(deviceA, hostA.data(), bytes);
    backend->CopyHostToDevice(deviceB, hostB.data(), bytes);
    
    // Determine backend type
    BackendType backendType = backend->GetType();
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        if (backendType == BackendType::CUDA) {
            launchMatrixMul((const float*)deviceA, (const float*)deviceB, (float*)deviceC, matrixSize);
        }
        backend->Synchronize();
    }
    
    // Timed execution
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        if (backendType == BackendType::CUDA) {
            launchMatrixMul((const float*)deviceA, (const float*)deviceB, (float*)deviceC, matrixSize);
        }
        backend->Synchronize();
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    // Copy results back
    backend->CopyDeviceToHost(hostC.data(), deviceC, bytes);
    
    // Calculate GFLOPS
    double numOperations = 2.0 * matrixSize * matrixSize * matrixSize;  // 2N^3 operations
    result.computeThroughputGFLOPS = (numOperations / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Verify (simple check)
    float expected = 2.0f * matrixSize;  // Each element should be N*1*2 = 2N
    int errors = 0;
    const float epsilon = 0.1f * matrixSize;
    for (int i = 0; i < std::min(100, (int)numElements); i++) {
        if (std::abs(hostC[i] - expected) > epsilon) {
            errors++;
        }
    }
    result.resultCorrect = (errors < 5);  // Allow some numerical errors
    
    // Cleanup
    backend->FreeMemory(deviceA);
    backend->FreeMemory(deviceB);
    backend->FreeMemory(deviceC);
    
    return result;
}

} // namespace GPUBenchmark
