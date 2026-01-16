/*******************************************************************************
 * FILE: main_working.cpp
 * 
 * PURPOSE:
 *   WORKING main application that properly uses all 3 backends
 *   (CUDA, OpenCL, DirectCompute) with their native kernel execution methods
 * 
 * AUTHOR: Soham Dave
 * DATE: January 2026
 ******************************************************************************/

#include "core/Logger.h"
#include "core/DeviceDiscovery.h"
#include "backends/cuda/CUDABackend.h"
#include "backends/opencl/OpenCLBackend.h"
#include "backends/directcompute/DirectComputeBackend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace GPUBenchmark;

// CUDA kernel launcher (only for CUDA)
extern "C" void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n);

// OpenCL vector add kernel source
const char* openclVectorAddSource = R"(
__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* c, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

// DirectCompute HLSL shader
const char* hlslVectorAddSource = R"(
RWStructuredBuffer<float> inputA : register(u0);
RWStructuredBuffer<float> inputB : register(u1);
RWStructuredBuffer<float> output : register(u2);

cbuffer Constants : register(b0) {
    uint numElements;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx < numElements) {
        output[idx] = inputA[idx] + inputB[idx];
    }
}
)";

// Run VectorAdd benchmark on CUDA
BenchmarkResult RunVectorAddCUDA(CUDABackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = "CUDA";
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    Logger& logger = Logger::GetInstance();
    size_t bytes = numElements * sizeof(float);
    
    // Allocate and initialize host data
    std::vector<float> hostA(numElements), hostB(numElements), hostC(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    // Copy to device
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        launchVectorAdd((const float*)devA, (const float*)devB, (float*)devC, numElements);
        backend->Synchronize();
    }
    
    // Benchmark
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        launchVectorAdd((const float*)devA, (const float*)devB, (float*)devC, numElements);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    // Copy back and verify
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    int errors = 0;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        if (std::abs(hostC[i] - (hostA[i] + hostB[i])) > 1e-5f) errors++;
    }
    result.resultCorrect = (errors == 0);
    
    double totalBytes = 3.0 * bytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Cleanup
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

// Run VectorAdd benchmark on OpenCL
BenchmarkResult RunVectorAddOpenCL(OpenCLBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = "OpenCL";
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    Logger& logger = Logger::GetInstance();
    size_t bytes = numElements * sizeof(float);
    
    // Compile kernel
    if (!backend->CompileKernel("vectorAdd", openclVectorAddSource)) {
        logger.Error("Failed to compile OpenCL kernel");
        result.resultCorrect = false;
        return result;
    }
    
    // Allocate and initialize host data
    std::vector<float> hostA(numElements), hostB(numElements), hostC(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    // Copy to device
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    // Set kernel arguments
    backend->SetKernelArg("vectorAdd", 0, sizeof(cl_mem), &devA);
    backend->SetKernelArg("vectorAdd", 1, sizeof(cl_mem), &devB);
    backend->SetKernelArg("vectorAdd", 2, sizeof(cl_mem), &devC);
    int n = static_cast<int>(numElements);
    backend->SetKernelArg("vectorAdd", 3, sizeof(int), &n);
    
    // Query device for maximum work group size
    DeviceInfo deviceInfo = backend->GetDeviceInfo();
    size_t maxWorkGroupSize = deviceInfo.maxThreadsPerBlock;
    
    // Use a safe work group size (common values: 64, 128, 256)
    size_t localWorkSize = (256 < maxWorkGroupSize) ? 256 : maxWorkGroupSize;
    if (localWorkSize == 0) localWorkSize = 64;  // Fallback
    
    size_t globalWorkSize = ((numElements + localWorkSize - 1) / localWorkSize) * localWorkSize;
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        backend->ExecuteKernel("vectorAdd", &globalWorkSize, &localWorkSize, 1);
        backend->Synchronize();
    }
    
    // Benchmark
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->ExecuteKernel("vectorAdd", &globalWorkSize, &localWorkSize, 1);
    }
    backend->Synchronize();  // Ensure all kernels complete before stopping timer
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    // Copy back and verify
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    int errors = 0;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        if (std::abs(hostC[i] - (hostA[i] + hostB[i])) > 1e-5f) errors++;
    }
    result.resultCorrect = (errors == 0);
    
    double totalBytes = 3.0 * bytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Cleanup
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

// Run VectorAdd benchmark on DirectCompute
BenchmarkResult RunVectorAddDirectCompute(DirectComputeBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = "DirectCompute";
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    Logger& logger = Logger::GetInstance();
    size_t bytes = numElements * sizeof(float);
    
    // Compile shader
    if (!backend->CompileShader("VectorAdd", hlslVectorAddSource, "CSMain", "cs_5_0")) {
        logger.Error("Failed to compile DirectCompute shader");
        result.resultCorrect = false;
        return result;
    }
    
    // Allocate and initialize host data
    std::vector<float> hostA(numElements), hostB(numElements), hostC(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    // Copy to device
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    // Bind UAVs
    backend->BindBufferUAV(devA, 0);
    backend->BindBufferUAV(devB, 1);
    backend->BindBufferUAV(devC, 2);
    
    // Set constant buffer
    struct Constants { unsigned int numElements; } constants;
    constants.numElements = static_cast<unsigned int>(numElements);
    backend->SetConstantBuffer(&constants, sizeof(constants), 0);
    
    unsigned int threadGroupsX = (static_cast<unsigned int>(numElements) + 255) / 256;
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        backend->DispatchShader("VectorAdd", threadGroupsX, 1, 1);
        backend->Synchronize();
    }
    
    // Benchmark
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->DispatchShader("VectorAdd", threadGroupsX, 1, 1);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    // Copy back and verify
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    int errors = 0;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        if (std::abs(hostC[i] - (hostA[i] + hostB[i])) > 1e-5f) errors++;
    }
    result.resultCorrect = (errors == 0);
    
    double totalBytes = 3.0 * bytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Cleanup
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

int main(int argc, char** argv) {
    Logger& logger = Logger::GetInstance();
    logger.SetLogLevel(LogLevel::INFO);
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║       GPU COMPUTE BENCHMARK SUITE v2.0 (WORKING!)     ║\n";
    std::cout << "║       All 3 Backends: CUDA | OpenCL | DirectCompute    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    // Discover system
    logger.Info("Discovering system capabilities...");
    SystemCapabilities sysCaps = DeviceDiscovery::Discover();
    
    logger.Info("\n=== SYSTEM INFORMATION ===");
    GPUInfo primaryGPU = sysCaps.GetPrimaryGPU();
    logger.Info("GPU: " + primaryGPU.name);
    logger.Info("CUDA Available: " + std::string(sysCaps.cuda.available ? "YES" : "NO"));
    logger.Info("OpenCL Available: " + std::string(sysCaps.opencl.available ? "YES" : "NO"));
    logger.Info("DirectCompute Available: " + std::string(sysCaps.directCompute.available ? "YES" : "NO"));
    logger.Info("");
    
    std::vector<BenchmarkResult> allResults;
    
    // Test problem size (smaller for stability)
    size_t numElements = 1000000;  // 1M elements
    int iterations = 50;
    
    // ==================== CUDA ====================
    if (sysCaps.cuda.available) {
        logger.Info("\n╔════════════════════════════════════╗");
        logger.Info("║         TESTING CUDA BACKEND        ║");
        logger.Info("╚════════════════════════════════════╝\n");
        
        CUDABackend cudaBackend;
        if (cudaBackend.Initialize()) {
            BenchmarkResult result = RunVectorAddCUDA(&cudaBackend, numElements, iterations);
            allResults.push_back(result);
            
            logger.Info("✓ VectorAdd (CUDA): " + std::to_string(result.effectiveBandwidthGBs) + " GB/s");
            logger.Info("  Execution Time: " + std::to_string(result.executionTimeMS) + " ms");
            logger.Info("  Result: " + std::string(result.resultCorrect ? "PASS" : "FAIL"));
            
            cudaBackend.Shutdown();
        } else {
            logger.Error("Failed to initialize CUDA backend");
        }
    }
    
    // ==================== OPENCL ====================
    if (sysCaps.opencl.available) {
        logger.Info("\n╔════════════════════════════════════╗");
        logger.Info("║        TESTING OPENCL BACKEND       ║");
        logger.Info("╚════════════════════════════════════╝\n");
        
        OpenCLBackend openclBackend;
        if (openclBackend.Initialize()) {
            BenchmarkResult result = RunVectorAddOpenCL(&openclBackend, numElements, iterations);
            allResults.push_back(result);
            
            logger.Info("✓ VectorAdd (OpenCL): " + std::to_string(result.effectiveBandwidthGBs) + " GB/s");
            logger.Info("  Execution Time: " + std::to_string(result.executionTimeMS) + " ms");
            logger.Info("  Result: " + std::string(result.resultCorrect ? "PASS" : "FAIL"));
            
            openclBackend.Shutdown();
        } else {
            logger.Error("Failed to initialize OpenCL backend");
        }
    }
    
    // ==================== DIRECTCOMPUTE ====================
    if (sysCaps.directCompute.available) {
        logger.Info("\n╔════════════════════════════════════╗");
        logger.Info("║    TESTING DIRECTCOMPUTE BACKEND    ║");
        logger.Info("╚════════════════════════════════════╝\n");
        
        DirectComputeBackend dcBackend;
        if (dcBackend.Initialize()) {
            BenchmarkResult result = RunVectorAddDirectCompute(&dcBackend, numElements, iterations);
            allResults.push_back(result);
            
            logger.Info("✓ VectorAdd (DirectCompute): " + std::to_string(result.effectiveBandwidthGBs) + " GB/s");
            logger.Info("  Execution Time: " + std::to_string(result.executionTimeMS) + " ms");
            logger.Info("  Result: " + std::string(result.resultCorrect ? "PASS" : "FAIL"));
            
            dcBackend.Shutdown();
        } else {
            logger.Error("Failed to initialize DirectCompute backend");
        }
    }
    
    // ==================== SUMMARY ====================
    logger.Info("\n╔════════════════════════════════════════════════════════╗");
    logger.Info("║                    BENCHMARK SUMMARY                   ║");
    logger.Info("╚════════════════════════════════════════════════════════╝\n");
    
    for (const auto& result : allResults) {
        std::cout << result.backendName << " VectorAdd: "
                  << result.effectiveBandwidthGBs << " GB/s  ["
                  << (result.resultCorrect ? "PASS" : "FAIL") << "]\n";
    }
    
    // Export to CSV
    std::ofstream csv("benchmark_results_working.csv");
    if (csv.is_open()) {
        csv << "Backend,Benchmark,Elements,Time_ms,Bandwidth_GBs,Status\n";
        for (const auto& result : allResults) {
            csv << result.backendName << ","
                << result.benchmarkName << ","
                << result.problemSize << ","
                << result.executionTimeMS << ","
                << result.effectiveBandwidthGBs << ","
                << (result.resultCorrect ? "PASS" : "FAIL") << "\n";
        }
        csv.close();
        logger.Info("\n✓ Results exported to: benchmark_results_working.csv");
    }
    
    std::cout << "\n";
    logger.Info("Benchmark complete!");
    logger.Info("Press Enter to exit...");
    std::cin.get();
    
    return 0;
}
