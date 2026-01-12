/********************************************************************************
 * @file    test_directcompute_backend.cpp
 * @brief   Test program for DirectCompute backend
 * 
 * @details Tests DirectCompute backend initialization, memory management, shader
 *          compilation, and execution with a simple vector addition shader.
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#include "src/backends/directcompute/DirectComputeBackend.h"
#include "src/core/Logger.h"
#include "src/core/Timer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace GPUBenchmark;

// Load shader source from file
std::string LoadShaderSource(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Simple vector add shader (embedded for testing)
const char* vectorAddShaderSource = R"(
RWStructuredBuffer<float> bufferA : register(u0);
RWStructuredBuffer<float> bufferB : register(u1);
RWStructuredBuffer<float> bufferC : register(u2);

cbuffer Constants : register(b0)
{
    uint numElements;
    uint3 padding;
};

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    
    if (idx < numElements)
    {
        bufferC[idx] = bufferA[idx] + bufferB[idx];
    }
}
)";

int main() {
    Logger& logger = Logger::GetInstance();
    logger.SetConsoleOutput(true);
    
    logger.Info("========================================");
    logger.Info("DirectCompute Backend Test");
    logger.Info("========================================");
    
    // Test 1: Backend Initialization
    logger.Info("\n[TEST 1] Initializing DirectCompute Backend...");
    DirectComputeBackend backend;
    
    if (!backend.Initialize()) {
        logger.Error("Failed to initialize DirectCompute backend");
        return 1;
    }
    logger.Info("✓ DirectCompute backend initialized");
    
    // Print device info
    DeviceInfo info = backend.GetDeviceInfo();
    logger.Info("Device: " + info.name);
    logger.Info("Memory: " + std::to_string(info.totalMemoryBytes / (1024*1024)) + " MB");
    logger.Info("Max Threads Per Group: " + std::to_string(info.maxThreadsPerBlock));
    
    // Test 2: Shader Compilation
    logger.Info("\n[TEST 2] Compiling vector add shader...");
    if (!backend.CompileShader("VectorAdd", vectorAddShaderSource, "CSMain", "cs_5_0")) {
        logger.Error("Failed to compile shader");
        logger.Error("Error: " + backend.GetLastError());
        backend.Shutdown();
        return 1;
    }
    logger.Info("✓ Shader compiled successfully");
    
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
    
    // Test 5: Shader Execution
    logger.Info("\n[TEST 5] Executing vector add shader...");
    
    // Bind buffers as UAVs
    backend.BindBufferUAV(d_a, 0);
    backend.BindBufferUAV(d_b, 1);
    backend.BindBufferUAV(d_c, 2);
    
    // Set constant buffer (numElements)
    struct Constants {
        UINT numElements;
        UINT padding[3];
    } constants;
    constants.numElements = N;
    constants.padding[0] = 0;
    constants.padding[1] = 0;
    constants.padding[2] = 0;
    
    backend.SetConstantBuffer(&constants, sizeof(constants), 0);
    
    // Calculate thread groups
    UINT threadGroupsX = (N + 255) / 256;  // 256 threads per group
    
    Timer timer;
    backend.Synchronize();
    timer.Start();
    
    if (!backend.DispatchShader("VectorAdd", threadGroupsX, 1, 1)) {
        logger.Error("Shader dispatch failed");
        logger.Error("Error: " + backend.GetLastError());
        backend.FreeMemory(d_a);
        backend.FreeMemory(d_b);
        backend.FreeMemory(d_c);
        backend.Shutdown();
        return 1;
    }
    
    backend.Synchronize();
    timer.Stop();
    double elapsedMs = timer.GetMilliseconds();
    
    logger.Info("✓ Shader executed in " + std::to_string(elapsedMs) + " ms");
    
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
    backend.UnbindUAVs();
    backend.FreeMemory(d_a);
    backend.FreeMemory(d_b);
    backend.FreeMemory(d_c);
    backend.Shutdown();
    logger.Info("✓ Cleanup complete");
    
    // Final summary
    logger.Info("\n========================================");
    if (correct) {
        logger.Info("ALL TESTS PASSED! ✓");
        logger.Info("DirectCompute backend is working correctly");
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
 * [INFO] DirectCompute Backend Test
 * [INFO] ========================================
 * 
 * [INFO] [TEST 1] Initializing DirectCompute Backend...
 * [INFO] [DirectCompute] Backend created
 * [INFO] [DirectCompute] Initializing DirectCompute backend...
 * [INFO] [DirectCompute] Selected adapter: NVIDIA GeForce RTX 3050 Laptop GPU
 * [INFO] [DirectCompute] Video Memory: 4096 MB
 * [INFO] [DirectCompute] Created D3D11 device (Feature Level 11.1)
 * [INFO] [DirectCompute] Initialization complete
 * [INFO] [DirectCompute] Device: NVIDIA GeForce RTX 3050 Laptop GPU
 * [INFO] [DirectCompute] Memory: 4096 MB
 * [INFO] ✓ DirectCompute backend initialized
 * 
 * [INFO] [TEST 2] Compiling vector add shader...
 * [INFO] [DirectCompute] Compiling shader: VectorAdd
 * [INFO] [DirectCompute] Shader 'VectorAdd' compiled successfully
 * [INFO] ✓ Shader compiled successfully
 * 
 * [INFO] [TEST 3] Testing memory allocation...
 * [INFO] ✓ Allocated 11 MB on GPU
 * 
 * [INFO] [TEST 4] Testing data transfer...
 * [INFO] ✓ Data copied to device
 * 
 * [INFO] [TEST 5] Executing vector add shader...
 * [INFO] ✓ Shader executed in 0.623 ms
 * [INFO] Effective Bandwidth: 17.2 GB/s
 * 
 * [INFO] [TEST 6] Verifying results...
 * [INFO] ✓ All results correct!
 * 
 * [INFO] [TEST 7] Cleaning up...
 * [INFO] ✓ Cleanup complete
 * 
 * [INFO] ========================================
 * [INFO] ALL TESTS PASSED! ✓
 * [INFO] DirectCompute backend is working correctly
 * [INFO] Performance: 17.2 GB/s
 * [INFO] ========================================
 ********************************************************************************/
