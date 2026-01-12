/*******************************************************************************
 * FILE: ReductionBenchmark.cpp
 * 
 * PURPOSE:
 *   Implementation of parallel reduction benchmark wrapper.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#include "ReductionBenchmark.h"
#include <cmath>
#include <numeric>

// CUDA kernel launcher (defined in reduction.cu)
extern "C" void launchReductionWarpShuffle(const float* d_input, float* d_output,
                                            int n, void* stream);

namespace GPUBenchmark {

ReductionBenchmark::ReductionBenchmark(size_t numElements)
    : m_numElements(numElements)
    , m_iterations(100)
    , m_warmupIterations(3)
    , m_verifyResults(true)
    , m_deviceInput(nullptr)
    , m_deviceOutput(nullptr)
    , m_logger(Logger::GetInstance())
{
    InitializeData();
}

ReductionBenchmark::~ReductionBenchmark() {
    // Device memory is managed by backend, cleaned up in Run()
}

void ReductionBenchmark::SetProblemSize(size_t numElements) {
    m_numElements = numElements;
    InitializeData();
}

void ReductionBenchmark::InitializeData() {
    m_hostInput.resize(m_numElements);
    
    // Initialize with simple values for easier verification
    for (size_t i = 0; i < m_numElements; i++) {
        m_hostInput[i] = 1.0f; // All ones for simple sum
    }
    
    // Calculate expected sum
    m_expectedSum = std::accumulate(m_hostInput.begin(), m_hostInput.end(), 0.0f);
}

BenchmarkResult ReductionBenchmark::Run(IComputeBackend* backend) {
    BenchmarkResult result;
    result.benchmarkName = GetName();
    result.backendName = backend->GetBackendName();
    result.gpuName = backend->GetDeviceInfo().name;
    result.problemSize = m_numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    m_logger.Debug("=== Parallel Reduction Benchmark ===");
    m_logger.Debug("Array size: " + std::to_string(m_numElements) + " elements");
    m_logger.Debug("Iterations: " + std::to_string(m_iterations));
    
    size_t inputBytes = m_numElements * sizeof(float);
    size_t outputBytes = sizeof(float); // Single result value
    
    // Allocate device memory
    m_logger.Debug("Allocating GPU memory...");
    m_deviceInput = backend->AllocateMemory(inputBytes);
    m_deviceOutput = backend->AllocateMemory(outputBytes);
    
    if (!m_deviceInput || !m_deviceOutput) {
        result.resultCorrect = false;
        m_logger.Error("Failed to allocate device memory");
        return result;
    }
    
    // Copy data to device
    m_logger.Debug("Copying data to device...");
    backend->CopyHostToDevice(m_deviceInput, m_hostInput.data(), inputBytes);
    
    // Warmup
    for (int i = 0; i < m_warmupIterations; i++) {
        launchReductionWarpShuffle((const float*)m_deviceInput, (float*)m_deviceOutput,
                                    m_numElements, nullptr);
        backend->Synchronize();
    }
    
    // Benchmark execution
    backend->StartTimer();
    for (int i = 0; i < m_iterations; i++) {
        launchReductionWarpShuffle((const float*)m_deviceInput, (float*)m_deviceOutput,
                                    m_numElements, nullptr);
        backend->Synchronize();
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / m_iterations;
    
    // Copy result back
    float gpuResult = 0.0f;
    backend->CopyDeviceToHost(&gpuResult, m_deviceOutput, outputBytes);
    
    // Calculate bandwidth (read entire array)
    double totalBytes = inputBytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Verify results
    if (m_verifyResults) {
        result.resultCorrect = VerifyResults(gpuResult);
    } else {
        result.resultCorrect = true;
    }
    
    // Cleanup
    backend->FreeMemory(m_deviceInput);
    backend->FreeMemory(m_deviceOutput);
    
    m_deviceInput = m_deviceOutput = nullptr;
    
    return result;
}

bool ReductionBenchmark::VerifyResults(float gpuResult) {
    // Allow for floating-point tolerance
    float diff = std::abs(gpuResult - m_expectedSum);
    float epsilon = 0.001f * m_expectedSum; // 0.1% tolerance
    
    if (diff > epsilon) {
        m_logger.Debug("Verification failed: expected " + std::to_string(m_expectedSum) + 
                      ", got " + std::to_string(gpuResult) + 
                      " (diff: " + std::to_string(diff) + ")");
        return false;
    }
    
    return true;
}

} // namespace GPUBenchmark
