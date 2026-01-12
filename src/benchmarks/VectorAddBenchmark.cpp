/*******************************************************************************
 * FILE: VectorAddBenchmark.cpp
 * 
 * PURPOSE:
 *   Implementation of vector addition benchmark.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#include "VectorAddBenchmark.h"
#include <cmath>
#include <algorithm>

// CUDA kernel launcher (defined in vector_add.cu)
extern "C" void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n);

namespace GPUBenchmark {

VectorAddBenchmark::VectorAddBenchmark(size_t numElements)
    : m_numElements(numElements)
    , m_iterations(10)
    , m_warmupIterations(3)
    , m_verifyResults(true)
    , m_deviceA(nullptr)
    , m_deviceB(nullptr)
    , m_deviceC(nullptr)
{
    InitializeData();
}

VectorAddBenchmark::~VectorAddBenchmark() {
    // Device memory cleanup handled by backend
}

void VectorAddBenchmark::SetProblemSize(size_t numElements) {
    m_numElements = numElements;
    InitializeData();
}

void VectorAddBenchmark::InitializeData() {
    m_hostA.resize(m_numElements);
    m_hostB.resize(m_numElements);
    m_hostC.resize(m_numElements);
    
    // Initialize with simple values for easy verification
    for (size_t i = 0; i < m_numElements; i++) {
        m_hostA[i] = static_cast<float>(i);
        m_hostB[i] = static_cast<float>(i * 2);
    }
}

BenchmarkResult VectorAddBenchmark::Run(IComputeBackend* backend) {
    BenchmarkResult result;
    result.benchmarkName = GetName();
    result.backendName = backend->GetBackendName();
    result.problemSize = m_numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    Logger& logger = Logger::GetInstance();
    
    size_t bytes = m_numElements * sizeof(float);
    
    // Allocate device memory
    logger.Debug("Allocating GPU memory...");
    m_deviceA = backend->AllocateMemory(bytes);
    m_deviceB = backend->AllocateMemory(bytes);
    m_deviceC = backend->AllocateMemory(bytes);
    
    if (!m_deviceA || !m_deviceB || !m_deviceC) {
        result.resultCorrect = false;
        logger.Error("Failed to allocate device memory");
        return result;
    }
    
    // Copy data to device
    logger.Debug("Copying data to device...");
    backend->CopyHostToDevice(m_deviceA, m_hostA.data(), bytes);
    backend->CopyHostToDevice(m_deviceB, m_hostB.data(), bytes);
    
    // Warmup
    for (int i = 0; i < m_warmupIterations; i++) {
        launchVectorAdd((const float*)m_deviceA, (const float*)m_deviceB, (float*)m_deviceC, m_numElements);
        backend->Synchronize();
    }
    
    // Benchmark execution
    backend->StartTimer();
    for (int i = 0; i < m_iterations; i++) {
        launchVectorAdd((const float*)m_deviceA, (const float*)m_deviceB, (float*)m_deviceC, m_numElements);
        backend->Synchronize();
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / m_iterations;
    
    // Copy results back
    backend->CopyDeviceToHost(m_hostC.data(), m_deviceC, bytes);
    
    // Calculate bandwidth
    double totalBytes = 3.0 * bytes;  // Read A, B, write C
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Verify results
    if (m_verifyResults) {
        result.resultCorrect = VerifyResults(m_hostC.data(), m_numElements);
    } else {
        result.resultCorrect = true;
    }
    
    // Cleanup
    backend->FreeMemory(m_deviceA);
    backend->FreeMemory(m_deviceB);
    backend->FreeMemory(m_deviceC);
    
    m_deviceA = m_deviceB = m_deviceC = nullptr;
    
    return result;
}

bool VectorAddBenchmark::VerifyResults(const float* C, size_t size) {
    int errors = 0;
    const float epsilon = 1e-5f;
    
    for (size_t i = 0; i < size && errors < 10; i++) {
        float expected = m_hostA[i] + m_hostB[i];
        if (std::abs(C[i] - expected) > epsilon) {
            errors++;
        }
    }
    
    return (errors == 0);
}

} // namespace GPUBenchmark
