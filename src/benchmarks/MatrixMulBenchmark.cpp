/*******************************************************************************
 * FILE: MatrixMulBenchmark.cpp
 * 
 * PURPOSE:
 *   Implementation of matrix multiplication benchmark wrapper.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#include "MatrixMulBenchmark.h"
#include <cmath>
#include <algorithm>

// CUDA kernel launcher (defined in matrix_mul.cu)
extern "C" void launchMatrixMulOptimized(const float* d_A, const float* d_B, float* d_C,
                                          int M, int N, int P, void* stream);

namespace GPUBenchmark {

MatrixMulBenchmark::MatrixMulBenchmark(size_t matrixSize)
    : m_matrixSize(matrixSize)
    , m_iterations(100)
    , m_warmupIterations(3)
    , m_verifyResults(true)
    , m_deviceA(nullptr)
    , m_deviceB(nullptr)
    , m_deviceC(nullptr)
    , m_logger(Logger::GetInstance())
{
    InitializeData();
}

MatrixMulBenchmark::~MatrixMulBenchmark() {
    // Device memory is managed by backend, cleaned up in Run()
}

void MatrixMulBenchmark::SetMatrixSize(size_t size) {
    m_matrixSize = size;
    InitializeData();
}

void MatrixMulBenchmark::InitializeData() {
    size_t totalElements = m_matrixSize * m_matrixSize;
    
    m_hostA.resize(totalElements);
    m_hostB.resize(totalElements);
    m_hostC.resize(totalElements);
    
    // Initialize matrices with test data
    for (size_t i = 0; i < totalElements; i++) {
        m_hostA[i] = static_cast<float>(rand()) / RAND_MAX;
        m_hostB[i] = static_cast<float>(rand()) / RAND_MAX;
        m_hostC[i] = 0.0f;
    }
}

BenchmarkResult MatrixMulBenchmark::Run(IComputeBackend* backend) {
    BenchmarkResult result;
    result.benchmarkName = GetName();
    result.backendName = backend->GetBackendName();
    result.gpuName = backend->GetDeviceInfo().name;
    result.problemSize = m_matrixSize * m_matrixSize;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    m_logger.Debug("=== Matrix Multiplication Benchmark ===");
    m_logger.Debug("Matrix size: " + std::to_string(m_matrixSize) + "×" + std::to_string(m_matrixSize));
    m_logger.Debug("Iterations: " + std::to_string(m_iterations));
    
    size_t bytes = m_matrixSize * m_matrixSize * sizeof(float);
    
    // Allocate device memory
    m_logger.Debug("Allocating GPU memory...");
    m_deviceA = backend->AllocateMemory(bytes);
    m_deviceB = backend->AllocateMemory(bytes);
    m_deviceC = backend->AllocateMemory(bytes);
    
    if (!m_deviceA || !m_deviceB || !m_deviceC) {
        result.resultCorrect = false;
        m_logger.Error("Failed to allocate device memory");
        return result;
    }
    
    // Copy data to device
    m_logger.Debug("Copying data to device...");
    backend->CopyHostToDevice(m_deviceA, m_hostA.data(), bytes);
    backend->CopyHostToDevice(m_deviceB, m_hostB.data(), bytes);
    
    // Warmup
    for (int i = 0; i < m_warmupIterations; i++) {
        launchMatrixMulOptimized((const float*)m_deviceA, (const float*)m_deviceB, (float*)m_deviceC,
                                  m_matrixSize, m_matrixSize, m_matrixSize, nullptr);
        backend->Synchronize();
    }
    
    // Benchmark execution
    backend->StartTimer();
    for (int i = 0; i < m_iterations; i++) {
        launchMatrixMulOptimized((const float*)m_deviceA, (const float*)m_deviceB, (float*)m_deviceC,
                                  m_matrixSize, m_matrixSize, m_matrixSize, nullptr);
        backend->Synchronize();
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / m_iterations;
    
    // Copy results back
    backend->CopyDeviceToHost(m_hostC.data(), m_deviceC, bytes);
    
    // Calculate GFLOPS
    // Matrix multiplication: 2*N^3 operations (N^2 multiplies + N^2*(N-1) adds)
    double operations = 2.0 * m_matrixSize * m_matrixSize * m_matrixSize;
    result.effectiveBandwidthGBs = (operations / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Verify results
    if (m_verifyResults) {
        result.resultCorrect = VerifyResults(m_hostC.data(), m_matrixSize * m_matrixSize);
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

bool MatrixMulBenchmark::VerifyResults(const float* C, size_t size) {
    // Compute reference result on CPU for first row only (for speed)
    const size_t N = m_matrixSize;
    
    for (size_t i = 0; i < std::min(N, size_t(10)); i++) {
        for (size_t j = 0; j < std::min(N, size_t(10)); j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < N; k++) {
                sum += m_hostA[i * N + k] * m_hostB[k * N + j];
            }
            
            float diff = std::abs(C[i * N + j] - sum);
            float epsilon = 0.01f * std::max(std::abs(sum), 1.0f);
            
            if (diff > epsilon) {
                m_logger.Debug("Verification failed at (" + std::to_string(i) + "," + std::to_string(j) + 
                              "): expected " + std::to_string(sum) + ", got " + std::to_string(C[i * N + j]));
                return false;
            }
        }
    }
    
    return true;
}

} // namespace GPUBenchmark
