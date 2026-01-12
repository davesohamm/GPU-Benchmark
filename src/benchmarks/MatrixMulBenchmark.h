/*******************************************************************************
 * FILE: MatrixMulBenchmark.h
 * 
 * PURPOSE:
 *   Benchmark wrapper for matrix multiplication kernel.
 *   Integrates CUDA kernel with benchmark framework.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#ifndef GPU_BENCHMARK_MATRIX_MUL_BENCHMARK_H
#define GPU_BENCHMARK_MATRIX_MUL_BENCHMARK_H

#include "../core/IComputeBackend.h"
#include "../core/Logger.h"
#include <vector>
#include <string>

namespace GPUBenchmark {

class MatrixMulBenchmark {
public:
    MatrixMulBenchmark(size_t matrixSize = 1024);
    ~MatrixMulBenchmark();
    
    // Run benchmark on specified backend
    BenchmarkResult Run(IComputeBackend* backend);
    
    // Configure benchmark
    void SetMatrixSize(size_t size);
    void SetIterations(int iterations) { m_iterations = iterations; }
    void SetWarmupIterations(int warmup) { m_warmupIterations = warmup; }
    void SetVerifyResults(bool verify) { m_verifyResults = verify; }
    
    // Get benchmark info
    std::string GetName() const { return "MatrixMultiplication"; }
    std::string GetDescription() const { return "Matrix multiplication: C = A × B"; }
    
private:
    // Initialize data
    void InitializeData();
    
    // Verify results
    bool VerifyResults(const float* C, size_t size);
    
    // Member variables
    size_t m_matrixSize;
    int m_iterations;
    int m_warmupIterations;
    bool m_verifyResults;
    
    // Host data
    std::vector<float> m_hostA;
    std::vector<float> m_hostB;
    std::vector<float> m_hostC;
    
    // Device pointers
    void* m_deviceA;
    void* m_deviceB;
    void* m_deviceC;
    
    Logger& m_logger;
};

} // namespace GPUBenchmark

#endif // GPU_BENCHMARK_MATRIX_MUL_BENCHMARK_H
