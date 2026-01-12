/*******************************************************************************
 * FILE: VectorAddBenchmark.h
 * 
 * PURPOSE:
 *   Benchmark wrapper for vector addition kernel.
 *   Integrates CUDA kernel with benchmark framework.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#ifndef GPU_BENCHMARK_VECTOR_ADD_BENCHMARK_H
#define GPU_BENCHMARK_VECTOR_ADD_BENCHMARK_H

#include "../core/IComputeBackend.h"
#include "../core/Logger.h"
#include <vector>
#include <string>

namespace GPUBenchmark {

class VectorAddBenchmark {
public:
    VectorAddBenchmark(size_t numElements = 10000000);
    ~VectorAddBenchmark();
    
    // Run benchmark on specified backend
    BenchmarkResult Run(IComputeBackend* backend);
    
    // Configure benchmark
    void SetProblemSize(size_t numElements);
    void SetIterations(int iterations) { m_iterations = iterations; }
    void SetWarmupIterations(int warmup) { m_warmupIterations = warmup; }
    void SetVerifyResults(bool verify) { m_verifyResults = verify; }
    
    // Get benchmark info
    std::string GetName() const { return "VectorAdd"; }
    std::string GetDescription() const { return "Element-wise vector addition: C[i] = A[i] + B[i]"; }
    
private:
    // Initialize data
    void InitializeData();
    
    // Verify results
    bool VerifyResults(const float* C, size_t size);
    
    // Member variables
    size_t m_numElements;
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
};

} // namespace GPUBenchmark

#endif // GPU_BENCHMARK_VECTOR_ADD_BENCHMARK_H
