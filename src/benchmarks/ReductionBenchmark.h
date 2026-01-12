/*******************************************************************************
 * FILE: ReductionBenchmark.h
 * 
 * PURPOSE:
 *   Benchmark wrapper for parallel reduction kernel.
 *   Integrates CUDA kernel with benchmark framework.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#ifndef GPU_BENCHMARK_REDUCTION_BENCHMARK_H
#define GPU_BENCHMARK_REDUCTION_BENCHMARK_H

#include "../core/IComputeBackend.h"
#include "../core/Logger.h"
#include <vector>
#include <string>

namespace GPUBenchmark {

class ReductionBenchmark {
public:
    ReductionBenchmark(size_t numElements = 10000000);
    ~ReductionBenchmark();
    
    // Run benchmark on specified backend
    BenchmarkResult Run(IComputeBackend* backend);
    
    // Configure benchmark
    void SetProblemSize(size_t numElements);
    void SetIterations(int iterations) { m_iterations = iterations; }
    void SetWarmupIterations(int warmup) { m_warmupIterations = warmup; }
    void SetVerifyResults(bool verify) { m_verifyResults = verify; }
    
    // Get benchmark info
    std::string GetName() const { return "ParallelReduction"; }
    std::string GetDescription() const { return "Sum all elements using parallel reduction"; }
    
private:
    // Initialize data
    void InitializeData();
    
    // Verify results
    bool VerifyResults(float gpuResult);
    
    // Member variables
    size_t m_numElements;
    int m_iterations;
    int m_warmupIterations;
    bool m_verifyResults;
    
    // Host data
    std::vector<float> m_hostInput;
    float m_expectedSum;
    
    // Device pointers
    void* m_deviceInput;
    void* m_deviceOutput;
    
    Logger& m_logger;
};

} // namespace GPUBenchmark

#endif // GPU_BENCHMARK_REDUCTION_BENCHMARK_H
