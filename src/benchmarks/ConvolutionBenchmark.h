/*******************************************************************************
 * FILE: ConvolutionBenchmark.h
 * 
 * PURPOSE:
 *   Benchmark wrapper for 2D image convolution kernel.
 *   Integrates CUDA kernel with benchmark framework.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#ifndef GPU_BENCHMARK_CONVOLUTION_BENCHMARK_H
#define GPU_BENCHMARK_CONVOLUTION_BENCHMARK_H

#include "../core/IComputeBackend.h"
#include "../core/Logger.h"
#include <vector>
#include <string>

namespace GPUBenchmark {

class ConvolutionBenchmark {
public:
    ConvolutionBenchmark(size_t width = 1920, size_t height = 1080);
    ~ConvolutionBenchmark();
    
    // Run benchmark on specified backend
    BenchmarkResult Run(IComputeBackend* backend);
    
    // Configure benchmark
    void SetImageSize(size_t width, size_t height);
    void SetIterations(int iterations) { m_iterations = iterations; }
    void SetWarmupIterations(int warmup) { m_warmupIterations = warmup; }
    void SetVerifyResults(bool verify) { m_verifyResults = verify; }
    
    // Get benchmark info
    std::string GetName() const { return "Convolution2D"; }
    std::string GetDescription() const { return "2D image convolution with 3×3 kernel"; }
    
private:
    // Initialize data
    void InitializeData();
    
    // Verify results
    bool VerifyResults(const float* output);
    
    // Member variables
    size_t m_width;
    size_t m_height;
    int m_iterations;
    int m_warmupIterations;
    bool m_verifyResults;
    
    // Host data
    std::vector<float> m_hostInput;
    std::vector<float> m_hostOutput;
    float m_kernel[9]; // 3x3 kernel
    
    // Device pointers
    void* m_deviceInput;
    void* m_deviceOutput;
    
    Logger& m_logger;
};

} // namespace GPUBenchmark

#endif // GPU_BENCHMARK_CONVOLUTION_BENCHMARK_H
