/*******************************************************************************
 * FILE: BenchmarkRunner.h
 * 
 * PURPOSE:
 *   Orchestrates benchmark execution across multiple backends and problem sizes.
 *   Manages warmup, timing, result aggregation, and CSV export.
 * 
 * RESPONSIBILITIES:
 *   - Initialize backends (CUDA, OpenCL, DirectCompute)
 *   - Run benchmarks with proper warmup
 *   - Collect and aggregate timing data
 *   - Verify correctness of results
 *   - Export results to CSV
 *   - Handle errors gracefully
 * 
 * USAGE:
 *   BenchmarkRunner runner;
 *   runner.DiscoverBackends();
 *   runner.RunAllBenchmarks();
 *   runner.ExportResults("results.csv");
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#ifndef GPU_BENCHMARK_BENCHMARK_RUNNER_H
#define GPU_BENCHMARK_BENCHMARK_RUNNER_H

#include "IComputeBackend.h"
#include "DeviceDiscovery.h"  // For SystemCapabilities
#include "Logger.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace GPUBenchmark {

/*******************************************************************************
 * BENCHMARK CONFIGURATION
 * 
 * PURPOSE:
 *   Defines parameters for a single benchmark run.
 ******************************************************************************/
struct BenchmarkConfig {
    std::string name;              // "VectorAdd", "MatrixMul", etc.
    size_t problemSize;            // Number of elements/dimension
    int iterations;                // Number of runs for averaging
    int warmupIterations;          // Warmup runs before timing
    bool verifyResults;            // Check correctness
    
    // Default configuration
    BenchmarkConfig()
        : name("Unknown")
        , problemSize(1000000)
        , iterations(10)
        , warmupIterations(3)
        , verifyResults(true)
    {}
};

/*******************************************************************************
 * BENCHMARK RUNNER CLASS
 * 
 * PURPOSE:
 *   Main orchestrator for benchmark execution.
 * 
 * DESIGN PATTERN:
 *   - Facade Pattern: Simplifies complex benchmark execution
 *   - Strategy Pattern: Different backends implement same interface
 *   - Template Method: Benchmark workflow is standardized
 ******************************************************************************/
class BenchmarkRunner {
public:
    BenchmarkRunner();
    ~BenchmarkRunner();
    
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    
    // Discover and initialize all available backends
    SystemCapabilities DiscoverBackends();
    
    // Initialize specific backend
    bool InitializeBackend(BackendType type);
    
    // Check if backend is available
    bool IsBackendAvailable(BackendType type) const;
    
    // Get backend instance (returns nullptr if not initialized)
    IComputeBackend* GetBackend(BackendType type);
    
    // =========================================================================
    // BENCHMARK EXECUTION
    // =========================================================================
    
    // Run all benchmarks on all available backends
    void RunAllBenchmarks();
    
    // Run specific benchmark on all backends
    void RunBenchmark(const std::string& benchmarkName);
    
    // Run specific benchmark on specific backend
    BenchmarkResult RunBenchmark(const std::string& benchmarkName, 
                                  BackendType backend,
                                  const BenchmarkConfig& config);
    
    // Run benchmarks with custom configurations
    void RunBenchmarksWithConfig(const std::vector<BenchmarkConfig>& configs);
    
    // =========================================================================
    // PREDEFINED BENCHMARK SUITES
    // =========================================================================
    
    // Quick test: Small problem sizes, few iterations
    void RunQuickTest();
    
    // Standard suite: Medium problem sizes, standard iterations
    void RunStandardSuite();
    
    // Performance suite: Large problem sizes, many iterations
    void RunPerformanceSuite();
    
    // Scaling analysis: Multiple problem sizes
    void RunScalingAnalysis();
    
    // =========================================================================
    // RESULT MANAGEMENT
    // =========================================================================
    
    // Get all results
    const std::vector<BenchmarkResult>& GetResults() const { return m_results; }
    
    // Get results for specific benchmark
    std::vector<BenchmarkResult> GetResults(const std::string& benchmarkName) const;
    
    // Get results for specific backend
    std::vector<BenchmarkResult> GetResults(BackendType backend) const;
    
    // Export results to CSV
    bool ExportResults(const std::string& filepath);
    
    // Print results summary to console
    void PrintResultsSummary();
    
    // Compare backends for specific benchmark
    void PrintBackendComparison(const std::string& benchmarkName);
    
    // =========================================================================
    // CONFIGURATION
    // =========================================================================
    
    // Set default problem sizes
    void SetDefaultProblemSize(size_t size) { m_defaultProblemSize = size; }
    
    // Set number of iterations
    void SetIterations(int iterations) { m_iterations = iterations; }
    
    // Set warmup iterations
    void SetWarmupIterations(int warmup) { m_warmupIterations = warmup; }
    
    // Enable/disable result verification
    void SetVerifyResults(bool verify) { m_verifyResults = verify; }
    
    // Set verbose output
    void SetVerbose(bool verbose) { m_verbose = verbose; }
    
    // =========================================================================
    // ERROR HANDLING
    // =========================================================================
    
    std::string GetLastError() const { return m_lastError; }
    
private:
    // =========================================================================
    // INTERNAL HELPERS
    // =========================================================================
    
    // Run single benchmark iteration
    BenchmarkResult RunBenchmarkIteration(const std::string& benchmarkName,
                                           IComputeBackend* backend,
                                           const BenchmarkConfig& config);
    
    // Perform warmup runs
    void WarmupBackend(IComputeBackend* backend, int iterations);
    
    // Verify benchmark results
    bool VerifyResults(const std::string& benchmarkName,
                       const void* results,
                       const void* expected,
                       size_t size);
    
    // Calculate statistics from multiple runs
    void CalculateStatistics(std::vector<BenchmarkResult>& results);
    
    // =========================================================================
    // MEMBER VARIABLES
    // =========================================================================
    
    // Backend instances
    std::map<BackendType, std::unique_ptr<IComputeBackend>> m_backends;
    
    // System capabilities (includes backend availability)
    SystemCapabilities m_systemCaps;
    
    // Results storage
    std::vector<BenchmarkResult> m_results;
    
    // Configuration
    size_t m_defaultProblemSize;
    int m_iterations;
    int m_warmupIterations;
    bool m_verifyResults;
    bool m_verbose;
    
    // Error tracking
    std::string m_lastError;
    
    // Logger instance
    Logger& m_logger;
};

/*******************************************************************************
 * BENCHMARK SUITE CONFIGURATIONS
 * 
 * PURPOSE:
 *   Predefined configurations for common benchmark scenarios.
 ******************************************************************************/

// Quick test: Fast validation
inline std::vector<BenchmarkConfig> GetQuickTestConfigs() {
    std::vector<BenchmarkConfig> configs;
    
    BenchmarkConfig config;
    config.iterations = 3;
    config.warmupIterations = 1;
    config.verifyResults = true;
    
    // Vector Add: 100K elements
    config.name = "VectorAdd";
    config.problemSize = 100000;
    configs.push_back(config);
    
    // Matrix Mul: 256x256
    config.name = "MatrixMul";
    config.problemSize = 256;
    configs.push_back(config);
    
    return configs;
}

// Standard suite: Typical workloads
inline std::vector<BenchmarkConfig> GetStandardSuiteConfigs() {
    std::vector<BenchmarkConfig> configs;
    
    BenchmarkConfig config;
    config.iterations = 10;
    config.warmupIterations = 3;
    config.verifyResults = true;
    
    // Vector Add: 10M elements
    config.name = "VectorAdd";
    config.problemSize = 10000000;
    configs.push_back(config);
    
    // Matrix Mul: 1024x1024
    config.name = "MatrixMul";
    config.problemSize = 1024;
    configs.push_back(config);
    
    // Convolution: 1920x1080 (Full HD)
    config.name = "Convolution";
    config.problemSize = 1920;  // width (height implied)
    configs.push_back(config);
    
    // Reduction: 16M elements
    config.name = "Reduction";
    config.problemSize = 16000000;
    configs.push_back(config);
    
    return configs;
}

// Performance suite: Large workloads
inline std::vector<BenchmarkConfig> GetPerformanceSuiteConfigs() {
    std::vector<BenchmarkConfig> configs;
    
    BenchmarkConfig config;
    config.iterations = 20;
    config.warmupIterations = 5;
    config.verifyResults = false;  // Skip verification for speed
    
    config.name = "VectorAdd";
    config.problemSize = 100000000;  // 100M
    configs.push_back(config);
    
    config.name = "MatrixMul";
    config.problemSize = 2048;
    configs.push_back(config);
    
    config.name = "Convolution";
    config.problemSize = 3840;  // 4K
    configs.push_back(config);
    
    config.name = "Reduction";
    config.problemSize = 64000000;  // 64M
    configs.push_back(config);
    
    return configs;
}

} // namespace GPUBenchmark

#endif // GPU_BENCHMARK_BENCHMARK_RUNNER_H

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
