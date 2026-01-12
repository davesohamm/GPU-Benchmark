/*******************************************************************************
 * FILE: BenchmarkRunner.cpp
 * 
 * PURPOSE:
 *   Implementation of BenchmarkRunner orchestration class.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#include "BenchmarkRunner.h"
#include "../backends/cuda/CUDABackend.h"

#ifdef USE_OPENCL
#include "../backends/opencl/OpenCLBackend.h"
#endif

#ifdef USE_DIRECTCOMPUTE
#include "../backends/directcompute/DirectComputeBackend.h"
#endif
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <cmath>

namespace GPUBenchmark {

/*******************************************************************************
 * CONSTRUCTOR / DESTRUCTOR
 ******************************************************************************/

BenchmarkRunner::BenchmarkRunner()
    : m_defaultProblemSize(1000000)
    , m_iterations(10)
    , m_warmupIterations(3)
    , m_verifyResults(true)
    , m_verbose(true)
    , m_logger(Logger::GetInstance())
{
    m_logger.Info("BenchmarkRunner initialized");
}

BenchmarkRunner::~BenchmarkRunner() {
    m_logger.Info("BenchmarkRunner shutting down");
}

/*******************************************************************************
 * BACKEND DISCOVERY
 ******************************************************************************/

SystemCapabilities BenchmarkRunner::DiscoverBackends() {
    m_logger.Info("=== Discovering Available Backends ===");
    
    // Use DeviceDiscovery to get system capabilities
    m_systemCaps = DeviceDiscovery::Discover();
    
    // Try to initialize each available backend
    if (m_systemCaps.cuda.available) {
        if (InitializeBackend(BackendType::CUDA)) {
            m_logger.Info("✓ CUDA Backend: Available (" + m_systemCaps.cuda.version + ")");
        } else {
            m_logger.Warning("✗ CUDA Backend: Failed to initialize");
        }
    }
    
    // TODO: OpenCL and DirectCompute backends
    // if (m_systemCaps.opencl.available) { InitializeBackend(BackendType::OpenCL); }
    // if (m_systemCaps.directCompute.available) { InitializeBackend(BackendType::DirectCompute); }
    
    if (m_backends.empty()) {
        m_logger.Error("No GPU backends available! Cannot run benchmarks.");
        m_lastError = "No GPU backends available";
    } else {
        m_logger.Info("Backend discovery complete");
    }
    
    return m_systemCaps;
}

bool BenchmarkRunner::InitializeBackend(BackendType type) {
    try {
        std::unique_ptr<IComputeBackend> backend;
        
        switch (type) {
            case BackendType::CUDA: {
                backend = std::make_unique<CUDABackend>();
                if (backend->Initialize()) {
                    DeviceInfo info = backend->GetDeviceInfo();
                    m_logger.Info("  CUDA Device: " + info.name);
                    m_logger.Info("  Memory: " + std::to_string(info.totalMemoryBytes / (1024*1024)) + " MB");
                    m_logger.Info("  Compute Capability: " + std::to_string(info.computeCapabilityMajor) + 
                                 "." + std::to_string(info.computeCapabilityMinor));
                    m_backends[type] = std::move(backend);
                    return true;
                } else {
                    return false;
                }
            }
            
            case BackendType::OpenCL:
            {
#ifdef USE_OPENCL
                m_logger.Info("Initializing OpenCL backend...");
                auto backend = std::make_unique<OpenCLBackend>();
                if (backend->Initialize()) {
                    DeviceInfo info = backend->GetDeviceInfo();
                    m_logger.Info("  OpenCL Device: " + info.name);
                    m_logger.Info("  Memory: " + std::to_string(info.totalMemoryBytes / (1024*1024)) + " MB");
                    m_logger.Info("  Max Work Group Size: " + std::to_string(info.maxThreadsPerBlock));
                    m_backends[type] = std::move(backend);
                    return true;
                } else {
                    return false;
                }
#else
                m_logger.Warning("OpenCL backend not compiled (USE_OPENCL not defined)");
                return false;
#endif
            }
            
            case BackendType::DirectCompute:
            {
#ifdef USE_DIRECTCOMPUTE
                m_logger.Info("Initializing DirectCompute backend...");
                auto backend = std::make_unique<DirectComputeBackend>();
                if (backend->Initialize()) {
                    DeviceInfo info = backend->GetDeviceInfo();
                    m_logger.Info("  DirectCompute Device: " + info.name);
                    m_logger.Info("  Memory: " + std::to_string(info.totalMemoryBytes / (1024*1024)) + " MB");
                    m_backends[type] = std::move(backend);
                    return true;
                } else {
                    return false;
                }
#else
                m_logger.Warning("DirectCompute backend not compiled (USE_DIRECTCOMPUTE not defined)");
                return false;
#endif
            }
            
            default:
                return false;
        }
    }
    catch (const std::exception& e) {
        m_logger.Error("Exception during backend initialization: " + std::string(e.what()));
        return false;
    }
}

bool BenchmarkRunner::IsBackendAvailable(BackendType type) const {
    return m_backends.find(type) != m_backends.end();
}

IComputeBackend* BenchmarkRunner::GetBackend(BackendType type) {
    auto it = m_backends.find(type);
    return (it != m_backends.end()) ? it->second.get() : nullptr;
}

/*******************************************************************************
 * BENCHMARK EXECUTION
 ******************************************************************************/

void BenchmarkRunner::RunAllBenchmarks() {
    m_logger.Info("\n=== Running All Benchmarks ===");
    
    auto configs = GetStandardSuiteConfigs();
    RunBenchmarksWithConfig(configs);
}

void BenchmarkRunner::RunBenchmark(const std::string& benchmarkName) {
    m_logger.Info("\n=== Running Benchmark: " + benchmarkName + " ===");
    
    BenchmarkConfig config;
    config.name = benchmarkName;
    config.problemSize = m_defaultProblemSize;
    config.iterations = m_iterations;
    config.warmupIterations = m_warmupIterations;
    config.verifyResults = m_verifyResults;
    
    // Run on all available backends
    for (auto& pair : m_backends) {
        BackendType type = pair.first;
        m_logger.Info("\n--- Backend: " + pair.second->GetBackendName() + " ---");
        
        BenchmarkResult result = RunBenchmark(benchmarkName, type, config);
        m_results.push_back(result);
    }
}

BenchmarkResult BenchmarkRunner::RunBenchmark(const std::string& benchmarkName,
                                               BackendType backend,
                                               const BenchmarkConfig& config) {
    BenchmarkResult finalResult;
    finalResult.benchmarkName = benchmarkName;
    finalResult.backendName = GetBackend(backend)->GetBackendName();
    finalResult.problemSize = config.problemSize;
    finalResult.timestamp = Logger::GetCurrentTimestamp();
    
    IComputeBackend* backendPtr = GetBackend(backend);
    if (!backendPtr) {
        m_logger.Error("Backend not available: " + std::to_string(static_cast<int>(backend)));
        finalResult.resultCorrect = false;
        return finalResult;
    }
    
    // Get device info
    DeviceInfo info = backendPtr->GetDeviceInfo();
    finalResult.gpuName = info.name;
    finalResult.gpuDriver = info.driverVersion;
    
    // Warmup
    if (config.warmupIterations > 0) {
        if (m_verbose) {
            m_logger.Info("Performing " + std::to_string(config.warmupIterations) + " warmup iterations...");
        }
        WarmupBackend(backendPtr, config.warmupIterations);
    }
    
    // Run benchmark multiple iterations
    std::vector<double> executionTimes;
    std::vector<double> transferTimes;
    
    for (int i = 0; i < config.iterations; i++) {
        BenchmarkResult result = RunBenchmarkIteration(benchmarkName, backendPtr, config);
        
        executionTimes.push_back(result.executionTimeMS);
        transferTimes.push_back(result.hostToDeviceTimeMS + result.deviceToHostTimeMS);
        
        if (m_verbose && ((i + 1) % 5 == 0 || i == 0)) {
            m_logger.Info("  Iteration " + std::to_string(i + 1) + "/" + 
                         std::to_string(config.iterations) + ": " + 
                         std::to_string(result.executionTimeMS) + " ms");
        }
    }
    
    // Calculate statistics
    double meanExecution = std::accumulate(executionTimes.begin(), executionTimes.end(), 0.0) / executionTimes.size();
    double meanTransfer = std::accumulate(transferTimes.begin(), transferTimes.end(), 0.0) / transferTimes.size();
    
    // Calculate standard deviation
    double sqSum = 0.0;
    for (double time : executionTimes) {
        sqSum += (time - meanExecution) * (time - meanExecution);
    }
    double stdDev = std::sqrt(sqSum / executionTimes.size());
    
    // Find min/max
    double minTime = *std::min_element(executionTimes.begin(), executionTimes.end());
    double maxTime = *std::max_element(executionTimes.begin(), executionTimes.end());
    
    // Fill in final result
    finalResult.executionTimeMS = meanExecution;
    finalResult.hostToDeviceTimeMS = meanTransfer / 2.0;  // Approximate
    finalResult.deviceToHostTimeMS = meanTransfer / 2.0;
    finalResult.totalTimeMS = meanExecution + meanTransfer;
    
    // Log results
    m_logger.Info("\nResults:");
    m_logger.Info("  Mean execution time: " + std::to_string(meanExecution) + " ms");
    m_logger.Info("  Min: " + std::to_string(minTime) + " ms");
    m_logger.Info("  Max: " + std::to_string(maxTime) + " ms");
    m_logger.Info("  Std Dev: " + std::to_string(stdDev) + " ms");
    
    if (finalResult.effectiveBandwidthGBs > 0.0) {
        m_logger.Info("  Bandwidth: " + std::to_string(finalResult.effectiveBandwidthGBs) + " GB/s");
    }
    if (finalResult.computeThroughputGFLOPS > 0.0) {
        m_logger.Info("  Throughput: " + std::to_string(finalResult.computeThroughputGFLOPS) + " GFLOPS");
    }
    
    finalResult.resultCorrect = true;  // Assume correct for now
    
    return finalResult;
}

BenchmarkResult BenchmarkRunner::RunBenchmarkIteration(const std::string& benchmarkName,
                                                        IComputeBackend* backend,
                                                        const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.benchmarkName = benchmarkName;
    result.problemSize = config.problemSize;
    
    // This is a placeholder - actual benchmarks will override this
    // For now, just run a simple kernel execution
    backend->StartTimer();
    backend->Synchronize();
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime();
    result.resultCorrect = true;
    
    return result;
}

void BenchmarkRunner::WarmupBackend(IComputeBackend* backend, int iterations) {
    for (int i = 0; i < iterations; i++) {
        backend->Synchronize();
    }
}

void BenchmarkRunner::RunBenchmarksWithConfig(const std::vector<BenchmarkConfig>& configs) {
    for (const auto& config : configs) {
        RunBenchmark(config.name);
    }
}

/*******************************************************************************
 * PREDEFINED SUITES
 ******************************************************************************/

void BenchmarkRunner::RunQuickTest() {
    m_logger.Info("\n=== Running Quick Test Suite ===");
    auto configs = GetQuickTestConfigs();
    RunBenchmarksWithConfig(configs);
}

void BenchmarkRunner::RunStandardSuite() {
    m_logger.Info("\n=== Running Standard Benchmark Suite ===");
    auto configs = GetStandardSuiteConfigs();
    RunBenchmarksWithConfig(configs);
}

void BenchmarkRunner::RunPerformanceSuite() {
    m_logger.Info("\n=== Running Performance Suite ===");
    auto configs = GetPerformanceSuiteConfigs();
    RunBenchmarksWithConfig(configs);
}

void BenchmarkRunner::RunScalingAnalysis() {
    m_logger.Info("\n=== Running Scaling Analysis ===");
    
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000, 10000000};
    
    for (size_t size : sizes) {
        BenchmarkConfig config;
        config.name = "VectorAdd";
        config.problemSize = size;
        config.iterations = 10;
        config.warmupIterations = 2;
        config.verifyResults = false;
        
        m_logger.Info("\nProblem size: " + std::to_string(size));
        
        for (auto& pair : m_backends) {
            BenchmarkResult result = RunBenchmark(config.name, pair.first, config);
            m_results.push_back(result);
        }
    }
}

/*******************************************************************************
 * RESULT MANAGEMENT
 ******************************************************************************/

std::vector<BenchmarkResult> BenchmarkRunner::GetResults(const std::string& benchmarkName) const {
    std::vector<BenchmarkResult> filtered;
    for (const auto& result : m_results) {
        if (result.benchmarkName == benchmarkName) {
            filtered.push_back(result);
        }
    }
    return filtered;
}

std::vector<BenchmarkResult> BenchmarkRunner::GetResults(BackendType backend) const {
    IComputeBackend* backendPtr = const_cast<BenchmarkRunner*>(this)->GetBackend(backend);
    if (!backendPtr) return {};
    
    std::string backendName = backendPtr->GetBackendName();
    std::vector<BenchmarkResult> filtered;
    
    for (const auto& result : m_results) {
        if (result.backendName == backendName) {
            filtered.push_back(result);
        }
    }
    return filtered;
}

bool BenchmarkRunner::ExportResults(const std::string& filepath) {
    m_logger.Info("Exporting results to: " + filepath);
    
    if (m_logger.InitializeCSV(filepath)) {
        m_logger.LogResults(m_results);
        m_logger.Info("Export complete!");
        return true;
    }
    
    m_logger.Error("Failed to export results");
    return false;
}

void BenchmarkRunner::PrintResultsSummary() {
    m_logger.Info("\n=== Benchmark Results Summary ===\n");
    
    if (m_results.empty()) {
        m_logger.Info("No results to display");
        return;
    }
    
    // Group by benchmark
    std::map<std::string, std::vector<BenchmarkResult>> grouped;
    for (const auto& result : m_results) {
        grouped[result.benchmarkName].push_back(result);
    }
    
    // Print each benchmark
    for (const auto& pair : grouped) {
        m_logger.Info("--- " + pair.first + " ---");
        for (const auto& result : pair.second) {
            m_logger.LogResult(result);
        }
        m_logger.Info("");
    }
}

void BenchmarkRunner::PrintBackendComparison(const std::string& benchmarkName) {
    auto results = GetResults(benchmarkName);
    
    if (results.empty()) {
        m_logger.Info("No results for benchmark: " + benchmarkName);
        return;
    }
    
    m_logger.Info("\n=== Backend Comparison: " + benchmarkName + " ===\n");
    
    // Find fastest
    auto fastest = std::min_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.executionTimeMS < b.executionTimeMS;
        });
    
    double fastestTime = fastest->executionTimeMS;
    
    // Print comparison
    for (const auto& result : results) {
        double speedup = fastestTime / result.executionTimeMS;
        m_logger.Info(result.backendName + ": " + 
                     std::to_string(result.executionTimeMS) + " ms " +
                     "(Speedup: " + std::to_string(speedup) + "x)");
    }
}

} // namespace GPUBenchmark

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
