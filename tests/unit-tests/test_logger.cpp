// Test program for Logger implementation
#include "src/core/Logger.h"
#include <iostream>

using namespace GPUBenchmark;

int main() {
    std::cout << "Testing Logger Implementation...\n" << std::endl;
    
    Logger& logger = Logger::GetInstance();
    
    // Test log levels
    logger.Debug("This is a DEBUG message");
    logger.Info("This is an INFO message");
    logger.Warning("This is a WARNING message");
    logger.Error("This is an ERROR message");
    
    // Test CSV
    logger.InitializeCSV("test_results.csv");
    
    // Test benchmark result
    BenchmarkResult result;
    result.timestamp = Logger::GetCurrentTimestamp();
    result.backendName = "CUDA";
    result.benchmarkName = "VectorAdd";
    result.problemSize = 1000000;
    result.executionTimeMS = 0.234;
    result.effectiveBandwidthGBs = 51.3;
    result.resultCorrect = true;
    result.gpuName = "RTX 3050";
    
    logger.LogResult(result);
    
    std::cout << "\n✓ Logger test complete!" << std::endl;
    std::cout << "Check test_results.csv for CSV output" << std::endl;
    
    return 0;
}
