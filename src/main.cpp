/*******************************************************************************
 * FILE: main.cpp
 * 
 * PURPOSE:
 *   Main entry point for the GPU Benchmark application.
 *   
 *   This application provides a comprehensive GPU compute benchmarking suite
 *   that works across multiple GPU APIs (CUDA, OpenCL, DirectCompute) and
 *   tests various computational patterns:
 *     - Memory bandwidth (Vector Addition)
 *     - Compute throughput (Matrix Multiplication)
 *     - Mixed workloads (Convolution, Reduction)
 * 
 * DESIGN PHILOSOPHY:
 *   - **Hardware Agnostic:** Same executable adapts to NVIDIA, AMD, Intel GPUs
 *   - **User Friendly:** Clear output, no cryptic error messages
 *   - **Comprehensive:** Tests multiple aspects of GPU performance
 *   - **Verifiable:** All results validated against CPU reference
 * 
 * USER WORKFLOW:
 *   1. Application detects available GPUs and APIs
 *   2. User selects benchmark suite (Quick/Standard/Full)
 *   3. Benchmarks execute with progress display
 *   4. Results shown on screen and exported to CSV
 *   5. Summary with performance analysis
 * 
 * COMMAND-LINE USAGE:
 *   GPU-Benchmark.exe                    # Interactive mode
 *   GPU-Benchmark.exe --quick            # Quick benchmark suite
 *   GPU-Benchmark.exe --standard         # Standard benchmark suite
 *   GPU-Benchmark.exe --full             # Full benchmark suite
 *   GPU-Benchmark.exe --help             # Show usage information
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#include "core/Logger.h"
#include "core/BenchmarkRunner.h"
#include "core/DeviceDiscovery.h"
#include "benchmarks/VectorAddBenchmark.h"
#include "benchmarks/MatrixMulBenchmark.h"
#include "benchmarks/ConvolutionBenchmark.h"
#include "benchmarks/ReductionBenchmark.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

using namespace GPUBenchmark;

/*******************************************************************************
 * BENCHMARK SUITE DEFINITIONS
 ******************************************************************************/

// Benchmark suite types
enum class BenchmarkSuite {
    QUICK,      // Fast overview (~30 seconds)
    STANDARD,   // Balanced testing (~2 minutes)
    FULL,       // Comprehensive analysis (~5 minutes)
    CUSTOM      // User-selected benchmarks
};

/*******************************************************************************
 * FUNCTION PROTOTYPES
 ******************************************************************************/

void PrintBanner();
void PrintHelp();
void PrintSystemInfo(const SystemCapabilities& caps);
void PrintBackendAvailability(const SystemCapabilities& caps);
BenchmarkSuite ParseCommandLine(int argc, char* argv[]);
std::vector<BenchmarkResult> RunQuickSuite(IComputeBackend* backend);
std::vector<BenchmarkResult> RunStandardSuite(IComputeBackend* backend);
std::vector<BenchmarkResult> RunFullSuite(IComputeBackend* backend);
void PrintSummary(const std::vector<BenchmarkResult>& results);

/*******************************************************************************
 * MAIN FUNCTION
 ******************************************************************************/

int main(int argc, char* argv[]) {
    Logger& logger = Logger::GetInstance();
    
    // Print application banner
    PrintBanner();
    
    // Parse command-line arguments
    BenchmarkSuite suite = ParseCommandLine(argc, argv);
    
    if (suite == BenchmarkSuite::CUSTOM && argc > 1 && 
        (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        PrintHelp();
        return 0;
    }
    
    logger.Info("==========================================================");
    logger.Info("           GPU COMPUTE BENCHMARK SUITE");
    logger.Info("==========================================================");
    logger.Info("");
    
    // Step 1: Discover system capabilities
    logger.Info("Step 1/4: Discovering system capabilities...");
    logger.Info("");
    
    SystemCapabilities caps = DeviceDiscovery::Discover();
    
    PrintSystemInfo(caps);
    PrintBackendAvailability(caps);
    
    // Step 2: Check if any backend is available
    if (!caps.cuda.available && !caps.opencl.available && !caps.directCompute.available) {
        logger.Error("No GPU compute backends available!");
        logger.Error("Please ensure you have:");
        logger.Error("  - A GPU installed");
        logger.Error("  - Latest GPU drivers");
        logger.Error("  - CUDA Toolkit (for NVIDIA GPUs)");
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }
    
    logger.Info("");
    logger.Info("Step 2/4: Initializing GPU backend...");
    logger.Info("");
    
    // Step 3: Initialize the best available backend
    BenchmarkRunner runner;
    SystemCapabilities discovered = runner.DiscoverBackends();
    
    // Get the initialized backend (prioritize CUDA > OpenCL > DirectCompute)
    // Note: DiscoverBackends() already initialized available backends
    BackendType selectedBackend = BackendType::Unknown;
    IComputeBackend* backend = nullptr;
    
    if (caps.cuda.available) {
        backend = runner.GetBackend(BackendType::CUDA);
        if (backend) {
            logger.Info("✓ Using CUDA backend");
            selectedBackend = BackendType::CUDA;
        }
    } else if (caps.opencl.available) {
        // TODO: Implement OpenCL
        logger.Warning("OpenCL detected but not yet implemented");
    } else if (caps.directCompute.available) {
        // TODO: Implement DirectCompute
        logger.Warning("DirectCompute detected but not yet implemented");
    }
    
    if (!backend || selectedBackend == BackendType::Unknown) {
        logger.Error("Failed to initialize any GPU backend!");
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }
    logger.Info("");
    logger.Info("Step 3/4: Running benchmark suite...");
    logger.Info("");
    
    // Initialize CSV logging
    logger.InitializeCSV("benchmark_results.csv");
    
    // Run the selected benchmark suite
    logger.Info("==========================================================");
    logger.Info("                BENCHMARK EXECUTION");
    logger.Info("==========================================================");
    logger.Info("");
    
    std::vector<BenchmarkResult> results;
    
    switch (suite) {
        case BenchmarkSuite::QUICK:
            logger.Info("Running QUICK benchmark suite (estimated time: 30 seconds)");
            results = RunQuickSuite(backend);
            break;
        case BenchmarkSuite::STANDARD:
            logger.Info("Running STANDARD benchmark suite (estimated time: 2 minutes)");
            results = RunStandardSuite(backend);
            break;
        case BenchmarkSuite::FULL:
            logger.Info("Running FULL benchmark suite (estimated time: 5 minutes)");
            results = RunFullSuite(backend);
            break;
        default:
            logger.Info("Running STANDARD benchmark suite (default)");
            results = RunStandardSuite(backend);
            break;
    }
    
    logger.Info("");
    
    // Print summary
    PrintSummary(results);
    
    logger.Info("");
    logger.Info("Step 4/4: Benchmark complete!");
    logger.Info("");
    
    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    
    return 0;
}

/*******************************************************************************
 * PRINT BANNER
 * 
 * Displays application title and version.
 ******************************************************************************/
void PrintBanner() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║           GPU COMPUTE BENCHMARK SUITE v1.0                 ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Comprehensive GPU performance testing across multiple     ║\n";
    std::cout << "║  compute APIs (CUDA, OpenCL, DirectCompute)                ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Author: Soham                                             ║\n";
    std::cout << "║  Date: January 2026                                        ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

/*******************************************************************************
 * PRINT HELP
 * 
 * Shows command-line usage information.
 ******************************************************************************/
void PrintHelp() {
    std::cout << "\nUSAGE:\n";
    std::cout << "  GPU-Benchmark.exe [options]\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --quick        Run quick benchmark suite (~30 seconds)\n";
    std::cout << "                 Tests: Vector Add (1M), Matrix Mul (512x512)\n\n";
    std::cout << "  --standard     Run standard benchmark suite (~2 minutes) [DEFAULT]\n";
    std::cout << "                 Tests: All benchmarks at moderate sizes\n\n";
    std::cout << "  --full         Run full benchmark suite (~5 minutes)\n";
    std::cout << "                 Tests: All benchmarks at large sizes with scaling analysis\n\n";
    std::cout << "  --help, -h     Show this help message\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  GPU-Benchmark.exe              # Run with default settings\n";
    std::cout << "  GPU-Benchmark.exe --quick      # Quick performance check\n";
    std::cout << "  GPU-Benchmark.exe --full       # Comprehensive analysis\n\n";
    std::cout << "OUTPUT:\n";
    std::cout << "  Results are displayed on screen and exported to 'benchmark_results.csv'\n\n";
}

/*******************************************************************************
 * PRINT SYSTEM INFO
 * 
 * Displays detected hardware and system information.
 ******************************************************************************/
void PrintSystemInfo(const SystemCapabilities& caps) {
    Logger& logger = Logger::GetInstance();
    
    logger.Info("=== SYSTEM INFORMATION ===");
    logger.Info("Operating System: " + caps.operatingSystem);
    logger.Info("CPU: " + caps.cpuName);
    logger.Info("RAM: " + std::to_string(caps.systemRAMMB / 1024) + " GB");
    logger.Info("");
    
    if (caps.gpus.empty()) {
        logger.Warning("No GPUs detected!");
        return;
    }
    
    logger.Info("=== DETECTED GPUs ===");
    for (size_t i = 0; i < caps.gpus.size(); ++i) {
        const GPUInfo& gpu = caps.gpus[i];
        logger.Info("GPU " + std::to_string(i + 1) + ": " + gpu.name);
        logger.Info("  Vendor: " + gpu.vendor);
        logger.Info("  Memory: " + std::to_string(gpu.totalMemoryMB / 1024) + " GB");
        logger.Info("  Driver: " + gpu.driverVersion);
        if (gpu.isPrimaryGPU) {
            logger.Info("  [PRIMARY GPU]");
        }
    }
    logger.Info("");
}

/*******************************************************************************
 * PRINT BACKEND AVAILABILITY
 * 
 * Shows which GPU compute APIs are available.
 ******************************************************************************/
void PrintBackendAvailability(const SystemCapabilities& caps) {
    Logger& logger = Logger::GetInstance();
    
    logger.Info("=== COMPUTE API AVAILABILITY ===");
    
    // CUDA
    if (caps.cuda.available) {
        logger.Info("✓ CUDA: Available (" + caps.cuda.version + ")");
    } else {
        logger.Warning("✗ CUDA: " + caps.cuda.unavailableReason);
    }
    
    // OpenCL
    if (caps.opencl.available) {
        logger.Info("✓ OpenCL: Available (" + caps.opencl.version + ")");
    } else {
        logger.Warning("✗ OpenCL: " + caps.opencl.unavailableReason);
    }
    
    // DirectCompute
    if (caps.directCompute.available) {
        logger.Info("✓ DirectCompute: Available (" + caps.directCompute.version + ")");
    } else {
        logger.Warning("✗ DirectCompute: " + caps.directCompute.unavailableReason);
    }
    
    logger.Info("");
}

/*******************************************************************************
 * PARSE COMMAND LINE
 * 
 * Parses command-line arguments to determine benchmark suite.
 ******************************************************************************/
BenchmarkSuite ParseCommandLine(int argc, char* argv[]) {
    if (argc < 2) {
        return BenchmarkSuite::STANDARD; // Default
    }
    
    std::string arg = argv[1];
    
    if (arg == "--quick" || arg == "-q") {
        return BenchmarkSuite::QUICK;
    } else if (arg == "--standard" || arg == "-s") {
        return BenchmarkSuite::STANDARD;
    } else if (arg == "--full" || arg == "-f") {
        return BenchmarkSuite::FULL;
    } else if (arg == "--help" || arg == "-h") {
        return BenchmarkSuite::CUSTOM;
    }
    
    return BenchmarkSuite::STANDARD;
}

/*******************************************************************************
 * BENCHMARK SUITE IMPLEMENTATIONS
 ******************************************************************************/

std::vector<BenchmarkResult> RunQuickSuite(IComputeBackend* backend) {
    Logger& logger = Logger::GetInstance();
    std::vector<BenchmarkResult> results;
    
    logger.Info("Quick Suite: Small problem sizes for rapid testing");
    logger.Info("");
    
    // 1. Vector Add - 1M elements
    logger.Info("[1/2] Vector Addition (1M elements, 10 iterations)");
    try {
        VectorAddBenchmark vecBench(1000000);
        vecBench.SetIterations(10);
        BenchmarkResult result = vecBench.Run(backend);
        results.push_back(result);
        logger.LogResult(result);
    } catch (const std::exception& e) {
        logger.Error("Vector Add failed: " + std::string(e.what()));
    }
    logger.Info("");
    
    // 2. Matrix Mul - 512x512
    logger.Info("[2/2] Matrix Multiplication (512×512, 10 iterations)");
    try {
        MatrixMulBenchmark matBench(512);
        matBench.SetIterations(10);
        BenchmarkResult result = matBench.Run(backend);
        results.push_back(result);
        logger.LogResult(result);
    } catch (const std::exception& e) {
        logger.Error("Matrix Multiplication failed: " + std::string(e.what()));
    }
    logger.Info("");
    
    return results;
}

std::vector<BenchmarkResult> RunStandardSuite(IComputeBackend* backend) {
    Logger& logger = Logger::GetInstance();
    std::vector<BenchmarkResult> results;
    
    logger.Info("Standard Suite: Moderate problem sizes for comprehensive testing");
    logger.Info("");
    
    // 1. Vector Add - 10M elements
    logger.Info("[1/4] Vector Addition (10M elements, 100 iterations)");
    try {
        VectorAddBenchmark vecBench(10000000);
        vecBench.SetIterations(100);
        BenchmarkResult result = vecBench.Run(backend);
        results.push_back(result);
        logger.LogResult(result);
    } catch (const std::exception& e) {
        logger.Error("Vector Add failed: " + std::string(e.what()));
    }
    logger.Info("");
    
    // 2. Matrix Mul - 1024x1024
    logger.Info("[2/4] Matrix Multiplication (1024×1024, 100 iterations)");
    try {
        MatrixMulBenchmark matBench(1024);
        matBench.SetIterations(100);
        BenchmarkResult result = matBench.Run(backend);
        results.push_back(result);
        logger.LogResult(result);
    } catch (const std::exception& e) {
        logger.Error("Matrix Multiplication failed: " + std::string(e.what()));
    }
    logger.Info("");
    
    // 3. Convolution - 1920x1080 (Full HD)
    logger.Info("[3/4] 2D Convolution (1920×1080, 100 iterations)");
    try {
        ConvolutionBenchmark convBench(1920, 1080);
        convBench.SetIterations(100);
        BenchmarkResult result = convBench.Run(backend);
        results.push_back(result);
        logger.LogResult(result);
    } catch (const std::exception& e) {
        logger.Error("Convolution failed: " + std::string(e.what()));
    }
    logger.Info("");
    
    // 4. Reduction - 10M elements
    logger.Info("[4/4] Parallel Reduction (10M elements, 100 iterations)");
    try {
        ReductionBenchmark redBench(10000000);
        redBench.SetIterations(100);
        BenchmarkResult result = redBench.Run(backend);
        results.push_back(result);
        logger.LogResult(result);
    } catch (const std::exception& e) {
        logger.Error("Reduction failed: " + std::string(e.what()));
    }
    logger.Info("");
    
    return results;
}

std::vector<BenchmarkResult> RunFullSuite(IComputeBackend* backend) {
    Logger& logger = Logger::GetInstance();
    std::vector<BenchmarkResult> results;
    
    logger.Info("Full Suite: Large problem sizes with scaling analysis");
    logger.Info("");
    
    // 1. Vector Add - Multiple sizes
    logger.Info("=== Vector Addition Scaling ===");
    for (size_t size : {1000000, 10000000, 50000000}) {
        logger.Info("[Vector Add] " + std::to_string(size) + " elements");
        try {
            VectorAddBenchmark vecBench(size);
            vecBench.SetIterations(100);
            BenchmarkResult result = vecBench.Run(backend);
            results.push_back(result);
            logger.LogResult(result);
        } catch (const std::exception& e) {
            logger.Error("Failed: " + std::string(e.what()));
        }
    }
    logger.Info("");
    
    // 2. Matrix Mul - Multiple sizes
    logger.Info("=== Matrix Multiplication Scaling ===");
    for (size_t size : {512, 1024, 2048}) {
        logger.Info("[Matrix Mul] " + std::to_string(size) + "×" + std::to_string(size));
        try {
            MatrixMulBenchmark matBench(size);
            matBench.SetIterations(100);
            BenchmarkResult result = matBench.Run(backend);
            results.push_back(result);
            logger.LogResult(result);
        } catch (const std::exception& e) {
            logger.Error("Failed: " + std::string(e.what()));
        }
    }
    logger.Info("");
    
    // 3. Convolution - Multiple resolutions
    logger.Info("=== 2D Convolution Scaling ===");
    std::vector<std::pair<size_t, size_t>> resolutions = {{640, 480}, {1920, 1080}, {3840, 2160}};
    for (auto [width, height] : resolutions) {
        logger.Info("[Convolution] " + std::to_string(width) + "×" + std::to_string(height));
        try {
            ConvolutionBenchmark convBench(width, height);
            convBench.SetIterations(100);
            BenchmarkResult result = convBench.Run(backend);
            results.push_back(result);
            logger.LogResult(result);
        } catch (const std::exception& e) {
            logger.Error("Failed: " + std::string(e.what()));
        }
    }
    logger.Info("");
    
    // 4. Reduction - Multiple sizes
    logger.Info("=== Parallel Reduction Scaling ===");
    for (size_t size : {1000000, 10000000, 50000000}) {
        logger.Info("[Reduction] " + std::to_string(size) + " elements");
        try {
            ReductionBenchmark redBench(size);
            redBench.SetIterations(100);
            BenchmarkResult result = redBench.Run(backend);
            results.push_back(result);
            logger.LogResult(result);
        } catch (const std::exception& e) {
            logger.Error("Failed: " + std::string(e.what()));
        }
    }
    logger.Info("");
    
    return results;
}

void PrintSummary(const std::vector<BenchmarkResult>& results) {
    Logger& logger = Logger::GetInstance();
    
    logger.Info("==========================================================");
    logger.Info("                  BENCHMARK SUMMARY");
    logger.Info("==========================================================");
    logger.Info("");
    
    if (results.empty()) {
        logger.Warning("No results to summarize");
        return;
    }
    
    // Group results by benchmark type
    double totalVectorBandwidth = 0.0;
    double totalMatrixGFLOPS = 0.0;
    double totalConvolutionBandwidth = 0.0;
    double totalReductionBandwidth = 0.0;
    
    int vectorCount = 0, matrixCount = 0, convolutionCount = 0, reductionCount = 0;
    
    for (const auto& result : results) {
        if (result.benchmarkName.find("Vector") != std::string::npos) {
            totalVectorBandwidth += result.effectiveBandwidthGBs;
            vectorCount++;
        } else if (result.benchmarkName.find("Matrix") != std::string::npos) {
            totalMatrixGFLOPS += result.effectiveBandwidthGBs; // GFLOPS stored here
            matrixCount++;
        } else if (result.benchmarkName.find("Convolution") != std::string::npos) {
            totalConvolutionBandwidth += result.effectiveBandwidthGBs;
            convolutionCount++;
        } else if (result.benchmarkName.find("Reduction") != std::string::npos) {
            totalReductionBandwidth += result.effectiveBandwidthGBs;
            reductionCount++;
        }
    }
    
    logger.Info("GPU: " + results[0].gpuName);
    logger.Info("Backend: " + results[0].backendName);
    logger.Info("");
    
    if (vectorCount > 0) {
        logger.Info("Vector Addition:    " + std::to_string(totalVectorBandwidth / vectorCount) + " GB/s (avg)");
    }
    if (matrixCount > 0) {
        logger.Info("Matrix Multiply:    " + std::to_string(totalMatrixGFLOPS / matrixCount) + " GFLOPS (avg)");
    }
    if (convolutionCount > 0) {
        logger.Info("2D Convolution:     " + std::to_string(totalConvolutionBandwidth / convolutionCount) + " GB/s (avg)");
    }
    if (reductionCount > 0) {
        logger.Info("Parallel Reduction: " + std::to_string(totalReductionBandwidth / reductionCount) + " GB/s (avg)");
    }
    
    logger.Info("");
    logger.Info("Total benchmarks run: " + std::to_string(results.size()));
    logger.Info("Results exported to: benchmark_results.csv");
    logger.Info("==========================================================");
}

/*******************************************************************************
 * NEXT STEPS FOR COMPLETION:
 * 
 * 1. **Backend Access:** Modify BenchmarkRunner to provide access to initialized
 *    backends so they can be passed to benchmark wrapper classes.
 * 
 * 2. **Suite Implementation:** Complete RunQuickSuite, RunStandardSuite, and
 *    RunFullSuite functions to execute benchmark wrapper classes.
 * 
 * 3. **Results Collection:** Collect BenchmarkResult objects from each benchmark
 *    and pass to PrintSummary for analysis.
 * 
 * 4. **Error Handling:** Add try-catch blocks for graceful error handling.
 * 
 * 5. **Progress Display:** Add progress indicators during long-running benchmarks.
 * 
 * 6. **GUI Integration:** Once command-line version is complete, wrap in ImGui
 *    for graphical interface.
 * 
 ******************************************************************************/
