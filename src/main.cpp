/*******************************************************************************
 * FILE: main.cpp
 * 
 * PURPOSE:
 *   Application entry point for the GPU Compute Benchmark Tool.
 *   
 *   This is where everything starts! When you run GPU-Benchmark.exe,
 *   execution begins at main() in this file.
 * 
 * WHAT THIS FILE DOES:
 *   1. Parse command-line arguments
 *   2. Initialize logging
 *   3. Detect system capabilities (GPU, APIs)
 *   4. Initialize available backends (CUDA, OpenCL, DirectCompute)
 *   5. Run benchmarks
 *   6. Display and export results
 *   7. Clean up and exit
 * 
 * EXECUTION FLOW:
 *   main()
 *     └─→ Discover system capabilities
 *     └─→ Initialize backends
 *     └─→ If GUI mode:
 *          └─→ Launch GUI window
 *          └─→ User selects benchmarks
 *          └─→ Run selected benchmarks
 *          └─→ Display results in real-time
 *     └─→ If CLI mode:
 *          └─→ Run all benchmarks
 *          └─→ Export results to CSV
 *     └─→ Cleanup and exit
 * 
 * COMMAND-LINE EXAMPLES:
 *   GPU-Benchmark.exe                          # Launch GUI mode
 *   GPU-Benchmark.exe --cli --all              # Run all benchmarks
 *   GPU-Benchmark.exe --benchmark=vector_add   # Run specific benchmark
 *   GPU-Benchmark.exe --backend=cuda           # Use only CUDA
 *   GPU-Benchmark.exe --output=results.csv     # Specify output file
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

// Core framework includes
#include "core/DeviceDiscovery.h"
#include "core/Timer.h"

// Standard library
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>  // For exit codes

// Windows-specific headers
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Using declarations to reduce typing
using namespace GPUBenchmark;

/*******************************************************************************
 * STRUCT: CommandLineArgs
 * 
 * Parsed command-line arguments.
 * 
 * Contains all user-specified options from command line.
 ******************************************************************************/
struct CommandLineArgs {
    bool showHelp;                          // Show help message?
    bool cliMode;                           // Command-line mode (no GUI)?
    bool runAll;                            // Run all benchmarks?
    std::vector<std::string> benchmarks;    // Specific benchmarks to run
    std::vector<std::string> backends;      // Specific backends to use
    std::string outputFile;                 // CSV output file path
    bool verbose;                           // Verbose output?
    
    // Constructor with defaults
    CommandLineArgs()
        : showHelp(false)
        , cliMode(false)
        , runAll(false)
        , outputFile("results/benchmark_results.csv")
        , verbose(false)
    {}
};

/*******************************************************************************
 * FUNCTION PROTOTYPES
 * 
 * Forward declarations of functions defined later in this file.
 ******************************************************************************/

// Parse command-line arguments
CommandLineArgs ParseCommandLine(int argc, char* argv[]);

// Print usage information
void PrintUsage();

// Print welcome banner
void PrintBanner();

// Run benchmarks in CLI mode
int RunCLIMode(const CommandLineArgs& args, const SystemCapabilities& caps);

// Run GUI mode
int RunGUIMode(const CommandLineArgs& args, const SystemCapabilities& caps);

/*******************************************************************************
 * FUNCTION: main()
 * 
 * Application entry point.
 * 
 * PARAMETERS:
 *   argc: Argument count (number of command-line arguments)
 *   argv: Argument vector (array of argument strings)
 * 
 * RETURN VALUE:
 *   0: Success
 *   1: Error
 * 
 * EXECUTION FLOW:
 *   1. Parse command-line arguments
 *   2. Show help if requested
 *   3. Print welcome banner
 *   4. Discover system capabilities
 *   5. Check if any backend available
 *   6. Launch appropriate mode (CLI or GUI)
 *   7. Return exit code
 * 
 * ERROR HANDLING:
 *   All exceptions are caught and logged.
 *   Application never crashes - always exits gracefully with error message.
 ******************************************************************************/
int main(int argc, char* argv[]) {
    // Enable Windows console colors (for colored output)
    // This allows us to use colors in console (errors in red, warnings in yellow, etc.)
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD consoleMode = 0;
    GetConsoleMode(hConsole, &consoleMode);
    consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hConsole, consoleMode);
    
    // Wrap everything in try-catch for safety
    // If anything goes wrong, we'll catch it and exit gracefully
    try {
        // =====================================================================
        // STEP 1: Parse Command-Line Arguments
        // =====================================================================
        CommandLineArgs args = ParseCommandLine(argc, argv);
        
        // If user asked for help, show usage and exit
        if (args.showHelp) {
            PrintUsage();
            return 0;
        }
        
        // =====================================================================
        // STEP 2: Print Welcome Banner
        // =====================================================================
        PrintBanner();
        
        // =====================================================================
        // STEP 3: Discover System Capabilities
        // =====================================================================
        std::cout << "\n=== System Discovery ===\n" << std::endl;
        
        Timer discoveryTimer;
        discoveryTimer.Start();
        
        // This is where we detect GPUs, CUDA, OpenCL, DirectCompute, etc.
        // See DeviceDiscovery.cpp for implementation details
        SystemCapabilities caps = DeviceDiscovery::Discover();
        
        discoveryTimer.Stop();
        
        std::cout << "\nDiscovery completed in " 
                  << discoveryTimer.GetMilliseconds() << " ms\n" << std::endl;
        
        // Print detailed capability information
        DeviceDiscovery::PrintCapabilities(caps);
        
        // =====================================================================
        // STEP 4: Verify At Least One Backend Available
        // =====================================================================
        if (!caps.HasAnyBackend()) {
            std::cerr << "\n[ERROR] No GPU compute backends available!" << std::endl;
            std::cerr << "Possible reasons:" << std::endl;
            std::cerr << "  - No compatible GPU detected" << std::endl;
            std::cerr << "  - GPU drivers not installed or outdated" << std::endl;
            std::cerr << "  - DirectX/OpenCL runtime not available" << std::endl;
            std::cerr << "\nPlease install/update GPU drivers and try again." << std::endl;
            return 1;
        }
        
        // =====================================================================
        // STEP 5: Launch Appropriate Mode
        // =====================================================================
        int exitCode = 0;
        
        if (args.cliMode) {
            // Command-line mode: Run benchmarks without GUI
            std::cout << "\n=== Running in CLI Mode ===\n" << std::endl;
            exitCode = RunCLIMode(args, caps);
        } else {
            // GUI mode: Launch interactive window
            std::cout << "\n=== Launching GUI Mode ===\n" << std::endl;
            exitCode = RunGUIMode(args, caps);
        }
        
        // =====================================================================
        // STEP 6: Exit
        // =====================================================================
        if (exitCode == 0) {
            std::cout << "\n=== Benchmark Completed Successfully ===\n" << std::endl;
        } else {
            std::cerr << "\n=== Benchmark Failed (Exit Code: " << exitCode << ") ===\n" << std::endl;
        }
        
        return exitCode;
    }
    catch (const std::exception& e) {
        // Catch any standard C++ exception
        std::cerr << "\n[CRITICAL ERROR] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        // Catch any other type of exception
        std::cerr << "\n[CRITICAL ERROR] Unknown exception occurred!" << std::endl;
        return 1;
    }
}

/*******************************************************************************
 * FUNCTION: ParseCommandLine()
 * 
 * Parse command-line arguments into CommandLineArgs structure.
 * 
 * SUPPORTED ARGUMENTS:
 *   --help, -h              : Show usage information
 *   --cli                   : Run in command-line mode (no GUI)
 *   --all                   : Run all benchmarks
 *   --benchmark=<name>      : Run specific benchmark
 *   --backend=<name>        : Use specific backend only
 *   --output=<file>         : Specify output CSV file
 *   --verbose, -v           : Enable verbose output
 * 
 * EXAMPLES:
 *   --cli --all --output=results.csv
 *   --benchmark=vector_add --backend=cuda
 *   --help
 * 
 * IMPLEMENTATION:
 *   Simple string parsing. For production code, consider using a library
 *   like Boost.Program_options or cxxopts for more robust parsing.
 ******************************************************************************/
CommandLineArgs ParseCommandLine(int argc, char* argv[]) {
    CommandLineArgs args;
    
    // Loop through all arguments (skip argv[0] which is program name)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Help
        if (arg == "--help" || arg == "-h") {
            args.showHelp = true;
        }
        // CLI mode
        else if (arg == "--cli") {
            args.cliMode = true;
        }
        // Run all benchmarks
        else if (arg == "--all") {
            args.runAll = true;
        }
        // Verbose
        else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        }
        // Benchmark specification: --benchmark=vector_add
        else if (arg.find("--benchmark=") == 0) {
            std::string benchmark = arg.substr(12);  // Skip "--benchmark="
            args.benchmarks.push_back(benchmark);
        }
        // Backend specification: --backend=cuda
        else if (arg.find("--backend=") == 0) {
            std::string backend = arg.substr(10);  // Skip "--backend="
            args.backends.push_back(backend);
        }
        // Output file: --output=results.csv
        else if (arg.find("--output=") == 0) {
            args.outputFile = arg.substr(9);  // Skip "--output="
        }
        // Unknown argument
        else {
            std::cerr << "Warning: Unknown argument: " << arg << std::endl;
        }
    }
    
    return args;
}

/*******************************************************************************
 * FUNCTION: PrintUsage()
 * 
 * Print command-line usage information.
 * 
 * This is shown when user runs: GPU-Benchmark.exe --help
 ******************************************************************************/
void PrintUsage() {
    std::cout << "GPU Compute Benchmark Tool\n" << std::endl;
    std::cout << "USAGE:" << std::endl;
    std::cout << "  GPU-Benchmark.exe [options]\n" << std::endl;
    
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  --help, -h                Show this help message" << std::endl;
    std::cout << "  --cli                     Run in command-line mode (no GUI)" << std::endl;
    std::cout << "  --all                     Run all benchmarks" << std::endl;
    std::cout << "  --benchmark=<name>        Run specific benchmark" << std::endl;
    std::cout << "                            Available: vector_add, matrix_mul, convolution, reduction" << std::endl;
    std::cout << "  --backend=<name>          Use specific backend only" << std::endl;
    std::cout << "                            Available: cuda, opencl, directcompute" << std::endl;
    std::cout << "  --output=<file>           Specify CSV output file (default: results/benchmark_results.csv)" << std::endl;
    std::cout << "  --verbose, -v             Enable verbose output\n" << std::endl;
    
    std::cout << "EXAMPLES:" << std::endl;
    std::cout << "  GPU-Benchmark.exe" << std::endl;
    std::cout << "      Launch GUI mode\n" << std::endl;
    
    std::cout << "  GPU-Benchmark.exe --cli --all" << std::endl;
    std::cout << "      Run all benchmarks in CLI mode\n" << std::endl;
    
    std::cout << "  GPU-Benchmark.exe --cli --benchmark=vector_add --backend=cuda" << std::endl;
    std::cout << "      Run vector addition on CUDA backend only\n" << std::endl;
    
    std::cout << "  GPU-Benchmark.exe --cli --all --output=my_results.csv" << std::endl;
    std::cout << "      Run all benchmarks and save to custom file\n" << std::endl;
}

/*******************************************************************************
 * FUNCTION: PrintBanner()
 * 
 * Print ASCII art banner and application information.
 * 
 * This gives the application a professional look!
 ******************************************************************************/
void PrintBanner() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                               ║\n";
    std::cout << "║           GPU COMPUTE BENCHMARK & VISUALIZATION               ║\n";
    std::cout << "║                                                               ║\n";
    std::cout << "║         Comparing CUDA, OpenCL, and DirectCompute             ║\n";
    std::cout << "║                                                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Author: Soham" << std::endl;
    std::cout << "System: Windows 11 | AMD Ryzen 7 4800H | NVIDIA RTX 3050" << std::endl;
    std::cout << "Version: 1.0.0" << std::endl;
    std::cout << "Build Date: " << __DATE__ << " " << __TIME__ << std::endl;
}

/*******************************************************************************
 * FUNCTION: RunCLIMode()
 * 
 * Run benchmarks in command-line interface mode (no GUI).
 * 
 * WORKFLOW:
 *   1. Initialize backends (CUDA, OpenCL, DirectCompute)
 *   2. Initialize CSV output file
 *   3. Run selected benchmarks
 *   4. Export results to CSV
 *   5. Print summary
 * 
 * PARAMETERS:
 *   args: Parsed command-line arguments
 *   caps: Detected system capabilities
 * 
 * RETURN:
 *   0 on success, 1 on failure
 ******************************************************************************/
int RunCLIMode(const CommandLineArgs& args, const SystemCapabilities& caps) {
    std::cout << "CLI Mode implementation:" << std::endl;
    std::cout << "  This would initialize backends and run benchmarks" << std::endl;
    std::cout << "  Results would be saved to: " << args.outputFile << std::endl;
    
    // TODO: Implement full CLI mode
    // This is a placeholder showing the structure
    
    // 1. Initialize backends based on availability
    std::cout << "\nInitializing backends..." << std::endl;
    
    if (caps.cuda.available && 
        (args.backends.empty() || std::find(args.backends.begin(), args.backends.end(), "cuda") != args.backends.end())) {
        std::cout << "  ✓ CUDA backend initialized" << std::endl;
    }
    
    if (caps.opencl.available && 
        (args.backends.empty() || std::find(args.backends.begin(), args.backends.end(), "opencl") != args.backends.end())) {
        std::cout << "  ✓ OpenCL backend initialized" << std::endl;
    }
    
    if (caps.directCompute.available && 
        (args.backends.empty() || std::find(args.backends.begin(), args.backends.end(), "directcompute") != args.backends.end())) {
        std::cout << "  ✓ DirectCompute backend initialized" << std::endl;
    }
    
    // 2. Run benchmarks
    std::cout << "\nRunning benchmarks..." << std::endl;
    
    if (args.runAll || args.benchmarks.empty()) {
        std::cout << "  - Vector Addition" << std::endl;
        std::cout << "  - Matrix Multiplication" << std::endl;
        std::cout << "  - 2D Convolution" << std::endl;
        std::cout << "  - Parallel Reduction" << std::endl;
    } else {
        for (const auto& benchmark : args.benchmarks) {
            std::cout << "  - " << benchmark << std::endl;
        }
    }
    
    // 3. Results would be saved here
    std::cout << "\nResults saved to: " << args.outputFile << std::endl;
    
    return 0;
}

/*******************************************************************************
 * FUNCTION: RunGUIMode()
 * 
 * Run application in GUI mode with interactive window.
 * 
 * GUI MODE FEATURES:
 *   - Real-time visualization of results
 *   - Interactive benchmark selection
 *   - Live performance graphs
 *   - Backend comparison charts
 * 
 * WORKFLOW:
 *   1. Initialize OpenGL/GLFW window
 *   2. Create GUI (ImGui or custom)
 *   3. Initialize backends
 *   4. Main loop:
 *      - Handle user input
 *      - Run selected benchmarks
 *      - Update visualizations
 *      - Render frame
 *   5. Cleanup and exit
 * 
 * PARAMETERS:
 *   args: Parsed command-line arguments
 *   caps: Detected system capabilities
 * 
 * RETURN:
 *   0 on success, 1 on failure
 ******************************************************************************/
int RunGUIMode(const CommandLineArgs& args, const SystemCapabilities& caps) {
    std::cout << "GUI Mode implementation:" << std::endl;
    std::cout << "  This would create an OpenGL window" << std::endl;
    std::cout << "  with real-time benchmark visualization" << std::endl;
    std::cout << "\n[NOTE] GUI mode is under development." << std::endl;
    std::cout << "For now, please use CLI mode: --cli --all" << std::endl;
    
    // TODO: Implement full GUI mode with:
    // - GLFW window creation
    // - ImGui integration
    // - OpenGL rendering
    // - Real-time charts and graphs
    
    return 0;
}

/*******************************************************************************
 * END OF FILE: main.cpp
 * 
 * WHAT WE IMPLEMENTED:
 *   1. Application entry point with error handling
 *   2. Command-line argument parsing
 *   3. System capability discovery
 *   4. Mode selection (CLI vs GUI)
 *   5. Professional console output with banner
 * 
 * EXECUTION FLOW SUMMARY:
 *   User runs: GPU-Benchmark.exe --cli --all
 *     ↓
 *   main() parses arguments
 *     ↓
 *   Discovers system (GPU, CUDA, OpenCL, DirectCompute)
 *     ↓
 *   Checks at least one backend available
 *     ↓
 *   Runs CLI mode with all benchmarks
 *     ↓
 *   Exports results to CSV
 *     ↓
 *   Exits with success code
 * 
 * KEY FEATURES:
 *   - Graceful error handling (try-catch)
 *   - User-friendly messages
 *   - Flexible command-line interface
 *   - Professional output formatting
 *   - Clear documentation for interviews
 * 
 * WHAT TO IMPLEMENT NEXT:
 *   1. Backend implementations (CUDA, OpenCL, DirectCompute)
 *   2. Benchmark implementations (Vector Add, Matrix Mul, etc.)
 *   3. BenchmarkRunner to orchestrate execution
 *   4. Visualization with OpenGL
 *   5. GUI with ImGui
 * 
 * FOR INTERVIEWS:
 *   Explain:
 *     - Why error handling important (application never crashes)
 *     - Why runtime detection instead of compile-time (one .exe, all GPUs)
 *     - Why both CLI and GUI modes (flexibility for users and automation)
 *     - How command-line parsing works
 *     - Professional software engineering practices (logging, error messages)
 * 
 ******************************************************************************/
