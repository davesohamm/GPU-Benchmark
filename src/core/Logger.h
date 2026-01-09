/*******************************************************************************
 * FILE: Logger.h
 * 
 * PURPOSE:
 *   Logging and result export functionality for benchmark results.
 *   
 *   This class handles:
 *     1. Console output (pretty-printed results with colors)
 *     2. CSV file export (for Excel, data analysis)
 *     3. Error logging
 *     4. Timestamped result files
 * 
 * WHY LOGGING MATTERS:
 *   Benchmarks generate a LOT of data. Without proper logging:
 *     - Results disappear after program closes
 *     - Can't compare runs from different times
 *     - Can't analyze trends or generate graphs
 * 
 * CSV FORMAT:
 *   Comma-separated values file that can be opened in:
 *     - Microsoft Excel
 *     - Google Sheets
 *     - Python pandas
 *     - R programming language
 * 
 * EXAMPLE CSV OUTPUT:
 *   Timestamp,Backend,Benchmark,Size,ExecutionMS,TransferMS,BandwidthGBs,Correct
 *   2026-01-09 12:30:45,CUDA,VectorAdd,1000000,0.234,1.23,51.3,true
 *   2026-01-09 12:30:46,OpenCL,VectorAdd,1000000,0.289,1.45,49.1,true
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

#ifndef LOGGER_H
#define LOGGER_H

// Standard library
#include <string>
#include <fstream>  // For file I/O
#include <vector>
#include <memory>   // For std::unique_ptr

/*******************************************************************************
 * NAMESPACE: GPUBenchmark
 ******************************************************************************/
namespace GPUBenchmark {

/*******************************************************************************
 * ENUM: LogLevel
 * 
 * Severity levels for log messages.
 * 
 * Used to filter messages and color-code console output.
 ******************************************************************************/
enum class LogLevel {
    DEBUG,      // Detailed diagnostic information (developer-only)
    INFO,       // Normal informational messages
    WARNING,    // Warning messages (degraded functionality)
    ERROR,      // Error messages (operation failed)
    CRITICAL    // Critical errors (application cannot continue)
};

/*******************************************************************************
 * STRUCT: BenchmarkResult
 * 
 * Complete result data for a single benchmark run.
 * 
 * This structure contains all measurements and metadata for one benchmark
 * execution on one backend.
 ******************************************************************************/
struct BenchmarkResult {
    // Metadata
    std::string timestamp;          // When benchmark ran (ISO 8601 format)
    std::string backendName;        // "CUDA", "OpenCL", or "DirectCompute"
    std::string benchmarkName;      // "VectorAdd", "MatrixMul", etc.
    size_t problemSize;             // Number of elements or problem dimension
    
    // Timing (milliseconds)
    double executionTimeMS;         // GPU kernel execution time
    double hostToDeviceTimeMS;      // CPU→GPU transfer time
    double deviceToHostTimeMS;      // GPU→CPU transfer time
    double totalTimeMS;             // Total time (execution + transfers)
    
    // Performance metrics
    double effectiveBandwidthGBs;   // Achieved memory bandwidth (GB/s)
    double computeThroughputGFLOPS; // Achieved compute throughput (GFLOPS)
    
    // Correctness
    bool resultCorrect;             // Did output match expected values?
    
    // System information (for comparison across runs)
    std::string gpuName;            // GPU used for this run
    std::string gpuDriver;          // Driver version
    
    // Constructor with default values
    BenchmarkResult()
        : problemSize(0)
        , executionTimeMS(0.0)
        , hostToDeviceTimeMS(0.0)
        , deviceToHostTimeMS(0.0)
        , totalTimeMS(0.0)
        , effectiveBandwidthGBs(0.0)
        , computeThroughputGFLOPS(0.0)
        , resultCorrect(false)
    {}
};

/*******************************************************************************
 * CLASS: Logger
 * 
 * Singleton logger for the benchmark application.
 * 
 * WHY SINGLETON?
 *   We want ONE logger throughout the application that writes to ONE
 *   output file. Multiple Logger instances would cause file access conflicts.
 * 
 * SINGLETON PATTERN:
 *   Logger& log = Logger::GetInstance();
 *   log.Info("Message");  // All code uses the same Logger instance
 * 
 * USAGE:
 *   // Get logger instance
 *   Logger& logger = Logger::GetInstance();
 *   
 *   // Initialize output file
 *   logger.InitializeCSV("results/benchmark_results.csv");
 *   
 *   // Log messages
 *   logger.Info("Starting benchmarks...");
 *   logger.Warning("CUDA unavailable on this GPU");
 *   
 *   // Log benchmark results
 *   BenchmarkResult result;
 *   // ... fill in result ...
 *   logger.LogResult(result);
 *   
 *   // Close (automatically happens at program exit)
 *   logger.Close();
 * 
 ******************************************************************************/
class Logger {
public:
    /**************************************************************************
     * SINGLETON ACCESS
     *************************************************************************/
    
    /**
     * Get the singleton Logger instance.
     * 
     * @return Reference to the one and only Logger instance
     * 
     * This creates the Logger on first call, returns same instance thereafter.
     * 
     * Thread-safe since C++11 (static local variables initialized once).
     * 
     * Example:
     *   Logger& log = Logger::GetInstance();
     *   log.Info("Hello, world!");
     */
    static Logger& GetInstance();
    
    // Delete copy constructor and assignment operator
    // This prevents creating copies of the singleton
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    /**************************************************************************
     * INITIALIZATION
     *************************************************************************/
    
    /**
     * Initialize CSV output file.
     * 
     * @param filepath  Path to CSV file (will be created or overwritten)
     * @return          true if file opened successfully
     * 
     * Creates CSV file and writes header row.
     * 
     * Example:
     *   logger.InitializeCSV("results/benchmark_2026-01-09.csv");
     */
    bool InitializeCSV(const std::string& filepath);
    
    /**
     * Set minimum log level for console output.
     * 
     * @param level  Minimum level to display
     * 
     * Messages below this level are not printed to console
     * (but still written to file if CSV logging enabled).
     * 
     * Example:
     *   logger.SetLogLevel(LogLevel::WARNING);  // Only warnings and errors
     */
    void SetLogLevel(LogLevel level);
    
    /**
     * Enable or disable console output.
     * 
     * @param enabled  true to print to console, false to suppress
     * 
     * Useful for automated testing where console output is noise.
     */
    void SetConsoleOutput(bool enabled);
    
    /**************************************************************************
     * LOGGING METHODS
     *************************************************************************/
    
    /**
     * Log a debug message.
     * 
     * @param message  Message text
     * 
     * Debug messages are for developer use, not shown to users by default.
     * 
     * Example:
     *   logger.Debug("Allocated 1024 bytes of device memory at 0x7f3a2b1c");
     */
    void Debug(const std::string& message);
    
    /**
     * Log an informational message.
     * 
     * @param message  Message text
     * 
     * Normal operational messages.
     * 
     * Example:
     *   logger.Info("Starting vector addition benchmark...");
     */
    void Info(const std::string& message);
    
    /**
     * Log a warning message.
     * 
     * @param message  Message text
     * 
     * Something unexpected but not fatal.
     * 
     * Example:
     *   logger.Warning("OpenCL backend unavailable - skipping");
     */
    void Warning(const std::string& message);
    
    /**
     * Log an error message.
     * 
     * @param message  Message text
     * 
     * Operation failed but application can continue.
     * 
     * Example:
     *   logger.Error("Failed to allocate GPU memory: out of memory");
     */
    void Error(const std::string& message);
    
    /**
     * Log a critical error message.
     * 
     * @param message  Message text
     * 
     * Severe error, application may need to terminate.
     * 
     * Example:
     *   logger.Critical("No GPU compute backends available - cannot continue");
     */
    void Critical(const std::string& message);
    
    /**************************************************************************
     * BENCHMARK RESULT LOGGING
     *************************************************************************/
    
    /**
     * Log a benchmark result.
     * 
     * @param result  BenchmarkResult structure
     * 
     * Writes result to both console (formatted) and CSV file (raw data).
     * 
     * Console output example:
     *   [CUDA] VectorAdd (1M elements): 0.234 ms | 51.3 GB/s ✓
     * 
     * CSV output example:
     *   2026-01-09 12:30:45,CUDA,VectorAdd,1000000,0.234,1.23,51.3,true
     * 
     * Example:
     *   BenchmarkResult result;
     *   result.backendName = "CUDA";
     *   result.benchmarkName = "VectorAdd";
     *   result.executionTimeMS = 0.234;
     *   // ... fill in other fields ...
     *   logger.LogResult(result);
     */
    void LogResult(const BenchmarkResult& result);
    
    /**
     * Log multiple benchmark results (batch).
     * 
     * @param results  Vector of BenchmarkResult structures
     * 
     * More efficient than calling LogResult() in a loop.
     */
    void LogResults(const std::vector<BenchmarkResult>& results);
    
    /**************************************************************************
     * FILE MANAGEMENT
     *************************************************************************/
    
    /**
     * Flush output buffers to disk.
     * 
     * Ensures all logged data is written to file immediately.
     * 
     * Normally not needed (happens automatically), but useful:
     *   - Before long-running operation (ensure progress is saved)
     *   - After critical result (ensure not lost if crash)
     * 
     * Example:
     *   logger.LogResult(result);
     *   logger.Flush();  // Make sure it's saved to disk
     */
    void Flush();
    
    /**
     * Close the logger and output files.
     * 
     * Flushes buffers and closes file handles.
     * 
     * Called automatically at program exit, but can be called early
     * if you're done logging.
     */
    void Close();
    
    /**************************************************************************
     * UTILITY METHODS
     *************************************************************************/
    
    /**
     * Get current timestamp as formatted string.
     * 
     * @return Timestamp in ISO 8601 format: "YYYY-MM-DD HH:MM:SS"
     * 
     * Example: "2026-01-09 12:30:45"
     * 
     * Used for timestamping benchmark results.
     */
    static std::string GetCurrentTimestamp();
    
    /**
     * Format a BenchmarkResult as a string (for console display).
     * 
     * @param result  Result to format
     * @return        Formatted string
     * 
     * Example output:
     *   "[CUDA] VectorAdd (1M): 0.234 ms | 51.3 GB/s | ✓ Correct"
     */
    static std::string FormatResult(const BenchmarkResult& result);
    
    /**
     * Convert LogLevel to string.
     * 
     * @param level  LogLevel enum value
     * @return       String representation
     * 
     * DEBUG -> "DEBUG", INFO -> "INFO", etc.
     */
    static std::string LogLevelToString(LogLevel level);
    
private:
    /**************************************************************************
     * PRIVATE CONSTRUCTOR (Singleton pattern)
     *************************************************************************/
    Logger();
    ~Logger();
    
    /**************************************************************************
     * PRIVATE MEMBER VARIABLES
     *************************************************************************/
    
    // CSV output file stream
    std::unique_ptr<std::ofstream> m_csvFile;
    
    // Minimum log level (messages below this are filtered out)
    LogLevel m_minLogLevel;
    
    // Console output enabled?
    bool m_consoleEnabled;
    
    // Is CSV file open and ready?
    bool m_csvInitialized;
    
    /**************************************************************************
     * PRIVATE HELPER METHODS
     *************************************************************************/
    
    /**
     * Log a message with specified level.
     * 
     * @param level    Log level
     * @param message  Message text
     * 
     * Internal method used by Debug(), Info(), Warning(), etc.
     */
    void Log(LogLevel level, const std::string& message);
    
    /**
     * Write CSV header row.
     * 
     * Called by InitializeCSV() to write column names.
     */
    void WriteCSVHeader();
    
    /**
     * Write a result to CSV file.
     * 
     * @param result  Result to write
     * 
     * Formats result as CSV row and writes to file.
     */
    void WriteCSVRow(const BenchmarkResult& result);
    
    /**
     * Get console color code for log level.
     * 
     * @param level  Log level
     * @return       Windows console color code
     * 
     * Windows console colors:
     *   DEBUG: Gray
     *   INFO: White
     *   WARNING: Yellow
     *   ERROR: Red
     *   CRITICAL: Bright Red + Bold
     */
    static int GetConsoleColor(LogLevel level);
    
    /**
     * Set console text color (Windows-specific).
     * 
     * @param color  Color code
     * 
     * Uses Windows SetConsoleTextAttribute() function.
     */
    static void SetConsoleColor(int color);
    
    /**
     * Reset console color to default.
     * 
     * Should be called after colored output to avoid messing up terminal.
     */
    static void ResetConsoleColor();
};

} // namespace GPUBenchmark

#endif // LOGGER_H

/*******************************************************************************
 * END OF FILE: Logger.h
 * 
 * WHAT WE LEARNED:
 *   1. Singleton pattern ensures one logger throughout application
 *   2. Logging serves multiple purposes: console feedback, file export, debugging
 *   3. CSV format enables data analysis in external tools
 *   4. Log levels allow filtering messages by importance
 * 
 * KEY DESIGN PATTERNS:
 *   - Singleton: One instance shared globally
 *   - RAII: File handle managed by unique_ptr
 *   - Builder pattern: Flexible result construction
 * 
 * WHY THIS MATTERS:
 *   - Professional applications need logging for debugging
 *   - Benchmark results are useless if not saved
 *   - CSV export enables data analysis and graphing
 *   - Color-coded console output improves user experience
 * 
 * NEXT FILE TO READ:
 *   - Logger.cpp : Implementation of logging functionality
 * 
 ******************************************************************************/
