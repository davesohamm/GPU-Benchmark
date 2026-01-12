/*******************************************************************************
 * FILE: Logger.cpp
 * 
 * PURPOSE:
 *   Implementation of the Logger class for benchmark result logging and output.
 *   
 * FEATURES:
 *   - Console output with colored messages (Windows API)
 *   - CSV file export for data analysis
 *   - Benchmark result formatting
 *   - Timestamping
 *   - Singleton pattern for global access
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#include "Logger.h"

// Windows API for console colors
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Undefine Windows ERROR macro to avoid conflicts
#ifdef ERROR
#undef ERROR
#endif

// Standard library
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <chrono>

namespace GPUBenchmark {

/*******************************************************************************
 * SINGLETON INSTANCE
 ******************************************************************************/
Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

/*******************************************************************************
 * CONSTRUCTOR
 ******************************************************************************/
Logger::Logger()
    : m_minLogLevel(LogLevel::INFO)
    , m_consoleEnabled(true)
    , m_csvInitialized(false)
{
}

/*******************************************************************************
 * DESTRUCTOR
 ******************************************************************************/
Logger::~Logger() {
    Close();
}

/*******************************************************************************
 * CSV INITIALIZATION
 ******************************************************************************/
bool Logger::InitializeCSV(const std::string& filepath) {
    m_csvFile = std::make_unique<std::ofstream>(filepath);
    
    if (!m_csvFile->is_open()) {
        Error("Failed to open CSV file: " + filepath);
        return false;
    }
    
    WriteCSVHeader();
    m_csvInitialized = true;
    Info("CSV logging initialized: " + filepath);
    
    return true;
}

/*******************************************************************************
 * CONFIGURATION
 ******************************************************************************/
void Logger::SetLogLevel(LogLevel level) {
    m_minLogLevel = level;
}

void Logger::SetConsoleOutput(bool enabled) {
    m_consoleEnabled = enabled;
}

/*******************************************************************************
 * LOGGING METHODS
 ******************************************************************************/
void Logger::Debug(const std::string& message) {
    Log(LogLevel::DEBUG, message);
}

void Logger::Info(const std::string& message) {
    Log(LogLevel::INFO, message);
}

void Logger::Warning(const std::string& message) {
    Log(LogLevel::WARNING, message);
}

void Logger::Error(const std::string& message) {
    Log(LogLevel::ERROR, message);
}

void Logger::Critical(const std::string& message) {
    Log(LogLevel::CRITICAL, message);
}

/*******************************************************************************
 * BENCHMARK RESULT LOGGING
 ******************************************************************************/
void Logger::LogResult(const BenchmarkResult& result) {
    if (m_consoleEnabled) {
        std::string formatted = FormatResult(result);
        std::cout << formatted << std::endl;
    }
    
    if (m_csvInitialized && m_csvFile && m_csvFile->is_open()) {
        WriteCSVRow(result);
    }
}

void Logger::LogResults(const std::vector<BenchmarkResult>& results) {
    for (const auto& result : results) {
        LogResult(result);
    }
}

/*******************************************************************************
 * FILE OPERATIONS
 ******************************************************************************/
void Logger::Flush() {
    if (m_csvFile && m_csvFile->is_open()) {
        m_csvFile->flush();
    }
    std::cout.flush();
}

void Logger::Close() {
    if (m_csvFile && m_csvFile->is_open()) {
        m_csvFile->close();
        m_csvInitialized = false;
    }
}

/*******************************************************************************
 * TIMESTAMP GENERATION
 ******************************************************************************/
std::string Logger::GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_now;
    localtime_s(&tm_now, &time_t_now);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
    
    return oss.str();
}

/*******************************************************************************
 * RESULT FORMATTING
 ******************************************************************************/
std::string Logger::FormatResult(const BenchmarkResult& result) {
    std::ostringstream oss;
    
    // [Backend] BenchmarkName (size): time ms | bandwidth GB/s | status
    oss << "[" << result.backendName << "] " << result.benchmarkName << " ";
    
    // Problem size with units
    if (result.problemSize >= 1000000) {
        oss << "(" << (result.problemSize / 1000000) << "M): ";
    } else if (result.problemSize >= 1000) {
        oss << "(" << (result.problemSize / 1000) << "K): ";
    } else {
        oss << "(" << result.problemSize << "): ";
    }
    
    // Timing
    oss << std::fixed << std::setprecision(3) << result.executionTimeMS << " ms";
    
    // Bandwidth (if available)
    if (result.effectiveBandwidthGBs > 0.0) {
        oss << " | " << std::setprecision(1) << result.effectiveBandwidthGBs << " GB/s";
    }
    
    // GFLOPS (if available)
    if (result.computeThroughputGFLOPS > 0.0) {
        oss << " | " << std::setprecision(1) << result.computeThroughputGFLOPS << " GFLOPS";
    }
    
    // Correctness
    if (result.resultCorrect) {
        oss << " | ✓ Correct";
    } else {
        oss << " | ✗ INCORRECT!";
    }
    
    return oss.str();
}

/*******************************************************************************
 * LOG LEVEL TO STRING
 ******************************************************************************/
std::string Logger::LogLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO";
        case LogLevel::WARNING:  return "WARNING";
        case LogLevel::ERROR:    return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default:                 return "UNKNOWN";
    }
}

/*******************************************************************************
 * INTERNAL LOGGING
 ******************************************************************************/
void Logger::Log(LogLevel level, const std::string& message) {
    if (level < m_minLogLevel) {
        return;
    }
    
    if (!m_consoleEnabled) {
        return;
    }
    
    SetConsoleColor(GetConsoleColor(level));
    std::cout << "[" << LogLevelToString(level) << "] " << message << std::endl;
    ResetConsoleColor();
}

/*******************************************************************************
 * CSV WRITING
 ******************************************************************************/
void Logger::WriteCSVHeader() {
    if (!m_csvFile || !m_csvFile->is_open()) {
        return;
    }
    
    *m_csvFile << "Timestamp,Backend,Benchmark,ProblemSize,"
               << "ExecutionTimeMS,HostToDeviceMS,DeviceToHostMS,TotalTimeMS,"
               << "BandwidthGBs,ThroughputGFLOPS,ResultCorrect,"
               << "GPUName,DriverVersion"
               << std::endl;
}

void Logger::WriteCSVRow(const BenchmarkResult& result) {
    if (!m_csvFile || !m_csvFile->is_open()) {
        return;
    }
    
    *m_csvFile << result.timestamp << ","
               << result.backendName << ","
               << result.benchmarkName << ","
               << result.problemSize << ","
               << result.executionTimeMS << ","
               << result.hostToDeviceTimeMS << ","
               << result.deviceToHostTimeMS << ","
               << result.totalTimeMS << ","
               << result.effectiveBandwidthGBs << ","
               << result.computeThroughputGFLOPS << ","
               << (result.resultCorrect ? "true" : "false") << ","
               << result.gpuName << ","
               << result.gpuDriver
               << std::endl;
}

/*******************************************************************************
 * CONSOLE COLOR MANAGEMENT
 ******************************************************************************/
int Logger::GetConsoleColor(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:    return 8;  // Gray
        case LogLevel::INFO:     return 7;  // White
        case LogLevel::WARNING:  return 14; // Yellow
        case LogLevel::ERROR:    return 12; // Red
        case LogLevel::CRITICAL: return 12 | FOREGROUND_INTENSITY; // Bright Red
        default:                 return 7;  // White
    }
}

void Logger::SetConsoleColor(int color) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, static_cast<WORD>(color));
}

void Logger::ResetConsoleColor() {
    SetConsoleColor(7);
}

} // namespace GPUBenchmark

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
