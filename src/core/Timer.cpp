/*******************************************************************************
 * FILE: Timer.cpp
 * 
 * PURPOSE:
 *   Implementation of the high-resolution Timer class for Windows.
 *   
 *   This file contains the actual code that measures time using the Windows
 *   Performance Counter API.
 * 
 * WINDOWS PERFORMANCE COUNTER API:
 *   - QueryPerformanceFrequency(): Get timer resolution (once at startup)
 *   - QueryPerformanceCounter(): Get current time (each measurement)
 * 
 * HOW IT WORKS INTERNALLY:
 *   Modern CPUs have a hardware counter called TSC (Time Stamp Counter) that
 *   increments every CPU cycle. Windows uses this (or HPET on older systems)
 *   to provide high-resolution timing.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

#include "Timer.h"

// Standard library includes
#include <sstream>    // For std::ostringstream (string formatting)
#include <iomanip>    // For std::fixed, std::setprecision (number formatting)
#include <chrono>     // For GetTimestampMS() implementation

// Windows API for timing
// Already included in Timer.h: <windows.h>

/*******************************************************************************
 * NAMESPACE: GPUBenchmark
 ******************************************************************************/
namespace GPUBenchmark {

/*******************************************************************************
 * CONSTRUCTOR: Timer::Timer()
 * 
 * Initializes the timer by querying the performance counter frequency.
 * 
 * The frequency tells us how many "ticks" happen per second on this system.
 * This is hardware-dependent but constant for the lifetime of the program.
 * 
 * WHAT HAPPENS:
 *   1. QueryPerformanceFrequency() fills m_frequency with ticks/second
 *   2. Initialize times to zero
 *   3. Set running state to false
 * 
 * NOTE: QueryPerformanceFrequency() always succeeds on Windows XP and later.
 *       On ancient systems (Windows 95/98), it might fail, but we don't
 *       support those systems (this is a modern GPU benchmarking tool!).
 ******************************************************************************/
Timer::Timer() 
    : m_startTime{}
    , m_endTime{}
    , m_running(false)
{
    // Query the performance counter frequency (ticks per second)
    // This is done once at construction and stored in m_frequency
    // 
    // Example values:
    //   - Modern systems: 10,000,000 (10 MHz) = 100 nanosecond resolution
    //   - Some systems:   3,579,545 (~3.58 MHz) = ~280 nanosecond resolution
    QueryPerformanceFrequency(&m_frequency);
    
    // Initialize start and end times to zero
    m_startTime.QuadPart = 0;
    m_endTime.QuadPart = 0;
}

/*******************************************************************************
 * METHOD: Timer::Start()
 * 
 * Start the timer by recording the current performance counter value.
 * 
 * WHAT HAPPENS:
 *   1. QueryPerformanceCounter() reads current hardware counter
 *   2. Value is stored in m_startTime
 *   3. Set running flag to true
 * 
 * PERFORMANCE:
 *   QueryPerformanceCounter() is very fast (~100-200 CPU cycles)
 *   Don't worry about calling it frequently!
 * 
 * THREAD SAFETY:
 *   Not thread-safe! Each thread should have its own Timer instance.
 ******************************************************************************/
void Timer::Start() {
    // Record the current performance counter value as the start time
    QueryPerformanceCounter(&m_startTime);
    
    // Mark the timer as running
    m_running = true;
}

/*******************************************************************************
 * METHOD: Timer::Stop()
 * 
 * Stop the timer by recording the current performance counter value.
 * 
 * WHAT HAPPENS:
 *   1. QueryPerformanceCounter() reads current hardware counter
 *   2. Value is stored in m_endTime
 *   3. Set running flag to false
 * 
 * AFTER CALLING:
 *   The elapsed time is now available via Get*() methods.
 *   Elapsed = (m_endTime - m_startTime) / m_frequency
 ******************************************************************************/
void Timer::Stop() {
    // Record the current performance counter value as the end time
    QueryPerformanceCounter(&m_endTime);
    
    // Mark the timer as stopped
    m_running = false;
}

/*******************************************************************************
 * METHOD: Timer::Reset()
 * 
 * Reset the timer to initial state.
 * 
 * WHAT HAPPENS:
 *   1. Clear start and end times
 *   2. Set running state to false
 * 
 * WHEN TO USE:
 *   - Reusing timer object for new measurement
 *   - Clearing previous measurement
 * 
 * NOTE: Not required before calling Start() again - Start() implicitly
 *       begins a new measurement.
 ******************************************************************************/
void Timer::Reset() {
    m_startTime.QuadPart = 0;
    m_endTime.QuadPart = 0;
    m_running = false;
}

/*******************************************************************************
 * METHOD: Timer::GetMilliseconds()
 * 
 * Get the elapsed time in milliseconds.
 * 
 * CALCULATION:
 *   1. Calculate elapsed seconds: (end - start) / frequency
 *   2. Convert to milliseconds: seconds * 1000
 * 
 * PRECISION:
 *   - Hardware resolution: ~0.0001 ms (100 nanoseconds)
 *   - Returned as double for sub-millisecond precision
 * 
 * EXAMPLE:
 *   If frequency = 10,000,000 Hz (10 MHz)
 *   And elapsed ticks = 152,345
 *   Then elapsed seconds = 152,345 / 10,000,000 = 0.0152345 s
 *   And milliseconds = 0.0152345 * 1000 = 15.2345 ms
 ******************************************************************************/
double Timer::GetMilliseconds() const {
    // Calculate elapsed time in seconds, then convert to milliseconds
    return CalculateElapsedSeconds() * 1000.0;
}

/*******************************************************************************
 * METHOD: Timer::GetMicroseconds()
 * 
 * Get the elapsed time in microseconds.
 * 
 * CALCULATION:
 *   1. Calculate elapsed seconds: (end - start) / frequency
 *   2. Convert to microseconds: seconds * 1,000,000
 * 
 * WHEN TO USE:
 *   For very fast operations that take < 1 millisecond
 * 
 * EXAMPLE:
 *   0.0152345 seconds = 15,234.5 microseconds
 ******************************************************************************/
double Timer::GetMicroseconds() const {
    // Calculate elapsed time in seconds, then convert to microseconds
    return CalculateElapsedSeconds() * 1000000.0;
}

/*******************************************************************************
 * METHOD: Timer::GetSeconds()
 * 
 * Get the elapsed time in seconds.
 * 
 * CALCULATION:
 *   elapsed_seconds = (end_ticks - start_ticks) / frequency
 * 
 * WHEN TO USE:
 *   For long-running operations or total benchmark duration
 * 
 * EXAMPLE:
 *   0.0152345 seconds = ~0.015 s
 ******************************************************************************/
double Timer::GetSeconds() const {
    return CalculateElapsedSeconds();
}

/*******************************************************************************
 * METHOD: Timer::GetNanoseconds()
 * 
 * Get the elapsed time in nanoseconds.
 * 
 * CALCULATION:
 *   1. Calculate elapsed seconds: (end - start) / frequency
 *   2. Convert to nanoseconds: seconds * 1,000,000,000
 * 
 * PRECISION NOTE:
 *   Even though we return nanoseconds, the actual resolution is limited
 *   by the hardware timer (typically ~100 ns).
 * 
 * EXAMPLE:
 *   0.0152345 seconds = 15,234,500 nanoseconds
 ******************************************************************************/
uint64_t Timer::GetNanoseconds() const {
    // Calculate elapsed time in seconds, then convert to nanoseconds
    return static_cast<uint64_t>(CalculateElapsedSeconds() * 1000000000.0);
}

/*******************************************************************************
 * METHOD: Timer::GetFrequency()
 * 
 * Get the performance counter frequency (ticks per second).
 * 
 * RETURNS:
 *   Number of ticks per second (Hz)
 * 
 * WHY THIS IS USEFUL:
 *   - Tells you the timer resolution
 *   - Higher frequency = better precision
 *   - Can be used to calculate overhead of timing itself
 * 
 * EXAMPLE VALUES:
 *   - 10,000,000 Hz = 100 ns per tick
 *   - 3,579,545 Hz ≈ 280 ns per tick
 ******************************************************************************/
uint64_t Timer::GetFrequency() const {
    return m_frequency.QuadPart;
}

/*******************************************************************************
 * METHOD: Timer::IsRunning()
 * 
 * Check if the timer is currently running.
 * 
 * RETURNS:
 *   true if Start() called but Stop() not yet called
 *   false otherwise
 * 
 * USE CASE:
 *   Can check state before calling Stop() or Get*() methods
 ******************************************************************************/
bool Timer::IsRunning() const {
    return m_running;
}

/*******************************************************************************
 * STATIC METHOD: Timer::GetTimestampMS()
 * 
 * Get current timestamp in milliseconds since Unix epoch.
 * 
 * UNIX EPOCH:
 *   January 1, 1970, 00:00:00 UTC
 * 
 * WHAT IT'S USED FOR:
 *   - Timestamping benchmark results
 *   - Creating unique filenames
 *   - Logging when tests were run
 * 
 * EXAMPLE OUTPUT:
 *   1704801600000 (some time in January 2024)
 * 
 * IMPLEMENTATION:
 *   Uses std::chrono::system_clock which is based on wall clock time
 *   (not monotonic, can jump if user changes system time)
 ******************************************************************************/
uint64_t Timer::GetTimestampMS() {
    // Get current time point from system clock
    auto now = std::chrono::system_clock::now();
    
    // Convert to milliseconds since epoch
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    
    return static_cast<uint64_t>(millis.count());
}

/*******************************************************************************
 * STATIC METHOD: Timer::FormatDuration()
 * 
 * Format a duration in milliseconds to a human-readable string.
 * 
 * AUTOMATIC UNIT SELECTION:
 *   - < 1 ms: Display in microseconds (μs)
 *   - 1-1000 ms: Display in milliseconds (ms)
 *   - 1-60 seconds: Display in seconds (s)
 *   - > 60 seconds: Display in minutes and seconds (m:s)
 * 
 * EXAMPLES:
 *   FormatDuration(0.123)    -> "123.0 μs"
 *   FormatDuration(1.234)    -> "1.234 ms"
 *   FormatDuration(1234.5)   -> "1.235 s"
 *   FormatDuration(75000)    -> "1m 15s"
 * 
 * WHY THIS IS USEFUL:
 *   Makes benchmark output much more readable!
 *   User sees "5.2 ms" instead of "0.0052 s" or "5200 μs"
 ******************************************************************************/
std::string Timer::FormatDuration(double milliseconds) {
    std::ostringstream oss;  // String stream for formatted output
    oss << std::fixed;       // Fixed-point notation (not scientific)
    
    // Case 1: Very fast (< 1 millisecond) - show in microseconds
    if (milliseconds < 1.0) {
        double microseconds = milliseconds * 1000.0;
        oss << std::setprecision(1) << microseconds << " μs";
    }
    // Case 2: Fast (1-1000 milliseconds) - show in milliseconds
    else if (milliseconds < 1000.0) {
        oss << std::setprecision(3) << milliseconds << " ms";
    }
    // Case 3: Moderate (1-60 seconds) - show in seconds
    else if (milliseconds < 60000.0) {
        double seconds = milliseconds / 1000.0;
        oss << std::setprecision(3) << seconds << " s";
    }
    // Case 4: Long (> 60 seconds) - show in minutes and seconds
    else {
        int minutes = static_cast<int>(milliseconds / 60000.0);
        double seconds = (milliseconds - minutes * 60000.0) / 1000.0;
        oss << minutes << "m " << std::setprecision(1) << seconds << "s";
    }
    
    return oss.str();
}

/*******************************************************************************
 * PRIVATE METHOD: Timer::CalculateElapsedSeconds()
 * 
 * Core calculation: convert tick difference to seconds.
 * 
 * FORMULA:
 *   ElapsedSeconds = (EndTicks - StartTicks) / TicksPerSecond
 * 
 * EXPLANATION:
 *   If the timer ticks 10,000,000 times per second (10 MHz),
 *   and 152,345 ticks elapsed,
 *   then 152,345 / 10,000,000 = 0.0152345 seconds passed.
 * 
 * WHY QUADPART?
 *   LARGE_INTEGER is a union that can be accessed as:
 *     - .QuadPart: 64-bit signed integer
 *     - .LowPart, .HighPart: Two 32-bit integers
 *   
 *   We use QuadPart for simplicity (direct 64-bit arithmetic).
 * 
 * TYPE CONVERSION:
 *   Cast to double for precise division (avoid integer division!)
 *   Integer division would give wrong results:
 *     152345 / 10000000 = 0 (integer division)
 *     152345.0 / 10000000.0 = 0.0152345 (correct!)
 ******************************************************************************/
double Timer::CalculateElapsedSeconds() const {
    // Calculate the difference in ticks (end - start)
    int64_t elapsedTicks = m_endTime.QuadPart - m_startTime.QuadPart;
    
    // Convert ticks to seconds
    // Cast to double to ensure floating-point division (not integer division!)
    double elapsedSeconds = static_cast<double>(elapsedTicks) / 
                           static_cast<double>(m_frequency.QuadPart);
    
    return elapsedSeconds;
}

} // namespace GPUBenchmark

/*******************************************************************************
 * END OF FILE: Timer.cpp
 * 
 * WHAT WE IMPLEMENTED:
 *   1. High-resolution CPU-side timing using Windows Performance Counter
 *   2. Multiple time unit conversions (ms, μs, ns, s)
 *   3. Human-readable duration formatting
 *   4. Timestamp generation for logging
 * 
 * KEY TAKEAWAYS:
 *   - QueryPerformanceCounter gives ~100 ns resolution on modern systems
 *   - Always cast to double before division to avoid integer truncation
 *   - Frequency is hardware-dependent but constant during execution
 *   - LARGE_INTEGER.QuadPart provides easy 64-bit access
 * 
 * PERFORMANCE CHARACTERISTICS:
 *   - QueryPerformanceCounter: ~100-200 CPU cycles (~0.00002 ms overhead)
 *   - Start/Stop operations are extremely lightweight
 *   - Safe to call thousands of times per second
 * 
 * TESTING THIS CODE:
 *   Timer t;
 *   t.Start();
 *   Sleep(100);  // Windows Sleep for 100 ms
 *   t.Stop();
 *   double elapsed = t.GetMilliseconds();
 *   // elapsed should be approximately 100 ms (within 1-2 ms)
 * 
 * NEXT FILES TO READ:
 *   - DeviceDiscovery.h/cpp : GPU and API detection
 *   - BenchmarkRunner.h/cpp : Orchestrating benchmarks with timing
 * 
 ******************************************************************************/
