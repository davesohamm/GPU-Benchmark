/*******************************************************************************
 * FILE: Timer.h
 * 
 * PURPOSE:
 *   High-resolution timing utility for measuring GPU and CPU performance.
 *   
 *   This class provides two timing mechanisms:
 *     1. CPU-side timing using Windows Performance Counter
 *     2. Wrapper for GPU-side timing (delegated to backends)
 * 
 * WHY TWO TIMING METHODS?
 *   GPU operations execute asynchronously from the CPU. If you use a normal
 *   stopwatch on the CPU, you'll measure only the time to SUBMIT work to
 *   the GPU, not the time for the GPU to actually EXECUTE it.
 * 
 *   Example of WRONG timing:
 *     auto start = std::chrono::now();
 *     LaunchGPUKernel();              // Returns immediately!
 *     auto end = std::chrono::now();  // Measures ~0.001 ms (just submission)
 * 
 *   Actual GPU execution might take 10 ms, but we measured 0.001 ms!
 * 
 * SOLUTION:
 *   Use GPU-specific timing events that record timestamps on the GPU timeline.
 * 
 * USAGE:
 *   // CPU timing (for host overhead, memory transfers)
 *   Timer cpuTimer;
 *   cpuTimer.Start();
 *   // ... CPU work ...
 *   cpuTimer.Stop();
 *   std::cout << "CPU time: " << cpuTimer.GetMilliseconds() << " ms\n";
 * 
 *   // GPU timing is handled by IComputeBackend::StartTimer/StopTimer
 *   // This class can be used for CPU-side measurements
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

#ifndef TIMER_H
#define TIMER_H

// Windows-specific high-resolution timer
// QueryPerformanceCounter is the most accurate timer on Windows
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#include <windows.h>          // For LARGE_INTEGER, QueryPerformanceCounter

// Standard library
#include <cstdint>  // For uint64_t
#include <string>   // For error messages

/*******************************************************************************
 * NAMESPACE: GPUBenchmark
 ******************************************************************************/
namespace GPUBenchmark {

/*******************************************************************************
 * CLASS: Timer
 * 
 * High-resolution timer using Windows Performance Counter.
 * 
 * WINDOWS PERFORMANCE COUNTER:
 *   - Resolution: Typically ~100 nanoseconds (10 million ticks per second)
 *   - Much better than: std::chrono (millisecond precision)
 *   - Based on: Hardware timestamp counter (TSC) or HPET
 * 
 * HOW IT WORKS:
 *   1. QueryPerformanceFrequency(): Get ticks per second (done once)
 *   2. QueryPerformanceCounter(): Get current tick count
 *   3. Delta = (EndTicks - StartTicks) / Frequency = Time in seconds
 * 
 * WHY NOT std::chrono?
 *   std::chrono::high_resolution_clock on Windows often has only ~1ms precision.
 *   QueryPerformanceCounter gives us ~0.0001ms precision!
 * 
 ******************************************************************************/
class Timer {
public:
    /**************************************************************************
     * CONSTRUCTOR
     * 
     * Initializes the timer and queries the performance counter frequency.
     * 
     * The frequency tells us how many "ticks" happen per second.
     * On modern systems, this is typically 10,000,000 (10 MHz) or higher.
     *************************************************************************/
    Timer();
    
    /**************************************************************************
     * DESTRUCTOR
     * 
     * Default destructor (nothing to clean up).
     *************************************************************************/
    ~Timer() = default;
    
    /**************************************************************************
     * TIMING CONTROL
     *************************************************************************/
    
    /**
     * Start the timer.
     * 
     * Records the current performance counter value as the start time.
     * 
     * THREAD SAFETY: Not thread-safe! Each thread should have its own Timer.
     * 
     * Example:
     *   Timer t;
     *   t.Start();
     *   // ... do work ...
     *   t.Stop();
     */
    void Start();
    
    /**
     * Stop the timer.
     * 
     * Records the current performance counter value as the end time.
     * 
     * After calling Stop(), you can retrieve the elapsed time with:
     *   - GetMilliseconds()
     *   - GetMicroseconds()
     *   - GetSeconds()
     * 
     * You can call Start() and Stop() multiple times to measure different
     * sections of code with the same timer object.
     */
    void Stop();
    
    /**
     * Reset the timer.
     * 
     * Clears start and end times. Useful for reusing the timer object.
     * 
     * Note: Not required before calling Start() again - Start() implicitly
     *       resets the timer.
     */
    void Reset();
    
    /**************************************************************************
     * TIME RETRIEVAL
     * 
     * Get the elapsed time in different units.
     * Must call Stop() before calling these!
     *************************************************************************/
    
    /**
     * Get elapsed time in milliseconds.
     * 
     * @return Time in milliseconds (ms)
     * 
     * Milliseconds are the standard unit for GPU benchmarking.
     * Most kernels take 0.1 - 100 ms to execute.
     * 
     * Example output: 15.234 ms
     */
    double GetMilliseconds() const;
    
    /**
     * Get elapsed time in microseconds.
     * 
     * @return Time in microseconds (μs)
     * 
     * Useful for very fast operations (< 1 ms).
     * 
     * 1 millisecond = 1000 microseconds
     * 
     * Example output: 15234.567 μs
     */
    double GetMicroseconds() const;
    
    /**
     * Get elapsed time in seconds.
     * 
     * @return Time in seconds (s)
     * 
     * Useful for long-running operations or total benchmark time.
     * 
     * Example output: 0.015234 s
     */
    double GetSeconds() const;
    
    /**
     * Get elapsed time in nanoseconds.
     * 
     * @return Time in nanoseconds (ns)
     * 
     * Maximum precision available. Useful for analyzing extremely
     * fast operations or measuring overhead.
     * 
     * 1 microsecond = 1000 nanoseconds
     * 
     * Note: Even though we return nanoseconds, the actual precision
     *       depends on the hardware timer (~100 ns on most systems).
     */
    uint64_t GetNanoseconds() const;
    
    /**************************************************************************
     * UTILITY FUNCTIONS
     *************************************************************************/
    
    /**
     * Get the timer frequency in ticks per second.
     * 
     * @return Frequency in Hz (ticks per second)
     * 
     * This tells you the resolution of the timer.
     * Higher = better precision.
     * 
     * Typical values:
     *   - Modern Intel/AMD: 10,000,000 Hz (10 MHz) = 100 ns resolution
     *   - Some systems: 3,579,545 Hz (~3.58 MHz) = 280 ns resolution
     * 
     * This is queried once at construction and doesn't change.
     */
    uint64_t GetFrequency() const;
    
    /**
     * Check if the timer is currently running.
     * 
     * @return true if Start() called but Stop() not yet called
     */
    bool IsRunning() const;
    
    /**************************************************************************
     * STATIC UTILITY FUNCTIONS
     *************************************************************************/
    
    /**
     * Get current timestamp in milliseconds since epoch.
     * 
     * @return Current time in milliseconds
     * 
     * Useful for timestamping benchmark results.
     * "Epoch" = January 1, 1970, 00:00:00 UTC (Unix time)
     * 
     * Example: 1704801600000 (some time in 2024)
     */
    static uint64_t GetTimestampMS();
    
    /**
     * Format a duration in milliseconds to human-readable string.
     * 
     * @param milliseconds  Duration to format
     * @return              Formatted string with appropriate unit
     * 
     * Examples:
     *   FormatDuration(0.123)    -> "123.0 μs"
     *   FormatDuration(1.234)    -> "1.234 ms"
     *   FormatDuration(1234.5)   -> "1.235 s"
     *   FormatDuration(75000)    -> "1m 15s"
     */
    static std::string FormatDuration(double milliseconds);
    
private:
    /**************************************************************************
     * PRIVATE MEMBER VARIABLES
     *************************************************************************/
    
    // Performance counter frequency (ticks per second)
    // This is the hardware-dependent timer resolution
    // Queried once at construction via QueryPerformanceFrequency()
    LARGE_INTEGER m_frequency;
    
    // Start time (in performance counter ticks)
    // Set by Start()
    LARGE_INTEGER m_startTime;
    
    // End time (in performance counter ticks)
    // Set by Stop()
    LARGE_INTEGER m_endTime;
    
    // Is the timer currently running? (Start() called, Stop() not yet called)
    bool m_running;
    
    /**************************************************************************
     * PRIVATE HELPER FUNCTIONS
     *************************************************************************/
    
    /**
     * Calculate elapsed time in seconds.
     * 
     * @return Elapsed time in seconds (double precision)
     * 
     * This is the fundamental calculation:
     *   ElapsedSeconds = (EndTicks - StartTicks) / Frequency
     * 
     * All other time units (ms, μs, ns) are derived from this.
     */
    double CalculateElapsedSeconds() const;
};

} // namespace GPUBenchmark

#endif // TIMER_H

/*******************************************************************************
 * END OF FILE: Timer.h
 * 
 * WHAT WE LEARNED:
 *   1. CPU and GPU timers serve different purposes
 *   2. QueryPerformanceCounter provides high precision on Windows
 *   3. GPU timing must use API-specific events (delegated to backends)
 *   4. Timer resolution affects measurement accuracy
 * 
 * KEY CONCEPTS:
 *   - Performance counter frequency (ticks per second)
 *   - Time = Ticks / Frequency
 *   - Asynchronous GPU execution requires special timing
 * 
 * NEXT FILE TO READ:
 *   - Timer.cpp : Implementation of these functions
 * 
 ******************************************************************************/
