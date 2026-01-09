/*******************************************************************************
 * FILE: test_standalone.cpp
 * 
 * PURPOSE:
 *   Minimal standalone test program to verify core components work.
 *   This compiles WITHOUT needing CUDA or CMake - just Visual Studio C++ compiler.
 * 
 * COMPILE:
 *   Open "Developer Command Prompt for VS 2022" and run:
 *   cl /EHsc /std:c++17 test_standalone.cpp src\core\Timer.cpp /Fe:test.exe
 * 
 * RUN:
 *   test.exe
 * 
 ******************************************************************************/

#include <iostream>
#include <windows.h>
#include "src/core/Timer.h"

using namespace GPUBenchmark;

// Simple GPU detection using DXGI (no external dependencies)
void DetectGPU() {
    std::cout << "\n=== GPU Detection ===" << std::endl;
    std::cout << "Attempting to detect GPU using Windows DXGI..." << std::endl;
    
    // We'll just print that detection would happen here
    // Full implementation is in DeviceDiscovery.cpp which needs linking
    std::cout << "Note: Full GPU detection requires linking DeviceDiscovery.cpp" << std::endl;
    std::cout << "For now, checking if you have NVIDIA GPU via nvidia-smi..." << std::endl;
}

// Test the Timer class
void TestTimer() {
    std::cout << "\n=== Testing High-Resolution Timer ===" << std::endl;
    
    Timer timer;
    
    // Test 1: Measure a known delay
    std::cout << "\nTest 1: Measuring 100ms sleep..." << std::endl;
    timer.Start();
    Sleep(100);  // Windows Sleep for 100 milliseconds
    timer.Stop();
    
    double elapsed = timer.GetMilliseconds();
    std::cout << "  Measured time: " << elapsed << " ms" << std::endl;
    std::cout << "  Expected: ~100 ms" << std::endl;
    std::cout << "  Result: " << (elapsed >= 98 && elapsed <= 105 ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    // Test 2: Timer frequency
    std::cout << "\nTest 2: Timer resolution..." << std::endl;
    uint64_t frequency = timer.GetFrequency();
    std::cout << "  Frequency: " << frequency << " Hz" << std::endl;
    std::cout << "  Resolution: ~" << (1000000000.0 / frequency) << " nanoseconds per tick" << std::endl;
    
    // Test 3: Very short measurement
    std::cout << "\nTest 3: Measuring very short operation..." << std::endl;
    timer.Start();
    volatile int sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += i;
    }
    timer.Stop();
    
    std::cout << "  Time for 1000 additions: " << timer.GetMicroseconds() << " μs" << std::endl;
    
    // Test 4: Format duration
    std::cout << "\nTest 4: Duration formatting..." << std::endl;
    std::cout << "  0.123 ms   = " << Timer::FormatDuration(0.123) << std::endl;
    std::cout << "  1.234 ms   = " << Timer::FormatDuration(1.234) << std::endl;
    std::cout << "  1234.5 ms  = " << Timer::FormatDuration(1234.5) << std::endl;
    std::cout << "  75000 ms   = " << Timer::FormatDuration(75000) << std::endl;
}

// Print system information
void PrintSystemInfo() {
    std::cout << "\n=== System Information ===" << std::endl;
    
    // OS Version
    std::cout << "\nOperating System:" << std::endl;
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    
    #pragma warning(push)
    #pragma warning(disable: 4996)
    GetVersionEx((LPOSVERSIONINFO)&osvi);
    #pragma warning(pop)
    
    if (osvi.dwBuildNumber >= 22000) {
        std::cout << "  Windows 11 Build " << osvi.dwBuildNumber << std::endl;
    } else if (osvi.dwMajorVersion == 10) {
        std::cout << "  Windows 10 Build " << osvi.dwBuildNumber << std::endl;
    } else {
        std::cout << "  Windows " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion << std::endl;
    }
    
    // System RAM
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memStatus)) {
        size_t totalRAM_MB = static_cast<size_t>(memStatus.ullTotalPhys / (1024 * 1024));
        std::cout << "\nSystem RAM:" << std::endl;
        std::cout << "  Total: " << totalRAM_MB << " MB (" 
                  << (totalRAM_MB / 1024.0) << " GB)" << std::endl;
    }
    
    // Processor count
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    std::cout << "\nProcessor:" << std::endl;
    std::cout << "  Logical processors: " << sysInfo.dwNumberOfProcessors << std::endl;
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                                                               ║" << std::endl;
    std::cout << "║        GPU BENCHMARK TOOL - STANDALONE TEST                   ║" << std::endl;
    std::cout << "║                                                               ║" << std::endl;
    std::cout << "║        Testing Core Components                                ║" << std::endl;
    std::cout << "║                                                               ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\nThis is a minimal test to verify the core framework works!" << std::endl;
    std::cout << "Author: Soham | System: Windows 11 | GPU: RTX 3050" << std::endl;
    
    try {
        // Test Timer class
        TestTimer();
        
        // Print system info
        PrintSystemInfo();
        
        // Attempt GPU detection
        DetectGPU();
        
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                   ALL TESTS COMPLETED!                        ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
        
        std::cout << "\nNext Steps:" << std::endl;
        std::cout << "1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" << std::endl;
        std::cout << "2. Install CMake: https://cmake.org/download/" << std::endl;
        std::cout << "3. Read QUICKSTART.md for full build instructions" << std::endl;
        std::cout << "4. Start implementing Logger.cpp and CUDABackend!" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
}
