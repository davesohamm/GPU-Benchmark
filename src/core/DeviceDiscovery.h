/*******************************************************************************
 * FILE: DeviceDiscovery.h
 * 
 * PURPOSE:
 *   Detect available GPUs and GPU compute APIs at runtime.
 *   
 *   This is how the application adapts to different hardware WITHOUT
 *   recompilation. The same executable works on:
 *     - NVIDIA GPU: Enables CUDA, OpenCL, DirectCompute
 *     - AMD GPU: Enables OpenCL, DirectCompute (no CUDA)
 *     - Intel GPU: Enables OpenCL, DirectCompute (no CUDA)
 * 
 * KEY CONCEPT: Runtime Detection vs Compile-Time Configuration
 *   
 *   WRONG APPROACH (Compile-time):
 *     #ifdef NVIDIA_GPU
 *         Use CUDA
 *     #elif AMD_GPU
 *         Use OpenCL
 *     #endif
 *   
 *   Problem: Need different executables for each GPU!
 * 
 *   RIGHT APPROACH (Runtime):
 *     Detect GPU at startup
 *     Enable appropriate backends
 *     Same .exe works on all systems!
 * 
 * WHAT THIS DOES:
 *   1. Enumerate all GPUs in the system
 *   2. Check which compute APIs are available (CUDA, OpenCL, DirectCompute)
 *   3. Query GPU properties (memory, compute capability, etc.)
 *   4. Provide friendly error messages for unavailable APIs
 * 
 * EXAMPLE OUTPUT:
 *   "Detected: NVIDIA GeForce RTX 3050 Laptop GPU"
 *   "  CUDA: Available (Compute Capability 8.6)"
 *   "  OpenCL: Available (OpenCL 3.0)"
 *   "  DirectCompute: Available (DirectX 11.0)"
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

#ifndef DEVICE_DISCOVERY_H
#define DEVICE_DISCOVERY_H

// Core framework
#include "IComputeBackend.h"  // For BackendType enum

// Standard library
#include <string>   // For device names, error messages
#include <vector>   // For lists of devices
#include <map>      // For backend availability mapping

/*******************************************************************************
 * NAMESPACE: GPUBenchmark
 ******************************************************************************/
namespace GPUBenchmark {

/*******************************************************************************
 * STRUCT: GPUInfo
 * 
 * Information about a detected GPU.
 * 
 * A system can have multiple GPUs:
 *   - Laptop: Integrated (Intel) + Discrete (NVIDIA RTX 3050)
 *   - Desktop: Multiple discrete GPUs for multi-GPU rendering
 * 
 * This structure holds info about one GPU.
 ******************************************************************************/
struct GPUInfo {
    std::string name;              // GPU name (e.g., "NVIDIA GeForce RTX 3050")
    std::string vendor;            // Vendor (NVIDIA, AMD, Intel, etc.)
    size_t totalMemoryMB;          // Total GPU memory in megabytes
    size_t availableMemoryMB;      // Currently available memory
    std::string driverVersion;     // Driver version string
    bool isPrimaryGPU;             // Is this the main GPU? (for multi-GPU systems)
    
    // Vendor IDs (standard PCI vendor IDs)
    // These are industry-standard identifiers
    static constexpr unsigned int VENDOR_ID_NVIDIA = 0x10DE;
    static constexpr unsigned int VENDOR_ID_AMD    = 0x1002;
    static constexpr unsigned int VENDOR_ID_INTEL  = 0x8086;
    
    unsigned int vendorID;         // PCI vendor ID
    unsigned int deviceID;         // PCI device ID
    
    // Constructor
    GPUInfo()
        : totalMemoryMB(0)
        , availableMemoryMB(0)
        , isPrimaryGPU(false)
        , vendorID(0)
        , deviceID(0)
    {}
};

/*******************************************************************************
 * STRUCT: BackendAvailability
 * 
 * Information about whether a specific backend is available.
 * 
 * For each backend (CUDA, OpenCL, DirectCompute), we track:
 *   - Is it available?
 *   - If not, why not? (for user-friendly error messages)
 *   - Version information
 ******************************************************************************/
struct BackendAvailability {
    bool available;                // Is this backend available?
    std::string version;           // Version string (e.g., "CUDA 12.0", "OpenCL 3.0")
    std::string unavailableReason; // Why unavailable? (e.g., "No NVIDIA GPU detected")
    
    // Constructor
    BackendAvailability()
        : available(false)
    {}
    
    BackendAvailability(bool avail, const std::string& ver = "", const std::string& reason = "")
        : available(avail)
        , version(ver)
        , unavailableReason(reason)
    {}
};

/*******************************************************************************
 * STRUCT: SystemCapabilities
 * 
 * Complete system capability information.
 * 
 * This is the result of the discovery process - everything the application
 * needs to know about the system's GPU compute capabilities.
 ******************************************************************************/
struct SystemCapabilities {
    // List of all detected GPUs
    std::vector<GPUInfo> gpus;
    
    // Primary GPU (the one we'll use for benchmarking)
    // Index into 'gpus' vector
    size_t primaryGPUIndex;
    
    // Backend availability
    BackendAvailability cuda;
    BackendAvailability opencl;
    BackendAvailability directCompute;
    
    // System information
    std::string operatingSystem;   // "Windows 11"
    std::string cpuName;            // "AMD Ryzen 7 4800H"
    size_t systemRAMMB;            // System RAM in megabytes
    
    // Constructor
    SystemCapabilities()
        : primaryGPUIndex(0)
        , systemRAMMB(0)
    {}
    
    // Helper: Get primary GPU info
    const GPUInfo& GetPrimaryGPU() const {
        if (gpus.empty()) {
            static GPUInfo dummy;
            return dummy;
        }
        return gpus[primaryGPUIndex];
    }
    
    // Helper: Check if any backend is available
    bool HasAnyBackend() const {
        return cuda.available || opencl.available || directCompute.available;
    }
    
    // Helper: Get list of available backend names
    std::vector<std::string> GetAvailableBackendNames() const {
        std::vector<std::string> names;
        if (cuda.available) names.push_back("CUDA");
        if (opencl.available) names.push_back("OpenCL");
        if (directCompute.available) names.push_back("DirectCompute");
        return names;
    }
};

/*******************************************************************************
 * CLASS: DeviceDiscovery
 * 
 * Static class (all methods are static) that performs system capability
 * detection.
 * 
 * WHY STATIC?
 *   Discovery is a one-time operation at application startup.
 *   No need to create instances - just call DeviceDiscovery::Discover().
 * 
 * USAGE:
 *   // At application startup
 *   SystemCapabilities caps = DeviceDiscovery::Discover();
 *   
 *   if (!caps.HasAnyBackend()) {
 *       std::cerr << "Error: No GPU compute backends available!" << std::endl;
 *       return -1;
 *   }
 *   
 *   std::cout << "Detected GPU: " << caps.GetPrimaryGPU().name << std::endl;
 *   
 *   if (caps.cuda.available) {
 *       std::cout << "  CUDA: " << caps.cuda.version << std::endl;
 *   } else {
 *       std::cout << "  CUDA unavailable: " << caps.cuda.unavailableReason << std::endl;
 *   }
 * 
 ******************************************************************************/
class DeviceDiscovery {
public:
    /**************************************************************************
     * MAIN DISCOVERY FUNCTION
     *************************************************************************/
    
    /**
     * Discover all system capabilities.
     * 
     * This is the main entry point. It performs complete system detection:
     *   1. Enumerate GPUs
     *   2. Check CUDA availability
     *   3. Check OpenCL availability
     *   4. Check DirectCompute availability
     *   5. Query system information
     * 
     * @return SystemCapabilities structure with all detected information
     * 
     * IMPORTANT: This function is expensive (may take 100-500 ms)!
     *            Call it once at startup, not repeatedly.
     * 
     * WHY SO SLOW?
     *   - GPU detection requires querying drivers
     *   - API initialization attempts (CUDA, OpenCL) take time
     *   - System information queries access hardware
     * 
     * Example:
     *   auto caps = DeviceDiscovery::Discover();
     *   // Now 'caps' contains everything we need to know
     */
    static SystemCapabilities Discover();
    
    /**************************************************************************
     * INDIVIDUAL DETECTION FUNCTIONS
     * 
     * These are called by Discover() but can also be used individually
     * if you only need specific information.
     *************************************************************************/
    
    /**
     * Enumerate all GPUs in the system.
     * 
     * @return Vector of GPUInfo structures (one per GPU)
     * 
     * Uses Windows DXGI (DirectX Graphics Infrastructure) to enumerate
     * display adapters.
     * 
     * DXGI is part of DirectX and available on all Windows systems.
     * It provides vendor-neutral GPU enumeration.
     * 
     * Multi-GPU systems will return multiple entries:
     *   - Laptop: Integrated + Discrete
     *   - Desktop: Multiple discrete GPUs
     *   - SLI/CrossFire: Multiple identical GPUs
     */
    static std::vector<GPUInfo> EnumerateGPUs();
    
    /**
     * Check if CUDA is available.
     * 
     * @return BackendAvailability with CUDA status
     * 
     * CUDA is available only if:
     *   1. NVIDIA GPU present
     *   2. CUDA-capable GPU (compute capability >= 3.0)
     *   3. CUDA driver installed
     *   4. CUDA runtime can initialize
     * 
     * If unavailable, the 'unavailableReason' field explains why:
     *   - "No NVIDIA GPU detected"
     *   - "CUDA driver not installed"
     *   - "GPU compute capability too old (need 3.0+)"
     */
    static BackendAvailability CheckCUDAAvailability();
    
    /**
     * Check if OpenCL is available.
     * 
     * @return BackendAvailability with OpenCL status
     * 
     * OpenCL is available if:
     *   1. OpenCL runtime installed (ICD loader)
     *   2. At least one OpenCL platform detected
     *   3. At least one GPU device available
     * 
     * OpenCL is vendor-neutral and works on:
     *   - NVIDIA GPUs (via CUDA backend)
     *   - AMD GPUs (native OpenCL)
     *   - Intel GPUs (native OpenCL)
     * 
     * On NVIDIA systems, OpenCL is provided by CUDA drivers.
     */
    static BackendAvailability CheckOpenCLAvailability();
    
    /**
     * Check if DirectCompute is available.
     * 
     * @return BackendAvailability with DirectCompute status
     * 
     * DirectCompute (Direct3D 11 Compute Shaders) is available if:
     *   1. Windows Vista or later (we require Windows 11)
     *   2. DirectX 11 or later
     *   3. GPU supports Feature Level 11.0+
     * 
     * DirectCompute works on all modern GPUs from all vendors.
     * It's part of DirectX and integrated into Windows.
     * 
     * Rarely unavailable on modern systems!
     */
    static BackendAvailability CheckDirectComputeAvailability();
    
    /**************************************************************************
     * SYSTEM INFORMATION
     *************************************************************************/
    
    /**
     * Get operating system information.
     * 
     * @return OS name and version (e.g., "Windows 11 Build 22000")
     * 
     * Uses Windows API to query OS version.
     */
    static std::string GetOperatingSystemInfo();
    
    /**
     * Get CPU information.
     * 
     * @return CPU name (e.g., "AMD Ryzen 7 4800H with Radeon Graphics")
     * 
     * Queries Windows registry for CPU brand string.
     */
    static std::string GetCPUInfo();
    
    /**
     * Get system RAM size.
     * 
     * @return Total system RAM in megabytes
     * 
     * Uses GlobalMemoryStatusEx() to query available physical memory.
     */
    static size_t GetSystemRAMMB();
    
    /**************************************************************************
     * UTILITY FUNCTIONS
     *************************************************************************/
    
    /**
     * Convert vendor ID to vendor name.
     * 
     * @param vendorID  PCI vendor ID
     * @return          Vendor name string
     * 
     * Maps standard PCI vendor IDs to names:
     *   0x10DE -> "NVIDIA"
     *   0x1002 -> "AMD"
     *   0x8086 -> "Intel"
     */
    static std::string VendorIDToString(unsigned int vendorID);
    
    /**
     * Print system capabilities to console.
     * 
     * @param caps  SystemCapabilities to print
     * 
     * Formats and displays all detected capabilities in a user-friendly way.
     * 
     * Example output:
     *   === System Capabilities ===
     *   OS: Windows 11 Build 22000
     *   CPU: AMD Ryzen 7 4800H
     *   RAM: 16384 MB
     *   
     *   GPU: NVIDIA GeForce RTX 3050 Laptop GPU
     *     Memory: 4096 MB
     *     Driver: 546.12
     *   
     *   Available Backends:
     *     ✓ CUDA 12.0
     *     ✓ OpenCL 3.0
     *     ✓ DirectCompute (DirectX 11.0)
     */
    static void PrintCapabilities(const SystemCapabilities& caps);
    
private:
    // Private constructor - this is a static-only class
    DeviceDiscovery() = delete;
    
    // Helper functions for DXGI enumeration
    static GPUInfo QueryGPUInfoDXGI(void* adapter);  // adapter is IDXGIAdapter*
};

} // namespace GPUBenchmark

#endif // DEVICE_DISCOVERY_H

/*******************************************************************************
 * END OF FILE: DeviceDiscovery.h
 * 
 * WHAT WE LEARNED:
 *   1. Runtime detection enables hardware-agnostic executables
 *   2. Multiple detection strategies (DXGI, CUDA API, OpenCL API, D3D11)
 *   3. Friendly error messages guide users when backends unavailable
 *   4. PCI vendor IDs provide standard hardware identification
 * 
 * KEY CONCEPTS:
 *   - Runtime vs compile-time configuration
 *   - DXGI for vendor-neutral GPU enumeration
 *   - Backend-specific detection (CUDA requires NVIDIA, etc.)
 *   - Graceful degradation (use what's available)
 * 
 * WHY THIS MATTERS FOR INTERVIEWS:
 *   Shows understanding of:
 *     - Cross-platform/cross-hardware design
 *     - Professional error handling
 *     - Windows GPU architecture (DXGI, drivers)
 *     - Defensive programming (check availability before use)
 * 
 * NEXT FILE TO READ:
 *   - DeviceDiscovery.cpp : Implementation of detection logic
 * 
 ******************************************************************************/
