/*******************************************************************************
 * FILE: DeviceDiscovery.cpp
 * 
 * PURPOSE:
 *   Implementation of system capability detection.
 *   
 *   This file contains the actual Windows API calls that detect:
 *     - GPUs using DXGI (DirectX Graphics Infrastructure)
 *     - CUDA availability using CUDA Runtime API
 *     - OpenCL availability using OpenCL API
 *     - DirectCompute availability using Direct3D 11
 *     - System information using Windows API
 * 
 * WINDOWS APIs USED:
 *   - DXGI: GPU enumeration (vendor-neutral)
 *   - CUDA Runtime: cudaGetDeviceCount(), cudaGetDeviceProperties()
 *   - OpenCL: clGetPlatformIDs(), clGetDeviceIDs()
 *   - Direct3D 11: D3D11CreateDevice()
 *   - Registry: CPU information
 *   - GlobalMemoryStatusEx: RAM information
 * 
 * ERROR HANDLING PHILOSOPHY:
 *   Detection failures are EXPECTED, not errors!
 *   Example: CUDA unavailable on AMD GPU is normal operation.
 *   We return availability status, not throw exceptions.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

#include "DeviceDiscovery.h"

// Windows API headers
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// DXGI for GPU enumeration (DirectX Graphics Infrastructure)
#include <dxgi.h>
#include <d3d11.h>

// Link with DXGI and D3D11 libraries
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")

// Standard library
#include <iostream>
#include <sstream>
#include <iomanip>

// CUDA headers (if CUDA backend is compiled)
// These are included conditionally - if CUDA toolkit not installed,
// CUDA detection will simply report "unavailable"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// OpenCL headers (if OpenCL backend is compiled)
#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

/*******************************************************************************
 * NAMESPACE: GPUBenchmark
 ******************************************************************************/
namespace GPUBenchmark {

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::Discover()
 * 
 * Main discovery function - orchestrates all detection operations.
 * 
 * WORKFLOW:
 *   1. Enumerate GPUs using DXGI
 *   2. Select primary GPU (usually first discrete GPU, or first GPU if no discrete)
 *   3. Check CUDA availability
 *   4. Check OpenCL availability  
 *   5. Check DirectCompute availability
 *   6. Query system information (OS, CPU, RAM)
 *   7. Return complete SystemCapabilities structure
 * 
 * PERFORMANCE:
 *   Takes 100-500 ms depending on number of GPUs and driver response time.
 *   This is acceptable because it's called once at startup.
 ******************************************************************************/
SystemCapabilities DeviceDiscovery::Discover() {
    SystemCapabilities caps;
    
    std::cout << "Detecting system capabilities..." << std::endl;
    
    // Step 1: Enumerate all GPUs in the system
    std::cout << "  Enumerating GPUs..." << std::endl;
    caps.gpus = EnumerateGPUs();
    
    if (caps.gpus.empty()) {
        std::cerr << "  WARNING: No GPUs detected! This is unusual." << std::endl;
        // Continue anyway - maybe integrated graphics not detected properly
    } else {
        std::cout << "  Found " << caps.gpus.size() << " GPU(s)" << std::endl;
        
        // Determine primary GPU
        // Strategy: Use first discrete GPU if available, otherwise use first GPU
        caps.primaryGPUIndex = 0;
        for (size_t i = 0; i < caps.gpus.size(); i++) {
            if (caps.gpus[i].vendor == "NVIDIA" || caps.gpus[i].vendor == "AMD") {
                // Found a discrete GPU - use this one
                caps.primaryGPUIndex = i;
                break;
            }
        }
        
        std::cout << "  Primary GPU: " << caps.GetPrimaryGPU().name << std::endl;
    }
    
    // Step 2: Check CUDA availability
    std::cout << "  Checking CUDA..." << std::endl;
    caps.cuda = CheckCUDAAvailability();
    if (caps.cuda.available) {
        std::cout << "    ✓ CUDA available: " << caps.cuda.version << std::endl;
    } else {
        std::cout << "    ✗ CUDA unavailable: " << caps.cuda.unavailableReason << std::endl;
    }
    
    // Step 3: Check OpenCL availability
    std::cout << "  Checking OpenCL..." << std::endl;
    caps.opencl = CheckOpenCLAvailability();
    if (caps.opencl.available) {
        std::cout << "    ✓ OpenCL available: " << caps.opencl.version << std::endl;
    } else {
        std::cout << "    ✗ OpenCL unavailable: " << caps.opencl.unavailableReason << std::endl;
    }
    
    // Step 4: Check DirectCompute availability
    std::cout << "  Checking DirectCompute..." << std::endl;
    caps.directCompute = CheckDirectComputeAvailability();
    if (caps.directCompute.available) {
        std::cout << "    ✓ DirectCompute available: " << caps.directCompute.version << std::endl;
    } else {
        std::cout << "    ✗ DirectCompute unavailable: " << caps.directCompute.unavailableReason << std::endl;
    }
    
    // Step 5: Query system information
    std::cout << "  Querying system information..." << std::endl;
    caps.operatingSystem = GetOperatingSystemInfo();
    caps.cpuName = GetCPUInfo();
    caps.systemRAMMB = GetSystemRAMMB();
    
    std::cout << "Discovery complete!" << std::endl;
    
    return caps;
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::EnumerateGPUs()
 * 
 * Enumerate all GPUs using DXGI (DirectX Graphics Infrastructure).
 * 
 * DXGI is a Windows API that provides vendor-neutral access to display adapters.
 * It's part of DirectX and available on all modern Windows systems.
 * 
 * WORKFLOW:
 *   1. Create DXGI factory (IDXGIFactory)
 *   2. Enumerate adapters (IDXGIAdapter) - each represents a GPU
 *   3. Query adapter description (name, memory, vendor ID, device ID)
 *   4. Fill GPUInfo structure
 * 
 * WHY DXGI?
 *   - Vendor-neutral (works for NVIDIA, AMD, Intel)
 *   - Always available on Windows
 *   - Provides standard PCI IDs
 *   - Doesn't require CUDA/OpenCL to be installed
 ******************************************************************************/
std::vector<GPUInfo> DeviceDiscovery::EnumerateGPUs() {
    std::vector<GPUInfo> gpus;
    
    // Create DXGI factory
    // The factory is used to enumerate display adapters
    IDXGIFactory* factory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGI factory (error 0x" << std::hex << hr << ")" << std::endl;
        return gpus;  // Return empty vector
    }
    
    // Enumerate adapters (GPUs)
    IDXGIAdapter* adapter = nullptr;
    UINT adapterIndex = 0;
    
    while (factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND) {
        // Query adapter description
        DXGI_ADAPTER_DESC adapterDesc;
        hr = adapter->GetDesc(&adapterDesc);
        
        if (SUCCEEDED(hr)) {
            GPUInfo gpuInfo;
            
            // Convert wide string (wchar_t*) to regular string
            // Windows uses wide strings for Unicode support
            int len = WideCharToMultiByte(CP_UTF8, 0, adapterDesc.Description, -1, nullptr, 0, nullptr, nullptr);
            if (len > 0) {
                std::vector<char> buffer(len);
                WideCharToMultiByte(CP_UTF8, 0, adapterDesc.Description, -1, buffer.data(), len, nullptr, nullptr);
                gpuInfo.name = buffer.data();
            }
            
            // Fill in GPU information
            gpuInfo.totalMemoryMB = static_cast<size_t>(adapterDesc.DedicatedVideoMemory / (1024 * 1024));
            gpuInfo.availableMemoryMB = gpuInfo.totalMemoryMB;  // DXGI doesn't provide available memory
            gpuInfo.vendorID = adapterDesc.VendorId;
            gpuInfo.deviceID = adapterDesc.DeviceId;
            gpuInfo.vendor = VendorIDToString(adapterDesc.VendorId);
            gpuInfo.isPrimaryGPU = (adapterIndex == 0);  // First adapter is usually primary
            
            // Driver version (format: AAA.BB.CCC.DDDD)
            LARGE_INTEGER driverVersion = adapterDesc.DriverVersion;
            std::ostringstream driverStr;
            driverStr << ((driverVersion.QuadPart >> 48) & 0xFFFF) << "."
                      << ((driverVersion.QuadPart >> 32) & 0xFFFF) << "."
                      << ((driverVersion.QuadPart >> 16) & 0xFFFF) << "."
                      << (driverVersion.QuadPart & 0xFFFF);
            gpuInfo.driverVersion = driverStr.str();
            
            gpus.push_back(gpuInfo);
        }
        
        // Release adapter (COM reference counting)
        adapter->Release();
        adapterIndex++;
    }
    
    // Release factory
    factory->Release();
    
    return gpus;
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::CheckCUDAAvailability()
 * 
 * Check if CUDA is available on this system.
 * 
 * CUDA REQUIREMENTS:
 *   1. NVIDIA GPU present
 *   2. NVIDIA drivers installed
 *   3. GPU has compute capability >= 3.0 (modern GPUs)
 *   4. CUDA runtime can initialize
 * 
 * HOW IT WORKS:
 *   Call cudaGetDeviceCount() from CUDA Runtime API.
 *   If it returns > 0 devices, CUDA is available.
 *   
 * CONDITIONAL COMPILATION:
 *   If CUDA toolkit not installed (no cuda_runtime.h), we simply
 *   report CUDA as unavailable. This allows the project to build
 *   even without CUDA.
 ******************************************************************************/
BackendAvailability DeviceDiscovery::CheckCUDAAvailability() {
#ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        // CUDA call failed
        std::string reason = "CUDA initialization failed: ";
        reason += cudaGetErrorString(error);
        return BackendAvailability(false, "", reason);
    }
    
    if (deviceCount == 0) {
        return BackendAvailability(false, "", "No CUDA-capable GPU detected (NVIDIA GPU required)");
    }
    
    // Get CUDA version
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    
    // Format version: 12000 -> "12.0"
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    std::ostringstream versionStr;
    versionStr << "CUDA " << major << "." << minor;
    
    // Get compute capability of first device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.major < 3) {
        std::ostringstream reason;
        reason << "GPU compute capability too old (" << prop.major << "." << prop.minor << "), need 3.0+";
        return BackendAvailability(false, "", reason.str());
    }
    
    versionStr << " (Compute Capability " << prop.major << "." << prop.minor << ")";
    
    return BackendAvailability(true, versionStr.str(), "");
#else
    // CUDA not compiled in
    return BackendAvailability(false, "", "CUDA support not compiled (USE_CUDA not defined)");
#endif
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::CheckOpenCLAvailability()
 * 
 * Check if OpenCL is available on this system.
 * 
 * OPENCL REQUIREMENTS:
 *   1. OpenCL runtime installed (ICD loader: OpenCL.dll)
 *   2. At least one OpenCL platform (vendor driver)
 *   3. At least one GPU device available
 * 
 * HOW IT WORKS:
 *   1. Call clGetPlatformIDs() to get available platforms
 *      (A "platform" is a vendor's OpenCL implementation)
 *   2. For each platform, call clGetDeviceIDs() to get GPU devices
 *   3. If any GPU device found, OpenCL is available
 * 
 * PLATFORMS ON YOUR SYSTEM:
 *   NVIDIA GPU: "NVIDIA CUDA" platform (OpenCL via CUDA)
 *   AMD GPU: "AMD Accelerated Parallel Processing" platform
 *   Intel GPU: "Intel(R) OpenCL" platform
 ******************************************************************************/
BackendAvailability DeviceDiscovery::CheckOpenCLAvailability() {
#ifdef USE_OPENCL
    // Get number of platforms
    cl_uint numPlatforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
    
    if (error != CL_SUCCESS || numPlatforms == 0) {
        return BackendAvailability(false, "", "No OpenCL platforms found (OpenCL runtime not installed)");
    }
    
    // Get platform IDs
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    // Look for GPU devices on any platform
    bool foundGPU = false;
    std::string platformName;
    std::string version;
    
    for (cl_uint i = 0; i < numPlatforms; i++) {
        // Get number of GPU devices on this platform
        cl_uint numDevices = 0;
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        
        if (error == CL_SUCCESS && numDevices > 0) {
            foundGPU = true;
            
            // Get platform name
            size_t nameSize = 0;
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &nameSize);
            std::vector<char> nameBuffer(nameSize);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameSize, nameBuffer.data(), nullptr);
            platformName = nameBuffer.data();
            
            // Get platform version
            size_t versionSize = 0;
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, nullptr, &versionSize);
            std::vector<char> versionBuffer(versionSize);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, versionSize, versionBuffer.data(), nullptr);
            version = versionBuffer.data();
            
            break;  // Found GPU, no need to check other platforms
        }
    }
    
    if (!foundGPU) {
        return BackendAvailability(false, "", "No OpenCL GPU devices found");
    }
    
    // Success!
    std::ostringstream versionStr;
    versionStr << version << " (" << platformName << ")";
    
    return BackendAvailability(true, versionStr.str(), "");
#else
    // OpenCL not compiled in
    return BackendAvailability(false, "", "OpenCL support not compiled (USE_OPENCL not defined)");
#endif
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::CheckDirectComputeAvailability()
 * 
 * Check if DirectCompute is available on this system.
 * 
 * DIRECTCOMPUTE REQUIREMENTS:
 *   1. Windows Vista or later (we require Windows 11)
 *   2. DirectX 11 or later
 *   3. GPU supports Feature Level 11.0+ (all modern GPUs do)
 * 
 * HOW IT WORKS:
 *   Call D3D11CreateDevice() to create a Direct3D 11 device.
 *   If successful, DirectCompute is available.
 * 
 * WHY RARELY UNAVAILABLE:
 *   DirectX 11 is built into Windows and supported by all modern GPUs.
 *   Even old GPUs have Feature Level 11.0 support.
 *   You'd need a VERY old GPU (pre-2009) for this to fail!
 ******************************************************************************/
BackendAvailability DeviceDiscovery::CheckDirectComputeAvailability() {
    // Try to create a D3D11 device
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL featureLevel;
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Use default adapter (primary GPU)
        D3D_DRIVER_TYPE_HARDWARE,   // Hardware acceleration (not software)
        nullptr,                    // No software rasterizer
        0,                          // No flags
        nullptr,                    // Try all feature levels
        0,                          // Number of feature levels
        D3D11_SDK_VERSION,          // SDK version
        &device,                    // Output: device pointer
        &featureLevel,              // Output: achieved feature level
        &context                    // Output: device context
    );
    
    if (FAILED(hr)) {
        std::ostringstream reason;
        reason << "D3D11CreateDevice failed (error 0x" << std::hex << hr << ")";
        return BackendAvailability(false, "", reason.str());
    }
    
    // Check feature level
    if (featureLevel < D3D_FEATURE_LEVEL_11_0) {
        std::ostringstream reason;
        reason << "GPU feature level too old (need 11.0+, have " 
               << static_cast<int>(featureLevel >> 12) << "." 
               << static_cast<int>((featureLevel >> 8) & 0xF) << ")";
        
        device->Release();
        context->Release();
        
        return BackendAvailability(false, "", reason.str());
    }
    
    // Success! DirectCompute is available
    device->Release();
    context->Release();
    
    // Format feature level
    std::ostringstream versionStr;
    versionStr << "DirectX " 
               << static_cast<int>(featureLevel >> 12) << "." 
               << static_cast<int>((featureLevel >> 8) & 0xF);
    
    return BackendAvailability(true, versionStr.str(), "");
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::GetOperatingSystemInfo()
 * 
 * Get Windows version information.
 * 
 * WINDOWS VERSION DETECTION:
 *   Windows 11: Version 10.0, Build 22000+
 *   Windows 10: Version 10.0, Build < 22000
 *   Windows 8.1: Version 6.3
 *   Windows 7: Version 6.1
 * 
 * NOTE: GetVersionEx() is deprecated, but works fine for our purposes.
 *       Modern approach uses RtlGetVersion() or registry query.
 ******************************************************************************/
std::string DeviceDiscovery::GetOperatingSystemInfo() {
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    
    // Note: GetVersionEx() is deprecated but still works
    // For production code, use RtlGetVersion() or registry
    #pragma warning(push)
    #pragma warning(disable: 4996)  // Disable deprecation warning
    GetVersionEx((LPOSVERSIONINFO)&osvi);
    #pragma warning(pop)
    
    std::ostringstream oss;
    
    // Windows 11 detection (Build 22000+)
    if (osvi.dwMajorVersion == 10 && osvi.dwBuildNumber >= 22000) {
        oss << "Windows 11";
    }
    // Windows 10
    else if (osvi.dwMajorVersion == 10) {
        oss << "Windows 10";
    }
    // Older versions
    else if (osvi.dwMajorVersion == 6 && osvi.dwMinorVersion == 3) {
        oss << "Windows 8.1";
    }
    else if (osvi.dwMajorVersion == 6 && osvi.dwMinorVersion == 2) {
        oss << "Windows 8";
    }
    else if (osvi.dwMajorVersion == 6 && osvi.dwMinorVersion == 1) {
        oss << "Windows 7";
    }
    else {
        oss << "Windows (Unknown)";
    }
    
    oss << " Build " << osvi.dwBuildNumber;
    
    return oss.str();
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::GetCPUInfo()
 * 
 * Get CPU name from Windows registry.
 * 
 * REGISTRY PATH:
 *   HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0
 *   Value: ProcessorNameString
 * 
 * EXAMPLE OUTPUT:
 *   "AMD Ryzen 7 4800H with Radeon Graphics"
 *   "Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz"
 ******************************************************************************/
std::string DeviceDiscovery::GetCPUInfo() {
    HKEY hKey;
    LONG result = RegOpenKeyEx(
        HKEY_LOCAL_MACHINE,
        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
        0,
        KEY_READ,
        &hKey
    );
    
    if (result != ERROR_SUCCESS) {
        return "Unknown CPU";
    }
    
    CHAR cpuName[256];
    DWORD bufferSize = sizeof(cpuName);
    result = RegQueryValueExA(
        hKey,
        "ProcessorNameString",
        nullptr,
        nullptr,
        (LPBYTE)cpuName,
        &bufferSize
    );
    
    RegCloseKey(hKey);
    
    if (result == ERROR_SUCCESS) {
        return std::string(cpuName);
    }
    
    return "Unknown CPU";
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::GetSystemRAMMB()
 * 
 * Get total system RAM in megabytes.
 * 
 * USES: GlobalMemoryStatusEx() Windows API
 * 
 * RETURNS:
 *   Total physical memory in MB
 * 
 * EXAMPLE:
 *   16 GB RAM = 16384 MB
 ******************************************************************************/
size_t DeviceDiscovery::GetSystemRAMMB() {
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&memStatus)) {
        // Convert bytes to megabytes
        return static_cast<size_t>(memStatus.ullTotalPhys / (1024 * 1024));
    }
    
    return 0;  // Failed to query
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::VendorIDToString()
 * 
 * Convert PCI vendor ID to human-readable name.
 * 
 * STANDARD PCI VENDOR IDS:
 *   0x10DE = NVIDIA Corporation
 *   0x1002 = Advanced Micro Devices (AMD)
 *   0x8086 = Intel Corporation
 *   0x1414 = Microsoft (for software adapters)
 * 
 * These are standardized IDs assigned by PCI-SIG (PCI Special Interest Group).
 ******************************************************************************/
std::string DeviceDiscovery::VendorIDToString(unsigned int vendorID) {
    switch (vendorID) {
        case GPUInfo::VENDOR_ID_NVIDIA:
            return "NVIDIA";
        case GPUInfo::VENDOR_ID_AMD:
            return "AMD";
        case GPUInfo::VENDOR_ID_INTEL:
            return "Intel";
        case 0x1414:
            return "Microsoft";
        default:
            std::ostringstream oss;
            oss << "Unknown (0x" << std::hex << vendorID << ")";
            return oss.str();
    }
}

/*******************************************************************************
 * FUNCTION: DeviceDiscovery::PrintCapabilities()
 * 
 * Pretty-print system capabilities to console.
 * 
 * EXAMPLE OUTPUT:
 *   ============================================
 *   System Capabilities
 *   ============================================
 *   Operating System: Windows 11 Build 22000
 *   CPU: AMD Ryzen 7 4800H with Radeon Graphics
 *   System RAM: 16384 MB (16 GB)
 *   
 *   GPU: NVIDIA GeForce RTX 3050 Laptop GPU
 *     Vendor: NVIDIA
 *     Memory: 4096 MB
 *     Driver: 546.12.0.0
 *   
 *   Compute Backends:
 *     ✓ CUDA: CUDA 12.0 (Compute Capability 8.6)
 *     ✓ OpenCL: OpenCL 3.0 (NVIDIA CUDA)
 *     ✓ DirectCompute: DirectX 11.0
 *   ============================================
 ******************************************************************************/
void DeviceDiscovery::PrintCapabilities(const SystemCapabilities& caps) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "System Capabilities" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // System information
    std::cout << "Operating System: " << caps.operatingSystem << std::endl;
    std::cout << "CPU: " << caps.cpuName << std::endl;
    std::cout << "System RAM: " << caps.systemRAMMB << " MB (" 
              << std::fixed << std::setprecision(1) 
              << (caps.systemRAMMB / 1024.0) << " GB)" << std::endl;
    
    // GPU information
    if (!caps.gpus.empty()) {
        const GPUInfo& gpu = caps.GetPrimaryGPU();
        std::cout << "\nGPU: " << gpu.name << std::endl;
        std::cout << "  Vendor: " << gpu.vendor << std::endl;
        std::cout << "  Memory: " << gpu.totalMemoryMB << " MB" << std::endl;
        std::cout << "  Driver: " << gpu.driverVersion << std::endl;
    }
    
    // Compute backends
    std::cout << "\nCompute Backends:" << std::endl;
    
    if (caps.cuda.available) {
        std::cout << "  ✓ CUDA: " << caps.cuda.version << std::endl;
    } else {
        std::cout << "  ✗ CUDA: " << caps.cuda.unavailableReason << std::endl;
    }
    
    if (caps.opencl.available) {
        std::cout << "  ✓ OpenCL: " << caps.opencl.version << std::endl;
    } else {
        std::cout << "  ✗ OpenCL: " << caps.opencl.unavailableReason << std::endl;
    }
    
    if (caps.directCompute.available) {
        std::cout << "  ✓ DirectCompute: " << caps.directCompute.version << std::endl;
    } else {
        std::cout << "  ✗ DirectCompute: " << caps.directCompute.unavailableReason << std::endl;
    }
    
    std::cout << "============================================\n" << std::endl;
}

} // namespace GPUBenchmark

/*******************************************************************************
 * END OF FILE: DeviceDiscovery.cpp
 * 
 * WHAT WE IMPLEMENTED:
 *   1. GPU enumeration using DXGI (vendor-neutral)
 *   2. CUDA detection using CUDA Runtime API
 *   3. OpenCL detection using OpenCL API
 *   4. DirectCompute detection using Direct3D 11
 *   5. System information queries (OS, CPU, RAM)
 * 
 * KEY WINDOWS APIs:
 *   - CreateDXGIFactory, IDXGIAdapter: GPU enumeration
 *   - D3D11CreateDevice: DirectCompute availability
 *   - RegOpenKeyEx: Registry access for CPU info
 *   - GlobalMemoryStatusEx: RAM query
 *   - GetVersionEx: OS version
 * 
 * TESTING THIS CODE:
 *   auto caps = DeviceDiscovery::Discover();
 *   DeviceDiscovery::PrintCapabilities(caps);
 * 
 * EXPECTED OUTPUT (Your RTX 3050 System):
 *   GPU: NVIDIA GeForce RTX 3050 Laptop GPU
 *   ✓ CUDA: CUDA 12.0 (Compute Capability 8.6)
 *   ✓ OpenCL: OpenCL 3.0 (NVIDIA CUDA)
 *   ✓ DirectCompute: DirectX 11.0
 * 
 * NEXT FILES TO READ:
 *   - Logger.h/cpp : Results logging and export
 *   - BenchmarkRunner.h/cpp : Orchestrating benchmarks
 * 
 ******************************************************************************/
