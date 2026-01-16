/********************************************************************************
 * @file    OpenCLBackend.h
 * @brief   OpenCL Backend Implementation for GPU Compute Benchmarking
 * 
 * @details This file provides a complete OpenCL backend that implements the
 *          IComputeBackend interface. It supports:
 *          - Multi-platform GPU enumeration (NVIDIA, AMD, Intel)
 *          - OpenCL context and command queue management
 *          - Runtime kernel compilation from source
 *          - Buffer management and data transfer
 *          - Event-based GPU timing
 *          - Error handling and device capability queries
 * 
 * @note    OpenCL Backend Implementation Strategy:
 *          - Uses OpenCL 1.2 API for maximum compatibility
 *          - Kernels compiled at runtime from .cl source files
 *          - Supports all major GPU vendors (cross-platform)
 *          - Event-based timing for accurate performance measurement
 * 
 * @architecture
 *          [OpenCLBackend] implements [IComputeBackend]
 *                  |
 *                  +-- Platform/Device Enumeration
 *                  +-- Context Management
 *                  +-- Command Queue (out-of-order, profiling enabled)
 *                  +-- Program/Kernel Compilation
 *                  +-- Buffer Management
 *                  +-- Event-based Timing
 * 
 * @performance
 *          Expected performance compared to CUDA:
 *          - Memory bandwidth: 95-100% (nearly identical)
 *          - Compute throughput: 90-95% (slightly lower due to compiler differences)
 *          - Latency: +5-10% overhead for kernel dispatch
 * 
 * @compatibility
 *          - Works on NVIDIA GPUs (via NVIDIA OpenCL ICD)
 *          - Works on AMD GPUs (via AMD ROCm OpenCL)
 *          - Works on Intel GPUs (via Intel OpenCL Runtime)
 *          - Cross-platform: Windows, Linux, macOS
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#pragma once

#define CL_TARGET_OPENCL_VERSION 120  // OpenCL 1.2 for maximum compatibility
#include <CL/cl.h>

#include "../../core/IComputeBackend.h"
#include "../../core/Logger.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace GPUBenchmark {

/********************************************************************************
 * @class OpenCLBackend
 * @brief Complete OpenCL implementation of IComputeBackend interface
 * 
 * @details This class provides a full-featured OpenCL backend that can:
 *          - Enumerate all OpenCL platforms and devices
 *          - Create contexts and command queues
 *          - Compile kernels from OpenCL C source
 *          - Allocate/free GPU memory buffers
 *          - Transfer data to/from GPU
 *          - Execute kernels with configurable work sizes
 *          - Measure execution time using OpenCL events
 * 
 * @thread_safety
 *          Not thread-safe. Each thread should create its own backend instance.
 * 
 * @resource_management
 *          Uses RAII pattern - all OpenCL resources released in Shutdown()
 ********************************************************************************/
class OpenCLBackend : public IComputeBackend {
public:
    //==========================================================================
    // CONSTRUCTOR & DESTRUCTOR
    //==========================================================================
    
    /**
     * @brief Construct OpenCL backend
     * @note Does not initialize OpenCL - call Initialize() explicitly
     */
    OpenCLBackend();
    
    /**
     * @brief Destructor - ensures cleanup
     */
    ~OpenCLBackend() override;

    //==========================================================================
    // ICOMPUTEBACKEND INTERFACE IMPLEMENTATION
    //==========================================================================
    
    /**
     * @brief Initialize OpenCL backend
     * @return true if initialization successful, false otherwise
     * 
     * @details Initialization sequence:
     *          1. Enumerate OpenCL platforms
     *          2. Select best available device (prefer GPU over CPU)
     *          3. Create OpenCL context
     *          4. Create command queue (out-of-order, profiling enabled)
     *          5. Query device capabilities
     *          6. Initialize timing events
     */
    bool Initialize() override;
    
    /**
     * @brief Shutdown OpenCL backend and release all resources
     * 
     * @details Cleanup sequence:
     *          1. Release all compiled programs/kernels
     *          2. Release all buffers
     *          3. Release command queue
     *          4. Release context
     *          5. Clear device info
     */
    void Shutdown() override;
    
    /**
     * @brief Get backend type
     * @return BackendType::OpenCL
     */
    BackendType GetBackendType() const override { return BackendType::OpenCL; }
    
    /**
     * @brief Get backend name
     * @return "OpenCL"
     */
    std::string GetBackendName() const override { return "OpenCL"; }
    
    /**
     * @brief Get device information
     * @return DeviceInfo structure with GPU details
     */
    DeviceInfo GetDeviceInfo() const override { return m_deviceInfo; }

    //==========================================================================
    // MEMORY MANAGEMENT
    //==========================================================================
    
    /**
     * @brief Allocate GPU memory
     * @param sizeBytes Size in bytes
     * @return Pointer to cl_mem buffer (cast to void*)
     * 
     * @note Returns nullptr on failure
     * @note Buffer is read-write by default
     */
    void* AllocateMemory(size_t sizeBytes) override;
    
    /**
     * @brief Free GPU memory
     * @param ptr Pointer to cl_mem buffer
     */
    void FreeMemory(void* ptr) override;
    
    /**
     * @brief Copy data from host to device
     * @param dst Device pointer (cl_mem)
     * @param src Host pointer
     * @param sizeBytes Size in bytes
     * 
     * @note Blocking transfer
     */
    void CopyHostToDevice(void* dst, const void* src, size_t sizeBytes) override;
    
    /**
     * @brief Copy data from device to host
     * @param dst Host pointer
     * @param src Device pointer (cl_mem)
     * @param sizeBytes Size in bytes
     * 
     * @note Blocking transfer
     */
    void CopyDeviceToHost(void* dst, const void* src, size_t sizeBytes) override;

    //==========================================================================
    // SYNCHRONIZATION & TIMING
    //==========================================================================
    
    /**
     * @brief Synchronize GPU - wait for all queued operations to complete
     */
    void Synchronize() override;
    
    /**
     * @brief Start GPU timer
     * @note Records timing event before subsequent kernel execution
     */
    void StartTimer() override;
    
    /**
     * @brief Stop GPU timer
     * @note Records timing event after kernel execution
     */
    void StopTimer() override;
    
    /**
     * @brief Get elapsed time between StartTimer/StopTimer
     * @return Elapsed time in milliseconds
     */
    double GetElapsedTime() override;
    
    /**
     * @brief Execute kernel with parameters (IComputeBackend interface)
     * @param kernelName Name of kernel to execute
     * @param params Kernel parameters
     * @return true if successful
     * 
     * @note This is a required method from IComputeBackend.
     *       OpenCL backend uses its own ExecuteKernel method internally.
     */
    bool ExecuteKernel(const std::string& kernelName, const KernelParams& params) override;
    
    /**
     * @brief Check if backend is available
     * @return true if OpenCL runtime available
     */
    bool IsAvailable() const override;
    
    /**
     * @brief Get last error message
     * @return Error string
     */
    std::string GetLastError() const override;

    //==========================================================================
    // OPENCL-SPECIFIC METHODS
    //==========================================================================
    
    /**
     * @brief Compile OpenCL kernel from source code
     * @param kernelName Unique identifier for this kernel
     * @param sourceCode OpenCL C source code
     * @param buildOptions Additional compiler options (e.g., "-cl-fast-relaxed-math")
     * @return true if compilation successful, false otherwise
     * 
     * @details Compiles and caches kernel for later execution.
     *          Build options typically include:
     *          - "-cl-fast-relaxed-math" for faster math
     *          - "-cl-mad-enable" for multiply-add fusion
     *          - "-cl-no-signed-zeros"
     * 
     * @note Kernel cached internally - subsequent calls with same name return existing kernel
     */
    bool CompileKernel(const std::string& kernelName, 
                       const std::string& sourceCode,
                       const std::string& buildOptions = "");
    
    /**
     * @brief Execute a compiled OpenCL kernel
     * @param kernelName Name of previously compiled kernel
     * @param globalWorkSize Array of global work sizes (1D, 2D, or 3D)
     * @param localWorkSize Array of local work sizes (nullptr for automatic selection)
     * @param workDim Number of dimensions (1, 2, or 3)
     * @return true if execution successful, false otherwise
     * 
     * @note Must set kernel arguments before calling this method
     */
    bool ExecuteKernel(const std::string& kernelName,
                       const size_t* globalWorkSize,
                       const size_t* localWorkSize,
                       size_t workDim);
    
    /**
     * @brief Set kernel argument
     * @param kernelName Name of compiled kernel
     * @param argIndex Argument index (0-based)
     * @param argSize Size of argument in bytes
     * @param argValue Pointer to argument value
     * @return true if successful, false otherwise
     * 
     * @example
     *     cl_mem buffer = (cl_mem)AllocateMemory(1024);
     *     SetKernelArg("myKernel", 0, sizeof(cl_mem), &buffer);
     *     int n = 256;
     *     SetKernelArg("myKernel", 1, sizeof(int), &n);
     */
    bool SetKernelArg(const std::string& kernelName, 
                      unsigned int argIndex,
                      size_t argSize,
                      const void* argValue);

private:
    //==========================================================================
    // PRIVATE HELPER METHODS
    //==========================================================================
    
    /**
     * @brief Enumerate all OpenCL platforms and select best one
     * @return true if platform selected successfully
     */
    bool SelectBestPlatform();
    
    /**
     * @brief Select best available device from platform
     * @return true if device selected successfully
     * 
     * @details Selection priority:
     *          1. GPU devices over CPU devices
     *          2. Highest compute units
     *          3. Highest clock frequency
     */
    bool SelectBestDevice();
    
    /**
     * @brief Query and populate device information
     */
    void QueryDeviceInfo();
    
    /**
     * @brief Check OpenCL error and log if failed
     * @param err OpenCL error code
     * @param operation Description of operation
     * @return true if no error, false otherwise
     */
    bool CheckCLError(cl_int err, const std::string& operation);
    
    /**
     * @brief Get human-readable error string from CL error code
     * @param err OpenCL error code
     * @return Error string
     */
    std::string GetCLErrorString(cl_int err);

    //==========================================================================
    // MEMBER VARIABLES
    //==========================================================================
    
    // OpenCL Core Objects
    cl_platform_id   m_platform;        ///< Selected OpenCL platform
    cl_device_id     m_device;          ///< Selected OpenCL device
    cl_context       m_context;         ///< OpenCL context
    cl_command_queue m_commandQueue;    ///< Command queue (out-of-order, profiling)
    
    // Device Information
    DeviceInfo m_deviceInfo;            ///< Cached device information
    
    // Kernel Cache
    std::unordered_map<std::string, cl_program> m_programs;  ///< Compiled programs
    std::unordered_map<std::string, cl_kernel>  m_kernels;   ///< Compiled kernels
    
    // Timing
    cl_event m_startEvent;              ///< Timer start event
    cl_event m_stopEvent;               ///< Timer stop event
    bool     m_timingActive;            ///< Whether timing is currently active
    double   m_accumulatedTime;         ///< Accumulated execution time in milliseconds

    // State
    bool m_initialized;                 ///< Whether backend is initialized
    std::string m_lastError;            ///< Last error message
    
    // Logger
    Logger& m_logger;                   ///< Reference to singleton logger
};

} // namespace GPUBenchmark

/********************************************************************************
 * IMPLEMENTATION NOTES:
 * 
 * 1. OPENCL API DIFFERENCES FROM CUDA:
 *    - Kernels compiled at runtime (no offline compilation)
 *    - More verbose API (explicit error checking required)
 *    - Different memory model (buffers vs pointers)
 *    - Work-items/work-groups vs threads/blocks
 * 
 * 2. PERFORMANCE CONSIDERATIONS:
 *    - First kernel launch slower due to JIT compilation
 *    - Subsequent launches similar performance to CUDA
 *    - Memory bandwidth typically 95-100% of CUDA
 *    - Compute performance 90-95% of CUDA (compiler dependent)
 * 
 * 3. PORTABILITY:
 *    - Same code runs on NVIDIA, AMD, Intel GPUs
 *    - Different vendors may have different optimal work-group sizes
 *    - Use NULL local work size for automatic selection
 * 
 * 4. ERROR HANDLING:
 *    - All OpenCL API calls should be wrapped with CheckCLError()
 *    - Log detailed error messages for debugging
 *    - Return false on errors (don't throw exceptions)
 * 
 * 5. RESOURCE MANAGEMENT:
 *    - All cl_* objects must be released with clRelease*()
 *    - Shutdown() must be called before destruction
 *    - RAII pattern used for automatic cleanup
 ********************************************************************************/
