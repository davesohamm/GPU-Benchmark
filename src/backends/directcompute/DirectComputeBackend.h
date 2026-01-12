/********************************************************************************
 * @file    DirectComputeBackend.h
 * @brief   DirectCompute Backend Implementation for GPU Compute Benchmarking
 * 
 * @details This file provides a complete DirectCompute backend that implements
 *          the IComputeBackend interface. DirectCompute is Windows-native GPU
 *          compute using Direct3D 11 and HLSL Compute Shaders.
 * 
 * @note    DirectCompute Backend Implementation Strategy:
 *          - Uses Direct3D 11 Compute Shaders
 *          - HLSL shader compilation (runtime or offline)
 *          - Windows-native (no external SDKs needed)
 *          - Works on NVIDIA, AMD, Intel GPUs
 * 
 * @architecture
 *          [DirectComputeBackend] implements [IComputeBackend]
 *                  |
 *                  +-- D3D11 Device Creation
 *                  +-- Compute Shader Compilation
 *                  +-- Structured Buffer Management
 *                  +-- Dispatch & Synchronization
 *                  +-- Query-based Timing
 * 
 * @performance
 *          Expected performance compared to CUDA:
 *          - Memory bandwidth: 90-95%
 *          - Compute throughput: 85-95%
 *          - Latency: +10-15% overhead for API calls
 * 
 * @compatibility
 *          - Works on NVIDIA GPUs (via D3D11 driver)
 *          - Works on AMD GPUs (native D3D11 support)
 *          - Works on Intel GPUs (native D3D11 support)
 *          - Requires: Windows 7+ with DirectX 11 support
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#pragma once

// Include C++ standard headers FIRST to avoid macro conflicts
#include <string>
#include <vector>
#include <unordered_map>

// Then include core headers
#include "../../core/IComputeBackend.h"
#include "../../core/Logger.h"

// Finally include Windows/DirectX headers
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>  // For ComPtr

// Link required libraries
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

namespace GPUBenchmark {

// ComPtr alias for cleaner code
template<typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

/********************************************************************************
 * @class DirectComputeBackend
 * @brief Complete DirectCompute implementation of IComputeBackend interface
 * 
 * @details This class provides a full-featured DirectCompute backend that:
 *          - Creates D3D11 device and context
 *          - Compiles HLSL compute shaders (runtime or from file)
 *          - Manages structured/raw buffers
 *          - Dispatches compute shaders with configurable thread groups
 *          - Measures execution time using D3D11 queries
 * 
 * @thread_safety
 *          Not thread-safe. Each thread should create its own backend instance.
 * 
 * @resource_management
 *          Uses ComPtr for automatic COM object cleanup
 ********************************************************************************/
class DirectComputeBackend : public IComputeBackend {
public:
    //==========================================================================
    // CONSTRUCTOR & DESTRUCTOR
    //==========================================================================
    
    /**
     * @brief Construct DirectCompute backend
     * @note Does not initialize D3D11 - call Initialize() explicitly
     */
    DirectComputeBackend();
    
    /**
     * @brief Destructor - ensures cleanup
     */
    ~DirectComputeBackend() override;

    //==========================================================================
    // ICOMPUTEBACKEND INTERFACE IMPLEMENTATION
    //==========================================================================
    
    /**
     * @brief Initialize DirectCompute backend
     * @return true if initialization successful, false otherwise
     * 
     * @details Initialization sequence:
     *          1. Enumerate DXGI adapters (GPUs)
     *          2. Select best available adapter
     *          3. Create D3D11 device and immediate context
     *          4. Query device capabilities
     *          5. Create timing queries
     */
    bool Initialize() override;
    
    /**
     * @brief Shutdown DirectCompute backend and release all resources
     * 
     * @details Cleanup sequence:
     *          1. Release all compiled shaders
     *          2. Release all buffers
     *          3. Release queries
     *          4. Release device context and device
     */
    void Shutdown() override;
    
    /**
     * @brief Get backend type
     * @return BackendType::DirectCompute
     */
    BackendType GetBackendType() const override { return BackendType::DirectCompute; }
    
    /**
     * @brief Get backend name
     * @return "DirectCompute"
     */
    std::string GetBackendName() const override { return "DirectCompute"; }
    
    /**
     * @brief Get device information
     * @return DeviceInfo structure with GPU details
     */
    DeviceInfo GetDeviceInfo() const override { return m_deviceInfo; }

    //==========================================================================
    // MEMORY MANAGEMENT
    //==========================================================================
    
    /**
     * @brief Allocate GPU memory (structured buffer)
     * @param sizeBytes Size in bytes
     * @return Pointer to ID3D11Buffer (cast to void*)
     * 
     * @note Creates a structured buffer with UAV access
     * @note Returns nullptr on failure
     */
    void* AllocateMemory(size_t sizeBytes) override;
    
    /**
     * @brief Free GPU memory
     * @param ptr Pointer to ID3D11Buffer
     */
    void FreeMemory(void* ptr) override;
    
    /**
     * @brief Copy data from host to device
     * @param dst Device pointer (ID3D11Buffer)
     * @param src Host pointer
     * @param sizeBytes Size in bytes
     * 
     * @note Uses staging buffer for efficient upload
     */
    void CopyHostToDevice(void* dst, const void* src, size_t sizeBytes) override;
    
    /**
     * @brief Copy data from device to host
     * @param dst Host pointer
     * @param src Device pointer (ID3D11Buffer)
     * @param sizeBytes Size in bytes
     * 
     * @note Uses staging buffer for readback
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
     * @note Records timestamp query before subsequent dispatch
     */
    void StartTimer() override;
    
    /**
     * @brief Stop GPU timer
     * @note Records timestamp query after dispatch
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
     * @note This is a wrapper for IComputeBackend compatibility
     */
    bool ExecuteKernel(const std::string& kernelName, const KernelParams& params) override;
    
    /**
     * @brief Check if backend is available
     * @return true if D3D11 device created successfully
     */
    bool IsAvailable() const override;
    
    /**
     * @brief Get last error message
     * @return Error string
     */
    std::string GetLastError() const override;

    //==========================================================================
    // DIRECTCOMPUTE-SPECIFIC METHODS
    //==========================================================================
    
    /**
     * @brief Compile HLSL compute shader from source code
     * @param shaderName Unique identifier for this shader
     * @param sourceCode HLSL source code
     * @param entryPoint Entry point function name (e.g., "CSMain")
     * @param shaderModel Shader model (e.g., "cs_5_0")
     * @return true if compilation successful, false otherwise
     * 
     * @details Compiles HLSL compute shader and caches it for later use.
     *          Shader models:
     *          - "cs_5_0" - DirectX 11.0 (most compatible)
     *          - "cs_5_1" - DirectX 11.3 (resource arrays)
     * 
     * @note Shader cached internally - subsequent calls return existing shader
     */
    bool CompileShader(const std::string& shaderName,
                       const std::string& sourceCode,
                       const std::string& entryPoint = "CSMain",
                       const std::string& shaderModel = "cs_5_0");
    
    /**
     * @brief Dispatch a compiled compute shader
     * @param shaderName Name of previously compiled shader
     * @param threadGroupsX Number of thread groups in X dimension
     * @param threadGroupsY Number of thread groups in Y dimension
     * @param threadGroupsZ Number of thread groups in Z dimension
     * @return true if dispatch successful, false otherwise
     * 
     * @note Must bind buffers before calling this method
     */
    bool DispatchShader(const std::string& shaderName,
                        UINT threadGroupsX,
                        UINT threadGroupsY = 1,
                        UINT threadGroupsZ = 1);
    
    /**
     * @brief Bind buffer to shader as UAV (Unordered Access View)
     * @param buffer Buffer pointer (ID3D11Buffer)
     * @param slot UAV slot (0-7 typically)
     * @return true if successful
     * 
     * @note UAVs allow read/write access from compute shaders
     */
    bool BindBufferUAV(void* buffer, UINT slot);
    
    /**
     * @brief Unbind all UAVs
     */
    void UnbindUAVs();
    
    /**
     * @brief Create UAV for a buffer
     * @param buffer D3D11 buffer
     * @param elementCount Number of elements
     * @param elementSize Size of each element
     * @return ComPtr to UAV
     */
    ComPtr<ID3D11UnorderedAccessView> CreateBufferUAV(ID3D11Buffer* buffer,
                                                        UINT elementCount,
                                                        UINT elementSize);
    
    /**
     * @brief Create and bind constant buffer
     * @param data Pointer to constant data
     * @param sizeBytes Size of data in bytes
     * @param slot Constant buffer slot (0-13 typically)
     * @return true if successful
     * 
     * @note Data must be 16-byte aligned
     * @note Creates a new constant buffer each time (inefficient but simple)
     */
    bool SetConstantBuffer(const void* data, size_t sizeBytes, UINT slot = 0);

private:
    //==========================================================================
    // PRIVATE HELPER METHODS
    //==========================================================================
    
    /**
     * @brief Enumerate DXGI adapters and select best one
     * @return true if adapter selected successfully
     */
    bool SelectBestAdapter();
    
    /**
     * @brief Query and populate device information
     */
    void QueryDeviceInfo();
    
    /**
     * @brief Create staging buffer for CPU↔GPU transfers
     * @param sizeBytes Buffer size
     * @return ComPtr to staging buffer
     */
    ComPtr<ID3D11Buffer> CreateStagingBuffer(size_t sizeBytes);
    
    /**
     * @brief Check HRESULT and log error if failed
     * @param hr HRESULT code
     * @param operation Description of operation
     * @return true if SUCCEEDED, false otherwise
     */
    bool CheckHR(HRESULT hr, const std::string& operation);
    
    /**
     * @brief Get human-readable error string from HRESULT
     * @param hr HRESULT code
     * @return Error string
     */
    std::string GetHRErrorString(HRESULT hr);

    //==========================================================================
    // MEMBER VARIABLES
    //==========================================================================
    
    // Direct3D 11 Objects
    ComPtr<IDXGIAdapter1>              m_adapter;         ///< Selected DXGI adapter (GPU)
    ComPtr<ID3D11Device>               m_device;          ///< D3D11 device
    ComPtr<ID3D11DeviceContext>        m_context;         ///< Immediate device context
    
    // Device Information
    DeviceInfo m_deviceInfo;                              ///< Cached device information
    DXGI_ADAPTER_DESC1 m_adapterDesc;                     ///< Adapter description
    
    // Shader Cache
    std::unordered_map<std::string, ComPtr<ID3D11ComputeShader>> m_shaders;  ///< Compiled shaders
    std::unordered_map<void*, ComPtr<ID3D11UnorderedAccessView>> m_uavCache; ///< UAV cache
    
    // Timing
    ComPtr<ID3D11Query> m_queryDisjoint;                  ///< Disjoint query (checks GPU freq)
    ComPtr<ID3D11Query> m_queryStart;                     ///< Start timestamp query
    ComPtr<ID3D11Query> m_queryEnd;                       ///< End timestamp query
    bool m_timingActive;                                  ///< Whether timing is active
    
    // State
    bool m_initialized;                                   ///< Whether backend is initialized
    std::string m_lastError;                              ///< Last error message
    
    // Logger
    Logger& m_logger;                                     ///< Reference to singleton logger
};

} // namespace GPUBenchmark

/********************************************************************************
 * IMPLEMENTATION NOTES:
 * 
 * 1. DIRECTCOMPUTE API DIFFERENCES FROM CUDA:
 *    - Uses COM objects (requires ComPtr for management)
 *    - More verbose API (explicit state management)
 *    - Buffers are opaque (ID3D11Buffer vs device pointers)
 *    - Thread groups vs thread blocks (similar concept)
 * 
 * 2. PERFORMANCE CONSIDERATIONS:
 *    - Shader compilation can be slow (cache compiled shaders)
 *    - Staging buffers add overhead for CPU↔GPU transfers
 *    - Query-based timing is accurate but adds sync overhead
 *    - UAV binding/unbinding has cost (batch operations)
 * 
 * 3. PORTABILITY:
 *    - Windows-only (uses Windows APIs)
 *    - Works on all Windows GPUs (NVIDIA, AMD, Intel)
 *    - No external SDKs needed (DirectX built into Windows)
 * 
 * 4. ERROR HANDLING:
 *    - All D3D11 API calls return HRESULT
 *    - Use CheckHR() to wrap calls and log errors
 *    - ComPtr automatically releases COM objects
 * 
 * 5. RESOURCE MANAGEMENT:
 *    - ComPtr handles reference counting
 *    - Must explicitly unbind UAVs before releasing buffers
 *    - Staging buffers created on-demand
 ********************************************************************************/
