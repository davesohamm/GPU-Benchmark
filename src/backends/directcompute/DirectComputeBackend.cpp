/********************************************************************************
 * @file    DirectComputeBackend.cpp
 * @brief   DirectCompute Backend Implementation
 * 
 * @details Complete implementation of DirectCompute backend for Windows-native
 *          GPU compute using Direct3D 11 and HLSL Compute Shaders.
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#include "DirectComputeBackend.h"
#include <dxgi1_2.h>
#include <sstream>
#include <iomanip>

namespace GPUBenchmark {

//==============================================================================
// CONSTRUCTOR & DESTRUCTOR
//==============================================================================

DirectComputeBackend::DirectComputeBackend()
    : m_timingActive(false)
    , m_initialized(false)
    , m_lastError("")
    , m_logger(Logger::GetInstance())
{
    ZeroMemory(&m_adapterDesc, sizeof(m_adapterDesc));
    m_logger.Info("[DirectCompute] Backend created");
}

DirectComputeBackend::~DirectComputeBackend() {
    if (m_initialized) {
        Shutdown();
    }
}

//==============================================================================
// INITIALIZATION & SHUTDOWN
//==============================================================================

bool DirectComputeBackend::Initialize() {
    m_logger.Info("[DirectCompute] Initializing DirectCompute backend...");
    
    // Step 1: Select best adapter
    if (!SelectBestAdapter()) {
        m_lastError = "Failed to select DXGI adapter";
        m_logger.Error("[DirectCompute] " + m_lastError);
        return false;
    }
    
    // Step 2: Create D3D11 device and context
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    D3D_FEATURE_LEVEL featureLevel;
    
    HRESULT hr = D3D11CreateDevice(
        m_adapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,  // Using explicit adapter
        nullptr,
        createDeviceFlags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &m_device,
        &featureLevel,
        &m_context
    );
    
    if (!CheckHR(hr, "D3D11CreateDevice")) {
        m_lastError = "Failed to create D3D11 device";
        return false;
    }
    
    m_logger.Info("[DirectCompute] Created D3D11 device (Feature Level " + 
                  std::to_string((featureLevel >> 12) & 0xF) + "." +
                  std::to_string((featureLevel >> 8) & 0xF) + ")");
    
    // Step 3: Create timing queries
    D3D11_QUERY_DESC queryDesc;
    
    queryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    queryDesc.MiscFlags = 0;
    hr = m_device->CreateQuery(&queryDesc, &m_queryDisjoint);
    if (!CheckHR(hr, "CreateQuery (disjoint)")) {
        return false;
    }
    
    queryDesc.Query = D3D11_QUERY_TIMESTAMP;
    hr = m_device->CreateQuery(&queryDesc, &m_queryStart);
    if (!CheckHR(hr, "CreateQuery (start)")) {
        return false;
    }
    
    hr = m_device->CreateQuery(&queryDesc, &m_queryEnd);
    if (!CheckHR(hr, "CreateQuery (end)")) {
        return false;
    }
    
    // Step 4: Query device info
    QueryDeviceInfo();
    
    m_initialized = true;
    m_logger.Info("[DirectCompute] Initialization complete");
    m_logger.Info("[DirectCompute] Device: " + m_deviceInfo.name);
    m_logger.Info("[DirectCompute] Memory: " + 
                  std::to_string(m_deviceInfo.totalMemoryBytes / (1024*1024)) + " MB");
    
    return true;
}

void DirectComputeBackend::Shutdown() {
    if (!m_initialized) return;
    
    m_logger.Info("[DirectCompute] Shutting down...");
    
    // Unbind all resources
    UnbindUAVs();
    
    // Clear caches
    m_shaders.clear();
    m_uavCache.clear();
    
    // Release queries
    m_queryEnd.Reset();
    m_queryStart.Reset();
    m_queryDisjoint.Reset();
    
    // Release context and device
    if (m_context) {
        m_context->ClearState();
        m_context->Flush();
        m_context.Reset();
    }
    
    m_device.Reset();
    m_adapter.Reset();
    
    m_initialized = false;
    m_logger.Info("[DirectCompute] Shutdown complete");
}

//==============================================================================
// MEMORY MANAGEMENT
//==============================================================================

void* DirectComputeBackend::AllocateMemory(size_t sizeBytes) {
    if (!m_initialized) {
        m_lastError = "Backend not initialized";
        m_logger.Error("[DirectCompute] " + m_lastError);
        return nullptr;
    }
    
    // Create structured buffer with UAV access
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.ByteWidth = static_cast<UINT>(sizeBytes);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bufferDesc.StructureByteStride = sizeof(float);  // Assume float for now
    
    ID3D11Buffer* buffer = nullptr;
    HRESULT hr = m_device->CreateBuffer(&bufferDesc, nullptr, &buffer);
    
    if (!CheckHR(hr, "CreateBuffer")) {
        m_lastError = "Failed to allocate GPU memory";
        return nullptr;
    }
    
    return buffer;
}

void DirectComputeBackend::FreeMemory(void* ptr) {
    if (!ptr) return;
    
    ID3D11Buffer* buffer = static_cast<ID3D11Buffer*>(ptr);
    
    // Remove from UAV cache if present
    m_uavCache.erase(ptr);
    
    buffer->Release();
}

void DirectComputeBackend::CopyHostToDevice(void* dst, const void* src, size_t sizeBytes) {
    if (!m_initialized) {
        m_lastError = "Backend not initialized";
        return;
    }
    
    ID3D11Buffer* dstBuffer = static_cast<ID3D11Buffer*>(dst);
    
    // Create staging buffer
    auto stagingBuffer = CreateStagingBuffer(sizeBytes);
    if (!stagingBuffer) {
        m_lastError = "Failed to create staging buffer";
        return;
    }
    
    // Map staging buffer and copy data
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = m_context->Map(stagingBuffer.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    if (CheckHR(hr, "Map staging buffer")) {
        memcpy(mapped.pData, src, sizeBytes);
        m_context->Unmap(stagingBuffer.Get(), 0);
        
        // Copy from staging to destination
        m_context->CopyResource(dstBuffer, stagingBuffer.Get());
    }
}

void DirectComputeBackend::CopyDeviceToHost(void* dst, const void* src, size_t sizeBytes) {
    if (!m_initialized) {
        m_lastError = "Backend not initialized";
        return;
    }
    
    ID3D11Buffer* srcBuffer = static_cast<ID3D11Buffer*>(const_cast<void*>(src));
    
    // Create staging buffer
    auto stagingBuffer = CreateStagingBuffer(sizeBytes);
    if (!stagingBuffer) {
        m_lastError = "Failed to create staging buffer";
        return;
    }
    
    // Copy from source to staging
    m_context->CopyResource(stagingBuffer.Get(), srcBuffer);
    
    // Map staging buffer and copy data
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = m_context->Map(stagingBuffer.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (CheckHR(hr, "Map staging buffer for read")) {
        memcpy(dst, mapped.pData, sizeBytes);
        m_context->Unmap(stagingBuffer.Get(), 0);
    }
}

//==============================================================================
// SYNCHRONIZATION & TIMING
//==============================================================================

void DirectComputeBackend::Synchronize() {
    if (!m_initialized) return;
    
    m_context->Flush();
    
    // Create a simple event query to wait for GPU
    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    
    ComPtr<ID3D11Query> eventQuery;
    HRESULT hr = m_device->CreateQuery(&queryDesc, &eventQuery);
    if (SUCCEEDED(hr)) {
        m_context->End(eventQuery.Get());
        
        // Wait for event
        BOOL data;
        while (m_context->GetData(eventQuery.Get(), &data, sizeof(BOOL), 0) == S_FALSE) {
            // Busy wait
        }
    }
}

void DirectComputeBackend::StartTimer() {
    if (!m_initialized) return;
    
    m_context->Begin(m_queryDisjoint.Get());
    m_context->End(m_queryStart.Get());
    m_timingActive = true;
}

void DirectComputeBackend::StopTimer() {
    if (!m_initialized || !m_timingActive) return;
    
    m_context->End(m_queryEnd.Get());
    m_context->End(m_queryDisjoint.Get());
}

double DirectComputeBackend::GetElapsedTime() {
    if (!m_initialized || !m_timingActive) {
        return 0.0;
    }
    
    // Wait for queries to complete
    while (m_context->GetData(m_queryDisjoint.Get(), nullptr, 0, 0) == S_FALSE) {
        // Busy wait
    }
    
    // Get disjoint data
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjointData;
    m_context->GetData(m_queryDisjoint.Get(), &disjointData, sizeof(disjointData), 0);
    
    if (disjointData.Disjoint) {
        m_logger.Warning("[DirectCompute] Timer disjoint - results may be inaccurate");
    }
    
    // Get timestamps
    UINT64 startTime, endTime;
    m_context->GetData(m_queryStart.Get(), &startTime, sizeof(UINT64), 0);
    m_context->GetData(m_queryEnd.Get(), &endTime, sizeof(UINT64), 0);
    
    // Calculate elapsed time in milliseconds
    UINT64 elapsed = endTime - startTime;
    double elapsedMs = (static_cast<double>(elapsed) / disjointData.Frequency) * 1000.0;
    
    m_timingActive = false;
    return elapsedMs;
}

bool DirectComputeBackend::ExecuteKernel(const std::string& kernelName, const KernelParams& params) {
    // This is a wrapper to satisfy the IComputeBackend interface
    // DirectCompute shaders are executed differently (bind UAVs + dispatch)
    m_lastError = "ExecuteKernel(KernelParams) not implemented for DirectCompute - use CompileShader + BindBufferUAV + DispatchShader";
    return false;
}

bool DirectComputeBackend::IsAvailable() const {
    return m_initialized;
}

std::string DirectComputeBackend::GetLastError() const {
    return m_lastError;
}

//==============================================================================
// DIRECTCOMPUTE SHADER MANAGEMENT
//==============================================================================

bool DirectComputeBackend::CompileShader(const std::string& shaderName,
                                          const std::string& sourceCode,
                                          const std::string& entryPoint,
                                          const std::string& shaderModel) {
    if (!m_initialized) {
        m_lastError = "Backend not initialized";
        m_logger.Error("[DirectCompute] " + m_lastError);
        return false;
    }
    
    // Check if already compiled
    if (m_shaders.find(shaderName) != m_shaders.end()) {
        m_logger.Info("[DirectCompute] Shader '" + shaderName + "' already compiled");
        return true;
    }
    
    m_logger.Info("[DirectCompute] Compiling shader: " + shaderName);
    
    // Compile shader from source
    ComPtr<ID3DBlob> shaderBlob;
    ComPtr<ID3DBlob> errorBlob;
    
    UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    compileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
    
    HRESULT hr = D3DCompile(
        sourceCode.c_str(),
        sourceCode.length(),
        shaderName.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entryPoint.c_str(),
        shaderModel.c_str(),
        compileFlags,
        0,
        &shaderBlob,
        &errorBlob
    );
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::string errorMsg = static_cast<const char*>(errorBlob->GetBufferPointer());
            m_lastError = "Shader compilation failed: " + errorMsg;
            m_logger.Error("[DirectCompute] " + m_lastError);
        }
        return false;
    }
    
    // Create compute shader
    ComPtr<ID3D11ComputeShader> shader;
    hr = m_device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &shader
    );
    
    if (!CheckHR(hr, "CreateComputeShader")) {
        m_lastError = "Failed to create compute shader";
        return false;
    }
    
    // Cache shader
    m_shaders[shaderName] = shader;
    
    m_logger.Info("[DirectCompute] Shader '" + shaderName + "' compiled successfully");
    return true;
}

bool DirectComputeBackend::DispatchShader(const std::string& shaderName,
                                           UINT threadGroupsX,
                                           UINT threadGroupsY,
                                           UINT threadGroupsZ) {
    if (!m_initialized) {
        m_lastError = "Backend not initialized";
        return false;
    }
    
    // Find shader
    auto it = m_shaders.find(shaderName);
    if (it == m_shaders.end()) {
        m_lastError = "Shader '" + shaderName + "' not found";
        m_logger.Error("[DirectCompute] " + m_lastError);
        return false;
    }
    
    // Set compute shader
    m_context->CSSetShader(it->second.Get(), nullptr, 0);
    
    // Dispatch
    m_context->Dispatch(threadGroupsX, threadGroupsY, threadGroupsZ);
    
    return true;
}

bool DirectComputeBackend::BindBufferUAV(void* buffer, UINT slot) {
    if (!m_initialized || !buffer) {
        return false;
    }
    
    ID3D11Buffer* d3dBuffer = static_cast<ID3D11Buffer*>(buffer);
    
    // Check cache for existing UAV
    auto it = m_uavCache.find(buffer);
    ComPtr<ID3D11UnorderedAccessView> uav;
    
    if (it != m_uavCache.end()) {
        uav = it->second;
    } else {
        // Get buffer description
        D3D11_BUFFER_DESC bufferDesc;
        d3dBuffer->GetDesc(&bufferDesc);
        
        // Create UAV
        UINT elementCount = bufferDesc.ByteWidth / bufferDesc.StructureByteStride;
        uav = CreateBufferUAV(d3dBuffer, elementCount, bufferDesc.StructureByteStride);
        
        if (!uav) {
            return false;
        }
        
        m_uavCache[buffer] = uav;
    }
    
    // Bind UAV
    m_context->CSSetUnorderedAccessViews(slot, 1, uav.GetAddressOf(), nullptr);
    
    return true;
}

void DirectComputeBackend::UnbindUAVs() {
    if (!m_initialized) return;
    
    ID3D11UnorderedAccessView* nullUAVs[D3D11_PS_CS_UAV_REGISTER_COUNT] = { nullptr };
    m_context->CSSetUnorderedAccessViews(0, D3D11_PS_CS_UAV_REGISTER_COUNT, nullUAVs, nullptr);
}

bool DirectComputeBackend::SetConstantBuffer(const void* data, size_t sizeBytes, UINT slot) {
    if (!m_initialized) {
        m_lastError = "Backend not initialized";
        return false;
    }
    
    // Round up to 16-byte boundary
    UINT alignedSize = static_cast<UINT>((sizeBytes + 15) & ~15);
    
    // Create constant buffer
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = alignedSize;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    ComPtr<ID3D11Buffer> constantBuffer;
    HRESULT hr = m_device->CreateBuffer(&cbDesc, nullptr, &constantBuffer);
    if (!CheckHR(hr, "CreateBuffer (constant buffer)")) {
        return false;
    }
    
    // Map and copy data
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = m_context->Map(constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (!CheckHR(hr, "Map constant buffer")) {
        return false;
    }
    
    memcpy(mapped.pData, data, sizeBytes);
    m_context->Unmap(constantBuffer.Get(), 0);
    
    // Bind to shader
    m_context->CSSetConstantBuffers(slot, 1, constantBuffer.GetAddressOf());
    
    return true;
}

ComPtr<ID3D11UnorderedAccessView> DirectComputeBackend::CreateBufferUAV(ID3D11Buffer* buffer,
                                                                          UINT elementCount,
                                                                          UINT elementSize) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = elementCount;
    uavDesc.Buffer.Flags = 0;
    
    ComPtr<ID3D11UnorderedAccessView> uav;
    HRESULT hr = m_device->CreateUnorderedAccessView(buffer, &uavDesc, &uav);
    
    if (!CheckHR(hr, "CreateUnorderedAccessView")) {
        return nullptr;
    }
    
    return uav;
}

//==============================================================================
// PRIVATE HELPER METHODS
//==============================================================================

bool DirectComputeBackend::SelectBestAdapter() {
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    
    if (!CheckHR(hr, "CreateDXGIFactory1")) {
        return false;
    }
    
    // Enumerate adapters
    UINT adapterIndex = 0;
    SIZE_T maxVideoMemory = 0;
    ComPtr<IDXGIAdapter1> bestAdapter;
    DXGI_ADAPTER_DESC1 bestDesc = {};
    
    while (true) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory->EnumAdapters1(adapterIndex++, &adapter);
        
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;  // No more adapters
        }
        
        if (FAILED(hr)) {
            continue;
        }
        
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        
        // Skip software adapters
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        
        // Select adapter with most video memory
        if (desc.DedicatedVideoMemory > maxVideoMemory) {
            maxVideoMemory = desc.DedicatedVideoMemory;
            bestAdapter = adapter;
            bestDesc = desc;
        }
    }
    
    if (!bestAdapter) {
        m_logger.Error("[DirectCompute] No suitable adapter found");
        return false;
    }
    
    m_adapter = bestAdapter;
    m_adapterDesc = bestDesc;
    
    // Convert wide string to string
    char adapterName[128];
    WideCharToMultiByte(CP_UTF8, 0, bestDesc.Description, -1, adapterName, sizeof(adapterName), nullptr, nullptr);
    
    m_logger.Info("[DirectCompute] Selected adapter: " + std::string(adapterName));
    m_logger.Info("[DirectCompute] Video Memory: " + 
                  std::to_string(bestDesc.DedicatedVideoMemory / (1024*1024)) + " MB");
    
    return true;
}

void DirectComputeBackend::QueryDeviceInfo() {
    // Convert adapter name
    char adapterName[128];
    WideCharToMultiByte(CP_UTF8, 0, m_adapterDesc.Description, -1, 
                        adapterName, sizeof(adapterName), nullptr, nullptr);
    
    m_deviceInfo.name = std::string(adapterName);
    m_deviceInfo.totalMemoryBytes = m_adapterDesc.DedicatedVideoMemory;
    m_deviceInfo.availableMemoryBytes = m_adapterDesc.DedicatedVideoMemory;  // Approximate
    
    // DirectCompute doesn't expose these directly - use reasonable defaults
    m_deviceInfo.maxThreadsPerBlock = 1024;  // D3D11 CS thread group limit
    m_deviceInfo.maxBlockDimX = 1024;
    m_deviceInfo.maxBlockDimY = 1024;
    m_deviceInfo.maxBlockDimZ = 64;
    
    // Driver version (format varies by vendor)
    std::stringstream ss;
    ss << "DirectX 11";
    m_deviceInfo.driverVersion = ss.str();
    
    // Compute capability not applicable
    m_deviceInfo.computeCapabilityMajor = 0;
    m_deviceInfo.computeCapabilityMinor = 0;
}

ComPtr<ID3D11Buffer> DirectComputeBackend::CreateStagingBuffer(size_t sizeBytes) {
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.ByteWidth = static_cast<UINT>(sizeBytes);
    bufferDesc.Usage = D3D11_USAGE_STAGING;
    bufferDesc.BindFlags = 0;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 0;
    
    ComPtr<ID3D11Buffer> buffer;
    HRESULT hr = m_device->CreateBuffer(&bufferDesc, nullptr, &buffer);
    
    if (!CheckHR(hr, "CreateBuffer (staging)")) {
        return nullptr;
    }
    
    return buffer;
}

bool DirectComputeBackend::CheckHR(HRESULT hr, const std::string& operation) {
    if (SUCCEEDED(hr)) {
        return true;
    }
    
    std::string errorMsg = "[DirectCompute] " + operation + " failed: " + GetHRErrorString(hr);
    m_logger.Error(errorMsg);
    m_lastError = errorMsg;
    return false;
}

std::string DirectComputeBackend::GetHRErrorString(HRESULT hr) {
    std::stringstream ss;
    ss << "HRESULT 0x" << std::hex << std::setw(8) << std::setfill('0') << hr;
    
    // Common HRESULT codes
    switch (hr) {
        case E_INVALIDARG: return ss.str() + " (E_INVALIDARG)";
        case E_OUTOFMEMORY: return ss.str() + " (E_OUTOFMEMORY)";
        case E_FAIL: return ss.str() + " (E_FAIL)";
        case DXGI_ERROR_DEVICE_REMOVED: return ss.str() + " (DEVICE_REMOVED)";
        case DXGI_ERROR_DEVICE_HUNG: return ss.str() + " (DEVICE_HUNG)";
        case DXGI_ERROR_DEVICE_RESET: return ss.str() + " (DEVICE_RESET)";
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR: return ss.str() + " (DRIVER_ERROR)";
        case DXGI_ERROR_INVALID_CALL: return ss.str() + " (INVALID_CALL)";
        default: return ss.str();
    }
}

} // namespace GPUBenchmark

/********************************************************************************
 * END OF FILE: DirectComputeBackend.cpp
 * 
 * SUMMARY:
 * - Complete DirectCompute backend implementation
 * - D3D11 device creation and management
 * - HLSL compute shader compilation
 * - Buffer management with UAV support
 * - Query-based timing
 * 
 * NEXT STEPS:
 * - Create HLSL compute shaders
 * - Port all 4 benchmark kernels
 * - Test on Windows GPU
 ********************************************************************************/
