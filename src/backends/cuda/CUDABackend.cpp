/*******************************************************************************
 * FILE: CUDABackend.cpp
 * 
 * PURPOSE:
 *   Implementation of CUDABackend class for NVIDIA GPU compute operations.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#include "CUDABackend.h"
#include "../../core/Logger.h"
#include <cuda_runtime.h>
#include <sstream>

namespace GPUBenchmark {

CUDABackend::CUDABackend()
    : m_isInitialized(false)
    , m_deviceId(0)
    , m_startEvent(nullptr)
    , m_stopEvent(nullptr)
    , m_timerStarted(false)
    , m_lastError("No error")
{
}

CUDABackend::~CUDABackend() {
    if (m_isInitialized) {
        Shutdown();
    }
}

bool CUDABackend::Initialize() {
    Logger& logger = Logger::GetInstance();
    logger.Info("Initializing CUDA backend...");
    
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (!CheckCUDAError(err, "cudaGetDeviceCount")) {
        logger.Error("Failed to query CUDA device count: " + m_lastError);
        return false;
    }
    
    if (deviceCount == 0) {
        m_lastError = "No CUDA-capable devices found";
        logger.Error(m_lastError);
        return false;
    }
    
    logger.Info("Found " + std::to_string(deviceCount) + " CUDA device(s)");
    
    // Use first device
    m_deviceId = 0;
    
    // Get device properties
    err = cudaGetDeviceProperties(&m_deviceProps, m_deviceId);
    if (!CheckCUDAError(err, "cudaGetDeviceProperties")) {
        logger.Error("Failed to get device properties: " + m_lastError);
        return false;
    }
    
    // Set device
    err = cudaSetDevice(m_deviceId);
    if (!CheckCUDAError(err, "cudaSetDevice")) {
        logger.Error("Failed to set device: " + m_lastError);
        return false;
    }
    
    // Create timing events
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
    
    // Log device info
    logger.Info("Selected Device: " + std::string(m_deviceProps.name));
    logger.Info("  Compute Capability: " + std::to_string(m_deviceProps.major) + "." + std::to_string(m_deviceProps.minor));
    logger.Info("  Total Memory: " + std::to_string(m_deviceProps.totalGlobalMem / (1024*1024)) + " MB");
    logger.Info("  Multiprocessors: " + std::to_string(m_deviceProps.multiProcessorCount));
    logger.Info("  Max Threads/Block: " + std::to_string(m_deviceProps.maxThreadsPerBlock));
    
    m_isInitialized = true;
    logger.Info("CUDA backend initialized successfully!");
    
    return true;
}

void CUDABackend::Shutdown() {
    if (!m_isInitialized) {
        return;
    }
    
    Logger& logger = Logger::GetInstance();
    logger.Info("Shutting down CUDA backend...");
    
    if (m_startEvent) {
        cudaEventDestroy(m_startEvent);
        m_startEvent = nullptr;
    }
    if (m_stopEvent) {
        cudaEventDestroy(m_stopEvent);
        m_stopEvent = nullptr;
    }
    
    cudaDeviceReset();
    m_isInitialized = false;
    
    logger.Info("CUDA backend shutdown complete");
}

DeviceInfo CUDABackend::GetDeviceInfo() const {
    DeviceInfo info;
    
    if (!m_isInitialized) {
        info.name = "Uninitialized";
        return info;
    }
    
    info.name = m_deviceProps.name;
    info.totalMemoryBytes = m_deviceProps.totalGlobalMem;
    info.availableMemoryBytes = m_deviceProps.totalGlobalMem;
    info.driverVersion = std::to_string(m_deviceProps.major) + "." + std::to_string(m_deviceProps.minor);
    info.computeCapabilityMajor = m_deviceProps.major;
    info.computeCapabilityMinor = m_deviceProps.minor;
    info.maxThreadsPerBlock = m_deviceProps.maxThreadsPerBlock;
    info.maxBlockDimX = m_deviceProps.maxThreadsDim[0];
    info.maxBlockDimY = m_deviceProps.maxThreadsDim[1];
    info.maxBlockDimZ = m_deviceProps.maxThreadsDim[2];
    
    return info;
}

void* CUDABackend::AllocateMemory(size_t sizeBytes) {
    if (!m_isInitialized) {
        m_lastError = "Backend not initialized";
        return nullptr;
    }
    
    void* devicePtr = nullptr;
    cudaError_t err = cudaMalloc(&devicePtr, sizeBytes);
    
    if (!CheckCUDAError(err, "cudaMalloc")) {
        Logger::GetInstance().Error("Memory allocation failed: " + m_lastError);
        return nullptr;
    }
    
    return devicePtr;
}

void CUDABackend::FreeMemory(void* ptr) {
    if (!m_isInitialized || ptr == nullptr) {
        return;
    }
    
    cudaError_t err = cudaFree(ptr);
    if (!CheckCUDAError(err, "cudaFree")) {
        Logger::GetInstance().Warning("Memory free failed: " + m_lastError);
    }
}

void CUDABackend::CopyHostToDevice(void* devicePtr, const void* hostPtr, size_t sizeBytes) {
    if (!m_isInitialized) {
        m_lastError = "Backend not initialized";
        return;
    }
    
    cudaError_t err = cudaMemcpy(devicePtr, hostPtr, sizeBytes, cudaMemcpyHostToDevice);
    if (!CheckCUDAError(err, "cudaMemcpy H2D")) {
        Logger::GetInstance().Error("Host-to-device copy failed: " + m_lastError);
    }
}

void CUDABackend::CopyDeviceToHost(void* hostPtr, const void* devicePtr, size_t sizeBytes) {
    if (!m_isInitialized) {
        m_lastError = "Backend not initialized";
        return;
    }
    
    cudaError_t err = cudaMemcpy(hostPtr, devicePtr, sizeBytes, cudaMemcpyDeviceToHost);
    if (!CheckCUDAError(err, "cudaMemcpy D2H")) {
        Logger::GetInstance().Error("Device-to-host copy failed: " + m_lastError);
    }
}

bool CUDABackend::ExecuteKernel(const std::string& kernelName, const KernelParams& params) {
    if (!m_isInitialized) {
        m_lastError = "Backend not initialized";
        return false;
    }
    
    // This is a placeholder - actual kernel execution happens in benchmark-specific code
    m_lastError = "ExecuteKernel not fully implemented - use benchmark classes";
    return false;
}

void CUDABackend::Synchronize() {
    if (!m_isInitialized) {
        return;
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (!CheckCUDAError(err, "cudaDeviceSynchronize")) {
        Logger::GetInstance().Warning("Synchronize failed: " + m_lastError);
    }
}

void CUDABackend::StartTimer() {
    if (!m_isInitialized || !m_startEvent) {
        return;
    }
    
    cudaEventRecord(m_startEvent);
    m_timerStarted = true;
}

void CUDABackend::StopTimer() {
    if (!m_isInitialized || !m_stopEvent || !m_timerStarted) {
        return;
    }
    
    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);
}

double CUDABackend::GetElapsedTime() {
    if (!m_isInitialized || !m_timerStarted) {
        return 0.0;
    }
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    
    m_timerStarted = false;
    return static_cast<double>(milliseconds);
}

bool CUDABackend::IsAvailable() const {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

bool CUDABackend::CheckCUDAError(cudaError_t error, const std::string& context) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << context << " failed: " << cudaGetErrorString(error);
        m_lastError = oss.str();
        return false;
    }
    
    m_lastError = "No error";
    return true;
}

} // namespace GPUBenchmark
