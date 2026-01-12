/*******************************************************************************
 * FILE: CUDABackend.h
 * 
 * PURPOSE:
 *   CUDA backend implementation for NVIDIA GPU compute benchmarking.
 *   Implements IComputeBackend interface using CUDA Runtime API.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#ifndef GPU_BENCHMARK_CUDA_BACKEND_H
#define GPU_BENCHMARK_CUDA_BACKEND_H

#include "../../core/IComputeBackend.h"
#include <cuda_runtime.h>
#include <string>

namespace GPUBenchmark {

class CUDABackend : public IComputeBackend {
public:
    CUDABackend();
    virtual ~CUDABackend();

    // IComputeBackend interface
    bool Initialize() override;
    void Shutdown() override;
    
    DeviceInfo GetDeviceInfo() const override;
    BackendType GetBackendType() const override { return BackendType::CUDA; }
    std::string GetBackendName() const override { return "CUDA"; }
    
    void* AllocateMemory(size_t sizeBytes) override;
    void FreeMemory(void* ptr) override;
    void CopyHostToDevice(void* devicePtr, const void* hostPtr, size_t sizeBytes) override;
    void CopyDeviceToHost(void* hostPtr, const void* devicePtr, size_t sizeBytes) override;
    
    bool ExecuteKernel(const std::string& kernelName, const KernelParams& params) override;
    void Synchronize() override;
    
    void StartTimer() override;
    void StopTimer() override;
    double GetElapsedTime() override;
    
    bool IsAvailable() const override;
    std::string GetLastError() const override { return m_lastError; }

private:
    bool m_isInitialized;
    int m_deviceId;
    cudaDeviceProp m_deviceProps;
    std::string m_lastError;
    
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    bool m_timerStarted;
    
    bool CheckCUDAError(cudaError_t error, const std::string& context);
};

} // namespace GPUBenchmark

#endif // GPU_BENCHMARK_CUDA_BACKEND_H
