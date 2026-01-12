/*******************************************************************************
 * FILE: ConvolutionBenchmark.cpp
 * 
 * PURPOSE:
 *   Implementation of 2D convolution benchmark wrapper.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#include "ConvolutionBenchmark.h"
#include <cmath>
#include <algorithm>

// CUDA kernel launcher (defined in convolution.cu)
extern "C" void launchConvolution2DShared(const float* d_input, float* d_output,
                                           int width, int height, void* stream);

namespace GPUBenchmark {

ConvolutionBenchmark::ConvolutionBenchmark(size_t width, size_t height)
    : m_width(width)
    , m_height(height)
    , m_iterations(100)
    , m_warmupIterations(3)
    , m_verifyResults(true)
    , m_deviceInput(nullptr)
    , m_deviceOutput(nullptr)
    , m_logger(Logger::GetInstance())
{
    // Initialize 3x3 Gaussian blur kernel
    m_kernel[0] = 1.0f/16.0f; m_kernel[1] = 2.0f/16.0f; m_kernel[2] = 1.0f/16.0f;
    m_kernel[3] = 2.0f/16.0f; m_kernel[4] = 4.0f/16.0f; m_kernel[5] = 2.0f/16.0f;
    m_kernel[6] = 1.0f/16.0f; m_kernel[7] = 2.0f/16.0f; m_kernel[8] = 1.0f/16.0f;
    
    InitializeData();
}

ConvolutionBenchmark::~ConvolutionBenchmark() {
    // Device memory is managed by backend, cleaned up in Run()
}

void ConvolutionBenchmark::SetImageSize(size_t width, size_t height) {
    m_width = width;
    m_height = height;
    InitializeData();
}

void ConvolutionBenchmark::InitializeData() {
    size_t totalPixels = m_width * m_height;
    
    m_hostInput.resize(totalPixels);
    m_hostOutput.resize(totalPixels);
    
    // Initialize with test pattern
    for (size_t i = 0; i < totalPixels; i++) {
        m_hostInput[i] = static_cast<float>(rand()) / RAND_MAX;
        m_hostOutput[i] = 0.0f;
    }
}

BenchmarkResult ConvolutionBenchmark::Run(IComputeBackend* backend) {
    BenchmarkResult result;
    result.benchmarkName = GetName();
    result.backendName = backend->GetBackendName();
    result.gpuName = backend->GetDeviceInfo().name;
    result.problemSize = m_width * m_height;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    m_logger.Debug("=== 2D Convolution Benchmark ===");
    m_logger.Debug("Image size: " + std::to_string(m_width) + "×" + std::to_string(m_height));
    m_logger.Debug("Iterations: " + std::to_string(m_iterations));
    
    size_t bytes = m_width * m_height * sizeof(float);
    
    // Allocate device memory
    m_logger.Debug("Allocating GPU memory...");
    m_deviceInput = backend->AllocateMemory(bytes);
    m_deviceOutput = backend->AllocateMemory(bytes);
    
    if (!m_deviceInput || !m_deviceOutput) {
        result.resultCorrect = false;
        m_logger.Error("Failed to allocate device memory");
        return result;
    }
    
    // Copy data to device
    m_logger.Debug("Copying data to device...");
    backend->CopyHostToDevice(m_deviceInput, m_hostInput.data(), bytes);
    
    // Warmup
    for (int i = 0; i < m_warmupIterations; i++) {
        launchConvolution2DShared((const float*)m_deviceInput, (float*)m_deviceOutput,
                                   m_width, m_height, nullptr);
        backend->Synchronize();
    }
    
    // Benchmark execution
    backend->StartTimer();
    for (int i = 0; i < m_iterations; i++) {
        launchConvolution2DShared((const float*)m_deviceInput, (float*)m_deviceOutput,
                                   m_width, m_height, nullptr);
        backend->Synchronize();
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / m_iterations;
    
    // Copy results back
    backend->CopyDeviceToHost(m_hostOutput.data(), m_deviceOutput, bytes);
    
    // Calculate bandwidth (bytes read + written)
    double totalBytes = 2.0 * bytes; // Read input, write output
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    // Verify results
    if (m_verifyResults) {
        result.resultCorrect = VerifyResults(m_hostOutput.data());
    } else {
        result.resultCorrect = true;
    }
    
    // Cleanup
    backend->FreeMemory(m_deviceInput);
    backend->FreeMemory(m_deviceOutput);
    
    m_deviceInput = m_deviceOutput = nullptr;
    
    return result;
}

bool ConvolutionBenchmark::VerifyResults(const float* output) {
    // Simple sanity check: output values should be in valid range
    size_t numSamples = std::min(m_width * m_height, size_t(1000));
    
    for (size_t i = 0; i < numSamples; i++) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            m_logger.Debug("Verification failed: invalid value at index " + std::to_string(i));
            return false;
        }
        if (output[i] < 0.0f || output[i] > 1.0f) {
            // Values should stay in [0,1] range after Gaussian blur
            m_logger.Debug("Verification warning: value out of range at index " + std::to_string(i));
        }
    }
    
    return true;
}

} // namespace GPUBenchmark
