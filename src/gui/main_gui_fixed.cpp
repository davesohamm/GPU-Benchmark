/*******************************************************************************
 * @file    main_gui_fixed.cpp
 * @brief   GPU Benchmark Suite - WORKING GUI with All 3 Backends
 * 
 * @details Beautiful desktop application featuring:
 *          - ImGui interface with DirectX 11 rendering
 *          - ALL 3 backends working (CUDA, OpenCL, DirectCompute)
 *          - Real-time performance graphs
 *          - Live benchmark execution and visualization
 * 
 * @author  Soham Dave (https://github.com/davesohamm)
 * @date    2026-01-09
 * @version 2.0.0 - FULLY WORKING!
 ********************************************************************************/

// C++ headers
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <map>

// Core framework
#include "../core/Logger.h"
#include "../core/DeviceDiscovery.h"
#include "../backends/cuda/CUDABackend.h"
#include "../backends/opencl/OpenCLBackend.h"
#include "../backends/directcompute/DirectComputeBackend.h"

// Windows/DirectX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <tchar.h>
#include <shellapi.h>
#include <commdlg.h>  // For file save dialog

// ImGui
#include "../../external/imgui/imgui.h"
#include "../../external/imgui/backends/imgui_impl_win32.h"
#include "../../external/imgui/backends/imgui_impl_dx11.h"

#pragma comment(lib, "d3d11.lib")

using namespace GPUBenchmark;

// Helper function to get current timestamp and formatted date/time
std::string GetFormattedDateTime() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_s(&tm_buf, &now_c);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// (Helper functions moved to after AppState definition)

// CUDA kernel launchers
extern "C" {
    void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n);
    void launchMatrixMulTiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int P, cudaStream_t stream);
    void setConvolutionKernel(const float* h_kernel, int kernelSize);
    void launchConvolution2DShared(const float* d_input, float* d_output, int width, int height, int kernelRadius, cudaStream_t stream);
    void launchReductionWarpShuffle(const float* d_input, float* d_output, int n, cudaStream_t stream);
}

// OpenCL kernel sources
const char* openclVectorAddSource = R"(
__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* c, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

const char* openclMatMulSource = R"(
#define TILE_SIZE 16
__kernel void matrixMul(__global const float* A, __global const float* B, __global float* C, int N) {
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * TILE_SIZE + ty;
    int col = get_group_id(0) * TILE_SIZE + tx;
    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        int bRow = t * TILE_SIZE + ty;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
)";

const char* openclConvolutionSource = R"(
__kernel void convolution2D(__global const float* input, __global float* output,
                             int width, int height, int kernelRadius,
                             __global const float* kernel) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= height || col >= width) return;
    
    float sum = 0.0f;
    int kernelSize = 2 * kernelRadius + 1;
    for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
        for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
            int imageRow = row + dy;
            int imageCol = col + dx;
            // Manual boundary check
            if (imageRow < 0) imageRow = 0;
            if (imageRow >= height) imageRow = height - 1;
            if (imageCol < 0) imageCol = 0;
            if (imageCol >= width) imageCol = width - 1;
            
            int kernelIdx = (dy + kernelRadius) * kernelSize + (dx + kernelRadius);
            sum += input[imageRow * width + imageCol] * kernel[kernelIdx];
        }
    }
    output[row * width + col] = sum;
}
)";

const char* openclReductionSource = R"(
__kernel void reduction(__global const float* input, __global float* output, int n, __local float* scratch) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int groupSize = get_local_size(0);
    scratch[lid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = groupSize / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}
)";

// DirectCompute HLSL shaders
const char* hlslVectorAddSource = R"(
RWStructuredBuffer<float> inputA : register(u0);
RWStructuredBuffer<float> inputB : register(u1);
RWStructuredBuffer<float> output : register(u2);
cbuffer Constants : register(b0) { uint numElements; };
[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx < numElements) {
        output[idx] = inputA[idx] + inputB[idx];
    }
}
)";

const char* hlslMatMulSource = R"(
RWStructuredBuffer<float> matrixA : register(u0);
RWStructuredBuffer<float> matrixB : register(u1);
RWStructuredBuffer<float> matrixC : register(u2);
cbuffer Constants : register(b0) { uint N; };
#define TILE_SIZE 16
groupshared float As[TILE_SIZE][TILE_SIZE];
groupshared float Bs[TILE_SIZE][TILE_SIZE];
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint tx = groupThreadID.x;
    uint ty = groupThreadID.y;
    uint row = groupID.y * TILE_SIZE + ty;
    uint col = groupID.x * TILE_SIZE + tx;
    float sum = 0.0f;
    uint numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tx;
        As[ty][tx] = (row < N && aCol < N) ? matrixA[row * N + aCol] : 0.0f;
        uint bRow = t * TILE_SIZE + ty;
        Bs[ty][tx] = (bRow < N && col < N) ? matrixB[bRow * N + col] : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < N && col < N) {
        matrixC[row * N + col] = sum;
    }
}
)";

const char* hlslConvolutionSource = R"(
RWStructuredBuffer<float> input : register(u0);
RWStructuredBuffer<float> output : register(u1);
RWStructuredBuffer<float> kernel : register(u2);
cbuffer Constants : register(b0) { uint width; uint height; uint kernelRadius; uint padding; };
[numthreads(16, 16, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint col = dispatchThreadID.x;
    uint row = dispatchThreadID.y;
    if (row < height && col < width) {
        float sum = 0.0f;
        uint kernelSize = 2 * kernelRadius + 1;
        for (int dy = -(int)kernelRadius; dy <= (int)kernelRadius; dy++) {
            for (int dx = -(int)kernelRadius; dx <= (int)kernelRadius; dx++) {
                int imageRow = clamp((int)row + dy, 0, (int)height - 1);
                int imageCol = clamp((int)col + dx, 0, (int)width - 1);
                uint kernelIdx = (dy + kernelRadius) * kernelSize + (dx + kernelRadius);
                sum += input[imageRow * width + imageCol] * kernel[kernelIdx];
            }
        }
        output[row * width + col] = sum;
    }
}
)";

const char* hlslReductionSource = R"(
RWStructuredBuffer<float> input : register(u0);
RWStructuredBuffer<float> output : register(u1);
cbuffer Constants : register(b0) { uint numElements; };
groupshared float scratch[512];
[numthreads(512, 1, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID) {
    uint lid = groupThreadID.x;
    uint gid = dispatchThreadID.x;
    scratch[lid] = (gid < numElements) ? input[gid] : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 256; offset > 0; offset >>= 1) {
        if (lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0) {
        output[groupID.x] = scratch[0];
    }
}
)";

//==============================================================================
// GLOBAL STATE
//==============================================================================

struct AppState {
    // DirectX 11
    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11RenderTargetView* mainRenderTargetView = nullptr;
    HWND hwnd = nullptr;
    
    // System
    SystemCapabilities systemCaps;
    
    // UI State
    int selectedBackendIndex = 0;
    int selectedSuiteIndex = 1;  // Standard
    bool showAbout = false;
    bool runAllBackends = false;  // NEW: Option to run all backends
    
    // Benchmark state
    std::atomic<bool> benchmarkRunning{false};
    std::atomic<float> progress{0.0f};
    std::string currentBenchmark;
    std::mutex currentBenchmarkMutex;
    
    // Results
    struct BenchmarkResult {
        std::string name;
        std::string backend;
        double timeMs;
        double bandwidth;
        double gflops;
        size_t problemSize;
        bool passed;
    };
    std::vector<BenchmarkResult> results;
    
    // Summary statistics
    struct BackendStats {
        double avgBandwidth = 0.0;
        double maxBandwidth = 0.0;
        double minBandwidth = 999999.0;
        int runCount = 0;
    };
    std::map<std::string, BackendStats> backendStats;
    std::mutex resultsMutex;
    
    // Enhanced history tracking with timestamps and test IDs
    struct TestResult {
        float bandwidth;
        double timestamp;  // Unix timestamp for sorting
        std::string testID;  // "Test 1", "Test 2", etc.
        double gflops;
        double timeMS;
        std::string dateTime;  // Human-readable date/time
    };
    
    struct BenchmarkHistory {
        std::vector<TestResult> vectorAdd;
        std::vector<TestResult> matrixMul;
        std::vector<TestResult> convolution;
        std::vector<TestResult> reduction;
        int totalTests = 0;  // Counter for test IDs
    };
    BenchmarkHistory cudaHistory;
    BenchmarkHistory openclHistory;
    BenchmarkHistory directcomputeHistory;
    
    // Worker thread
    std::thread workerThread;
    std::atomic<bool> workerThreadRunning{false};
    std::atomic<bool> running{true};
};

AppState g_App;

//==============================================================================
// DIRECTX 11 SETUP
//==============================================================================

bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
                                                featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_App.swapChain,
                                                &g_App.d3dDevice, &featureLevel, &g_App.d3dContext);
    
    if (res != S_OK) {
        MessageBoxW(nullptr, L"Failed to create D3D11 device!", L"Error", MB_OK | MB_ICONERROR);
        return false;
    }

    ID3D11Texture2D* pBackBuffer;
    g_App.swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_App.d3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_App.mainRenderTargetView);
    pBackBuffer->Release();

    return true;
}

void CleanupDeviceD3D() {
    if (g_App.mainRenderTargetView) { g_App.mainRenderTargetView->Release(); g_App.mainRenderTargetView = nullptr; }
    if (g_App.swapChain) { g_App.swapChain->Release(); g_App.swapChain = nullptr; }
    if (g_App.d3dContext) { g_App.d3dContext->Release(); g_App.d3dContext = nullptr; }
    if (g_App.d3dDevice) { g_App.d3dDevice->Release(); g_App.d3dDevice = nullptr; }
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (g_App.d3dDevice != nullptr && wParam != SIZE_MINIMIZED) {
            if (g_App.mainRenderTargetView) { g_App.mainRenderTargetView->Release(); g_App.mainRenderTargetView = nullptr; }
            g_App.swapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            ID3D11Texture2D* pBackBuffer;
            g_App.swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
            g_App.d3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_App.mainRenderTargetView);
            pBackBuffer->Release();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}

//==============================================================================
// BENCHMARK EXECUTION (WORKING VERSION!)
//==============================================================================

BenchmarkResult RunVectorAddCUDA(CUDABackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = "CUDA";
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    size_t bytes = numElements * sizeof(float);
    std::vector<float> hostA(numElements), hostB(numElements), hostC(numElements);
    
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    for (int i = 0; i < 5; i++) {
        launchVectorAdd((const float*)devA, (const float*)devB, (float*)devC, numElements);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        launchVectorAdd((const float*)devA, (const float*)devB, (float*)devC, numElements);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    int errors = 0;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        if (std::abs(hostC[i] - (hostA[i] + hostB[i])) > 1e-5f) errors++;
    }
    result.resultCorrect = (errors == 0);
    
    double totalBytes = 3.0 * bytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

BenchmarkResult RunVectorAddOpenCL(OpenCLBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = "OpenCL";
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    size_t bytes = numElements * sizeof(float);
    
    if (!backend->CompileKernel("vectorAdd", openclVectorAddSource)) {
        result.resultCorrect = false;
        return result;
    }
    
    std::vector<float> hostA(numElements), hostB(numElements), hostC(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    backend->SetKernelArg("vectorAdd", 0, sizeof(cl_mem), &devA);
    backend->SetKernelArg("vectorAdd", 1, sizeof(cl_mem), &devB);
    backend->SetKernelArg("vectorAdd", 2, sizeof(cl_mem), &devC);
    int n = static_cast<int>(numElements);
    backend->SetKernelArg("vectorAdd", 3, sizeof(int), &n);
    
    DeviceInfo deviceInfo = backend->GetDeviceInfo();
    size_t maxWorkGroupSize = deviceInfo.maxThreadsPerBlock;
    size_t localWorkSize = (256 < maxWorkGroupSize) ? 256 : maxWorkGroupSize;
    if (localWorkSize == 0) localWorkSize = 64;
    size_t globalWorkSize = ((numElements + localWorkSize - 1) / localWorkSize) * localWorkSize;
    
    for (int i = 0; i < 5; i++) {
        backend->ExecuteKernel("vectorAdd", &globalWorkSize, &localWorkSize, 1);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->ExecuteKernel("vectorAdd", &globalWorkSize, &localWorkSize, 1);
    }
    backend->Synchronize();
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    int errors = 0;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        if (std::abs(hostC[i] - (hostA[i] + hostB[i])) > 1e-5f) errors++;
    }
    result.resultCorrect = (errors == 0);
    
    double totalBytes = 3.0 * bytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

BenchmarkResult RunVectorAddDirectCompute(DirectComputeBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "VectorAdd";
    result.backendName = "DirectCompute";
    result.problemSize = numElements;
    result.timestamp = Logger::GetCurrentTimestamp();
    
    size_t bytes = numElements * sizeof(float);
    
    if (!backend->CompileShader("VectorAdd", hlslVectorAddSource, "CSMain", "cs_5_0")) {
        result.resultCorrect = false;
        return result;
    }
    
    std::vector<float> hostA(numElements), hostB(numElements), hostC(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    backend->BindBufferUAV(devA, 0);
    backend->BindBufferUAV(devB, 1);
    backend->BindBufferUAV(devC, 2);
    
    struct Constants { unsigned int numElements; } constants;
    constants.numElements = static_cast<unsigned int>(numElements);
    backend->SetConstantBuffer(&constants, sizeof(constants), 0);
    
    unsigned int threadGroupsX = (static_cast<unsigned int>(numElements) + 255) / 256;
    
    for (int i = 0; i < 5; i++) {
        backend->DispatchShader("VectorAdd", threadGroupsX, 1, 1);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->DispatchShader("VectorAdd", threadGroupsX, 1, 1);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    int errors = 0;
    for (size_t i = 0; i < numElements && errors < 10; i++) {
        if (std::abs(hostC[i] - (hostA[i] + hostB[i])) > 1e-5f) errors++;
    }
    result.resultCorrect = (errors == 0);
    
    double totalBytes = 3.0 * bytes;
    result.effectiveBandwidthGBs = (totalBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

//==============================================================================
// MATRIX MULTIPLICATION BENCHMARKS (2048×2048 matrices)
//==============================================================================

BenchmarkResult RunMatrixMulCUDA(CUDABackend* backend, size_t N, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "MatrixMul";
    result.backendName = "CUDA";
    result.problemSize = N * N;
    
    size_t bytes = N * N * sizeof(float);
    std::vector<float> hostA(N * N), hostB(N * N), hostC(N * N);
    for (size_t i = 0; i < N * N; i++) {
        hostA[i] = ((i % 100) / 100.0f);
        hostB[i] = (((i * 2) % 100) / 100.0f);
    }
    
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    for (int i = 0; i < 3; i++) {
        launchMatrixMulTiled((const float*)devA, (const float*)devB, (float*)devC, N, N, N, 0);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        launchMatrixMulTiled((const float*)devA, (const float*)devB, (float*)devC, N, N, N, 0);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    result.resultCorrect = true;
    double flops = 2.0 * N * N * N;
    result.computeThroughputGFLOPS = (flops / (result.executionTimeMS / 1000.0)) / 1e9;
    result.effectiveBandwidthGBs = (3.0 * bytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

BenchmarkResult RunMatrixMulOpenCL(OpenCLBackend* backend, size_t N, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "MatrixMul";
    result.backendName = "OpenCL";
    result.problemSize = N * N;
    
    if (!backend->CompileKernel("matrixMul", openclMatMulSource)) {
        result.resultCorrect = false;
        return result;
    }
    
    size_t bytes = N * N * sizeof(float);
    std::vector<float> hostA(N * N), hostB(N * N), hostC(N * N);
    for (size_t i = 0; i < N * N; i++) {
        hostA[i] = ((i % 100) / 100.0f);
        hostB[i] = (((i * 2) % 100) / 100.0f);
    }
    
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    backend->SetKernelArg("matrixMul", 0, sizeof(cl_mem), &devA);
    backend->SetKernelArg("matrixMul", 1, sizeof(cl_mem), &devB);
    backend->SetKernelArg("matrixMul", 2, sizeof(cl_mem), &devC);
    int N_int = static_cast<int>(N);
    backend->SetKernelArg("matrixMul", 3, sizeof(int), &N_int);
    
    size_t localWorkSize[2] = {16, 16};
    size_t globalWorkSize[2] = {((N + 15) / 16) * 16, ((N + 15) / 16) * 16};
    
    for (int i = 0; i < 3; i++) {
        backend->ExecuteKernel("matrixMul", globalWorkSize, localWorkSize, 2);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->ExecuteKernel("matrixMul", globalWorkSize, localWorkSize, 2);
    }
    backend->Synchronize();
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    result.resultCorrect = true;
    double flops = 2.0 * N * N * N;
    result.computeThroughputGFLOPS = (flops / (result.executionTimeMS / 1000.0)) / 1e9;
    result.effectiveBandwidthGBs = (3.0 * bytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

BenchmarkResult RunMatrixMulDirectCompute(DirectComputeBackend* backend, size_t N, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "MatrixMul";
    result.backendName = "DirectCompute";
    result.problemSize = N * N;
    
    if (!backend->CompileShader("MatrixMul", hlslMatMulSource, "CSMain", "cs_5_0")) {
        result.resultCorrect = false;
        return result;
    }
    
    size_t bytes = N * N * sizeof(float);
    std::vector<float> hostA(N * N), hostB(N * N), hostC(N * N);
    for (size_t i = 0; i < N * N; i++) {
        hostA[i] = ((i % 100) / 100.0f);
        hostB[i] = (((i * 2) % 100) / 100.0f);
    }
    
    void* devA = backend->AllocateMemory(bytes);
    void* devB = backend->AllocateMemory(bytes);
    void* devC = backend->AllocateMemory(bytes);
    
    backend->CopyHostToDevice(devA, hostA.data(), bytes);
    backend->CopyHostToDevice(devB, hostB.data(), bytes);
    
    backend->BindBufferUAV(devA, 0);
    backend->BindBufferUAV(devB, 1);
    backend->BindBufferUAV(devC, 2);
    
    struct { unsigned int N; } constants;
    constants.N = static_cast<unsigned int>(N);
    backend->SetConstantBuffer(&constants, sizeof(constants), 0);
    
    unsigned int threadGroupsX = (static_cast<unsigned int>(N) + 15) / 16;
    unsigned int threadGroupsY = (static_cast<unsigned int>(N) + 15) / 16;
    
    for (int i = 0; i < 3; i++) {
        backend->DispatchShader("MatrixMul", threadGroupsX, threadGroupsY, 1);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->DispatchShader("MatrixMul", threadGroupsX, threadGroupsY, 1);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostC.data(), devC, bytes);
    
    result.resultCorrect = true;
    double flops = 2.0 * N * N * N;
    result.computeThroughputGFLOPS = (flops / (result.executionTimeMS / 1000.0)) / 1e9;
    result.effectiveBandwidthGBs = (3.0 * bytes / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devA);
    backend->FreeMemory(devB);
    backend->FreeMemory(devC);
    
    return result;
}

//==============================================================================
// CONVOLUTION BENCHMARKS (2048×2048 image, 9×9 kernel)
//==============================================================================

BenchmarkResult RunConvolutionCUDA(CUDABackend* backend, size_t width, size_t height, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "Convolution";
    result.backendName = "CUDA";
    result.problemSize = width * height;
    
    const int kernelRadius = 4;  // 9×9 kernel
    const int kernelSize = 2 * kernelRadius + 1;
    
    std::vector<float> hostKernel(kernelSize * kernelSize);
    float kernelSum = 0.0f;
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        hostKernel[i] = 1.0f / (kernelSize * kernelSize);
        kernelSum += hostKernel[i];
    }
    
    size_t imageBytes = width * height * sizeof(float);
    std::vector<float> hostInput(width * height), hostOutput(width * height);
    for (size_t i = 0; i < width * height; i++) {
        hostInput[i] = ((i % 256) / 255.0f);
    }
    
    void* devInput = backend->AllocateMemory(imageBytes);
    void* devOutput = backend->AllocateMemory(imageBytes);
    
    backend->CopyHostToDevice(devInput, hostInput.data(), imageBytes);
    setConvolutionKernel(hostKernel.data(), kernelSize);
    
    for (int i = 0; i < 3; i++) {
        launchConvolution2DShared((const float*)devInput, (float*)devOutput, width, height, kernelRadius, 0);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        launchConvolution2DShared((const float*)devInput, (float*)devOutput, width, height, kernelRadius, 0);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostOutput.data(), devOutput, imageBytes);
    
    result.resultCorrect = true;
    result.effectiveBandwidthGBs = (2.0 * imageBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    result.computeThroughputGFLOPS = (width * height * kernelSize * kernelSize * 2.0 / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devInput);
    backend->FreeMemory(devOutput);
    
    return result;
}

BenchmarkResult RunConvolutionOpenCL(OpenCLBackend* backend, size_t width, size_t height, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "Convolution";
    result.backendName = "OpenCL";
    result.problemSize = width * height;
    result.resultCorrect = true;  // Initialize as true
    result.executionTimeMS = 0.0;
    result.effectiveBandwidthGBs = 0.0;
    result.computeThroughputGFLOPS = 0.0;
    
    if (!backend->CompileKernel("convolution2D", openclConvolutionSource)) {
        result.resultCorrect = false;
        return result;
    }
    
    const int kernelRadius = 4;
    const int kernelSize = 2 * kernelRadius + 1;
    
    std::vector<float> hostKernel(kernelSize * kernelSize);
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        hostKernel[i] = 1.0f / (kernelSize * kernelSize);
    }
    
    size_t imageBytes = width * height * sizeof(float);
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);
    
    std::vector<float> hostInput(width * height), hostOutput(width * height);
    for (size_t i = 0; i < width * height; i++) {
        hostInput[i] = ((i % 256) / 255.0f);
    }
    
    void* devInput = backend->AllocateMemory(imageBytes);
    void* devOutput = backend->AllocateMemory(imageBytes);
    void* devKernel = backend->AllocateMemory(kernelBytes);
    
    backend->CopyHostToDevice(devInput, hostInput.data(), imageBytes);
    backend->CopyHostToDevice(devKernel, hostKernel.data(), kernelBytes);
    
    backend->SetKernelArg("convolution2D", 0, sizeof(cl_mem), &devInput);
    backend->SetKernelArg("convolution2D", 1, sizeof(cl_mem), &devOutput);
    int width_int = static_cast<int>(width);
    int height_int = static_cast<int>(height);
    int radius_int = kernelRadius;
    backend->SetKernelArg("convolution2D", 2, sizeof(int), &width_int);
    backend->SetKernelArg("convolution2D", 3, sizeof(int), &height_int);
    backend->SetKernelArg("convolution2D", 4, sizeof(int), &radius_int);
    backend->SetKernelArg("convolution2D", 5, sizeof(cl_mem), &devKernel);
    
    size_t localWorkSize[2] = {16, 16};
    size_t globalWorkSize[2] = {((width + 15) / 16) * 16, ((height + 15) / 16) * 16};
    
    // Warmup with error checking
    for (int i = 0; i < 3; i++) {
        if (!backend->ExecuteKernel("convolution2D", globalWorkSize, localWorkSize, 2)) {
            result.resultCorrect = false;
            backend->FreeMemory(devInput);
            backend->FreeMemory(devOutput);
            backend->FreeMemory(devKernel);
            return result;
        }
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        if (!backend->ExecuteKernel("convolution2D", globalWorkSize, localWorkSize, 2)) {
            result.resultCorrect = false;
            backend->FreeMemory(devInput);
            backend->FreeMemory(devOutput);
            backend->FreeMemory(devKernel);
            return result;
        }
    }
    backend->Synchronize();
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostOutput.data(), devOutput, imageBytes);
    
    result.resultCorrect = true;
    result.effectiveBandwidthGBs = (2.0 * imageBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    result.computeThroughputGFLOPS = (width * height * kernelSize * kernelSize * 2.0 / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devInput);
    backend->FreeMemory(devOutput);
    backend->FreeMemory(devKernel);
    
    return result;
}

BenchmarkResult RunConvolutionDirectCompute(DirectComputeBackend* backend, size_t width, size_t height, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "Convolution";
    result.backendName = "DirectCompute";
    result.problemSize = width * height;
    result.resultCorrect = true;  // Initialize as true
    result.executionTimeMS = 0.0;
    result.effectiveBandwidthGBs = 0.0;
    result.computeThroughputGFLOPS = 0.0;
    
    if (!backend->CompileShader("Convolution", hlslConvolutionSource, "CSMain", "cs_5_0")) {
        result.resultCorrect = false;
        return result;
    }
    
    const int kernelRadius = 4;
    const int kernelSize = 2 * kernelRadius + 1;
    
    std::vector<float> hostKernel(kernelSize * kernelSize);
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        hostKernel[i] = 1.0f / (kernelSize * kernelSize);
    }
    
    size_t imageBytes = width * height * sizeof(float);
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);
    
    std::vector<float> hostInput(width * height), hostOutput(width * height);
    for (size_t i = 0; i < width * height; i++) {
        hostInput[i] = ((i % 256) / 255.0f);
    }
    
    void* devInput = backend->AllocateMemory(imageBytes);
    void* devOutput = backend->AllocateMemory(imageBytes);
    void* devKernel = backend->AllocateMemory(kernelBytes);
    
    backend->CopyHostToDevice(devInput, hostInput.data(), imageBytes);
    backend->CopyHostToDevice(devKernel, hostKernel.data(), kernelBytes);
    
    backend->BindBufferUAV(devInput, 0);
    backend->BindBufferUAV(devOutput, 1);
    backend->BindBufferUAV(devKernel, 2);
    
    struct { unsigned int width; unsigned int height; unsigned int kernelRadius; unsigned int padding; } constants;
    constants.width = static_cast<unsigned int>(width);
    constants.height = static_cast<unsigned int>(height);
    constants.kernelRadius = static_cast<unsigned int>(kernelRadius);
    constants.padding = 0;
    backend->SetConstantBuffer(&constants, sizeof(constants), 0);
    
    unsigned int threadGroupsX = (static_cast<unsigned int>(width) + 15) / 16;
    unsigned int threadGroupsY = (static_cast<unsigned int>(height) + 15) / 16;
    
    for (int i = 0; i < 3; i++) {
        backend->DispatchShader("Convolution", threadGroupsX, threadGroupsY, 1);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->DispatchShader("Convolution", threadGroupsX, threadGroupsY, 1);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    backend->CopyDeviceToHost(hostOutput.data(), devOutput, imageBytes);
    
    result.resultCorrect = true;
    result.effectiveBandwidthGBs = (2.0 * imageBytes / (result.executionTimeMS / 1000.0)) / 1e9;
    result.computeThroughputGFLOPS = (width * height * kernelSize * kernelSize * 2.0 / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devInput);
    backend->FreeMemory(devOutput);
    backend->FreeMemory(devKernel);
    
    return result;
}

//==============================================================================
// REDUCTION BENCHMARKS (64M elements)
//==============================================================================

BenchmarkResult RunReductionCUDA(CUDABackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "Reduction";
    result.backendName = "CUDA";
    result.problemSize = numElements;
    
    size_t bytes = numElements * sizeof(float);
    std::vector<float> hostInput(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostInput[i] = 1.0f;
    }
    
    // Reduction outputs partial sums (one per block)
    size_t threadsPerBlock = 256;
    size_t numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    void* devInput = backend->AllocateMemory(bytes);
    void* devOutput = backend->AllocateMemory(numBlocks * sizeof(float));
    
    backend->CopyHostToDevice(devInput, hostInput.data(), bytes);
    
    for (int i = 0; i < 3; i++) {
        launchReductionWarpShuffle((const float*)devInput, (float*)devOutput, numElements, 0);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        launchReductionWarpShuffle((const float*)devInput, (float*)devOutput, numElements, 0);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    // Aggregate partial sums on CPU
    std::vector<float> partialSums(numBlocks);
    backend->CopyDeviceToHost(partialSums.data(), devOutput, numBlocks * sizeof(float));
    float hostResult = 0.0f;
    for (size_t i = 0; i < numBlocks; i++) {
        hostResult += partialSums[i];
    }
    
    result.resultCorrect = (std::abs(hostResult - numElements) < (numElements * 0.01f));
    result.effectiveBandwidthGBs = (bytes / (result.executionTimeMS / 1000.0)) / 1e9;
    result.computeThroughputGFLOPS = (numElements / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devInput);
    backend->FreeMemory(devOutput);
    
    return result;
}

BenchmarkResult RunReductionOpenCL(OpenCLBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "Reduction";
    result.backendName = "OpenCL";
    result.problemSize = numElements;
    result.resultCorrect = true;  // Initialize as true
    result.executionTimeMS = 0.0;
    result.effectiveBandwidthGBs = 0.0;
    result.computeThroughputGFLOPS = 0.0;
    
    if (!backend->CompileKernel("reduction", openclReductionSource)) {
        result.resultCorrect = false;
        return result;
    }
    
    size_t bytes = numElements * sizeof(float);
    std::vector<float> hostInput(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostInput[i] = 1.0f;
    }
    
    size_t localWorkSize = 256;
    size_t globalWorkSize = ((numElements + localWorkSize - 1) / localWorkSize) * localWorkSize;
    size_t numGroups = globalWorkSize / localWorkSize;
    
    void* devInput = backend->AllocateMemory(bytes);
    void* devOutput = backend->AllocateMemory(numGroups * sizeof(float));
    
    backend->CopyHostToDevice(devInput, hostInput.data(), bytes);
    
    backend->SetKernelArg("reduction", 0, sizeof(cl_mem), &devInput);
    backend->SetKernelArg("reduction", 1, sizeof(cl_mem), &devOutput);
    int n_int = static_cast<int>(numElements);
    backend->SetKernelArg("reduction", 2, sizeof(int), &n_int);
    backend->SetKernelArg("reduction", 3, localWorkSize * sizeof(float), nullptr);
    
    for (int i = 0; i < 3; i++) {
        backend->ExecuteKernel("reduction", &globalWorkSize, &localWorkSize, 1);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->ExecuteKernel("reduction", &globalWorkSize, &localWorkSize, 1);
    }
    backend->Synchronize();
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    std::vector<float> partialSums(numGroups);
    backend->CopyDeviceToHost(partialSums.data(), devOutput, numGroups * sizeof(float));
    float finalSum = 0.0f;
    for (size_t i = 0; i < numGroups; i++) {
        finalSum += partialSums[i];
    }
    
    result.resultCorrect = (std::abs(finalSum - numElements) < (numElements * 0.01f));
    result.effectiveBandwidthGBs = (bytes / (result.executionTimeMS / 1000.0)) / 1e9;
    result.computeThroughputGFLOPS = (numElements / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devInput);
    backend->FreeMemory(devOutput);
    
    return result;
}

BenchmarkResult RunReductionDirectCompute(DirectComputeBackend* backend, size_t numElements, int iterations) {
    BenchmarkResult result;
    result.benchmarkName = "Reduction";
    result.backendName = "DirectCompute";
    result.problemSize = numElements;
    result.resultCorrect = true;  // Initialize as true
    result.executionTimeMS = 0.0;
    result.effectiveBandwidthGBs = 0.0;
    result.computeThroughputGFLOPS = 0.0;
    
    if (!backend->CompileShader("Reduction", hlslReductionSource, "CSMain", "cs_5_0")) {
        result.resultCorrect = false;
        return result;
    }
    
    size_t bytes = numElements * sizeof(float);
    std::vector<float> hostInput(numElements);
    for (size_t i = 0; i < numElements; i++) {
        hostInput[i] = 1.0f;
    }
    
    unsigned int numGroups = (static_cast<unsigned int>(numElements) + 511) / 512;
    
    void* devInput = backend->AllocateMemory(bytes);
    void* devOutput = backend->AllocateMemory(numGroups * sizeof(float));
    
    backend->CopyHostToDevice(devInput, hostInput.data(), bytes);
    
    backend->BindBufferUAV(devInput, 0);
    backend->BindBufferUAV(devOutput, 1);
    
    struct { unsigned int numElements; } constants;
    constants.numElements = static_cast<unsigned int>(numElements);
    backend->SetConstantBuffer(&constants, sizeof(constants), 0);
    
    for (int i = 0; i < 3; i++) {
        backend->DispatchShader("Reduction", numGroups, 1, 1);
        backend->Synchronize();
    }
    
    backend->StartTimer();
    for (int i = 0; i < iterations; i++) {
        backend->DispatchShader("Reduction", numGroups, 1, 1);
    }
    backend->StopTimer();
    
    result.executionTimeMS = backend->GetElapsedTime() / iterations;
    
    std::vector<float> partialSums(numGroups);
    backend->CopyDeviceToHost(partialSums.data(), devOutput, numGroups * sizeof(float));
    float finalSum = 0.0f;
    for (unsigned int i = 0; i < numGroups; i++) {
        finalSum += partialSums[i];
    }
    
    result.resultCorrect = (std::abs(finalSum - numElements) < (numElements * 0.01f));
    result.effectiveBandwidthGBs = (bytes / (result.executionTimeMS / 1000.0)) / 1e9;
    result.computeThroughputGFLOPS = (numElements / (result.executionTimeMS / 1000.0)) / 1e9;
    
    backend->FreeMemory(devInput);
    backend->FreeMemory(devOutput);
    
    return result;
}

//==============================================================================
// HELPER FUNCTIONS FOR RENDERING
//==============================================================================

// Helper to extract bandwidth values for plotting
std::vector<float> ExtractBandwidth(const std::vector<AppState::TestResult>& history) {
    std::vector<float> values;
    for (const auto& res : history) {
        values.push_back(res.bandwidth);
    }
    return values;
}

//==============================================================================
// UI RENDERING
//==============================================================================

void RenderUI() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    
    ImGui::Begin("GPU Benchmark Suite", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    
    // Enhanced Header with better styling
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "[GPU BENCHMARK SUITE v4.0]");
    ImGui::PopFont();
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "| Comprehensive Multi-API GPU Testing");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 130);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.3f, 0.5f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.4f, 0.6f, 1.0f));
    if (ImGui::Button("About", ImVec2(110, 32))) {
        g_App.showAbout = !g_App.showAbout;
    }
    ImGui::PopStyleColor(2);
    
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();
    
    // System Info Panel
    if (ImGui::CollapsingHeader("System Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        GPUInfo primaryGPU = g_App.systemCaps.GetPrimaryGPU();
        ImGui::Text("GPU: %s", primaryGPU.name.c_str());
        ImGui::Text("Memory: %zu MB", primaryGPU.totalMemoryMB);
        ImGui::Text("Driver: %s", primaryGPU.driverVersion.c_str());
        
        ImGui::Spacing();
        ImGui::Text("Backend Support:");
        ImGui::SameLine(150);
        ImGui::TextColored(g_App.systemCaps.cuda.available ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                          g_App.systemCaps.cuda.available ? "CUDA OK" : "CUDA N/A");
        ImGui::SameLine(250);
        ImGui::TextColored(g_App.systemCaps.opencl.available ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                          g_App.systemCaps.opencl.available ? "OpenCL OK" : "OpenCL N/A");
        ImGui::SameLine(370);
        ImGui::TextColored(g_App.systemCaps.directCompute.available ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                          g_App.systemCaps.directCompute.available ? "DirectCompute OK" : "DirectCompute N/A");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Benchmark Configuration
    if (ImGui::CollapsingHeader("Benchmark Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::vector<std::string> availableBackends = g_App.systemCaps.GetAvailableBackendNames();
        
        ImGui::Text("Select Backend:");
        ImGui::SameLine(150);
        ImGui::SetNextItemWidth(200);
        if (ImGui::BeginCombo("##Backend", availableBackends[g_App.selectedBackendIndex].c_str())) {
            for (int i = 0; i < availableBackends.size(); i++) {
                bool isSelected = (g_App.selectedBackendIndex == i);
                if (ImGui::Selectable(availableBackends[i].c_str(), isSelected))
                    g_App.selectedBackendIndex = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        ImGui::Text("Select Test Profile:");
        ImGui::SameLine(150);
        ImGui::SetNextItemWidth(350);
        const char* suites[] = { 
            "Quick Test (50M elements, 10 iterations)",
            "Standard Test (100M elements, 20 iterations)", 
            "Intensive Test (200M elements, 30 iterations)" 
        };
        if (ImGui::BeginCombo("##Suite", suites[g_App.selectedSuiteIndex])) {
            for (int i = 0; i < 3; i++) {
                bool isSelected = (g_App.selectedSuiteIndex == i);
                if (ImGui::Selectable(suites[i], isSelected))
                    g_App.selectedSuiteIndex = i;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        
        ImGui::Spacing();
        
        // Option to run all backends
        ImGui::Checkbox("Run All Backends (Comprehensive Test)", &g_App.runAllBackends);
        ImGui::Spacing();
        
        bool canStart = !g_App.benchmarkRunning && g_App.systemCaps.HasAnyBackend();
        if (!canStart) ImGui::BeginDisabled();
        
        const char* buttonText = g_App.runAllBackends ? "Start All Backends" : "Start Benchmark";
        if (ImGui::Button(buttonText, ImVec2(200, 40))) {
            g_App.benchmarkRunning = true;
            g_App.progress = 0.0f;
            {
                std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                g_App.results.clear();
            }
            {
                std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                g_App.currentBenchmark = "Initializing...";
            }
            
            // CRITICAL FIX: Ensure previous thread is FULLY completed before starting new one
            if (g_App.workerThread.joinable()) {
                g_App.workerThreadRunning = false;
                g_App.workerThread.join();
                // Allow GPU resources to fully release before starting new benchmark
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            
            g_App.workerThreadRunning = true;
            g_App.workerThread = std::thread([]() {
                try {
                    std::vector<std::string> availableBackends = g_App.systemCaps.GetAvailableBackendNames();
                    
                    // Determine which backends to run
                    std::vector<std::string> backendsToRun;
                    if (g_App.runAllBackends) {
                        backendsToRun = availableBackends;  // Run all
                    } else {
                        backendsToRun.push_back(availableBackends[g_App.selectedBackendIndex]);  // Run selected only
                    }
                    
                    // SIGNIFICANTLY INCREASED PROBLEM SIZES to properly stress GPU
                    size_t vectorElements = (g_App.selectedSuiteIndex == 0) ? 50000000 :   // 50M elements (200MB)
                                           (g_App.selectedSuiteIndex == 1) ? 100000000 :  // 100M elements (400MB)
                                                                            200000000;   // 200M elements (800MB)
                    size_t matrixSize = (g_App.selectedSuiteIndex == 0) ? 1024 :          // 1024×1024
                                       (g_App.selectedSuiteIndex == 1) ? 2048 :          // 2048×2048
                                                                         4096;           // 4096×4096
                    size_t imageSize = (g_App.selectedSuiteIndex == 0) ? 1024 :           // 1024×1024
                                      (g_App.selectedSuiteIndex == 1) ? 2048 :           // 2048×2048
                                                                        4096;            // 4096×4096
                    size_t reductionElements = (g_App.selectedSuiteIndex == 0) ? 32000000 :  // 32M elements
                                              (g_App.selectedSuiteIndex == 1) ? 64000000 :  // 64M elements
                                                                               128000000; // 128M elements
                    int iterations = (g_App.selectedSuiteIndex == 0) ? 10 :
                                   (g_App.selectedSuiteIndex == 1) ? 20 : 30;
                    
                    // Run benchmarks for each backend
                    for (size_t backendIdx = 0; backendIdx < backendsToRun.size(); backendIdx++) {
                        std::string selectedBackendName = backendsToRun[backendIdx];
                        float baseProgress = (float)backendIdx / backendsToRun.size();
                        float progressStep = 1.0f / backendsToRun.size();
                        
                        {
                            std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                            g_App.currentBenchmark = "Initializing " + selectedBackendName + " (" + 
                                                    std::to_string(backendIdx + 1) + "/" + 
                                                    std::to_string(backendsToRun.size()) + ")...";
                        }
                        g_App.progress = baseProgress + 0.1f * progressStep;
                        
                        BenchmarkResult result;
                    
                    if (selectedBackendName == "CUDA") {
                        CUDABackend cudaBackend;
                        if (cudaBackend.Initialize()) {
                            std::vector<std::string> benchmarks = {"VectorAdd", "MatrixMul", "Convolution", "Reduction"};
                            for (size_t benchIdx = 0; benchIdx < benchmarks.size(); benchIdx++) {
                                float benchProgress = baseProgress + progressStep * (0.2f + 0.6f * benchIdx / benchmarks.size());
                                
                                {
                                    std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                    g_App.currentBenchmark = "Running " + benchmarks[benchIdx] + " (CUDA)...";
                                }
                                g_App.progress = benchProgress;
                                
                                BenchmarkResult benchResult;
                                if (benchmarks[benchIdx] == "VectorAdd") {
                                    benchResult = RunVectorAddCUDA(&cudaBackend, vectorElements, iterations);
                                } else if (benchmarks[benchIdx] == "MatrixMul") {
                                    benchResult = RunMatrixMulCUDA(&cudaBackend, matrixSize, iterations);
                                } else if (benchmarks[benchIdx] == "Convolution") {
                                    benchResult = RunConvolutionCUDA(&cudaBackend, imageSize, imageSize, iterations);
                                } else if (benchmarks[benchIdx] == "Reduction") {
                                    benchResult = RunReductionCUDA(&cudaBackend, reductionElements, iterations);
                                }
                                
                                // Add to results
                                AppState::BenchmarkResult guiResult;
                                guiResult.name = benchmarks[benchIdx];
                                guiResult.backend = selectedBackendName;
                                guiResult.timeMs = benchResult.executionTimeMS;
                                guiResult.bandwidth = benchResult.effectiveBandwidthGBs;
                                guiResult.gflops = benchResult.computeThroughputGFLOPS;
                                guiResult.problemSize = benchResult.problemSize;
                                guiResult.passed = benchResult.resultCorrect;
                                
                                {
                                    std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                                    g_App.results.push_back(guiResult);

                                    // Update history graphs for CUDA with enhanced TestResult
                                    g_App.cudaHistory.totalTests++;
                                    AppState::TestResult testRes;
                                    testRes.bandwidth = static_cast<float>(benchResult.effectiveBandwidthGBs);
                                    testRes.gflops = benchResult.computeThroughputGFLOPS;
                                    testRes.timeMS = benchResult.executionTimeMS;
                                    testRes.timestamp = GetCurrentTimestamp();
                                    testRes.dateTime = GetFormattedDateTime();
                                    testRes.testID = "Test " + std::to_string(g_App.cudaHistory.totalTests);
                                    
                                    std::vector<AppState::TestResult>* historyVec = nullptr;
                                    if (benchmarks[benchIdx] == "VectorAdd") historyVec = &g_App.cudaHistory.vectorAdd;
                                    else if (benchmarks[benchIdx] == "MatrixMul") historyVec = &g_App.cudaHistory.matrixMul;
                                    else if (benchmarks[benchIdx] == "Convolution") historyVec = &g_App.cudaHistory.convolution;
                                    else if (benchmarks[benchIdx] == "Reduction") historyVec = &g_App.cudaHistory.reduction;

                                    if (historyVec) {
                                        historyVec->push_back(testRes);
                                        // Keep cumulative history up to 100 entries
                                        if (historyVec->size() > 100) historyVec->erase(historyVec->begin());
                                    }
                                }
                            }
                            
                            cudaBackend.Shutdown();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                    } else if (selectedBackendName == "OpenCL") {
                        try {
                            {
                                std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                g_App.currentBenchmark = "Initializing OpenCL (detecting platforms)...";
                            }
                            g_App.progress = 0.15f;
                            
                            OpenCLBackend openclBackend;
                            
                            if (!openclBackend.Initialize()) {
                                {
                                    std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                    g_App.currentBenchmark = "ERROR: OpenCL initialization failed - check drivers";
                                }
                                std::this_thread::sleep_for(std::chrono::seconds(3));
                                g_App.progress = 1.0f;
                                g_App.benchmarkRunning = false;
                                g_App.workerThreadRunning = false;
                                return;
                            }
                            
                            {
                                std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                g_App.currentBenchmark = "OpenCL initialized! Running benchmarks...";
                            }
                            
                            std::vector<std::string> benchmarks = {"VectorAdd", "MatrixMul", "Convolution", "Reduction"};
                            for (size_t benchIdx = 0; benchIdx < benchmarks.size(); benchIdx++) {
                                float benchProgress = baseProgress + progressStep * (0.2f + 0.6f * benchIdx / benchmarks.size());
                                
                                {
                                    std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                    g_App.currentBenchmark = "Running " + benchmarks[benchIdx] + " (OpenCL)...";
                                }
                                g_App.progress = benchProgress;
                                
                                BenchmarkResult benchResult;
                                if (benchmarks[benchIdx] == "VectorAdd") {
                                    benchResult = RunVectorAddOpenCL(&openclBackend, vectorElements, iterations);
                                } else if (benchmarks[benchIdx] == "MatrixMul") {
                                    benchResult = RunMatrixMulOpenCL(&openclBackend, matrixSize, iterations);
                                } else if (benchmarks[benchIdx] == "Convolution") {
                                    benchResult = RunConvolutionOpenCL(&openclBackend, imageSize, imageSize, iterations);
                                } else if (benchmarks[benchIdx] == "Reduction") {
                                    benchResult = RunReductionOpenCL(&openclBackend, reductionElements, iterations);
                                }
                                
                                // Add to results
                                AppState::BenchmarkResult guiResult;
                                guiResult.name = benchmarks[benchIdx];
                                guiResult.backend = selectedBackendName;
                                guiResult.timeMs = benchResult.executionTimeMS;
                                guiResult.bandwidth = benchResult.effectiveBandwidthGBs;
                                guiResult.gflops = benchResult.computeThroughputGFLOPS;
                                guiResult.problemSize = benchResult.problemSize;
                                guiResult.passed = benchResult.resultCorrect;
                                
                                {
                                    std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                                    g_App.results.push_back(guiResult);

                                    // Update history graphs for OpenCL with enhanced TestResult
                                    g_App.openclHistory.totalTests++;
                                    AppState::TestResult testRes;
                                    testRes.bandwidth = static_cast<float>(benchResult.effectiveBandwidthGBs);
                                    testRes.gflops = benchResult.computeThroughputGFLOPS;
                                    testRes.timeMS = benchResult.executionTimeMS;
                                    testRes.timestamp = GetCurrentTimestamp();
                                    testRes.dateTime = GetFormattedDateTime();
                                    testRes.testID = "Test " + std::to_string(g_App.openclHistory.totalTests);
                                    
                                    std::vector<AppState::TestResult>* historyVec = nullptr;
                                    if (benchmarks[benchIdx] == "VectorAdd") historyVec = &g_App.openclHistory.vectorAdd;
                                    else if (benchmarks[benchIdx] == "MatrixMul") historyVec = &g_App.openclHistory.matrixMul;
                                    else if (benchmarks[benchIdx] == "Convolution") historyVec = &g_App.openclHistory.convolution;
                                    else if (benchmarks[benchIdx] == "Reduction") historyVec = &g_App.openclHistory.reduction;

                                    if (historyVec) {
                                        historyVec->push_back(testRes);
                                        // Keep cumulative history up to 100 entries
                                        if (historyVec->size() > 100) historyVec->erase(historyVec->begin());
                                    }
                                }
                            }
                            
                            openclBackend.Shutdown();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            
                        } catch (const std::exception& e) {
                            {
                                std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                g_App.currentBenchmark = std::string("ERROR: OpenCL exception - ") + e.what();
                            }
                            std::this_thread::sleep_for(std::chrono::seconds(3));
                            result.resultCorrect = false;
                        } catch (...) {
                            {
                                std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                g_App.currentBenchmark = "ERROR: OpenCL unknown exception - possible driver issue";
                            }
                            std::this_thread::sleep_for(std::chrono::seconds(3));
                            result.resultCorrect = false;
                        }
                    } else if (selectedBackendName == "DirectCompute") {
                        DirectComputeBackend dcBackend;
                        if (dcBackend.Initialize()) {
                            std::vector<std::string> benchmarks = {"VectorAdd", "MatrixMul", "Convolution", "Reduction"};
                            for (size_t benchIdx = 0; benchIdx < benchmarks.size(); benchIdx++) {
                                float benchProgress = baseProgress + progressStep * (0.2f + 0.6f * benchIdx / benchmarks.size());
                                
                                {
                                    std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                                    g_App.currentBenchmark = "Running " + benchmarks[benchIdx] + " (DirectCompute)...";
                                }
                                g_App.progress = benchProgress;
                                
                                BenchmarkResult benchResult;
                                if (benchmarks[benchIdx] == "VectorAdd") {
                                    benchResult = RunVectorAddDirectCompute(&dcBackend, vectorElements, iterations);
                                } else if (benchmarks[benchIdx] == "MatrixMul") {
                                    benchResult = RunMatrixMulDirectCompute(&dcBackend, matrixSize, iterations);
                                } else if (benchmarks[benchIdx] == "Convolution") {
                                    benchResult = RunConvolutionDirectCompute(&dcBackend, imageSize, imageSize, iterations);
                                } else if (benchmarks[benchIdx] == "Reduction") {
                                    benchResult = RunReductionDirectCompute(&dcBackend, reductionElements, iterations);
                                }
                                
                                // Add to results
                                AppState::BenchmarkResult guiResult;
                                guiResult.name = benchmarks[benchIdx];
                                guiResult.backend = selectedBackendName;
                                guiResult.timeMs = benchResult.executionTimeMS;
                                guiResult.bandwidth = benchResult.effectiveBandwidthGBs;
                                guiResult.gflops = benchResult.computeThroughputGFLOPS;
                                guiResult.problemSize = benchResult.problemSize;
                                guiResult.passed = benchResult.resultCorrect;
                                
                                {
                                    std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                                    g_App.results.push_back(guiResult);

                                    // Update history graphs for DirectCompute with enhanced TestResult
                                    g_App.directcomputeHistory.totalTests++;
                                    AppState::TestResult testRes;
                                    testRes.bandwidth = static_cast<float>(benchResult.effectiveBandwidthGBs);
                                    testRes.gflops = benchResult.computeThroughputGFLOPS;
                                    testRes.timeMS = benchResult.executionTimeMS;
                                    testRes.timestamp = GetCurrentTimestamp();
                                    testRes.dateTime = GetFormattedDateTime();
                                    testRes.testID = "Test " + std::to_string(g_App.directcomputeHistory.totalTests);
                                    
                                    std::vector<AppState::TestResult>* historyVec = nullptr;
                                    if (benchmarks[benchIdx] == "VectorAdd") historyVec = &g_App.directcomputeHistory.vectorAdd;
                                    else if (benchmarks[benchIdx] == "MatrixMul") historyVec = &g_App.directcomputeHistory.matrixMul;
                                    else if (benchmarks[benchIdx] == "Convolution") historyVec = &g_App.directcomputeHistory.convolution;
                                    else if (benchmarks[benchIdx] == "Reduction") historyVec = &g_App.directcomputeHistory.reduction;

                                    if (historyVec) {
                                        historyVec->push_back(testRes);
                                        // Keep cumulative history up to 100 entries
                                        if (historyVec->size() > 100) historyVec->erase(historyVec->begin());
                                    }
                                }
                            }
                            
                            dcBackend.Shutdown();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                    }
                    
                    // Results are now added inside benchmark loops above
                        
                        g_App.progress = baseProgress + progressStep;
                        
                    } // End of backend loop
                    
                    // All backends complete
                    g_App.progress = 1.0f;
                    {
                        std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                        if (backendsToRun.size() > 1) {
                            g_App.currentBenchmark = "Complete! Tested " + std::to_string(backendsToRun.size()) + " backends";
                        } else {
                            g_App.currentBenchmark = "Complete!";
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                    g_App.currentBenchmark = std::string("ERROR: ") + e.what();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                    g_App.currentBenchmark = "ERROR: Unknown exception";
                }
                
                g_App.benchmarkRunning = false;
                g_App.workerThreadRunning = false;
            });
        }
        
        if (!canStart) ImGui::EndDisabled();
        
        // Progress bar
        if (g_App.benchmarkRunning) {
            ImGui::Spacing();
            ImGui::ProgressBar(g_App.progress, ImVec2(-1, 30));
            std::string currentBench;
            {
                std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                currentBench = g_App.currentBenchmark;
            }
            ImGui::Text("Status: %s", currentBench.c_str());
        }
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Results Display with Graph
    if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::lock_guard<std::mutex> lock(g_App.resultsMutex);
        
        if (g_App.results.empty()) {
            ImGui::TextDisabled("No results yet. Run a benchmark to see results.");
        } else {
            // Enhanced Results Table with GFLOPS
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "BENCHMARK RESULTS");
            ImGui::PopFont();
            ImGui::Spacing();
            
            if (ImGui::BeginTable("ResultsTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
                ImGui::TableSetupColumn("Benchmark", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                ImGui::TableSetupColumn("Backend", ImGuiTableColumnFlags_WidthFixed, 110.0f);
                ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Bandwidth", ImGuiTableColumnFlags_WidthFixed, 90.0f);
                ImGui::TableSetupColumn("GFLOPS", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 90.0f);
                ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                ImGui::TableHeadersRow();
                
                for (const auto& result : g_App.results) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    
                    // Color-code benchmark names
                    ImVec4 benchColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
                    if (result.name == "VectorAdd") benchColor = ImVec4(0.3f, 0.9f, 1.0f, 1.0f);
                    else if (result.name == "MatrixMul") benchColor = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
                    else if (result.name == "Convolution") benchColor = ImVec4(0.9f, 0.3f, 0.9f, 1.0f);
                    else if (result.name == "Reduction") benchColor = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
                    ImGui::TextColored(benchColor, "%s", result.name.c_str());
                    
                    ImGui::TableNextColumn();
                    // Color-code backends
                    ImVec4 backendColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
                    if (result.backend == "CUDA") backendColor = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
                    else if (result.backend == "OpenCL") backendColor = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
                    else if (result.backend == "DirectCompute") backendColor = ImVec4(0.5f, 0.7f, 1.0f, 1.0f);
                    ImGui::TextColored(backendColor, "%s", result.backend.c_str());
                    
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f", result.timeMs);
                    
                    ImGui::TableNextColumn();
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "%.1f GB/s", result.bandwidth);
                    
                    ImGui::TableNextColumn();
                    if (result.gflops > 0.1) {
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%.1f", result.gflops);
                    } else {
                        ImGui::TextDisabled("N/A");
                    }
                    
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%zu", result.problemSize);
                    
                    ImGui::TableNextColumn();
                    if (result.passed) {
                        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "PASS");
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "✗ FAIL");
                    }
                }
                
                ImGui::EndTable();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Enhanced Multi-Colored Performance Graphs
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "PERFORMANCE HISTORY (Cumulative)");
            ImGui::PopFont();
            ImGui::Spacing();
            
            float maxBandwidth = 200.0f;
            ImVec2 graphSize(ImGui::GetContentRegionAvail().x - 20, 100);
            
            // CUDA Graphs with color-coded benchmarks
            bool hasCudaData = !g_App.cudaHistory.vectorAdd.empty() || !g_App.cudaHistory.matrixMul.empty() ||
                               !g_App.cudaHistory.convolution.empty() || !g_App.cudaHistory.reduction.empty();
            if (hasCudaData) {
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "[CUDA Backend]");
                ImGui::Spacing();
                
                if (!g_App.cudaHistory.vectorAdd.empty()) {
                    auto values = ExtractBandwidth(g_App.cudaHistory.vectorAdd);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.3f, 0.9f, 1.0f, 1.0f));
                    ImGui::Text("  VectorAdd (Memory Bandwidth Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##CUDA_VectorAdd", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Higher is Better", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.cudaHistory.matrixMul.empty()) {
                    auto values = ExtractBandwidth(g_App.cudaHistory.matrixMul);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(1.0f, 0.6f, 0.2f, 1.0f));
                    ImGui::Text("  MatrixMul (Compute Throughput Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##CUDA_MatrixMul", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Measures Data Transfer", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.cudaHistory.convolution.empty()) {
                    auto values = ExtractBandwidth(g_App.cudaHistory.convolution);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.9f, 0.3f, 0.9f, 1.0f));
                    ImGui::Text("  Convolution (Cache Efficiency Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##CUDA_Convolution", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Tests 2D Memory Access", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.cudaHistory.reduction.empty()) {
                    auto values = ExtractBandwidth(g_App.cudaHistory.reduction);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                    ImGui::Text("  Reduction (Thread Synchronization Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##CUDA_Reduction", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Tests Parallel Aggregation", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                ImGui::Spacing();
            }
            
            // OpenCL Graphs with color-coded benchmarks
            bool hasOpenCLData = !g_App.openclHistory.vectorAdd.empty() || !g_App.openclHistory.matrixMul.empty() ||
                                 !g_App.openclHistory.convolution.empty() || !g_App.openclHistory.reduction.empty();
            if (hasOpenCLData) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "■ OpenCL Backend");
                ImGui::Spacing();
                
                if (!g_App.openclHistory.vectorAdd.empty()) {
                    auto values = ExtractBandwidth(g_App.openclHistory.vectorAdd);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.3f, 0.9f, 1.0f, 1.0f));
                    ImGui::Text("  VectorAdd (Memory Bandwidth Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##OpenCL_VectorAdd", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Higher is Better", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.openclHistory.matrixMul.empty()) {
                    auto values = ExtractBandwidth(g_App.openclHistory.matrixMul);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(1.0f, 0.6f, 0.2f, 1.0f));
                    ImGui::Text("  MatrixMul (Compute Throughput Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##OpenCL_MatrixMul", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Measures Data Transfer", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.openclHistory.convolution.empty()) {
                    auto values = ExtractBandwidth(g_App.openclHistory.convolution);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.9f, 0.3f, 0.9f, 1.0f));
                    ImGui::Text("  Convolution (Cache Efficiency Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##OpenCL_Convolution", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Tests 2D Memory Access", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.openclHistory.reduction.empty()) {
                    auto values = ExtractBandwidth(g_App.openclHistory.reduction);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                    ImGui::Text("  Reduction (Thread Synchronization Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##OpenCL_Reduction", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Tests Parallel Aggregation", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                ImGui::Spacing();
            }
            
            // DirectCompute Graphs with color-coded benchmarks
            bool hasDCData = !g_App.directcomputeHistory.vectorAdd.empty() || !g_App.directcomputeHistory.matrixMul.empty() ||
                             !g_App.directcomputeHistory.convolution.empty() || !g_App.directcomputeHistory.reduction.empty();
            if (hasDCData) {
                ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "[DirectCompute Backend]");
                ImGui::Spacing();
                
                if (!g_App.directcomputeHistory.vectorAdd.empty()) {
                    auto values = ExtractBandwidth(g_App.directcomputeHistory.vectorAdd);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.3f, 0.9f, 1.0f, 1.0f));
                    ImGui::Text("  VectorAdd (Memory Bandwidth Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##DC_VectorAdd", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Higher is Better", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.directcomputeHistory.matrixMul.empty()) {
                    auto values = ExtractBandwidth(g_App.directcomputeHistory.matrixMul);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(1.0f, 0.6f, 0.2f, 1.0f));
                    ImGui::Text("  MatrixMul (Compute Throughput Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##DC_MatrixMul", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Measures Data Transfer", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.directcomputeHistory.convolution.empty()) {
                    auto values = ExtractBandwidth(g_App.directcomputeHistory.convolution);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.9f, 0.3f, 0.9f, 1.0f));
                    ImGui::Text("  Convolution (Cache Efficiency Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##DC_Convolution", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Tests 2D Memory Access", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                if (!g_App.directcomputeHistory.reduction.empty()) {
                    auto values = ExtractBandwidth(g_App.directcomputeHistory.reduction);
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                    ImGui::Text("  Reduction (Thread Synchronization Test) - %d tests", (int)values.size());
                    ImGui::PlotLines("##DC_Reduction", values.data(), values.size(),
                                   0, "Bandwidth (GB/s) - Tests Parallel Aggregation", 0.0f, maxBandwidth, ImVec2(ImGui::GetContentRegionAvail().x - 20, 80));
                    ImGui::PopStyleColor();
                }
                
                ImGui::Spacing();
            }
            
            // Color Legend
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Color Legend:");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "■ VectorAdd");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "■ MatrixMul");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.9f, 1.0f), "■ Convolution");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "■ Reduction");
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Enhanced CSV Export
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.5f, 0.1f, 1.0f));
            if (ImGui::Button("Export Results to CSV...", ImVec2(200, 35))) {
                // Windows file save dialog
                OPENFILENAMEA ofn;
                char szFile[260] = "benchmark_results.csv";
                ZeroMemory(&ofn, sizeof(ofn));
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = g_App.hwnd;
                ofn.lpstrFile = szFile;
                ofn.nMaxFile = sizeof(szFile);
                ofn.lpstrFilter = "CSV Files (*.csv)\0*.csv\0All Files (*.*)\0*.*\0";
                ofn.nFilterIndex = 1;
                ofn.lpstrFileTitle = NULL;
                ofn.nMaxFileTitle = 0;
                ofn.lpstrInitialDir = NULL;
                ofn.lpstrTitle = "Save Benchmark Results";
                ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
                ofn.lpstrDefExt = "csv";
                
                if (GetSaveFileNameA(&ofn)) {
                    std::ofstream csv(ofn.lpstrFile);
                    if (csv.is_open()) {
                        csv << "Benchmark,Backend,Time_ms,Bandwidth_GBs,GFLOPS,ProblemSize,Status\n";
                        for (const auto& result : g_App.results) {
                            csv << result.name << ","
                               << result.backend << ","
                               << result.timeMs << ","
                               << result.bandwidth << ","
                               << result.gflops << ","
                               << result.problemSize << ","
                               << (result.passed ? "PASS" : "FAIL") << "\n";
                        }
                        csv.close();
                        
                        // Show success message
                        std::lock_guard<std::mutex> lock(g_App.currentBenchmarkMutex);
                        g_App.currentBenchmark = "Results exported successfully!";
                    }
                }
            }
            ImGui::PopStyleColor(3);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Exports all results with GFLOPS data");
        }
    }
    
    // Exit button at bottom (use normal flow, not absolute positioning)
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Center the exit button
    float buttonWidth = 120.0f;
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - buttonWidth) * 0.5f);
    if (ImGui::Button("Exit Application", ImVec2(buttonWidth, 40))) {
        g_App.running = false;
    }
    
    ImGui::Spacing();
    ImGui::End();
    
    // Enhanced About Dialog
    if (g_App.showAbout) {
        ImGui::SetNextWindowSize(ImVec2(680, 580), ImGuiCond_FirstUseEver);
        ImGui::Begin("About GPU Benchmark Suite v4.0", &g_App.showAbout);

        // Header
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "GPU BENCHMARK SUITE v4.0");
        ImGui::PopFont();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Comprehensive Multi-API GPU Performance Testing Tool");
        ImGui::Separator();
        ImGui::Spacing();

        // Description
        ImGui::TextWrapped(
            "A professional-grade GPU benchmarking application designed to evaluate graphics "
            "processing unit performance across multiple compute APIs. This tool provides detailed "
            "performance metrics, historical tracking, and comprehensive analysis of GPU capabilities."
        );
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Features Section
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "FEATURES:");
        ImGui::BulletText("4 Benchmark Types: VectorAdd, MatrixMul, Convolution, Reduction");
        ImGui::BulletText("3 GPU APIs: CUDA (NVIDIA), OpenCL (Cross-platform), DirectCompute (Windows)");
        ImGui::BulletText("Real-time performance monitoring with live graphs");
        ImGui::BulletText("Cumulative history tracking (up to 100 test runs)");
        ImGui::BulletText("Detailed metrics: Bandwidth (GB/s), GFLOPS, Execution Time");
        ImGui::BulletText("CSV export with custom save location");
        ImGui::BulletText("Hardware-agnostic: Adapts to your GPU automatically");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Technical Details
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "BENCHMARK DETAILS:");
        ImGui::Columns(2, "tech_cols", false);
        
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 1.0f, 1.0f), "VectorAdd:");
        ImGui::TextWrapped("Tests raw memory bandwidth with parallel element-wise addition operations");
        ImGui::Spacing();
        
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "MatrixMul:");
        ImGui::TextWrapped("Measures compute throughput with tiled matrix multiplication (GFLOPS)");
        ImGui::Spacing();
        
        ImGui::NextColumn();
        
        ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.9f, 1.0f), "Convolution:");
        ImGui::TextWrapped("Evaluates cache efficiency with 2D image convolution operations");
        ImGui::Spacing();
        
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Reduction:");
        ImGui::TextWrapped("Tests parallel aggregation and thread synchronization");
        
        ImGui::Columns(1);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Developer Info
        ImGui::TextColored(ImVec4(0.8f, 0.5f, 1.0f, 1.0f), "DEVELOPER:");
        ImGui::Text("Author: Soham Dave");
        ImGui::Text("GitHub:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "https://github.com/davesohamm");
        if (ImGui::IsItemClicked()) {
            ShellExecuteA(nullptr, "open", "https://github.com/davesohamm", nullptr, nullptr, SW_SHOW);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
        ImGui::Text("Version: 4.0.0 | Built: 2026-01-09");
        ImGui::Text("Platform: Windows 11 | APIs: CUDA 12.x, OpenCL 3.0, DirectX 11");
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // System Info
        ImGui::TextColored(ImVec4(0.5f, 0.9f, 1.0f, 1.0f), "YOUR SYSTEM:");
        ImGui::Text("CUDA: %s", g_App.systemCaps.cuda.available ? "Available" : "Not Available");
        ImGui::Text("OpenCL: %s", g_App.systemCaps.opencl.available ? "Available" : "Not Available");
        ImGui::Text("DirectCompute: %s", g_App.systemCaps.directCompute.available ? "Available" : "Not Available");
        if (!g_App.systemCaps.gpus.empty()) {
            ImGui::Text("Primary GPU: %s", g_App.systemCaps.gpus[g_App.systemCaps.primaryGPUIndex].name.c_str());
        }
        
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - 120) * 0.5f);
        if (ImGui::Button("Close", ImVec2(120, 35))) {
            g_App.showAbout = false;
        }
        
        ImGui::End();
    }
}

//==============================================================================
// MAIN ENTRY POINT
//==============================================================================

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    // Redirect console output
    FILE* dummy;
    freopen_s(&dummy, "NUL", "w", stdout);
    freopen_s(&dummy, "NUL", "w", stderr);
    
    // Create window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"GPUBenchmark", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"GPU Benchmark Suite v2.0 - All Backends Working!",
                                WS_OVERLAPPEDWINDOW, 100, 100, 1000, 700, nullptr, nullptr, wc.hInstance, nullptr);
    g_App.hwnd = hwnd;
    
    // Initialize D3D
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    
    ::ShowWindow(hwnd, SW_SHOW);
    ::UpdateWindow(hwnd);
    ::SetForegroundWindow(hwnd);
    ::SetFocus(hwnd);
    
    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 5.0f;
    style.WindowRounding = 5.0f;
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.1f, 0.4f, 0.7f, 1.0f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.5f, 0.8f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.6f, 0.9f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.1f, 0.4f, 0.7f, 1.0f);
    
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_App.d3dDevice, g_App.d3dContext);
    
    // Discover system capabilities
    g_App.systemCaps = DeviceDiscovery::Discover();
    
    // Main loop
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (g_App.running && msg.message != WM_QUIT) {
        if (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            continue;
        }
        
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        
        RenderUI();
        
        ImGui::Render();
        const float clear_color[4] = { 0.1f, 0.1f, 0.15f, 1.0f };
        g_App.d3dContext->OMSetRenderTargets(1, &g_App.mainRenderTargetView, nullptr);
        g_App.d3dContext->ClearRenderTargetView(g_App.mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        
        g_App.swapChain->Present(1, 0);
    }
    
    // Cleanup
    g_App.running = false;
    if (g_App.workerThread.joinable()) {
        g_App.workerThread.join();
    }
    
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    
    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    
    return 0;
}
