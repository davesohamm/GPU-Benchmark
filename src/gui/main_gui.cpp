/********************************************************************************
 * @file    main_gui.cpp
 * @brief   GPU Benchmark Suite - GUI Application Entry Point
 * 
 * @details Beautiful desktop application for GPU compute benchmarking with:
 *          - ImGui interface with DirectX 11 rendering
 *          - Multi-backend support (CUDA, OpenCL, DirectCompute)
 *          - Real-time benchmark execution and visualization
 *          - Interactive configuration and results display
 * 
 * @author  Soham Dave (https://github.com/davesohamm)
 * @date    2026-01-09
 * @version 1.0.0
 ********************************************************************************/

// Include C++ headers first
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <chrono>

// Include core framework
#include "../core/BenchmarkRunner.h"
#include "../core/DeviceDiscovery.h"
#include "../core/Logger.h"

// Include benchmark classes
#include "../benchmarks/VectorAddBenchmark.h"
#include "../benchmarks/MatrixMulBenchmark.h"
#include "../benchmarks/ConvolutionBenchmark.h"
#include "../benchmarks/ReductionBenchmark.h"

// Include Windows/DirectX headers
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <tchar.h>
#include <shellapi.h>  // For ShellExecuteA

// Include ImGui
#include "../../external/imgui/imgui.h"
#include "../../external/imgui/backends/imgui_impl_win32.h"
#include "../../external/imgui/backends/imgui_impl_dx11.h"

// Link libraries
#pragma comment(lib, "d3d11.lib")

using namespace GPUBenchmark;

//==============================================================================
// GLOBAL STATE
//==============================================================================

struct ApplicationState {
    // D3D11 objects
    ID3D11Device*            d3dDevice = nullptr;
    ID3D11DeviceContext*     d3dContext = nullptr;
    IDXGISwapChain*          swapChain = nullptr;
    ID3D11RenderTargetView*  mainRenderTargetView = nullptr;
    
    // Window
    HWND hwnd = nullptr;
    bool running = true;
    
    // Benchmark state
    std::unique_ptr<BenchmarkRunner> benchmarkRunner;
    SystemCapabilities systemCaps;
    
    // UI state
    int selectedBackendIndex = 0;
    int selectedSuiteIndex = 1;  // 0=Quick, 1=Standard, 2=Full
    bool benchmarkRunning = false;
    bool showAboutDialog = false;
    float progress = 0.0f;
    std::string currentBenchmark = "";
    
    // Results
    struct BenchmarkResult {
        std::string name;
        std::string backend;
        double timeMs;
        double performance;  // GB/s or GFLOPS
        std::string unit;
        bool passed;
    };
    std::vector<BenchmarkResult> results;
    
    // Threading
    std::atomic<bool> workerThreadRunning{false};
    std::thread workerThread;
    std::mutex resultsMutex;
};

static ApplicationState g_App;

//==============================================================================
// FORWARD DECLARATIONS
//==============================================================================

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

//==============================================================================
// D3D11 INITIALIZATION
//==============================================================================

bool CreateDeviceD3D(HWND hWnd) {
    // Setup swap chain
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
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_App.swapChain) { g_App.swapChain->Release(); g_App.swapChain = nullptr; }
    if (g_App.d3dContext) { g_App.d3dContext->Release(); g_App.d3dContext = nullptr; }
    if (g_App.d3dDevice) { g_App.d3dDevice->Release(); g_App.d3dDevice = nullptr; }
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer;
    g_App.swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_App.d3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_App.mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_App.mainRenderTargetView) { g_App.mainRenderTargetView->Release(); g_App.mainRenderTargetView = nullptr; }
}

//==============================================================================
// WINDOW PROCEDURE
//==============================================================================

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            if (g_App.d3dDevice != nullptr && wParam != SIZE_MINIMIZED) {
                CleanupRenderTarget();
                g_App.swapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
                return 0;
            break;
        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}

//==============================================================================
// GUI RENDERING
//==============================================================================

void RenderUI() {
    ImGuiIO& io = ImGui::GetIO();
    
    // Main window (fullscreen)
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("GPU Benchmark Suite", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | 
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
    
    // Header
    ImGui::PushFont(io.Fonts->Fonts[0]);  // Default font
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "GPU BENCHMARK SUITE");
    ImGui::PopFont();
    ImGui::SameLine(ImGui::GetWindowWidth() - 250);
    if (ImGui::Button("About", ImVec2(80, 30))) {
        g_App.showAboutDialog = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Exit", ImVec2(80, 30))) {
        g_App.running = false;
    }
    
    ImGui::Separator();
    
    // System Information Panel
    if (ImGui::CollapsingHeader("System Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!g_App.systemCaps.gpus.empty()) {
            const GPUInfo& gpu = g_App.systemCaps.GetPrimaryGPU();
            ImGui::Text("GPU:     %s", gpu.name.c_str());
            ImGui::Text("Memory:  %.0f MB", gpu.totalMemoryMB);
            ImGui::Text("CPU:     %s", g_App.systemCaps.cpuName.c_str());
            ImGui::Text("RAM:     %zu MB", g_App.systemCaps.systemRAMMB);
            ImGui::Text("OS:      %s", g_App.systemCaps.operatingSystem.c_str());
            
            ImGui::Separator();
            ImGui::Text("Backends Available:");
            ImGui::Indent();
            
            // CUDA
            const char* cudaIcon = g_App.systemCaps.cuda.available ? "[OK]" : "[X]";
            ImVec4 cudaColor = g_App.systemCaps.cuda.available ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            ImGui::TextColored(cudaColor, "%s CUDA %s", cudaIcon, 
                              g_App.systemCaps.cuda.available ? g_App.systemCaps.cuda.version.c_str() : "");
            
            // OpenCL
            const char* openclIcon = g_App.systemCaps.opencl.available ? "[OK]" : "[X]";
            ImVec4 openclColor = g_App.systemCaps.opencl.available ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            ImGui::TextColored(openclColor, "%s OpenCL %s", openclIcon,
                              g_App.systemCaps.opencl.available ? g_App.systemCaps.opencl.version.c_str() : "");
            
            // DirectCompute
            const char* dcIcon = g_App.systemCaps.directCompute.available ? "[OK]" : "[X]";
            ImVec4 dcColor = g_App.systemCaps.directCompute.available ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            ImGui::TextColored(dcColor, "%s DirectCompute %s", dcIcon,
                              g_App.systemCaps.directCompute.available ? g_App.systemCaps.directCompute.version.c_str() : "");
            
            ImGui::Unindent();
        }
    }
    
    ImGui::Separator();
    
    // Benchmark Configuration
    if (ImGui::CollapsingHeader("Benchmark Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Backend selection
        ImGui::Text("Backend:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        std::vector<std::string> availableBackendNames = g_App.systemCaps.GetAvailableBackendNames();
        
        if (!availableBackendNames.empty()) {
            // Convert to char* array for ImGui
            std::vector<const char*> backendCStrings;
            for (const auto& name : availableBackendNames) {
                backendCStrings.push_back(name.c_str());
            }
            ImGui::Combo("##Backend", &g_App.selectedBackendIndex, backendCStrings.data(), (int)backendCStrings.size());
        } else {
            ImGui::TextDisabled("No backends available");
        }
        
        // Suite selection
        ImGui::Text("Suite:  ");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        const char* suites[] = { "Quick", "Standard", "Full" };
        ImGui::Combo("##Suite", &g_App.selectedSuiteIndex, suites, 3);
        
        ImGui::Separator();
        
        // Start button
        bool canStart = !g_App.benchmarkRunning && g_App.systemCaps.HasAnyBackend();
        if (!canStart) ImGui::BeginDisabled();
        
        if (ImGui::Button("Start Benchmark", ImVec2(200, 40))) {
            g_App.benchmarkRunning = true;
            g_App.progress = 0.0f;
            g_App.results.clear();
            g_App.currentBenchmark = "Initializing...";
            
            // Launch benchmark in background thread
            if (g_App.workerThreadRunning) {
                if (g_App.workerThread.joinable()) {
                    g_App.workerThread.join();
                }
            }
            
            g_App.workerThreadRunning = true;
            g_App.workerThread = std::thread([]() {
                try {
                    // Determine which backend to use
                    std::vector<std::string> availableBackends = g_App.systemCaps.GetAvailableBackendNames();
                    if (availableBackends.empty()) {
                        g_App.currentBenchmark = "Error: No backends available";
                        g_App.benchmarkRunning = false;
                        g_App.workerThreadRunning = false;
                        return;
                    }
                    
                    std::string selectedBackendName = availableBackends[g_App.selectedBackendIndex];
                    BackendType backendType = BackendType::CUDA;
                    if (selectedBackendName == "OpenCL") backendType = BackendType::OpenCL;
                    else if (selectedBackendName == "DirectCompute") backendType = BackendType::DirectCompute;
                    
                    g_App.currentBenchmark = "Initializing " + selectedBackendName + "...";
                    
                    // Initialize backend
                    if (!g_App.benchmarkRunner->InitializeBackend(backendType)) {
                        g_App.currentBenchmark = "Error: Failed to initialize " + selectedBackendName;
                        g_App.benchmarkRunning = false;
                        g_App.workerThreadRunning = false;
                        return;
                    }
                
                    // Determine suite
                    const char* suites[] = { "Quick", "Standard", "Full" };
                    std::string suiteName = suites[g_App.selectedSuiteIndex];
                    
                    // Run benchmarks
                    std::vector<std::string> benchmarks;
                    if (suiteName == "Quick") {
                        benchmarks = {"VectorAdd"};
                    } else {
                        benchmarks = {"VectorAdd", "MatrixMul", "Convolution", "Reduction"};
                    }
                    
                    float progressStep = 1.0f / benchmarks.size();
                    
                    // Get the backend pointer
                    IComputeBackend* backend = g_App.benchmarkRunner->GetBackend(backendType);
                    if (!backend) {
                        g_App.currentBenchmark = "Error: Failed to get backend pointer";
                        g_App.benchmarkRunning = false;
                        g_App.workerThreadRunning = false;
                        return;
                    }
                    
                    for (size_t i = 0; i < benchmarks.size(); i++) {
                        g_App.currentBenchmark = "Running " + benchmarks[i] + "...";
                        g_App.progress = i * progressStep;
                        
                        BenchmarkResult result;
                        
                        try {
                            // Synchronize before each benchmark
                            backend->Synchronize();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            
                            // Run the appropriate benchmark
                            if (benchmarks[i] == "VectorAdd") {
                                size_t problemSize = (suiteName == "Quick") ? 1000000 : 
                                                   (suiteName == "Standard") ? 5000000 : 10000000;  // Reduced from 10M/100M
                                int iterations = (suiteName == "Quick") ? 10 : 50;  // Reduced from 100
                                
                                g_App.currentBenchmark = "VectorAdd (" + std::to_string(problemSize / 1000000) + "M elements)";
                                
                                VectorAddBenchmark bench(problemSize);
                                bench.SetIterations(iterations);
                                bench.SetWarmupIterations(suiteName == "Quick" ? 2 : 5);  // Reduced warmup
                                result = bench.Run(backend);
                                
                            } else if (benchmarks[i] == "MatrixMul") {
                                int matrixSize = (suiteName == "Quick") ? 512 : 
                                               (suiteName == "Standard") ? 512 : 1024;  // Reduced from 1024/2048
                                int iterations = (suiteName == "Quick") ? 10 : 50;  // Reduced from 100
                                
                                g_App.currentBenchmark = "MatrixMul (" + std::to_string(matrixSize) + "x" + std::to_string(matrixSize) + ")";
                                
                                MatrixMulBenchmark bench(matrixSize);
                                bench.SetIterations(iterations);
                                bench.SetWarmupIterations(suiteName == "Quick" ? 2 : 5);  // Reduced warmup
                                result = bench.Run(backend);
                                
                            } else if (benchmarks[i] == "Convolution") {
                                int width = (suiteName == "Quick") ? 1280 : 
                                          (suiteName == "Standard") ? 1280 : 1920;  // Reduced from 1920/3840
                                int height = (suiteName == "Quick") ? 720 : 
                                           (suiteName == "Standard") ? 720 : 1080;  // Reduced from 1080/2160
                                int iterations = (suiteName == "Quick") ? 10 : 50;  // Reduced from 100
                                
                                g_App.currentBenchmark = "Convolution (" + std::to_string(width) + "x" + std::to_string(height) + ")";
                                
                                ConvolutionBenchmark bench(width, height);
                                bench.SetIterations(iterations);
                                bench.SetWarmupIterations(suiteName == "Quick" ? 2 : 5);  // Reduced warmup
                                result = bench.Run(backend);
                                
                            } else if (benchmarks[i] == "Reduction") {
                                size_t problemSize = (suiteName == "Quick") ? 1000000 : 
                                                   (suiteName == "Standard") ? 5000000 : 10000000;  // Reduced from 10M/100M
                                int iterations = (suiteName == "Quick") ? 10 : 50;  // Reduced from 100
                                
                                g_App.currentBenchmark = "Reduction (" + std::to_string(problemSize / 1000000) + "M elements)";
                                
                                ReductionBenchmark bench(problemSize);
                                bench.SetIterations(iterations);
                                bench.SetWarmupIterations(suiteName == "Quick" ? 2 : 5);  // Reduced warmup
                                result = bench.Run(backend);
                            }
                            
                            // Clean up after each benchmark
                            backend->Synchronize();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            
                            // Add to results
                            ApplicationState::BenchmarkResult guiResult;
                            guiResult.name = benchmarks[i];
                            guiResult.backend = selectedBackendName;
                            guiResult.timeMs = result.executionTimeMS;
                            
                            if (benchmarks[i] == "MatrixMul") {
                                guiResult.performance = result.computeThroughputGFLOPS;
                                guiResult.unit = "GFLOPS";
                            } else {
                                guiResult.performance = result.effectiveBandwidthGBs;
                                guiResult.unit = "GB/s";
                            }
                            
                            guiResult.passed = result.resultCorrect;
                            
                            std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                            g_App.results.push_back(guiResult);
                            
                        } catch (const std::exception& e) {
                            std::string errorMsg = "Error in " + benchmarks[i] + ": " + e.what();
                            g_App.currentBenchmark = errorMsg;
                            
                            ApplicationState::BenchmarkResult errorResult;
                            errorResult.name = benchmarks[i];
                            errorResult.backend = selectedBackendName + " (ERROR)";
                            errorResult.timeMs = 0.0;
                            errorResult.performance = 0.0;
                            errorResult.unit = "ERROR";
                            errorResult.passed = false;
                            
                            std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                            g_App.results.push_back(errorResult);
                            
                            // Continue with next benchmark instead of stopping
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        } catch (...) {
                            g_App.currentBenchmark = "Unknown error in " + benchmarks[i];
                            
                            ApplicationState::BenchmarkResult errorResult;
                            errorResult.name = benchmarks[i];
                            errorResult.backend = selectedBackendName + " (CRASH)";
                            errorResult.timeMs = 0.0;
                            errorResult.performance = 0.0;
                            errorResult.unit = "CRASH";
                            errorResult.passed = false;
                            
                            std::lock_guard<std::mutex> lock(g_App.resultsMutex);
                            g_App.results.push_back(errorResult);
                            
                            // Continue with next benchmark
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        }
                    }
                    
                    g_App.progress = 1.0f;
                    g_App.currentBenchmark = "Complete!";
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    
                    g_App.benchmarkRunning = false;
                    g_App.workerThreadRunning = false;
                    
                } catch (const std::exception& e) {
                    g_App.currentBenchmark = std::string("ERROR: ") + e.what();
                    g_App.benchmarkRunning = false;
                    g_App.workerThreadRunning = false;
                } catch (...) {
                    g_App.currentBenchmark = "ERROR: Unknown exception occurred";
                    g_App.benchmarkRunning = false;
                    g_App.workerThreadRunning = false;
                }
            });
        }
        
        if (!canStart) ImGui::EndDisabled();
        
        // Progress
        if (g_App.benchmarkRunning) {
            ImGui::Text("Running: %s", g_App.currentBenchmark.c_str());
            ImGui::ProgressBar(g_App.progress, ImVec2(-1, 0));
        }
    }
    
    ImGui::Separator();
    
    // Results Display
    if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::lock_guard<std::mutex> lock(g_App.resultsMutex);
        
        if (g_App.results.empty()) {
            ImGui::TextDisabled("No results yet. Run a benchmark to see results.");
        } else {
            // Results table
            if (ImGui::BeginTable("ResultsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Benchmark");
                ImGui::TableSetupColumn("Backend");
                ImGui::TableSetupColumn("Time (ms)");
                ImGui::TableSetupColumn("Performance");
                ImGui::TableSetupColumn("Status");
                ImGui::TableHeadersRow();
                
                for (const auto& result : g_App.results) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", result.name.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", result.backend.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", result.timeMs);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.1f %s", result.performance, result.unit.c_str());
                    ImGui::TableNextColumn();
                    if (result.passed) {
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "PASS");
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "FAIL");
                    }
                }
                
                ImGui::EndTable();
            }
            
            ImGui::Separator();
            if (ImGui::Button("Export to CSV", ImVec2(150, 30))) {
                // Export results to CSV
                std::ofstream csvFile("benchmark_results_gui.csv");
                if (csvFile.is_open()) {
                    csvFile << "Benchmark,Backend,Time_ms,Performance,Unit,Status\n";
                    for (const auto& result : g_App.results) {
                        csvFile << result.name << ","
                               << result.backend << ","
                               << result.timeMs << ","
                               << result.performance << ","
                               << result.unit << ","
                               << (result.passed ? "PASS" : "FAIL") << "\n";
                    }
                    csvFile.close();
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(saves to benchmark_results_gui.csv)");
        }
    }
    
    ImGui::End();
    
    // About Dialog
    if (g_App.showAboutDialog) {
        ImGui::OpenPopup("About");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        
        if (ImGui::BeginPopupModal("About", &g_App.showAboutDialog, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("GPU Benchmark Suite");
            ImGui::Text("Version 1.0.0");
            ImGui::Separator();
            ImGui::Text("A professional GPU compute benchmarking tool");
            ImGui::Text("supporting CUDA, OpenCL, and DirectCompute.");
            ImGui::Separator();
            ImGui::Text("Developed by: Soham Dave");
            if (ImGui::SmallButton("GitHub: github.com/davesohamm")) {
                ShellExecuteA(nullptr, "open", "https://github.com/davesohamm", nullptr, nullptr, SW_SHOWNORMAL);
            }
            ImGui::Separator();
            ImGui::Text("Built with: ImGui, DirectX 11, C++");
            ImGui::Separator();
            if (ImGui::Button("Close", ImVec2(120, 0))) {
                g_App.showAboutDialog = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
}

//==============================================================================
// MAIN ENTRY POINT
//==============================================================================

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    
    // Redirect console output to null (we're a GUI app, not console)
    FILE* dummy;
    freopen_s(&dummy, "NUL", "w", stdout);
    freopen_s(&dummy, "NUL", "w", stderr);
    
    // Create window class
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"GPUBenchmark", nullptr };
    ::RegisterClassExW(&wc);
    g_App.hwnd = ::CreateWindowW(wc.lpszClassName, L"GPU Benchmark Suite", WS_OVERLAPPEDWINDOW, 100, 100, 1280, 720, nullptr, nullptr, wc.hInstance, nullptr);

    // Initialize Direct3D
    if (!CreateDeviceD3D(g_App.hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        MessageBoxW(nullptr, L"Failed to create Direct3D device!", L"Error", MB_OK | MB_ICONERROR);
        return 1;
    }

    // Show the window BEFORE initializing heavy stuff
    ::ShowWindow(g_App.hwnd, SW_SHOW);
    ::SetForegroundWindow(g_App.hwnd);
    ::SetFocus(g_App.hwnd);
    ::UpdateWindow(g_App.hwnd);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(g_App.hwnd);
    ImGui_ImplDX11_Init(g_App.d3dDevice, g_App.d3dContext);

    // Initialize benchmark runner (this is fast, just creates the object)
    g_App.benchmarkRunner = std::make_unique<BenchmarkRunner>();
    
    // Discover system capabilities (this might take a moment)
    g_App.systemCaps = DeviceDiscovery::Discover();

    // Main loop
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (g_App.running && msg.message != WM_QUIT) {
        // Poll and handle messages
        if (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            continue;
        }

        // Start the ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Render our UI
        RenderUI();

        // Rendering
        ImGui::Render();
        const float clear_color[4] = { 0.1f, 0.1f, 0.12f, 1.0f };
        g_App.d3dContext->OMSetRenderTargets(1, &g_App.mainRenderTargetView, nullptr);
        g_App.d3dContext->ClearRenderTargetView(g_App.mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_App.swapChain->Present(1, 0); // Present with vsync
    }

    // Cleanup
    // Wait for worker thread to finish
    g_App.running = false;
    if (g_App.workerThread.joinable()) {
        g_App.workerThread.join();
    }
    
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(g_App.hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

/********************************************************************************
 * END OF FILE: main_gui.cpp
 * 
 * This is the heart of your GPU Benchmark GUI!
 * 
 * Features implemented:
 * - Beautiful ImGui interface with DirectX 11
 * - System information display
 * - Backend selection (CUDA/OpenCL/DirectCompute)
 * - Benchmark configuration
 * - Results table
 * - About dialog with GitHub link
 * 
 * Next steps:
 * - Implement background benchmark execution
 * - Add CSV export functionality
 * - Add performance charts
 * - Polish UI with better styling
 ********************************************************************************/
