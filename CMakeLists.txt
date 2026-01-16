cmake_minimum_required(VERSION 3.18)

project(GPUBenchmark CUDA CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture for RTX 3050 (Ampere, compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find OpenCL
find_package(OpenCL)

# Define preprocessor macros for enabled backends
add_compile_definitions(USE_CUDA)

if(OpenCL_FOUND)
    add_compile_definitions(USE_OPENCL)
    message(STATUS "OpenCL Found: ${OpenCL_VERSION_STRING}")
    message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
else()
    message(WARNING "OpenCL not found - OpenCL backend will not be available")
endif()

# DirectCompute is always available on Windows (via DirectX 11)
if(WIN32)
    add_compile_definitions(USE_DIRECTCOMPUTE)
    message(STATUS "DirectCompute: Enabled (Windows platform)")
endif()

# Core sources
set(CORE_SOURCES
    src/core/Timer.cpp
    src/core/Logger.cpp
    src/core/BenchmarkRunner.cpp
    src/core/DeviceDiscovery.cpp
)

# CUDA backend sources
set(CUDA_SOURCES
    src/backends/cuda/CUDABackend.cpp
)

# OpenCL backend sources (conditionally included)
if(OpenCL_FOUND)
    set(OPENCL_SOURCES
        src/backends/opencl/OpenCLBackend.cpp
    )
endif()

# DirectCompute backend sources (Windows only)
if(WIN32)
    set(DIRECTCOMPUTE_SOURCES
        src/backends/directcompute/DirectComputeBackend.cpp
    )
endif()

# Benchmark wrapper classes
set(BENCHMARK_SOURCES
    src/benchmarks/VectorAddBenchmark.cpp
    src/benchmarks/MatrixMulBenchmark.cpp
    src/benchmarks/ConvolutionBenchmark.cpp
    src/benchmarks/ReductionBenchmark.cpp
)

# CUDA kernel sources
set(CUDA_KERNELS
    src/backends/cuda/kernels/vector_add.cu
    src/backends/cuda/kernels/matrix_mul.cu
    src/backends/cuda/kernels/convolution.cu
    src/backends/cuda/kernels/reduction.cu
)

# Test 1: Logger test (C++ only, no CUDA)
# Only needs Logger and Timer, not the full framework
add_executable(test_logger 
    test_logger.cpp 
    src/core/Timer.cpp
    src/core/Logger.cpp
)
target_include_directories(test_logger PRIVATE ${CMAKE_SOURCE_DIR})

# Test 2: Simple CUDA test
add_executable(test_cuda_simple test_cuda_simple.cu)
set_target_properties(test_cuda_simple PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

# Test 3: Full CUDA backend test
add_executable(test_cuda_backend test_cuda_backend.cu ${CUDA_SOURCES} ${CORE_SOURCES})
set_target_properties(test_cuda_backend PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(test_cuda_backend PRIVATE ${CMAKE_SOURCE_DIR})

# Test 4: Matrix multiplication kernel test
add_executable(test_matmul 
    test_matmul.cu 
    src/backends/cuda/kernels/matrix_mul.cu
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
set_target_properties(test_matmul PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(test_matmul PRIVATE ${CMAKE_SOURCE_DIR})

# Test 5: Convolution kernel test
add_executable(test_convolution
    test_convolution.cu
    src/backends/cuda/kernels/convolution.cu
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
set_target_properties(test_convolution PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(test_convolution PRIVATE ${CMAKE_SOURCE_DIR})

# Test 6: Reduction kernel test
add_executable(test_reduction
    test_reduction.cu
    src/backends/cuda/kernels/reduction.cu
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
set_target_properties(test_reduction PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(test_reduction PRIVATE ${CMAKE_SOURCE_DIR})

message(STATUS "")
message(STATUS "=============================================")
message(STATUS "  GPU Benchmark - CMake Configuration")
message(STATUS "=============================================")
message(STATUS "C++ Compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA Compiler:   ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Arch:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "")
# Test 7: OpenCL backend test (if OpenCL available)
if(OpenCL_FOUND)
    # Create as CUDA project to link CUDA runtime properly
    add_executable(test_opencl_backend
        test_opencl_backend.cpp
        test_opencl_stub.cu
        ${OPENCL_SOURCES}
        ${CORE_SOURCES}
        ${CUDA_SOURCES}
    )
    set_target_properties(test_opencl_backend PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
    )
    target_include_directories(test_opencl_backend PRIVATE 
        ${CMAKE_SOURCE_DIR}
        ${OpenCL_INCLUDE_DIRS}
    )
    target_link_libraries(test_opencl_backend 
        ${OpenCL_LIBRARIES}
    )
endif()

# Test 8: DirectCompute backend test (Windows only)
if(WIN32)
    add_executable(test_directcompute_backend
        test_directcompute_backend.cpp
        test_opencl_stub.cu
        ${DIRECTCOMPUTE_SOURCES}
        ${CORE_SOURCES}
        ${CUDA_SOURCES}
        ${OPENCL_SOURCES}
    )
    set_target_properties(test_directcompute_backend PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
    )
    target_include_directories(test_directcompute_backend PRIVATE 
        ${CMAKE_SOURCE_DIR}
        ${OpenCL_INCLUDE_DIRS}
    )
    target_link_libraries(test_directcompute_backend 
        ${OpenCL_LIBRARIES}
    )
endif()

# Check for ImGui
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui")
if(EXISTS "${IMGUI_DIR}/imgui.h")
    message(STATUS "ImGui Found: ${IMGUI_DIR}")
    set(IMGUI_FOUND TRUE)
    
    # ImGui sources
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
        ${IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    )
else()
    message(WARNING "ImGui not found - GUI application will not be built. Run DOWNLOAD_IMGUI.cmd first.")
    set(IMGUI_FOUND FALSE)
endif()

# Main application (CLI)
add_executable(GPU-Benchmark 
    src/main_working.cpp
    src/cuda_stub.cu
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
    ${CUDA_KERNELS}
    ${OPENCL_SOURCES}
    ${DIRECTCOMPUTE_SOURCES}
)
set_target_properties(GPU-Benchmark PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(GPU-Benchmark PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${OpenCL_INCLUDE_DIRS}
)
target_link_libraries(GPU-Benchmark ${OpenCL_LIBRARIES})

# GUI Application
if(IMGUI_FOUND AND WIN32)
    add_executable(GPU-Benchmark-GUI WIN32
        src/gui/main_gui_fixed.cpp
        test_opencl_stub.cu
        ${IMGUI_SOURCES}
        ${CORE_SOURCES}
        ${CUDA_SOURCES}
        ${CUDA_KERNELS}
        ${OPENCL_SOURCES}
        ${DIRECTCOMPUTE_SOURCES}
    )
    set_target_properties(GPU-Benchmark-GUI PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
    )
    target_include_directories(GPU-Benchmark-GUI PRIVATE 
        ${CMAKE_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${OpenCL_INCLUDE_DIRS}
    )
    target_link_libraries(GPU-Benchmark-GUI 
        ${OpenCL_LIBRARIES}
        d3d11
        dxgi
    )
    message(STATUS "GUI application target added: GPU-Benchmark-GUI")
endif()

message(STATUS "Targets:")
message(STATUS "  - GPU-Benchmark        (Main CLI application)")
if(IMGUI_FOUND AND WIN32)
    message(STATUS "  - GPU-Benchmark-GUI    (GUI application) ⭐ NEW!")
endif()
message(STATUS "  - test_logger          (Logger test)")
message(STATUS "  - test_cuda_simple     (Simple CUDA)")
message(STATUS "  - test_cuda_backend    (Full backend)")
message(STATUS "  - test_matmul          (Matrix multiplication)")
message(STATUS "  - test_convolution     (2D convolution)")
message(STATUS "  - test_reduction       (Parallel reduction)")
if(OpenCL_FOUND)
    message(STATUS "  - test_opencl_backend  (OpenCL backend)")
endif()
if(WIN32)
    message(STATUS "  - test_directcompute_backend (DirectCompute backend)")
endif()
message(STATUS "=============================================")
message(STATUS "")
