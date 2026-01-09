################################################################################
# FILE: CMakeLists.txt
# 
# PURPOSE:
#   CMake build configuration for GPU Compute Benchmark Tool.
#   
#   CMake is a cross-platform build system generator. It generates:
#     - Visual Studio solutions (.sln) on Windows
#     - Makefiles on Linux
#     - Xcode projects on macOS
#   
#   We use CMake because it handles complex dependencies (CUDA, OpenCL, DirectX)
#   better than manual Visual Studio project configuration.
# 
# HOW TO BUILD:
#   1. Install CMake (cmake.org)
#   2. Open PowerShell in project directory
#   3. mkdir build
#   4. cd build
#   5. cmake .. -G "Visual Studio 17 2022" -A x64
#   6. cmake --build . --config Release
#   7. .\Release\GPU-Benchmark.exe
# 
# REQUIREMENTS:
#   - CMake 3.18+ (for CUDA support)
#   - Visual Studio 2019 or 2022
#   - CUDA Toolkit 11.0+ (optional, for CUDA backend)
#   - Windows SDK (for DirectCompute)
# 
# AUTHOR: Soham
# DATE: January 2026
# SYSTEM: Windows 11, Visual Studio 2022
# 
################################################################################

# Minimum CMake version
# 3.18 required for CUDA language support
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name and version
# LANGUAGES: CXX (C++), CUDA (NVIDIA CUDA)
project(GPU-Benchmark 
    VERSION 1.0.0
    DESCRIPTION "GPU Compute Benchmark and Visualization Tool"
    LANGUAGES CXX CUDA
)

# C++ Standard: C++17
# We use C++17 for:
#   - std::filesystem (file I/O)
#   - std::optional (optional values)
#   - Structured bindings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard: C++17 (match C++ standard)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

################################################################################
# CONFIGURATION OPTIONS
# 
# These can be set from command line:
#   cmake .. -DUSE_CUDA=ON -DUSE_OPENCL=OFF
################################################################################

# Enable/disable backends
option(USE_CUDA "Build with CUDA support" ON)
option(USE_OPENCL "Build with OpenCL support" ON)
option(USE_DIRECTCOMPUTE "Build with DirectCompute support" ON)

# Enable/disable GUI
option(BUILD_GUI "Build GUI (requires OpenGL/GLFW)" ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

################################################################################
# FIND REQUIRED PACKAGES
################################################################################

# Windows SDK (required for DirectCompute)
# This is usually automatically found on Windows
if(WIN32)
    message(STATUS "Windows platform detected")
endif()

# CUDA Toolkit (optional)
if(USE_CUDA)
    # FindCUDA module looks for CUDA installation
    # Sets variables: CUDA_FOUND, CUDA_INCLUDE_DIRS, CUDA_LIBRARIES
    find_package(CUDA QUIET)
    
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
        
        # Set CUDA compute capabilities
        # RTX 3050 = Ampere = sm_86
        # This tells CUDA compiler to generate code for these architectures
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
            -gencode arch=compute_86,code=sm_86  # Ampere (RTX 30xx, RTX 40xx)
            -gencode arch=compute_75,code=sm_75  # Turing (RTX 20xx, GTX 16xx)
            -gencode arch=compute_61,code=sm_61  # Pascal (GTX 10xx)
        )
        
        # Enable fast math for better performance (trade accuracy for speed)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)
        
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "CUDA not found - CUDA backend will be disabled")
        set(USE_CUDA OFF)
    endif()
endif()

# OpenCL (optional)
if(USE_OPENCL)
    # FindOpenCL module (built into CMake)
    find_package(OpenCL QUIET)
    
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
        add_definitions(-DUSE_OPENCL)
    else()
        message(WARNING "OpenCL not found - OpenCL backend will be disabled")
        set(USE_OPENCL OFF)
    endif()
endif()

# DirectX (DirectCompute) - Windows only
if(USE_DIRECTCOMPUTE AND WIN32)
    # DirectX is part of Windows SDK, no need to find it
    # We'll link with d3d11.lib and d3dcompiler.lib
    add_definitions(-DUSE_DIRECTCOMPUTE)
    message(STATUS "DirectCompute enabled (Windows platform)")
endif()

################################################################################
# SOURCE FILES
################################################################################

# Core framework sources
set(CORE_SOURCES
    src/core/Timer.cpp
    src/core/DeviceDiscovery.cpp
)

set(CORE_HEADERS
    src/core/IComputeBackend.h
    src/core/Timer.h
    src/core/DeviceDiscovery.h
)

# CUDA backend sources (only if CUDA enabled)
if(USE_CUDA)
    set(CUDA_BACKEND_SOURCES
        # These would be .cu files compiled by NVCC
        # Placeholder for now
    )
endif()

# Main application source
set(APP_SOURCES
    src/main.cpp
)

################################################################################
# CREATE EXECUTABLE
################################################################################

# Main executable target
add_executable(GPU-Benchmark
    ${APP_SOURCES}
    ${CORE_SOURCES}
    ${CORE_HEADERS}
)

# Include directories
target_include_directories(GPU-Benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(GPU-Benchmark PRIVATE
    # Windows libraries (always needed)
    # These provide: DXGI, Direct3D 11, registry access, etc.
)

if(WIN32)
    target_link_libraries(GPU-Benchmark PRIVATE
        dxgi.lib        # DirectX Graphics Infrastructure (GPU enumeration)
        d3d11.lib       # Direct3D 11 (DirectCompute)
        d3dcompiler.lib # HLSL shader compiler
    )
endif()

# CUDA libraries (if enabled)
if(USE_CUDA AND CUDA_FOUND)
    target_include_directories(GPU-Benchmark PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(GPU-Benchmark PRIVATE ${CUDA_LIBRARIES})
endif()

# OpenCL libraries (if enabled)
if(USE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(GPU-Benchmark PRIVATE OpenCL::OpenCL)
endif()

################################################################################
# COMPILER FLAGS
################################################################################

# Windows-specific flags
if(MSVC)
    # Enable multi-processor compilation (faster builds)
    target_compile_options(GPU-Benchmark PRIVATE /MP)
    
    # Warning level 4 (high)
    target_compile_options(GPU-Benchmark PRIVATE /W4)
    
    # Treat warnings as errors (enforce clean code)
    # target_compile_options(GPU-Benchmark PRIVATE /WX)
    
    # Optimization flags for Release build
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(GPU-Benchmark PRIVATE /O2)  # Maximize speed
        target_compile_options(GPU-Benchmark PRIVATE /Oi)  # Enable intrinsics
    endif()
endif()

################################################################################
# POST-BUILD STEPS
################################################################################

# Create results directory
add_custom_command(TARGET GPU-Benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory 
    ${CMAKE_BINARY_DIR}/results
    COMMENT "Creating results directory"
)

# Copy shaders to build directory (if DirectCompute enabled)
if(USE_DIRECTCOMPUTE)
    # This would copy .hlsl files
    # Placeholder for now
endif()

################################################################################
# INSTALLATION (optional)
################################################################################

# Install executable
install(TARGETS GPU-Benchmark
    RUNTIME DESTINATION bin
)

# Install documentation
install(FILES
    README.md
    BUILD_GUIDE.md
    ARCHITECTURE.md
    RESULTS_INTERPRETATION.md
    DESTINATION docs
)

################################################################################
# PRINT CONFIGURATION SUMMARY
################################################################################

message(STATUS "")
message(STATUS "==============================================")
message(STATUS "GPU Benchmark Tool - Build Configuration")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Backends:")
message(STATUS "  CUDA:          ${USE_CUDA}")
if(USE_CUDA AND CUDA_FOUND)
    message(STATUS "    Version:     ${CUDA_VERSION}")
endif()
message(STATUS "  OpenCL:        ${USE_OPENCL}")
if(USE_OPENCL AND OpenCL_FOUND)
    message(STATUS "    Version:     ${OpenCL_VERSION_STRING}")
endif()
message(STATUS "  DirectCompute: ${USE_DIRECTCOMPUTE}")
message(STATUS "")
message(STATUS "GUI:             ${BUILD_GUI}")
message(STATUS "==============================================")
message(STATUS "")

################################################################################
# END OF FILE: CMakeLists.txt
# 
# USAGE SUMMARY:
#   mkdir build
#   cd build
#   cmake .. -G "Visual Studio 17 2022" -A x64
#   cmake --build . --config Release
#   .\Release\GPU-Benchmark.exe
# 
# WHAT THIS FILE DOES:
#   1. Configures project for CMake
#   2. Finds required libraries (CUDA, OpenCL, DirectX)
#   3. Sets up include directories and link libraries
#   4. Defines compiler flags
#   5. Creates build targets
# 
# CMAKE BENEFITS:
#   - Handles complex dependencies automatically
#   - Generates Visual Studio solutions
#   - Cross-platform (could compile on Linux with minor changes)
#   - Tracks file dependencies (incremental builds)
# 
# FOR INTERVIEWS:
#   Explain:
#     - Why use build systems (vs manual compilation)
#     - How CMake finds libraries (FindCUDA, FindOpenCL)
#     - Conditional compilation (USE_CUDA, USE_OPENCL flags)
#     - Compiler optimization flags (-O2, /Oi)
#     - Linking libraries (static vs dynamic)
# 
################################################################################
