# GPU Compute Benchmark and Visualization Tool for Windows

## 📋 Project Overview

This is a professional-grade GPU compute benchmarking application designed to measure and compare performance across multiple GPU compute APIs on Windows systems. The tool provides deep insights into GPU architectural differences, memory behavior, and compute efficiency.

### 🎯 Purpose
- **Learning**: Understand how different GPU APIs work at a low level
- **Comparison**: Fair benchmarking across CUDA, OpenCL, and DirectCompute
- **Analysis**: Visualize performance characteristics and bottlenecks
- **Portfolio**: Demonstrate deep GPU programming knowledge for technical interviews

---

## 🖥️ Your System Specifications

This project was developed and tested on:
- **CPU**: AMD Ryzen 7 4800H (8 cores, 16 threads)
- **GPU**: NVIDIA RTX 3050 (4GB VRAM, Ampere architecture)
- **RAM**: 16 GB
- **OS**: Windows 11

**What this means for you:**
- ✅ **CUDA**: Fully supported (NVIDIA GPU detected)
- ✅ **OpenCL**: Supported (NVIDIA provides OpenCL drivers)
- ✅ **DirectCompute**: Supported (Windows native API)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│  (User Interface, Benchmark Controller, Result Logging)     │
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────────────────┐
│                    CORE FRAMEWORK                            │
│  (Abstract Interfaces, Timing, Device Discovery)            │
└─────────────┬───────────────────────────────────────────────┘
              │
      ┌───────┴───────┬───────────┬──────────┐
      │               │           │          │
┌─────▼─────┐  ┌─────▼─────┐  ┌──▼──────┐  ┌▼──────────┐
│   CUDA    │  │  OpenCL   │  │DirectCmp│  │  OpenGL   │
│  Backend  │  │  Backend  │  │ Backend │  │ Renderer  │
└───────────┘  └───────────┘  └─────────┘  └───────────┘
      │               │            │              │
      └───────┬───────┴────────────┴──────────────┘
              │
     ┌────────▼─────────┐
     │   GPU HARDWARE   │
     │  (RTX 3050)      │
     └──────────────────┘
```

---

## 📁 Project Structure

```
GPU-Benchmark/
│
├── README.md                          # This file - main project documentation
├── BUILD_GUIDE.md                     # Detailed build instructions
├── ARCHITECTURE.md                    # Deep dive into system architecture
├── RESULTS_INTERPRETATION.md          # How to understand benchmark results
│
├── src/                               # Source code
│   ├── main.cpp                       # Application entry point
│   │
│   ├── core/                          # Core framework (API-agnostic)
│   │   ├── README.md                  # Core framework documentation
│   │   ├── IComputeBackend.h          # Abstract interface for all backends
│   │   ├── BenchmarkRunner.h/cpp      # Orchestrates benchmark execution
│   │   ├── Timer.h/cpp                # High-resolution timing utilities
│   │   ├── DeviceDiscovery.h/cpp      # Runtime GPU/API detection
│   │   └── Logger.h/cpp               # Logging and result export
│   │
│   ├── backends/                      # Compute backend implementations
│   │   ├── README.md                  # Backend comparison guide
│   │   │
│   │   ├── cuda/                      # NVIDIA CUDA backend
│   │   │   ├── README.md              # CUDA-specific documentation
│   │   │   ├── CUDABackend.h/cpp      # CUDA implementation
│   │   │   └── kernels/               # CUDA kernel implementations
│   │   │       ├── vector_add.cu
│   │   │       ├── matrix_mul.cu
│   │   │       ├── convolution.cu
│   │   │       └── reduction.cu
│   │   │
│   │   ├── opencl/                    # OpenCL backend
│   │   │   ├── README.md              # OpenCL-specific documentation
│   │   │   ├── OpenCLBackend.h/cpp    # OpenCL implementation
│   │   │   └── kernels/               # OpenCL kernel source strings
│   │   │       ├── vector_add.cl
│   │   │       ├── matrix_mul.cl
│   │   │       ├── convolution.cl
│   │   │       └── reduction.cl
│   │   │
│   │   └── directcompute/             # DirectCompute backend
│   │       ├── README.md              # DirectCompute-specific docs
│   │       ├── DirectComputeBackend.h/cpp
│   │       └── shaders/               # HLSL compute shaders
│   │           ├── vector_add.hlsl
│   │           ├── matrix_mul.hlsl
│   │           ├── convolution.hlsl
│   │           └── reduction.hlsl
│   │
│   ├── benchmarks/                    # Benchmark definitions
│   │   ├── README.md                  # What each benchmark measures
│   │   ├── VectorAddBenchmark.h/cpp
│   │   ├── MatrixMulBenchmark.h/cpp
│   │   ├── ConvolutionBenchmark.h/cpp
│   │   └── ReductionBenchmark.h/cpp
│   │
│   ├── visualization/                 # OpenGL rendering and GUI
│   │   ├── README.md                  # Visualization architecture
│   │   ├── Renderer.h/cpp             # OpenGL renderer
│   │   ├── GUI.h/cpp                  # User interface
│   │   └── shaders/                   # OpenGL shaders for visualization
│   │       ├── vertex.glsl
│   │       └── fragment.glsl
│   │
│   └── utils/                         # Utility functions
│       ├── FileIO.h/cpp               # CSV export, file operations
│       └── SystemInfo.h/cpp           # Hardware information queries
│
├── include/                           # Third-party headers (GLFW, glad, etc.)
├── lib/                               # Third-party libraries
├── build/                             # Build output (generated)
└── results/                           # Benchmark results (CSV files)
```

---

## 🔬 What This Tool Measures

### 1. **Kernel Execution Time**
Pure GPU compute time, excluding all host-side overhead and data transfers.

### 2. **Memory Transfer Time**
- Host → Device (Upload time)
- Device → Host (Download time)

### 3. **Memory Bandwidth**
Effective data throughput in GB/s during transfers and compute operations.

### 4. **Dispatch Latency**
Time overhead to launch a kernel (important for understanding API differences).

### 5. **Scaling Behavior**
How performance changes as problem size increases (weak scaling analysis).

### 6. **Compute Efficiency**
Achieved FLOPS compared to theoretical GPU peak performance.

---

## 🧪 Benchmark Suite

### 1. **Vector Addition** (`C[i] = A[i] + B[i]`)
- **Tests**: Memory bandwidth, kernel launch overhead
- **Use Case**: Stream processing, element-wise operations
- **Memory Pattern**: Perfectly coalesced, streaming access

### 2. **Matrix Multiplication** (`C = A × B`)
- **Tests**: Compute intensity, cache utilization, shared memory
- **Use Case**: Deep learning, scientific computing
- **Memory Pattern**: Complex strided access, cache-dependent

### 3. **2D Convolution** (Image filtering)
- **Tests**: Memory access patterns, texture cache, constant memory
- **Use Case**: Computer vision, image processing
- **Memory Pattern**: Overlapping reads, halo regions

### 4. **Parallel Reduction** (Sum all elements)
- **Tests**: Synchronization, shared memory, warp/wavefront efficiency
- **Use Case**: Aggregations, statistics
- **Memory Pattern**: Tree-based reduction, bank conflicts

---

## 🚀 Quick Start

### Prerequisites
1. **Visual Studio 2019 or 2022** with C++ desktop development tools
2. **NVIDIA CUDA Toolkit** (version 11.0 or higher) - [Download here](https://developer.nvidia.com/cuda-downloads)
3. **Windows SDK** (included with Visual Studio)
4. **GPU Drivers** (latest from NVIDIA)

### Building the Project

```powershell
# 1. Clone or extract this project to a local directory
cd y:\GPU-Benchmark

# 2. Open Visual Studio solution
start GPU-Benchmark.sln

# 3. Build the solution (Release mode recommended)
# Press Ctrl+Shift+B or use Build → Build Solution

# 4. Run the executable
.\build\Release\GPU-Benchmark.exe
```

Detailed build instructions are in [`BUILD_GUIDE.md`](BUILD_GUIDE.md).

---

## 📊 Using the Application

### GUI Mode (Default)
1. Launch `GPU-Benchmark.exe`
2. The application will automatically detect your GPU and supported APIs
3. Select benchmarks from the list
4. Choose which backends to run (CUDA/OpenCL/DirectCompute)
5. Configure problem sizes (Small/Medium/Large)
6. Click "Run Benchmarks"
7. View real-time results in the visualization window
8. Export results to CSV for further analysis

### Command-Line Mode
```powershell
# Run all benchmarks on all available backends
GPU-Benchmark.exe --all

# Run specific benchmark
GPU-Benchmark.exe --benchmark=matrix_mul --backend=cuda

# Specify output file
GPU-Benchmark.exe --all --output=results.csv
```

---

## 📈 Understanding Results

### Sample Output Message
```
=== GPU Compute Benchmark Tool ===
Detected Hardware: NVIDIA GeForce RTX 3050 Laptop GPU
Driver Version: 546.12
CUDA Compute Capability: 8.6

Available Backends:
✓ CUDA: Enabled (NVIDIA GPU detected)
✓ OpenCL: Enabled (OpenCL 3.0)
✓ DirectCompute: Enabled (DirectX 11.0)

Running Vector Addition (1M elements)...
  CUDA:          0.234 ms (execution)  |  1.23 ms (transfer)  |  89.2 GB/s
  OpenCL:        0.289 ms (execution)  |  1.45 ms (transfer)  |  76.4 GB/s
  DirectCompute: 0.312 ms (execution)  |  1.67 ms (transfer)  |  71.8 GB/s
```

### What to Look For
- **CUDA typically fastest**: Direct hardware access on NVIDIA GPUs
- **OpenCL overhead**: Cross-platform abstraction cost
- **DirectCompute integration**: Best Windows API integration
- **Memory transfer bottleneck**: Often dominates small workloads

See [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) for detailed analysis guidance.

---

## 🎓 Learning Outcomes

By studying this project, you'll understand:

1. **GPU Architecture**: How modern GPUs execute parallel workloads
2. **Memory Hierarchies**: Global vs shared vs constant memory trade-offs
3. **API Differences**: Why CUDA, OpenCL, and DirectCompute exist
4. **Performance Analysis**: Identifying bottlenecks (compute vs memory bound)
5. **Low-Level Windows Programming**: DirectX integration, driver interaction
6. **Software Architecture**: Clean separation between backends and visualization
7. **Real-Time Rendering**: OpenGL integration with compute results

---

## 🔧 Customization and Extension

### Adding a New Benchmark
1. Create new class inheriting from `IBenchmark` in `src/benchmarks/`
2. Implement kernels for each backend (CUDA/OpenCL/DirectCompute)
3. Register benchmark in `BenchmarkRunner.cpp`
4. Document expected performance characteristics

### Adding a New Backend (e.g., Vulkan Compute)
1. Create new directory in `src/backends/vulkan/`
2. Implement `IComputeBackend` interface
3. Add detection logic in `DeviceDiscovery.cpp`
4. Implement equivalent kernels for all benchmarks

---

## 🐛 Troubleshooting

### "CUDA backend failed to initialize"
- **Cause**: CUDA toolkit not installed or NVIDIA driver outdated
- **Solution**: Install CUDA Toolkit and update GPU drivers

### "OpenCL backend unavailable"
- **Cause**: OpenCL ICD loader not found
- **Solution**: Update GPU drivers (vendors include OpenCL support)

### "DirectCompute backend failed"
- **Cause**: DirectX runtime issue
- **Solution**: Update Windows and run Windows Update

### Application crashes on launch
- **Cause**: Missing dependencies
- **Solution**: Install Visual C++ Redistributable 2022

---

## 📚 Additional Documentation

- **[BUILD_GUIDE.md](BUILD_GUIDE.md)**: Step-by-step compilation instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep dive into design decisions
- **[RESULTS_INTERPRETATION.md](RESULTS_INTERPRETATION.md)**: How to analyze results
- **[src/core/README.md](src/core/README.md)**: Core framework documentation
- **[src/backends/README.md](src/backends/README.md)**: Backend comparison guide

---

## 🙏 Acknowledgments

This project demonstrates understanding of:
- NVIDIA CUDA programming model
- Khronos OpenCL specification
- Microsoft DirectCompute and HLSL
- OpenGL rendering pipeline
- High-performance C++ development
- Windows driver model interaction

---

## 📝 Interview Talking Points

When discussing this project:

1. **Architecture**: Explain the modular backend design and abstraction layers
2. **Performance**: Discuss memory coalescing, occupancy, and memory bandwidth
3. **Trade-offs**: CUDA performance vs OpenCL portability vs DirectCompute integration
4. **Measurement**: How to accurately time GPU operations without polluting results
5. **Hardware**: How different GPU architectures (NVIDIA, AMD, Intel) behave
6. **Scalability**: How performance changes with problem size

---

## 📄 License

This project is created for educational and portfolio purposes.

---

**Author**: Soham  
**Hardware**: AMD Ryzen 7 4800H | NVIDIA RTX 3050 | 16GB RAM  
**System**: Windows 11  
**Created**: 2026

---

**Remember**: This tool shows relative performance differences between APIs on YOUR hardware. Results will vary on different systems - that's expected and demonstrates the hardware-dependent nature of GPU computing!
