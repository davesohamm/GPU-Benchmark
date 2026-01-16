# 📚 API References and Learning Resources

## Official API Documentation

### CUDA (NVIDIA)

#### Primary Documentation
- **CUDA C++ Programming Guide**
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - Comprehensive guide to CUDA programming
  - Essential reading for understanding our CUDA backend

- **CUDA Runtime API Reference**
  - https://docs.nvidia.com/cuda/cuda-runtime-api/
  - Complete API reference for all CUDA functions
  - Used extensively in `src/backends/cuda/CUDABackend.cpp`

- **CUDA Toolkit Documentation**
  - https://docs.nvidia.com/cuda/
  - Hub for all CUDA documentation

#### Best Practices Guides
- **CUDA C++ Best Practices Guide**
  - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
  - Optimization techniques we use in our kernels
  
- **Parallel Reduction**
  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  - Mark Harris's famous reduction optimization guide
  - Direct inspiration for our `reduction.cu` kernel

#### GPU Architecture
- **Ampere Architecture Whitepaper** (RTX 3000 series)
  - https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
  
- **CUDA GPU Compute Capability List**
  - https://developer.nvidia.com/cuda-gpus
  - Find your GPU's compute capability

### OpenCL (Khronos Group)

#### Primary Documentation
- **OpenCL 3.0 Specification**
  - https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html
  - Official API specification
  
- **OpenCL C Language Specification**
  - https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html
  - Kernel language syntax

- **OpenCL Quick Reference**
  - https://www.khronos.org/files/opencl30-reference-guide.pdf
  - Handy PDF reference card

#### Tutorials and Guides
- **Hands On OpenCL**
  - https://handsonopencl.github.io/
  - Excellent tutorial series
  
- **OpenCL Programming Guide** (Book)
  - By Aaftab Munshi, Benedict Gaster, et al.
  - ISBN: 978-0321749642

### DirectCompute / HLSL (Microsoft)

#### Primary Documentation
- **DirectCompute Overview**
  - https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader
  - Official Microsoft documentation

- **HLSL Reference**
  - https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-reference
  - Complete HLSL language reference
  
- **Compute Shader Overview**
  - https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-create
  - How to create and use compute shaders

#### DirectX 11 Programming
- **Direct3D 11 Documentation**
  - https://learn.microsoft.com/en-us/windows/win32/direct3d11/atoc-dx-graphics-direct3d-11
  - Full DirectX 11 API documentation

---

## Books (Highly Recommended)

### GPU Programming
1. **"Programming Massively Parallel Processors"**
   - Authors: David Kirk, Wen-mei Hwu
   - ISBN: 978-0124159921
   - **Best for:** Understanding GPU architecture fundamentals
   - **Used in this project:** Matrix multiplication optimization insights

2. **"CUDA by Example"**
   - Authors: Jason Sanders, Edward Kandrot
   - ISBN: 978-0131387683
   - **Best for:** Learning CUDA from scratch
   - **Used in this project:** Vector addition patterns

3. **"Professional CUDA C Programming"**
   - Author: John Cheng, Max Grossman, Ty McKercher
   - ISBN: 978-1118739327
   - **Best for:** Advanced optimization techniques
   - **Used in this project:** Warp shuffle primitives, bank conflict avoidance

4. **"Heterogeneous Computing with OpenCL 2.0"**
   - Authors: David Kaeli, Perhaad Mistry, et al.
   - ISBN: 978-0128014141
   - **Best for:** Cross-platform GPU programming
   - **Used in this project:** OpenCL backend design

### C++ and Software Engineering
5. **"Effective Modern C++"**
   - Author: Scott Meyers
   - ISBN: 978-1491903995
   - **Best for:** Modern C++ patterns (C++11/14/17)
   - **Used in this project:** Smart pointers, move semantics, RAII

6. **"Design Patterns"**
   - Authors: Gang of Four (Gamma, Helm, Johnson, Vlissides)
   - ISBN: 978-0201633610
   - **Best for:** Software architecture patterns
   - **Used in this project:** Strategy, Factory, Singleton patterns

---

## Online Courses and Tutorials

### CUDA
- **NVIDIA CUDA Training**
  - https://www.nvidia.com/en-us/training/
  - Free official NVIDIA courses

- **Udacity: Intro to Parallel Programming (CS344)**
  - https://www.udacity.com/course/intro-to-parallel-programming--cs344
  - Free course, excellent for beginners

- **Coursera: GPU Programming Specialization**
  - https://www.coursera.org/specializations/gpu-programming
  - Johns Hopkins University

### OpenCL
- **Hands-On OpenCL Course**
  - https://handsonopencl.github.io/
  - Free, interactive tutorial
  
### DirectCompute/DirectX
- **Microsoft Learn: DirectX**
  - https://learn.microsoft.com/en-us/windows/win32/directx
  - Official tutorials and samples

---

## Academic Papers (Advanced)

### Performance Optimization
1. **"Optimizing Parallel Reduction in CUDA"** (Mark Harris, 2007)
   - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   - Foundational paper on reduction optimization
   - **Direct influence on our `reduction.cu` implementation**

2. **"Roofline: An Insightful Visual Performance Model"** (Williams et al., 2009)
   - https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
   - Performance analysis framework
   - **Used to understand bottlenecks**

3. **"Matrix Multiplication on GPUs"** (Volkov & Demmel, 2008)
   - https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-111.pdf
   - Advanced matrix multiplication optimization
   - **Inspired our tiling strategy**

### GPU Architecture
4. **"NVIDIA GPU Architecture Whitepapers"**
   - Ampere: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
   - Turing: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
   - **Essential for understanding modern GPU capabilities**

---

## Tools and Utilities

### Profilers and Debuggers

#### NVIDIA Nsight (CUDA)
- **Nsight Compute**
  - https://developer.nvidia.com/nsight-compute
  - Kernel profiler (instruction-level analysis)
  - **Usage:** Profile our CUDA kernels to find bottlenecks

- **Nsight Systems**
  - https://developer.nvidia.com/nsight-systems
  - System-wide profiler (CPU+GPU timeline)
  - **Usage:** Understand host-device interactions

- **CUDA-GDB**
  - https://docs.nvidia.com/cuda/cuda-gdb/
  - GPU debugger
  - **Usage:** Debug kernel crashes

#### CodeXL (OpenCL/AMD)
- **AMD CodeXL**
  - https://github.com/GPUOpen-Archive/CodeXL
  - OpenCL profiler and debugger
  - **Usage:** Profile OpenCL kernels on AMD GPUs

#### PIX (DirectX)
- **PIX for Windows**
  - https://devblogs.microsoft.com/pix/download/
  - DirectX profiler and debugger
  - **Usage:** Profile DirectCompute shaders

### Performance Analysis
- **GPU-Z**
  - https://www.techpowerup.com/gpuz/
  - GPU monitoring tool (clocks, temperatures, utilization)

- **HWiNFO**
  - https://www.hwinfo.com/
  - Comprehensive system monitoring

---

## Community Resources

### Forums and Q&A
- **NVIDIA Developer Forums**
  - https://forums.developer.nvidia.com/c/gpu-graphics-and-game-dev/cuda/206
  - Ask CUDA questions

- **Khronos OpenCL Forums**
  - https://community.khronos.org/c/opencl/13
  - OpenCL discussions

- **Stack Overflow**
  - Tags: `[cuda]`, `[opencl]`, `[directcompute]`, `[hlsl]`
  - https://stackoverflow.com/questions/tagged/cuda

### GitHub Repositories
- **CUDA Samples**
  - https://github.com/NVIDIA/cuda-samples
  - Official NVIDIA examples
  
- **OpenCL Samples**
  - https://github.com/KhronosGroup/OpenCL-Guide
  - Official Khronos guide

- **DirectX Samples**
  - https://github.com/microsoft/DirectX-Graphics-Samples
  - Official Microsoft samples

---

## Blogs and Articles

### NVIDIA Blogs
- **NVIDIA Developer Blog**
  - https://developer.nvidia.com/blog/
  - Latest CUDA features and best practices

- **Parallel Forall** (Archive)
  - https://developer.nvidia.com/blog/category/parallel-forall/
  - Classic GPU programming articles

### Performance Optimization
- **Colfax Research: CUDA Optimization**
  - https://colfaxresearch.com/blog/
  - Deep-dive optimization guides

### Industry Blogs
- **Real-Time Rendering Blog**
  - http://www.realtimerendering.com/blog/
  - Graphics and compute shader insights

---

## Standards and Specifications

### OpenCL
- **OpenCL Registry**
  - https://www.khronos.org/registry/OpenCL/
  - All OpenCL specifications and extensions

### HLSL
- **HLSL Specifications**
  - https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl
  - Language specifications and shader models

### C++ Standards
- **C++17 Standard**
  - https://isocpp.org/std/the-standard
  - Language standard we use

---

## Video Resources

### YouTube Channels
- **NVIDIA Developer**
  - https://www.youtube.com/c/NVIDIADeveloper
  - GTC talks, tutorials

- **CppCon Talks**
  - https://www.youtube.com/user/CppCon
  - C++ best practices

### Recommended Talks
1. **"Intro to CUDA"** - NVIDIA
   - Basic CUDA programming concepts

2. **"Optimizing Parallel Reduction in CUDA"** - Mark Harris
   - Our reduction kernel is based on this

3. **"GPU Performance Analysis and Optimization"** - NVIDIA GTC
   - Profiling and optimization techniques

---

## How We Used These Resources in This Project

### During Initial Development
1. **CUDA C++ Programming Guide** → Core backend architecture
2. **"CUDA by Example"** → Vector addition implementation
3. **"Programming Massively Parallel Processors"** → Matrix multiplication tiling

### For Optimization
1. **Mark Harris's Reduction Paper** → Reduction kernel optimization
2. **NVIDIA Best Practices Guide** → Memory coalescing patterns
3. **Roofline Model Paper** → Performance analysis framework

### For Architecture Design
1. **"Design Patterns" (GoF)** → Strategy and Factory patterns
2. **"Effective Modern C++"** → RAII and smart pointers
3. **OpenCL Spec** → Cross-platform API design

### For Documentation
1. **Professional CUDA C Programming** → Technical explanations
2. **NVIDIA Documentation Style** → Code comments format
3. **GitHub Best Practices** → README structure

---

## Recommended Learning Path

### Beginner (0-3 months)
1. Read "CUDA by Example"
2. Complete Udacity CS344 course
3. Study our `vector_add.cu` kernel
4. Modify and experiment

### Intermediate (3-6 months)
1. Read "Programming Massively Parallel Processors"
2. Study our `matrix_mul.cu` optimizations
3. Profile with Nsight Compute
4. Implement your own benchmark

### Advanced (6-12 months)
1. Read "Professional CUDA C Programming"
2. Study advanced papers (Reduction, Roofline)
3. Optimize for specific GPU architectures
4. Contribute to this project!

---

## Contributing Your Knowledge

Found a great resource? Add it here!

1. Fork the repository
2. Edit this file
3. Submit a pull request
4. Help others learn!

---

**This is your roadmap to GPU programming mastery!** 🎓🚀

**Next:** Apply this knowledge by reading our [source code](../src/) and [documentation](../docs/)!

---

**Curated by:** Soham Dave  
**Date:** January 2026  
**For:** GPU Benchmark Suite v1.0  
**Purpose:** Comprehensive learning resource collection
