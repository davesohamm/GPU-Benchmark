# 🎮 GPU Benchmark Suite v2.0

**Professional GPU Compute Benchmarking Tool**  
Multi-API Support: CUDA | OpenCL | DirectCompute

---

## 🚀 **Quick Start**

### **GUI Application (Recommended)**
```
Double-click: GPU-Benchmark-GUI.exe
```
- Select your GPU backend
- Click "Start Benchmark"
- View results and graphs!

### **CLI Application**
```
Run: GPU-Benchmark.exe
```
- Automatically tests all 3 backends
- Exports results to CSV

---

## 📊 **What It Does**

Benchmarks your GPU's compute performance using:
- **Vector Addition** - Memory bandwidth test
- Real-time performance visualization
- Accurate timing measurements
- Result verification

### **Supported Backends:**
- ✅ **CUDA** - NVIDIA GPUs (fastest)
- ✅ **OpenCL** - All vendors (cross-platform)
- ✅ **DirectCompute** - Windows native (excellent)

---

## 💻 **System Requirements**

- **OS:** Windows 10 or Windows 11
- **GPU:** Any modern GPU with drivers installed
  - NVIDIA (GTX 900+, RTX series)
  - AMD (RX 400+, Radeon VII+)
  - Intel (HD 5000+, Arc)
- **RAM:** 2 GB minimum
- **Storage:** 10 MB

**No CUDA Toolkit or SDK installation required!**

---

## 🎯 **How to Use**

### **GUI Application:**

1. **Launch** - Double-click `GPU-Benchmark-GUI.exe`
2. **Check System Info** - Verify your GPU is detected
3. **Select Backend:**
   - **CUDA** - Best for NVIDIA GPUs
   - **OpenCL** - Works on all GPUs
   - **DirectCompute** - Windows optimized
4. **Select Suite:**
   - **Quick** - 10 seconds, fast test
   - **Standard** - 30 seconds, balanced
   - **Full** - 60 seconds, comprehensive
5. **Click "Start Benchmark"**
6. **View Results** - Performance metrics and graphs
7. **Export** - Save results to CSV

### **CLI Application:**

```cmd
GPU-Benchmark.exe
```

Automatically runs all 3 backends and exports results.

---

## 📈 **Understanding Results**

### **Bandwidth (GB/s):**
- Measures memory transfer speed
- Higher is better
- Typical values:
  - **Excellent:** 150-200 GB/s (High-end GPUs)
  - **Good:** 100-150 GB/s (Mid-range GPUs)
  - **Acceptable:** 50-100 GB/s (Entry-level GPUs)

### **Status:**
- **PASS** (Green) - Result verified correct
- **FAIL** (Red) - Computation error detected

### **Backend Comparison:**
- **CUDA** - Usually best on NVIDIA GPUs
- **DirectCompute** - Often fastest on Windows
- **OpenCL** - 85-95% of CUDA performance

---

## 🔧 **Troubleshooting**

### **GUI doesn't open:**
- Install Visual C++ Redistributable 2022
- Update GPU drivers
- Check if DirectX 11 is installed

### **Backend not available:**
- **CUDA** - Requires NVIDIA GPU with updated drivers
- **OpenCL** - Check GPU vendor's OpenCL support
- **DirectCompute** - Requires Windows 10+ and DirectX 11

### **Low performance:**
- Close other GPU-intensive applications
- Check GPU temperature (thermal throttling)
- Update GPU drivers
- Try different backend

---

## 📁 **Output Files**

### **GUI:**
- `benchmark_results_gui.csv` - Results from GUI runs

### **CLI:**
- `benchmark_results_working.csv` - Results from CLI runs

### **Format:**
```csv
Backend,Benchmark,Time_ms,Bandwidth_GBs,Status
CUDA,VectorAdd,0.069,174.7,PASS
OpenCL,VectorAdd,0.077,155.5,PASS
DirectCompute,VectorAdd,0.068,177.1,PASS
```

---

## 🎓 **About This Project**

### **Technology Stack:**
- **C++17** - Modern C++ with RAII and smart pointers
- **CUDA 13.1** - NVIDIA GPU compute
- **OpenCL 3.0** - Cross-vendor GPU compute
- **DirectCompute (DirectX 11)** - Windows GPU compute
- **ImGui** - Immediate mode GUI framework
- **CMake** - Cross-platform build system

### **Architecture:**
- **Backend Abstraction** - Unified interface for all APIs
- **Strategy Pattern** - Swappable backend implementations
- **Facade Pattern** - Simplified benchmark execution
- **RAII** - Automatic resource management
- **Multi-threading** - Non-blocking UI during benchmarks

### **Features:**
- Multi-API GPU benchmarking
- Real-time visualization
- Performance graphing
- CSV export
- Hardware-agnostic design
- Error handling and validation
- Comprehensive logging

---

## 👨‍💻 **Author**

**Soham Dave**  
GitHub: [github.com/davesohamm](https://github.com/davesohamm)

---

## 📄 **License**

This software is provided as-is for educational and benchmarking purposes.

---

## 🙏 **Credits**

- **ImGui** - Dear ImGui by Omar Cornut
- **CUDA** - NVIDIA CUDA Toolkit
- **OpenCL** - Khronos Group
- **DirectX** - Microsoft DirectX SDK

---

## 📞 **Support**

For issues or questions:
- Check `GUI_V2_COMPLETE.md` for detailed documentation
- Check `FINAL_COMPLETE_STATUS.md` for troubleshooting
- Visit: https://github.com/davesohamm

---

**Enjoy benchmarking your GPU!** 🚀
