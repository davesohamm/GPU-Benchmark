# Tests Directory

This folder contains all testing-related files.

## Subfolders

### 🧪 **unit-tests/** - Unit Test Files
Individual component test files (.cu, .cpp) for testing specific functionality.

**Contents:**
- `test_cuda_backend.cu` - CUDA backend test
- `test_opencl_backend.cpp` - OpenCL backend test
- `test_directcompute_backend.cpp` - DirectCompute test
- `test_matmul.cu` - Matrix multiplication test
- `test_convolution.cu` - Convolution test
- `test_reduction.cu` - Reduction test
- `test_logger.cpp` - Logger test
- And more...

**Compile these with:** CMake build system (see CMakeLists.txt)

### 📜 **test-scripts/** - Test Automation Scripts
Batch scripts (.cmd) for running automated tests.

**Contents:**
- Full suite tests
- Backend-specific tests
- GUI tests
- Fix verification scripts

**Run these from:** Windows command prompt or PowerShell

---

**Purpose:** Validate functionality, verify fixes, ensure quality
