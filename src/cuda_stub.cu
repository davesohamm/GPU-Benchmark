/*******************************************************************************
 * FILE: cuda_stub.cu
 * 
 * PURPOSE:
 *   Dummy CUDA file to force CMake to treat GPU-Benchmark as a CUDA target
 *   and link CUDA runtime libraries properly.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

// This file exists solely to ensure GPU-Benchmark.exe links with CUDA runtime
// CMake only automatically links CUDA libraries when a target includes .cu files

// Empty CUDA function - never called, just forces CUDA linking
__global__ void cudaStubKernel() {
    // This kernel is never executed
    // It exists only to make CMake link CUDA libraries
}

// Empty host function to avoid warnings about unused kernel
void cudaStubFunction() {
    // Never called - exists only for compilation
}
