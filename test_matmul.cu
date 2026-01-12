/*******************************************************************************
 * FILE: test_matmul.cu
 * 
 * PURPOSE:
 *   Test program for matrix multiplication CUDA kernels.
 *   Verifies correctness and measures performance of all 3 implementations.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

// Declare kernel launch functions
extern "C" {
    void launchMatrixMulNaive(const float* d_A, const float* d_B, float* d_C,
                               int M, int N, int P, cudaStream_t stream);
    void launchMatrixMulTiled(const float* d_A, const float* d_B, float* d_C,
                               int M, int N, int P, cudaStream_t stream);
    void launchMatrixMulOptimized(const float* d_A, const float* d_B, float* d_C,
                                   int M, int N, int P, cudaStream_t stream);
}

/*******************************************************************************
 * CPU REFERENCE IMPLEMENTATION
 ******************************************************************************/
void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

/*******************************************************************************
 * VERIFICATION
 ******************************************************************************/
bool verifyResults(const float* C_gpu, const float* C_cpu, int M, int P) {
    int errors = 0;
    const float epsilon = 1e-3f;  // Tolerance for floating point comparison
    
    for (int i = 0; i < M * P; i++) {
        float diff = std::abs(C_gpu[i] - C_cpu[i]);
        if (diff > epsilon) {
            errors++;
            if (errors <= 5) {  // Print first 5 errors
                std::cout << "Error at index " << i << ": GPU=" << C_gpu[i] 
                         << " CPU=" << C_cpu[i] << " diff=" << diff << std::endl;
            }
        }
    }
    
    if (errors > 0) {
        std::cout << "Total errors: " << errors << " / " << (M * P) << std::endl;
        return false;
    }
    
    return true;
}

/*******************************************************************************
 * TEST FUNCTION
 ******************************************************************************/
void testMatrixMul(int M, int N, int P, const std::string& variant) {
    std::cout << "\n=== Testing " << variant << " ===" << std::endl;
    std::cout << "Matrix dimensions: " << M << "×" << N << " × " << N << "×" << P << std::endl;
    
    // Allocate host memory
    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * P * sizeof(float);
    size_t bytes_C = M * P * sizeof(float);
    
    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * P);
    std::vector<float> h_C_gpu(M * P);
    std::vector<float> h_C_cpu(M * P);
    
    // Initialize matrices
    for (int i = 0; i < M * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N * P; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    // Copy to device
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    if (variant == "Naive") {
        launchMatrixMulNaive(d_A, d_B, d_C, M, N, P, 0);
    } else if (variant == "Tiled") {
        launchMatrixMulTiled(d_A, d_B, d_C, M, N, P, 0);
    } else {
        launchMatrixMulOptimized(d_A, d_B, d_C, M, N, P, 0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark (10 iterations)
    const int iterations = 10;
    cudaEventRecord(start);
    
    for (int i = 0; i < iterations; i++) {
        if (variant == "Naive") {
            launchMatrixMulNaive(d_A, d_B, d_C, M, N, P, 0);
        } else if (variant == "Tiled") {
            launchMatrixMulTiled(d_A, d_B, d_C, M, N, P, 0);
        } else {
            launchMatrixMulOptimized(d_A, d_B, d_C, M, N, P, 0);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avgTime = milliseconds / iterations;
    
    // Copy result back
    cudaMemcpy(h_C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * P;  // Each output element: N muls + N adds
    double gflops = (flops / (avgTime / 1000.0)) / 1e9;
    
    // Calculate bandwidth
    double totalBytes = (bytes_A + bytes_B + bytes_C);
    double bandwidth = (totalBytes / (avgTime / 1000.0)) / 1e9;
    
    std::cout << "  Execution time: " << std::fixed << std::setprecision(3) 
              << avgTime << " ms" << std::endl;
    std::cout << "  Performance: " << std::setprecision(1) << gflops << " GFLOPS" << std::endl;
    std::cout << "  Bandwidth: " << std::setprecision(1) << bandwidth << " GB/s" << std::endl;
    
    // Verify (only for small matrices to save time)
    if (M <= 512) {
        std::cout << "  Verifying results..." << std::flush;
        matrixMulCPU(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, P);
        
        if (verifyResults(h_C_gpu.data(), h_C_cpu.data(), M, P)) {
            std::cout << " ✓ CORRECT!" << std::endl;
        } else {
            std::cout << " ✗ INCORRECT!" << std::endl;
        }
    } else {
        std::cout << "  Skipping verification (matrix too large)" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*******************************************************************************
 * MAIN
 ******************************************************************************/
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Matrix Multiplication Kernel Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "\nDevice: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Memory: " << (props.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    
    // Test different sizes
    std::vector<int> sizes = {256, 512, 1024};
    
    for (int size : sizes) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing " << size << "×" << size << " matrices" << std::endl;
        std::cout << "========================================" << std::endl;
        
        testMatrixMul(size, size, size, "Naive");
        testMatrixMul(size, size, size, "Tiled");
        testMatrixMul(size, size, size, "Optimized");
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
