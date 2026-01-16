/*******************************************************************************
 * FILE: test_reduction.cu
 * 
 * PURPOSE:
 *   Test program for parallel reduction CUDA kernels.
 *   Tests all 5 reduction algorithms and compares performance.
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * 
 ******************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>

// Declare kernel launch functions
extern "C" {
    void launchReductionNaive(const float* d_input, float* d_output, int n,
                               cudaStream_t stream);
    void launchReductionSequential(const float* d_input, float* d_output, int n,
                                    cudaStream_t stream);
    void launchReductionBankConflictFree(const float* d_input, float* d_output, int n,
                                          cudaStream_t stream);
    void launchReductionWarpShuffle(const float* d_input, float* d_output, int n,
                                     cudaStream_t stream);
    void launchReductionMultiBlockAtomic(const float* d_input, float* d_output, int n,
                                          cudaStream_t stream);
}

/*******************************************************************************
 * CPU REFERENCE IMPLEMENTATION
 ******************************************************************************/
float reductionCPU(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

/*******************************************************************************
 * MULTI-PASS REDUCTION (for non-atomic kernels)
 ******************************************************************************/
float multiPassReduction(void (*launchFunc)(const float*, float*, int, cudaStream_t),
                          const float* d_input, int n, 
                          float* d_temp1, float* d_temp2) {
    int elementsRemaining = n;
    const float* currentInput = d_input;
    float* currentOutput = d_temp1;
    
    while (elementsRemaining > 1) {
        launchFunc(currentInput, currentOutput, elementsRemaining, 0);
        cudaDeviceSynchronize();
        
        // Calculate how many blocks were launched
        int threadsPerBlock = (elementsRemaining < 512) ? elementsRemaining : 512;
        int blocksLaunched = (elementsRemaining + threadsPerBlock - 1) / threadsPerBlock;
        
        // Swap buffers
        currentInput = currentOutput;
        currentOutput = (currentOutput == d_temp1) ? d_temp2 : d_temp1;
        elementsRemaining = blocksLaunched;
    }
    
    // Copy final result to host
    float result;
    cudaMemcpy(&result, currentInput, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

/*******************************************************************************
 * TEST FUNCTION
 ******************************************************************************/
void testReduction(int n, const std::string& variant, 
                    const std::vector<float>& h_input, float expected) {
    std::cout << "\n=== Testing " << variant << " ===" << std::endl;
    std::cout << "Array size: " << n << " elements" << std::endl;
    
    size_t bytes = n * sizeof(float);
    
    // Allocate device memory
    float *d_input, *d_output, *d_temp1 = nullptr, *d_temp2 = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));
    
    // For multi-pass reductions
    cudaMalloc(&d_temp1, bytes);
    cudaMalloc(&d_temp2, bytes);
    
    // Copy to device
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup + compute result
    float result = 0.0f;
    
    if (variant == "Naive") {
        result = multiPassReduction(launchReductionNaive, d_input, n, d_temp1, d_temp2);
    } else if (variant == "Sequential") {
        result = multiPassReduction(launchReductionSequential, d_input, n, d_temp1, d_temp2);
    } else if (variant == "BankConflictFree") {
        result = multiPassReduction(launchReductionBankConflictFree, d_input, n, d_temp1, d_temp2);
    } else if (variant == "WarpShuffle") {
        result = multiPassReduction(launchReductionWarpShuffle, d_input, n, d_temp1, d_temp2);
    } else { // MultiBlockAtomic
        // Zero out output first
        cudaMemset(d_output, 0, sizeof(float));
        launchReductionMultiBlockAtomic(d_input, d_output, n, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Benchmark
    const int iterations = 20;
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < iterations; i++) {
        if (variant == "MultiBlockAtomic") {
            cudaMemset(d_output, 0, sizeof(float));
            launchReductionMultiBlockAtomic(d_input, d_output, n, 0);
        } else if (variant == "Naive") {
            launchReductionNaive(d_input, d_temp1, n, 0);
        } else if (variant == "Sequential") {
            launchReductionSequential(d_input, d_temp1, n, 0);
        } else if (variant == "BankConflictFree") {
            launchReductionBankConflictFree(d_input, d_temp1, n, 0);
        } else {
            launchReductionWarpShuffle(d_input, d_temp1, n, 0);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avgTime = milliseconds / iterations;
    
    // Calculate bandwidth
    double bandwidth = (bytes / (avgTime / 1000.0)) / 1e9;
    
    // Verify result
    float error = std::abs(result - expected);
    float relError = error / std::abs(expected);
    bool correct = (relError < 1e-4f);
    
    std::cout << "  Result: " << std::fixed << std::setprecision(2) << result << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Relative error: " << std::scientific << std::setprecision(2) 
              << relError << std::endl;
    std::cout << "  Execution time: " << std::fixed << std::setprecision(3) 
              << avgTime << " ms" << std::endl;
    std::cout << "  Bandwidth: " << std::setprecision(1) << bandwidth << " GB/s" << std::endl;
    
    if (correct) {
        std::cout << "  ✓ CORRECT!" << std::endl;
    } else {
        std::cout << "  ✗ INCORRECT!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*******************************************************************************
 * MAIN
 ******************************************************************************/
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Parallel Reduction Kernel Test" << std::endl;
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
    std::cout << "Memory: " << (props.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    
    // Test different array sizes
    std::vector<int> sizes = {1000000, 10000000, 50000000};
    
    for (int n : sizes) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing with " << (n / 1000000) << "M elements" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Create test data
        std::vector<float> h_input(n);
        for (int i = 0; i < n; i++) {
            h_input[i] = 1.0f;  // Simple test: sum of all 1's = n
        }
        
        float expected = static_cast<float>(n);
        
        // Test all variants
        testReduction(n, "Naive", h_input, expected);
        testReduction(n, "Sequential", h_input, expected);
        testReduction(n, "BankConflictFree", h_input, expected);
        testReduction(n, "WarpShuffle", h_input, expected);
        testReduction(n, "MultiBlockAtomic", h_input, expected);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Comparison Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nExpected speedups (vs Naive):" << std::endl;
    std::cout << "  Sequential:        ~3x faster" << std::endl;
    std::cout << "  BankConflictFree:  ~5x faster" << std::endl;
    std::cout << "  WarpShuffle:       ~7-8x faster (FASTEST!)" << std::endl;
    std::cout << "  MultiBlockAtomic:  ~5-6x faster" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
