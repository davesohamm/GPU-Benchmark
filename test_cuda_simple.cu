// Minimal CUDA test - vector addition
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("=== Simple CUDA Test ===\n\n");
    
    // Check CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    // Get device info
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Memory: %.2f GB\n\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Vector addition test
    const int N = 1000000;
    const size_t bytes = N * sizeof(float);
    
    printf("Testing vector addition (%d elements)...\n", N);
    
    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel with timing
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.3f ms\n", ms);
    
    // Copy back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            errors++;
            if (errors < 5) {
                printf("Error at %d: %f != %f\n", i, h_c[i], expected);
            }
        }
    }
    
    if (errors == 0) {
        printf("\n✓ SUCCESS! All %d elements correct!\n", N);
        
        // Calculate bandwidth
        double totalBytes = 3.0 * bytes;
        double bandwidth = (totalBytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        printf("Bandwidth: %.1f GB/s\n", bandwidth);
    } else {
        printf("\n✗ FAILED! %d errors\n", errors);
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (errors == 0) ? 0 : 1;
}
