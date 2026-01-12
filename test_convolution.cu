/*******************************************************************************
 * FILE: test_convolution.cu
 * 
 * PURPOSE:
 *   Test program for 2D convolution CUDA kernels.
 *   Tests naive, shared memory, and separable implementations.
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

// Declare kernel launch functions
extern "C" {
    void setConvolutionKernel(const float* h_kernel, int kernelSize);
    void launchConvolution2DNaive(const float* d_input, float* d_output,
                                   int width, int height, int kernelRadius,
                                   cudaStream_t stream);
    void launchConvolution2DShared(const float* d_input, float* d_output,
                                    int width, int height, int kernelRadius,
                                    cudaStream_t stream);
    void launchConvolutionSeparable(const float* d_input, float* d_temp, float* d_output,
                                     int width, int height, int kernelRadius,
                                     cudaStream_t stream);
}

/*******************************************************************************
 * FILTER KERNELS
 ******************************************************************************/

// 3x3 Gaussian blur
float gaussian3x3[9] = {
    1/16.0f, 2/16.0f, 1/16.0f,
    2/16.0f, 4/16.0f, 2/16.0f,
    1/16.0f, 2/16.0f, 1/16.0f
};

// 5x5 Gaussian blur
float gaussian5x5[25] = {
    1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
    4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
    6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
    4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
    1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f
};

// 1D Gaussian for separable convolution (5-tap)
float gaussian1D[5] = {
    1/16.0f, 4/16.0f, 6/16.0f, 4/16.0f, 1/16.0f
};

/*******************************************************************************
 * CPU REFERENCE IMPLEMENTATION
 ******************************************************************************/
void convolution2DCPU(const float* input, float* output, 
                       const float* kernel, int width, int height, int radius) {
    int kernelSize = 2 * radius + 1;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int imageY = std::min(std::max(y + ky, 0), height - 1);
                    int imageX = std::min(std::max(x + kx, 0), width - 1);
                    
                    int kernelIdx = (ky + radius) * kernelSize + (kx + radius);
                    sum += input[imageY * width + imageX] * kernel[kernelIdx];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

/*******************************************************************************
 * VERIFICATION
 ******************************************************************************/
bool verifyResults(const float* gpu, const float* cpu, int width, int height) {
    int errors = 0;
    const float epsilon = 1e-3f;
    
    for (int i = 0; i < width * height; i++) {
        float diff = std::abs(gpu[i] - cpu[i]);
        if (diff > epsilon) {
            errors++;
            if (errors <= 5) {
                int y = i / width;
                int x = i % width;
                std::cout << "Error at (" << x << "," << y << "): GPU=" << gpu[i] 
                         << " CPU=" << cpu[i] << " diff=" << diff << std::endl;
            }
        }
    }
    
    if (errors > 0) {
        std::cout << "Total errors: " << errors << " / " << (width * height) << std::endl;
        return false;
    }
    
    return true;
}

/*******************************************************************************
 * TEST FUNCTION
 ******************************************************************************/
void testConvolution(int width, int height, int radius, const std::string& variant) {
    std::cout << "\n=== Testing " << variant << " ===" << std::endl;
    std::cout << "Image size: " << width << "×" << height << std::endl;
    std::cout << "Kernel radius: " << radius << " (size: " << (2*radius+1) << "×" << (2*radius+1) << ")" << std::endl;
    
    size_t imageBytes = width * height * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_input(width * height);
    std::vector<float> h_output_gpu(width * height);
    std::vector<float> h_output_cpu(width * height);
    
    // Initialize input image (random values)
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Set kernel in constant memory
    if (radius == 1) {
        setConvolutionKernel(gaussian3x3, 3);
    } else {
        setConvolutionKernel(gaussian5x5, 5);
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_temp = nullptr;
    cudaMalloc(&d_input, imageBytes);
    cudaMalloc(&d_output, imageBytes);
    if (variant == "Separable") {
        cudaMalloc(&d_temp, imageBytes);
    }
    
    // Copy to device
    cudaMemcpy(d_input, h_input.data(), imageBytes, cudaMemcpyHostToDevice);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    if (variant == "Naive") {
        launchConvolution2DNaive(d_input, d_output, width, height, radius, 0);
    } else if (variant == "Shared") {
        launchConvolution2DShared(d_input, d_output, width, height, radius, 0);
    } else {
        launchConvolutionSeparable(d_input, d_temp, d_output, width, height, radius, 0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    const int iterations = 20;
    cudaEventRecord(start);
    
    for (int i = 0; i < iterations; i++) {
        if (variant == "Naive") {
            launchConvolution2DNaive(d_input, d_output, width, height, radius, 0);
        } else if (variant == "Shared") {
            launchConvolution2DShared(d_input, d_output, width, height, radius, 0);
        } else {
            launchConvolutionSeparable(d_input, d_temp, d_output, width, height, radius, 0);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avgTime = milliseconds / iterations;
    
    // Copy result back
    cudaMemcpy(h_output_gpu.data(), d_output, imageBytes, cudaMemcpyDeviceToHost);
    
    // Calculate bandwidth
    int kernelSize = 2 * radius + 1;
    double readsPerPixel = kernelSize * kernelSize;  // Naive/shared
    if (variant == "Separable") {
        readsPerPixel = 2 * kernelSize;  // Two 1D passes
    }
    double totalBytes = imageBytes * (readsPerPixel + 1);  // Reads + 1 write
    double bandwidth = (totalBytes / (avgTime / 1000.0)) / 1e9;
    
    std::cout << "  Execution time: " << std::fixed << std::setprecision(3) 
              << avgTime << " ms" << std::endl;
    std::cout << "  Bandwidth: " << std::setprecision(1) << bandwidth << " GB/s" << std::endl;
    
    // Verify (for smaller images)
    if (width <= 1024 && height <= 1024) {
        std::cout << "  Verifying results..." << std::flush;
        
        const float* kernel = (radius == 1) ? gaussian3x3 : gaussian5x5;
        convolution2DCPU(h_input.data(), h_output_cpu.data(), kernel, width, height, radius);
        
        if (verifyResults(h_output_gpu.data(), h_output_cpu.data(), width, height)) {
            std::cout << " ✓ CORRECT!" << std::endl;
        } else {
            std::cout << " ✗ INCORRECT!" << std::endl;
        }
    } else {
        std::cout << "  Skipping verification (image too large)" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    if (d_temp) cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/*******************************************************************************
 * MAIN
 ******************************************************************************/
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  2D Convolution Kernel Test" << std::endl;
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
    
    // Test different image sizes
    struct TestConfig {
        int width;
        int height;
        int radius;
        std::string name;
    };
    
    std::vector<TestConfig> configs = {
        {640, 480, 1, "VGA (640×480), 3×3 kernel"},
        {1920, 1080, 1, "Full HD (1920×1080), 3×3 kernel"},
        {1920, 1080, 2, "Full HD (1920×1080), 5×5 kernel"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing: " << config.name << std::endl;
        std::cout << "========================================" << std::endl;
        
        testConvolution(config.width, config.height, config.radius, "Naive");
        testConvolution(config.width, config.height, config.radius, "Shared");
        testConvolution(config.width, config.height, config.radius, "Separable");
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
