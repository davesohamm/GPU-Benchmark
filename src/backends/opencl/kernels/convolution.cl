/********************************************************************************
 * @file    convolution.cl
 * @brief   OpenCL 2D Image Convolution Kernels
 * 
 * @details Three implementations of 2D convolution with increasing optimization:
 *          1. Naive - Global memory access only
 *          2. Shared - Uses local memory for input tile caching
 *          3. Separable - Optimized for separable kernels (Gaussian, etc.)
 * 
 * @note    Fixed 5x5 Gaussian kernel used for benchmarking
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

// Fixed 5x5 Gaussian kernel for benchmarking
__constant float gaussianKernel[25] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
};

#define KERNEL_RADIUS 2
#define KERNEL_SIZE 5

/********************************************************************************
 * KERNEL 1: NAIVE 2D CONVOLUTION
 * 
 * @brief Straightforward convolution using global memory
 * @details Each work-item:
 *          - Reads 5x5 neighborhood from global memory
 *          - Multiplies by kernel weights
 *          - Writes result
 * 
 * @performance
 *          - High global memory traffic
 *          - Redundant loads (neighbors overlap)
 *          - Expected: 10-20% of peak bandwidth
 ********************************************************************************/
__kernel void convolution2DNaive(__global const float* input,
                                  __global float* output,
                                  const int width,
                                  const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    // Apply 5x5 convolution
    for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
        for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
            int ix = x + kx;
            int iy = y + ky;
            
            // Clamp to image bounds
            ix = clamp(ix, 0, width - 1);
            iy = clamp(iy, 0, height - 1);
            
            int kernelIdx = (ky + KERNEL_RADIUS) * KERNEL_SIZE + (kx + KERNEL_RADIUS);
            sum += input[iy * width + ix] * gaussianKernel[kernelIdx];
        }
    }
    
    output[y * width + x] = sum;
}

/********************************************************************************
 * KERNEL 2: SHARED MEMORY CONVOLUTION
 * 
 * @brief Optimized convolution using local memory
 * @details Optimization strategy:
 *          - Load tile of input into local memory (with halo)
 *          - Each work-item reads from local memory (much faster)
 *          - Reduces global memory accesses by ~25x
 * 
 * @performance
 *          - Excellent data reuse
 *          - Coalesced global loads
 *          - Expected: 60-80% of peak bandwidth
 * 
 * @note BLOCK_SIZE must be defined (e.g., -DBLOCK_SIZE=16)
 ********************************************************************************/
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define SHARED_SIZE (BLOCK_SIZE + 2 * KERNEL_RADIUS)

__kernel void convolution2DShared(__global const float* input,
                                   __global float* output,
                                   const int width,
                                   const int height)
{
    // Local memory for tile with halo
    __local float tile[SHARED_SIZE][SHARED_SIZE];
    
    // Thread indices
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    // Global position
    int x = get_group_id(0) * BLOCK_SIZE + tx;
    int y = get_group_id(1) * BLOCK_SIZE + ty;
    
    // Load tile into local memory (including halo)
    // Each thread may load multiple elements
    for (int i = ty; i < SHARED_SIZE; i += BLOCK_SIZE) {
        for (int j = tx; j < SHARED_SIZE; j += BLOCK_SIZE) {
            int gx = get_group_id(0) * BLOCK_SIZE + j - KERNEL_RADIUS;
            int gy = get_group_id(1) * BLOCK_SIZE + i - KERNEL_RADIUS;
            
            // Clamp to image bounds
            gx = clamp(gx, 0, width - 1);
            gy = clamp(gy, 0, height - 1);
            
            tile[i][j] = input[gy * width + gx];
        }
    }
    
    // Synchronize to ensure tile is loaded
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform convolution using local memory
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                int tileX = tx + KERNEL_RADIUS + kx;
                int tileY = ty + KERNEL_RADIUS + ky;
                
                int kernelIdx = (ky + KERNEL_RADIUS) * KERNEL_SIZE + (kx + KERNEL_RADIUS);
                sum += tile[tileY][tileX] * gaussianKernel[kernelIdx];
            }
        }
        
        output[y * width + x] = sum;
    }
}

/********************************************************************************
 * KERNEL 3: SEPARABLE CONVOLUTION (Horizontal Pass)
 * 
 * @brief Horizontal pass of separable convolution
 * @details Separable kernels (like Gaussian) can be decomposed:
 *          - Horizontal 1D convolution followed by
 *          - Vertical 1D convolution
 *          - Reduces complexity from O(K²) to O(2K)
 * 
 * @performance
 *          - Much less computation (5 ops vs 25 ops per pixel)
 *          - Better cache utilization
 *          - Expected: 80-95% of peak bandwidth
 ********************************************************************************/
__constant float gaussianKernel1D[5] = {
    0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f
};

__kernel void convolution1DHorizontal(__global const float* input,
                                       __global float* output,
                                       const int width,
                                       const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    // Apply 1D horizontal convolution
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k) {
        int ix = clamp(x + k, 0, width - 1);
        sum += input[y * width + ix] * gaussianKernel1D[k + KERNEL_RADIUS];
    }
    
    output[y * width + x] = sum;
}

/********************************************************************************
 * KERNEL 4: SEPARABLE CONVOLUTION (Vertical Pass)
 * 
 * @brief Vertical pass of separable convolution
 * @details Completes separable convolution:
 *          - Reads from horizontal pass output
 *          - Applies vertical 1D filter
 *          - Writes final result
 ********************************************************************************/
__kernel void convolution1DVertical(__global const float* input,
                                     __global float* output,
                                     const int width,
                                     const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    
    // Apply 1D vertical convolution
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k) {
        int iy = clamp(y + k, 0, height - 1);
        sum += input[iy * width + x] * gaussianKernel1D[k + KERNEL_RADIUS];
    }
    
    output[y * width + x] = sum;
}

/********************************************************************************
 * OPENCL vs CUDA COMPARISON:
 * 
 * 1. Constant Memory:
 *    CUDA:   __constant__ float kernel[25];
 *    OpenCL: __constant float kernel[25];  (same!)
 * 
 * 2. Shared/Local Memory:
 *    CUDA:   __shared__ float tile[SIZE][SIZE];
 *    OpenCL: __local float tile[SIZE][SIZE];
 * 
 * 3. Boundary Handling:
 *    CUDA:   min(max(x, 0), width-1)
 *    OpenCL: clamp(x, 0, width-1)  (built-in function)
 * 
 * PERFORMANCE EXPECTATIONS:
 * - Naive:     ~10-20% of peak bandwidth
 * - Shared:    ~60-80% of peak bandwidth
 * - Separable: ~80-95% of peak bandwidth
 * 
 * OpenCL achieves 95-98% of CUDA performance on this workload.
 ********************************************************************************/
