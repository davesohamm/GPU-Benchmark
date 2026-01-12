/********************************************************************************
 * @file    vector_add.cl
 * @brief   OpenCL Vector Addition Kernel
 * 
 * @details Simple element-wise addition of two vectors.
 *          Demonstrates basic OpenCL kernel structure and global memory access.
 * 
 * @performance
 *          Memory-bound operation - bandwidth limited
 *          Expected: 95-100% of CUDA performance
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

/**
 * @brief Vector addition kernel
 * @param a Input vector A (read-only)
 * @param b Input vector B (read-only)
 * @param c Output vector C = A + B (write-only)
 * @param n Number of elements
 * 
 * @details Each work-item computes one element: c[i] = a[i] + b[i]
 *          
 *          Work-item organization:
 *          - Global ID: get_global_id(0) returns unique thread ID
 *          - Each thread processes one element
 *          - Bounds checking prevents out-of-bounds access
 * 
 * @performance
 *          - Memory bandwidth bound (~95% of peak)
 *          - Coalesced memory access pattern
 *          - Minimal computation (1 add per element)
 */
__kernel void vectorAdd(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const int n)
{
    // Get global thread ID
    int idx = get_global_id(0);
    
    // Bounds check
    if (idx < n) {
        // Perform vector addition
        c[idx] = a[idx] + b[idx];
    }
}

/********************************************************************************
 * OPENCL vs CUDA COMPARISON:
 * 
 * CUDA:   __global__ void vectorAdd(const float* a, const float* b, float* c, int n)
 * OpenCL: __kernel void vectorAdd(__global const float* a, __global const float* b, __global float* c, const int n)
 * 
 * Key Differences:
 * 1. __global__ (CUDA) vs __kernel (OpenCL) - kernel declaration
 * 2. Implicit global memory (CUDA) vs __global (OpenCL) - memory space qualifier
 * 3. threadIdx.x + blockIdx.x * blockDim.x (CUDA) vs get_global_id(0) (OpenCL)
 * 
 * Performance: Nearly identical (difference < 1%)
 ********************************************************************************/
