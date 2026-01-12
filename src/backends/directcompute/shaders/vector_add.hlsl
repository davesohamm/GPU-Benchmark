/********************************************************************************
 * @file    vector_add.hlsl
 * @brief   DirectCompute Vector Addition Compute Shader
 * 
 * @details Simple element-wise addition of two vectors using HLSL Compute Shader.
 *          Demonstrates basic DirectCompute shader structure.
 * 
 * @performance
 *          Memory-bound operation - bandwidth limited
 *          Expected: 90-95% of CUDA performance
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

// Input/output buffers
RWStructuredBuffer<float> bufferA : register(u0);  // Input A (read)
RWStructuredBuffer<float> bufferB : register(u1);  // Input B (read)
RWStructuredBuffer<float> bufferC : register(u2);  // Output C (write)

// Constants
cbuffer Constants : register(b0)
{
    uint numElements;      // Total number of elements
    uint3 padding;         // Padding to 16-byte boundary
};

/**
 * @brief Vector addition compute shader
 * 
 * @details Each thread computes one element: C[i] = A[i] + B[i]
 *          
 *          Thread organization:
 *          - [numthreads(256, 1, 1)] - 256 threads per thread group
 *          - SV_DispatchThreadID - unique global thread ID
 *          - Bounds checking prevents out-of-bounds access
 * 
 * @performance
 *          - Memory bandwidth bound (~90-95% of peak)
 *          - Coalesced memory access pattern
 *          - Minimal computation (1 add per element)
 */
[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Get global thread ID
    uint idx = dispatchThreadID.x;
    
    // Bounds check
    if (idx < numElements)
    {
        // Perform vector addition
        bufferC[idx] = bufferA[idx] + bufferB[idx];
    }
}

/********************************************************************************
 * HLSL vs CUDA vs OPENCL COMPARISON:
 * 
 * Thread Indexing:
 * - CUDA:   threadIdx.x + blockIdx.x * blockDim.x
 * - OpenCL: get_global_id(0)
 * - HLSL:   SV_DispatchThreadID.x
 * 
 * Memory Spaces:
 * - CUDA:   __global__ float*
 * - OpenCL: __global float*
 * - HLSL:   RWStructuredBuffer<float>
 * 
 * Thread Group Declaration:
 * - CUDA:   <<<gridDim, blockDim>>> (at launch time)
 * - OpenCL: globalWorkSize, localWorkSize (at enqueue time)
 * - HLSL:   [numthreads(X, Y, Z)] (in shader)
 * 
 * Dispatch:
 * - CUDA:   myKernel<<<blocks, threads>>>(args);
 * - OpenCL: clEnqueueNDRangeKernel(...)
 * - HLSL:   context->Dispatch(threadGroups, 1, 1);
 * 
 * Performance: 90-95% of CUDA for this memory-bound workload
 ********************************************************************************/
