/********************************************************************************
 * @file    reduction.cl
 * @brief   OpenCL Parallel Reduction Kernels
 * 
 * @details Five implementations of parallel reduction (sum) with increasing optimization:
 *          1. Naive - Simple but has divergence and bank conflicts
 *          2. Sequential Addressing - Removes divergence
 *          3. Bank Conflict Free - Optimizes shared memory access
 *          4. Warp Shuffle - Uses sub-group operations (if available)
 *          5. Atomic - Simple atomic reduction
 * 
 * @note    This is a compute-bound workload showcasing synchronization patterns
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

/********************************************************************************
 * KERNEL 1: NAIVE REDUCTION
 * 
 * @brief Simple but inefficient reduction
 * @details Problems:
 *          - Thread divergence (half threads idle each iteration)
 *          - Bank conflicts in shared memory
 *          - Poor instruction-level parallelism
 * 
 * @performance
 *          - Expected: 10-20% of peak
 ********************************************************************************/
#ifndef BLOCK_SIZE_REDUCE
#define BLOCK_SIZE_REDUCE 256
#endif

__kernel void reductionNaive(__global const float* input,
                              __global float* output,
                              const int n,
                              __local float* sdata)
{
    int tid = get_local_id(0);
    int idx = get_global_id(0);
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in shared memory (naive approach)
    for (int s = 1; s < get_local_size(0); s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this block
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

/********************************************************************************
 * KERNEL 2: SEQUENTIAL ADDRESSING REDUCTION
 * 
 * @brief Improved reduction with better memory access pattern
 * @details Optimizations:
 *          - Sequential addressing removes divergence
 *          - Better warp/wavefront utilization
 *          - Still has bank conflicts
 * 
 * @performance
 *          - Expected: 30-40% of peak
 ********************************************************************************/
__kernel void reductionSequential(__global const float* input,
                                   __global float* output,
                                   const int n,
                                   __local float* sdata)
{
    int tid = get_local_id(0);
    int idx = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    
    // Load and perform first level of reduction during load
    sdata[tid] = ((idx < n) ? input[idx] : 0.0f) + 
                 ((idx + get_local_size(0) < n) ? input[idx + get_local_size(0)] : 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Sequential addressing reduction
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

/********************************************************************************
 * KERNEL 3: BANK CONFLICT FREE REDUCTION
 * 
 * @brief Optimized reduction avoiding bank conflicts
 * @details Optimizations:
 *          - Sequential addressing
 *          - Conflict-free shared memory access
 *          - Loop unrolling for last warp
 * 
 * @performance
 *          - Expected: 60-70% of peak
 ********************************************************************************/
__kernel void reductionBankConflictFree(__global const float* input,
                                         __global float* output,
                                         const int n,
                                         __local float* sdata)
{
    int tid = get_local_id(0);
    int idx = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    
    // Load and reduce during load
    sdata[tid] = ((idx < n) ? input[idx] : 0.0f) + 
                 ((idx + get_local_size(0) < n) ? input[idx + get_local_size(0)] : 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Sequential addressing with no bank conflicts
    for (int s = get_local_size(0) / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Unroll last warp (no synchronization needed)
    if (tid < 32) {
        volatile __local float* smem = sdata;
        if (get_local_size(0) >= 64) smem[tid] += smem[tid + 32];
        if (get_local_size(0) >= 32) smem[tid] += smem[tid + 16];
        if (get_local_size(0) >= 16) smem[tid] += smem[tid + 8];
        if (get_local_size(0) >= 8)  smem[tid] += smem[tid + 4];
        if (get_local_size(0) >= 4)  smem[tid] += smem[tid + 2];
        if (get_local_size(0) >= 2)  smem[tid] += smem[tid + 1];
    }
    
    // Write result
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}

/********************************************************************************
 * KERNEL 4: WARP SHUFFLE REDUCTION (using sub-groups if available)
 * 
 * @brief Advanced reduction using sub-group operations
 * @details Uses OpenCL 2.0+ sub-group operations (similar to CUDA warp shuffle):
 *          - sub_group_reduce_add() for warp-level reduction
 *          - No shared memory needed for final reduction
 *          - Maximum efficiency
 * 
 * @performance
 *          - Expected: 80-90% of peak
 * 
 * @note Requires OpenCL 2.0+ with sub-group support
 *       Falls back to bank-conflict-free version if not available
 ********************************************************************************/
#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void reductionWarpShuffle(__global const float* input,
                                    __global float* output,
                                    const int n,
                                    __local float* sdata)
{
    int tid = get_local_id(0);
    int idx = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    
    // Load and reduce during load
    float val = ((idx < n) ? input[idx] : 0.0f) + 
                ((idx + get_local_size(0) < n) ? input[idx + get_local_size(0)] : 0.0f);
    
    // Sub-group reduction (warp-level)
    val = sub_group_reduce_add(val);
    
    // Write sub-group results to shared memory
    int lane = get_sub_group_local_id();
    int wid = get_sub_group_id();
    
    if (lane == 0) {
        sdata[wid] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Final reduction in first warp
    if (wid == 0) {
        val = (tid < get_num_sub_groups()) ? sdata[tid] : 0.0f;
        val = sub_group_reduce_add(val);
        
        if (tid == 0) {
            output[get_group_id(0)] = val;
        }
    }
}
#else
// Fallback to bank conflict free if sub-groups not available
__kernel void reductionWarpShuffle(__global const float* input,
                                    __global float* output,
                                    const int n,
                                    __local float* sdata)
{
    // Just call the bank conflict free version
    int tid = get_local_id(0);
    int idx = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
    
    sdata[tid] = ((idx < n) ? input[idx] : 0.0f) + 
                 ((idx + get_local_size(0) < n) ? input[idx + get_local_size(0)] : 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = get_local_size(0) / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid < 32) {
        volatile __local float* smem = sdata;
        if (get_local_size(0) >= 64) smem[tid] += smem[tid + 32];
        if (get_local_size(0) >= 32) smem[tid] += smem[tid + 16];
        if (get_local_size(0) >= 16) smem[tid] += smem[tid + 8];
        if (get_local_size(0) >= 8)  smem[tid] += smem[tid + 4];
        if (get_local_size(0) >= 4)  smem[tid] += smem[tid + 2];
        if (get_local_size(0) >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[get_group_id(0)] = sdata[0];
    }
}
#endif

/********************************************************************************
 * KERNEL 5: ATOMIC REDUCTION
 * 
 * @brief Simple atomic reduction
 * @details Uses atomic operations for straightforward reduction:
 *          - Each thread adds to global result atomically
 *          - Very simple code
 *          - Serialization at atomic operation
 * 
 * @performance
 *          - Expected: 5-10% of peak (atomic serialization)
 *          - Good for small arrays or when simplicity matters
 * 
 * @note Only useful for small reductions or as fallback
 ********************************************************************************/
__kernel void reductionAtomic(__global const float* input,
                               __global float* output,
                               const int n)
{
    int idx = get_global_id(0);
    
    if (idx < n) {
        // Use atomic add to global memory
        // Note: OpenCL doesn't have atomic_add for floats in all versions
        // This is a workaround using atomic_cmpxchg
        union {
            float f;
            unsigned int i;
        } old_val, new_val;
        
        __global unsigned int* output_as_int = (__global unsigned int*)output;
        
        do {
            old_val.i = *output_as_int;
            new_val.f = old_val.f + input[idx];
        } while (atomic_cmpxchg(output_as_int, old_val.i, new_val.i) != old_val.i);
    }
}

/********************************************************************************
 * OPENCL vs CUDA COMPARISON:
 * 
 * 1. Warp/Wavefront Operations:
 *    CUDA:   __shfl_down_sync() warp intrinsics
 *    OpenCL: sub_group_reduce_add() (OpenCL 2.0+)
 * 
 * 2. Atomic Operations:
 *    CUDA:   atomicAdd() for floats (native)
 *    OpenCL: atomic_cmpxchg() workaround (OpenCL 1.2)
 *            atomic_fetch_add_explicit() (OpenCL 2.0+ for floats)
 * 
 * 3. Volatile Keyword:
 *    CUDA:   volatile __shared__ float* ptr;
 *    OpenCL: volatile __local float* ptr;  (same semantics)
 * 
 * PERFORMANCE COMPARISON:
 * - Naive:         ~10-20% of peak
 * - Sequential:    ~30-40% of peak
 * - Bank Conflict Free: ~60-70% of peak
 * - Warp Shuffle:  ~80-90% of peak
 * - Atomic:        ~5-10% of peak
 * 
 * OpenCL achieves 85-95% of CUDA performance on optimized reductions.
 * The gap is due to slightly less mature compiler optimizations.
 ********************************************************************************/
