/*******************************************************************************
 * FILE: reduction.cu
 * 
 * PURPOSE:
 *   CUDA kernel implementations for parallel reduction (sum all elements).
 *   Demonstrates synchronization, divergence, and shared memory optimization.
 * 
 * WHAT THIS BENCHMARK MEASURES:
 *   - Warp-level synchronization
 *   - Shared memory bank conflicts
 *   - Thread divergence impact
 *   - Atomic operations performance
 *   - Multi-level parallelism
 * 
 * WHY REDUCTION IS IMPORTANT:
 *   - Statistics (sum, mean, variance)
 *   - Machine learning (loss calculation, gradient aggregation)
 *   - Scientific computing (norms, inner products)
 *   - Data analysis (totals, aggregations)
 * 
 * ALGORITHM:
 *   Given array [a0, a1, a2, ..., an-1], compute:
 *   result = a0 + a1 + a2 + ... + an-1
 * 
 * CHALLENGE:
 *   Sequential addition is O(n) - how to parallelize?
 * 
 * SOLUTION:
 *   Tree-based reduction in log2(n) steps:
 *   Step 0: [a0, a1, a2, a3, a4, a5, a6, a7]
 *   Step 1: [a0+a1, a2+a3, a4+a5, a6+a7]
 *   Step 2: [a0+a1+a2+a3, a4+a5+a6+a7]
 *   Step 3: [a0+a1+a2+a3+a4+a5+a6+a7]
 * 
 * OPTIMIZATION LEVELS:
 *   1. Naive: Divergent threads, poor memory access
 *   2. Sequential addressing: No divergence
 *   3. Bank conflict free: Optimized shared memory
 *   4. Warp shuffle: No shared memory needed
 *   5. Multi-block: Handle arbitrary sizes
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022, RTX 3050, CUDA 13.1
 * 
 ******************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*******************************************************************************
 * KERNEL 1: NAIVE REDUCTION
 * 
 * PURPOSE:
 *   Simple tree-based reduction with interleaved addressing.
 * 
 * ALGORITHM:
 *   stride = 1
 *   while stride < blockSize:
 *     if threadIdx.x % (2*stride) == 0:
 *       sdata[tid] += sdata[tid + stride]
 *     stride *= 2
 * 
 * PROBLEMS:
 *   1. DIVERGENCE: Threads in same warp take different paths
 *   2. MEMORY: Non-coalesced shared memory access
 *   3. IDLE THREADS: Half threads idle at each step
 * 
 * PERFORMANCE:
 *   RTX 3050: ~20-30 GB/s (vs 224 GB/s peak)
 ******************************************************************************/
__global__ void reductionNaive(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        // BAD: Divergent threads (threadIdx.x % (2*stride) causes divergence)
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/*******************************************************************************
 * KERNEL 2: SEQUENTIAL ADDRESSING
 * 
 * PURPOSE:
 *   Eliminate thread divergence by using sequential addressing.
 * 
 * KEY INSIGHT:
 *   Instead of checking threadIdx.x % (2*stride),
 *   use sequential addressing: threads 0 to stride-1 are active.
 * 
 * ADVANTAGE:
 *   - First half of threads always active together (no divergence)
 *   - Warps either fully active or fully inactive
 *   - Better instruction throughput
 * 
 * PERFORMANCE:
 *   RTX 3050: ~60-80 GB/s (3x faster than naive)
 ******************************************************************************/
__global__ void reductionSequential(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // GOOD: Sequential addressing - no divergence
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/*******************************************************************************
 * KERNEL 3: BANK CONFLICT FREE
 * 
 * PURPOSE:
 *   Avoid shared memory bank conflicts during reduction.
 * 
 * SHARED MEMORY BANKS:
 *   - Shared memory divided into 32 banks
 *   - Successive 4-byte words map to successive banks
 *   - Bank conflict: Multiple threads access same bank
 * 
 * PROBLEM IN SEQUENTIAL:
 *   When stride < 32, threads access same bank
 *   Example: stride=16, thread 0 and 16 access banks 0 and 16
 * 
 * SOLUTION:
 *   First add with larger strides (>32), then use warp-level primitives
 * 
 * PERFORMANCE:
 *   RTX 3050: ~100-140 GB/s (5x faster than naive)
 ******************************************************************************/
__global__ void reductionBankConflictFree(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and perform first level of reduction during load
    // (reduces global memory reads by 2x!)
    sdata[tid] = 0.0f;
    if (i < n) sdata[tid] = input[i];
    if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
    __syncthreads();
    
    // Reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no __syncthreads needed, implicit sync)
    if (tid < 32) {
        // Last warp reduces without __syncthreads (warp executes in lockstep)
        volatile float* vsmem = sdata;  // Volatile to prevent optimization
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
        if (blockDim.x >= 16) vsmem[tid] += vsmem[tid + 8];
        if (blockDim.x >= 8) vsmem[tid] += vsmem[tid + 4];
        if (blockDim.x >= 4) vsmem[tid] += vsmem[tid + 2];
        if (blockDim.x >= 2) vsmem[tid] += vsmem[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/*******************************************************************************
 * KERNEL 4: WARP SHUFFLE
 * 
 * PURPOSE:
 *   Use modern warp-level primitives (__shfl_down_sync) for fastest reduction.
 * 
 * WARP SHUFFLE:
 *   - Threads within a warp can directly read registers of other threads
 *   - No shared memory needed!
 *   - Faster and simpler
 * 
 * __shfl_down_sync(mask, value, delta):
 *   - mask: Active thread mask (0xffffffff = all 32 threads)
 *   - value: Value to exchange
 *   - delta: Offset to read from (threadIdx + delta)
 * 
 * ALGORITHM:
 *   Step 0: Each thread has value
 *   Step 1: thread[i] += thread[i+16]  (warp of 32)
 *   Step 2: thread[i] += thread[i+8]
 *   Step 3: thread[i] += thread[i+4]
 *   Step 4: thread[i] += thread[i+2]
 *   Step 5: thread[i] += thread[i+1]
 *   Result in thread[0]
 * 
 * PERFORMANCE:
 *   RTX 3050: ~180-220 GB/s (approaching peak bandwidth!)
 ******************************************************************************/
__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reductionWarpShuffle(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and perform first reduction
    float sum = 0.0f;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Shared memory for inter-warp reduction
    __shared__ float warpSums[32];  // Max 32 warps per block (1024/32)
    
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    // Write warp result to shared memory
    if (laneId == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();
    
    // Final reduction of warp results
    // Only first warp does this
    if (tid < (blockDim.x + 31) / 32) {
        sum = warpSums[tid];
    } else {
        sum = 0.0f;
    }
    
    if (warpId == 0) {
        sum = warpReduceSum(sum);
    }
    
    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

/*******************************************************************************
 * KERNEL 5: MULTI-BLOCK WITH ATOMICS
 * 
 * PURPOSE:
 *   Handle arrays larger than can fit in one block using atomic operations.
 * 
 * CHALLENGE:
 *   - Previous kernels output one value per block
 *   - Need recursive reduction for multiple blocks
 *   - OR use atomic operations
 * 
 * ATOMIC OPERATIONS:
 *   atomicAdd(address, value):
 *   - Atomically: *address = *address + value
 *   - Hardware-guaranteed no race conditions
 *   - Serializes conflicting updates
 * 
 * TRADE-OFF:
 *   - Simplifies code (single-pass)
 *   - Slightly slower than two-pass reduction
 *   - Good for moderate contention
 * 
 * PERFORMANCE:
 *   RTX 3050: ~140-180 GB/s (slower due to atomic contention)
 ******************************************************************************/
__global__ void reductionMultiBlockAtomic(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and perform first reduction
    float sum = 0.0f;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Accumulate warp sums
    __shared__ float warpSums[32];
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    if (laneId == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();
    
    // Final warp reduction
    if (tid < (blockDim.x + 31) / 32) {
        sum = warpSums[tid];
    } else {
        sum = 0.0f;
    }
    
    if (warpId == 0) {
        sum = warpReduceSum(sum);
    }
    
    // Atomic add to global output (all blocks contribute to single result)
    if (tid == 0) {
        atomicAdd(output, sum);
    }
}

/*******************************************************************************
 * HOST WRAPPER FUNCTIONS
 ******************************************************************************/

extern "C" {

// Launch naive reduction
void launchReductionNaive(const float* d_input, float* d_output, int n,
                           cudaStream_t stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int smemSize = threadsPerBlock * sizeof(float);
    
    reductionNaive<<<blocksPerGrid, threadsPerBlock, smemSize, stream>>>(
        d_input, d_output, n);
}

// Launch sequential addressing reduction
void launchReductionSequential(const float* d_input, float* d_output, int n,
                                cudaStream_t stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int smemSize = threadsPerBlock * sizeof(float);
    
    reductionSequential<<<blocksPerGrid, threadsPerBlock, smemSize, stream>>>(
        d_input, d_output, n);
}

// Launch bank conflict free reduction
void launchReductionBankConflictFree(const float* d_input, float* d_output, int n,
                                      cudaStream_t stream) {
    int threadsPerBlock = 512;  // Can use more threads now
    int blocksPerGrid = (n + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    int smemSize = threadsPerBlock * sizeof(float);
    
    reductionBankConflictFree<<<blocksPerGrid, threadsPerBlock, smemSize, stream>>>(
        d_input, d_output, n);
}

// Launch warp shuffle reduction
void launchReductionWarpShuffle(const float* d_input, float* d_output, int n,
                                 cudaStream_t stream) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    
    reductionWarpShuffle<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_input, d_output, n);
}

// Launch multi-block atomic reduction (single-pass)
void launchReductionMultiBlockAtomic(const float* d_input, float* d_output, int n,
                                      cudaStream_t stream) {
    // Initialize output to zero first
    cudaMemsetAsync(d_output, 0, sizeof(float), stream);
    
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    
    reductionMultiBlockAtomic<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_input, d_output, n);
}

} // extern "C"

/*******************************************************************************
 * PERFORMANCE ANALYSIS
 * 
 * For reducing 16M elements on RTX 3050:
 * 
 * NAIVE:
 *   - Time: ~2-3 ms
 *   - Bandwidth: ~20-30 GB/s
 *   - Bottleneck: Thread divergence, poor memory access
 * 
 * SEQUENTIAL ADDRESSING:
 *   - Time: ~0.8-1.2 ms
 *   - Bandwidth: ~60-80 GB/s
 *   - Improvement: 3x faster (no divergence)
 * 
 * BANK CONFLICT FREE:
 *   - Time: ~0.4-0.6 ms
 *   - Bandwidth: ~100-140 GB/s
 *   - Improvement: 5x faster (optimized shared memory)
 * 
 * WARP SHUFFLE:
 *   - Time: ~0.3-0.4 ms
 *   - Bandwidth: ~180-220 GB/s
 *   - Improvement: 7-8x faster (no shared memory, warp intrinsics)
 *   - Achieves 80-90% of peak bandwidth!
 * 
 * MULTI-BLOCK ATOMIC:
 *   - Time: ~0.4-0.5 ms
 *   - Bandwidth: ~140-180 GB/s
 *   - Single-pass convenience vs slight performance cost
 * 
 * KEY LEARNINGS:
 *   1. Divergence kills performance (50-70% loss)
 *   2. Shared memory bank conflicts matter (20-30% loss)
 *   3. Warp-level primitives are powerful (2x faster)
 *   4. First add during load doubles efficiency
 *   5. Atomic operations are reasonable for moderate contention
 * 
 * REAL-WORLD APPLICATIONS:
 *   - Deep learning: Loss calculation (sum of errors)
 *   - Scientific computing: Vector norms, inner products
 *   - Statistics: Sum, mean, variance
 *   - Data analytics: Aggregations
 * 
 ******************************************************************************/

/*******************************************************************************
 * END OF FILE
 ******************************************************************************/
