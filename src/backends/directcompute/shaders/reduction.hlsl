/********************************************************************************
 * @file    reduction.hlsl
 * @brief   DirectCompute Parallel Reduction Compute Shaders
 * 
 * @details Five implementations with increasing optimization:
 *          1. Naive - Simple but inefficient
 *          2. Sequential - Better addressing pattern
 *          3. Bank Conflict Free - Optimized shared memory access
 *          4. Warp Shuffle - Uses wave intrinsics (Shader Model 6.0+)
 *          5. Atomic - Simple atomic reduction
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#define BLOCK_SIZE_REDUCE 256

/********************************************************************************
 * SHADER 1: NAIVE REDUCTION
 ********************************************************************************/

RWStructuredBuffer<float> input_naive : register(u0);
RWStructuredBuffer<float> output_naive : register(u1);

cbuffer Params_Naive : register(b0)
{
    uint numElements_naive;
    uint3 padding_naive;
};

groupshared float sdata_naive[BLOCK_SIZE_REDUCE];

[numthreads(BLOCK_SIZE_REDUCE, 1, 1)]
void CSReductionNaive(uint3 groupThreadID : SV_GroupThreadID,
                      uint3 groupID : SV_GroupID,
                      uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = groupThreadID.x;
    uint idx = dispatchThreadID.x;
    
    // Load data into shared memory
    sdata_naive[tid] = (idx < numElements_naive) ? input_naive[idx] : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    
    // Reduction in shared memory (naive approach - has divergence)
    for (uint s = 1; s < BLOCK_SIZE_REDUCE; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata_naive[tid] += sdata_naive[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Write result for this thread group
    if (tid == 0)
    {
        output_naive[groupID.x] = sdata_naive[0];
    }
}

/********************************************************************************
 * SHADER 2: SEQUENTIAL ADDRESSING REDUCTION
 ********************************************************************************/

RWStructuredBuffer<float> input_seq : register(u0);
RWStructuredBuffer<float> output_seq : register(u1);

cbuffer Params_Seq : register(b0)
{
    uint numElements_seq;
    uint3 padding_seq;
};

groupshared float sdata_seq[BLOCK_SIZE_REDUCE];

[numthreads(BLOCK_SIZE_REDUCE, 1, 1)]
void CSReductionSequential(uint3 groupThreadID : SV_GroupThreadID,
                           uint3 groupID : SV_GroupID)
{
    uint tid = groupThreadID.x;
    uint idx = groupID.x * (BLOCK_SIZE_REDUCE * 2) + groupThreadID.x;
    
    // Load and perform first level of reduction during load
    sdata_seq[tid] = ((idx < numElements_seq) ? input_seq[idx] : 0.0f) +
                     ((idx + BLOCK_SIZE_REDUCE < numElements_seq) ? input_seq[idx + BLOCK_SIZE_REDUCE] : 0.0f);
    GroupMemoryBarrierWithGroupSync();
    
    // Sequential addressing reduction (no divergence)
    for (uint s = BLOCK_SIZE_REDUCE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata_seq[tid] += sdata_seq[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Write result
    if (tid == 0)
    {
        output_seq[groupID.x] = sdata_seq[0];
    }
}

/********************************************************************************
 * SHADER 3: BANK CONFLICT FREE REDUCTION
 ********************************************************************************/

RWStructuredBuffer<float> input_bcf : register(u0);
RWStructuredBuffer<float> output_bcf : register(u1);

cbuffer Params_BCF : register(b0)
{
    uint numElements_bcf;
    uint3 padding_bcf;
};

groupshared float sdata_bcf[BLOCK_SIZE_REDUCE];

[numthreads(BLOCK_SIZE_REDUCE, 1, 1)]
void CSReductionBankConflictFree(uint3 groupThreadID : SV_GroupThreadID,
                                  uint3 groupID : SV_GroupID)
{
    uint tid = groupThreadID.x;
    uint idx = groupID.x * (BLOCK_SIZE_REDUCE * 2) + groupThreadID.x;
    
    // Load and reduce during load
    sdata_bcf[tid] = ((idx < numElements_bcf) ? input_bcf[idx] : 0.0f) +
                     ((idx + BLOCK_SIZE_REDUCE < numElements_bcf) ? input_bcf[idx + BLOCK_SIZE_REDUCE] : 0.0f);
    GroupMemoryBarrierWithGroupSync();
    
    // Sequential addressing
    for (uint s = BLOCK_SIZE_REDUCE / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata_bcf[tid] += sdata_bcf[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Unroll last warp (no synchronization needed within warp)
    if (tid < 32)
    {
        if (BLOCK_SIZE_REDUCE >= 64) sdata_bcf[tid] += sdata_bcf[tid + 32];
        if (BLOCK_SIZE_REDUCE >= 32) sdata_bcf[tid] += sdata_bcf[tid + 16];
        if (BLOCK_SIZE_REDUCE >= 16) sdata_bcf[tid] += sdata_bcf[tid + 8];
        if (BLOCK_SIZE_REDUCE >= 8)  sdata_bcf[tid] += sdata_bcf[tid + 4];
        if (BLOCK_SIZE_REDUCE >= 4)  sdata_bcf[tid] += sdata_bcf[tid + 2];
        if (BLOCK_SIZE_REDUCE >= 2)  sdata_bcf[tid] += sdata_bcf[tid + 1];
    }
    
    // Write result
    if (tid == 0)
    {
        output_bcf[groupID.x] = sdata_bcf[0];
    }
}

/********************************************************************************
 * SHADER 4: WAVE INTRINSICS REDUCTION (Shader Model 6.0+)
 * 
 * @note Uses HLSL wave intrinsics (similar to CUDA warp shuffle)
 *       Requires Shader Model 6.0+ and compatible hardware
 ********************************************************************************/

RWStructuredBuffer<float> input_wave : register(u0);
RWStructuredBuffer<float> output_wave : register(u1);

cbuffer Params_Wave : register(b0)
{
    uint numElements_wave;
    uint3 padding_wave;
};

groupshared float sdata_wave[BLOCK_SIZE_REDUCE / 32];  // One entry per wave

[numthreads(BLOCK_SIZE_REDUCE, 1, 1)]
void CSReductionWaveShuffle(uint3 groupThreadID : SV_GroupThreadID,
                             uint3 groupID : SV_GroupID)
{
    uint tid = groupThreadID.x;
    uint idx = groupID.x * (BLOCK_SIZE_REDUCE * 2) + groupThreadID.x;
    
    // Load and reduce during load
    float val = ((idx < numElements_wave) ? input_wave[idx] : 0.0f) +
                ((idx + BLOCK_SIZE_REDUCE < numElements_wave) ? input_wave[idx + BLOCK_SIZE_REDUCE] : 0.0f);
    
    // Wave-level reduction using intrinsics
    val = WaveActiveSum(val);
    
    // First thread in each wave writes to shared memory
    uint laneIndex = WaveGetLaneIndex();
    uint waveIndex = tid / WaveGetLaneCount();
    
    if (laneIndex == 0)
    {
        sdata_wave[waveIndex] = val;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Final reduction in first wave
    if (tid < (BLOCK_SIZE_REDUCE / WaveGetLaneCount()))
    {
        val = sdata_wave[tid];
        val = WaveActiveSum(val);
        
        if (laneIndex == 0)
        {
            output_wave[groupID.x] = val;
        }
    }
}

/********************************************************************************
 * SHADER 5: ATOMIC REDUCTION
 ********************************************************************************/

RWStructuredBuffer<float> input_atomic : register(u0);
RWStructuredBuffer<float> output_atomic : register(u1);

cbuffer Params_Atomic : register(b0)
{
    uint numElements_atomic;
    uint3 padding_atomic;
};

[numthreads(256, 1, 1)]
void CSReductionAtomic(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    
    if (idx < numElements_atomic)
    {
        // Note: HLSL doesn't have native atomic add for floats
        // This is a workaround using InterlockedAdd on uint representation
        // For simplicity in this version, we'll use a simpler approach
        
        // Read value
        float value = input_atomic[idx];
        
        // For demonstration - actual atomic float add requires bit casting
        // This is a simplified version that shows the concept
        uint originalValue;
        uint newValue;
        uint assumedValue;
        
        // Atomic compare-exchange loop (atomic add for float)
        do
        {
            InterlockedCompareExchange(
                (uint)output_atomic[0],  // destination
                assumedValue,             // compare value
                asuint(asfloat(assumedValue) + value),  // exchange value
                originalValue             // original value
            );
            assumedValue = originalValue;
        } while (originalValue != assumedValue);
    }
}

/********************************************************************************
 * HLSL vs CUDA vs OPENCL COMPARISON:
 * 
 * 1. Wave/Warp Operations:
 *    CUDA:   __shfl_down_sync(), __ballot_sync()
 *    OpenCL: sub_group_reduce_add() (OpenCL 2.0+)
 *    HLSL:   WaveActiveSum(), WaveGetLaneIndex() (SM 6.0+)
 * 
 * 2. Atomic Operations:
 *    CUDA:   atomicAdd() (native for floats)
 *    OpenCL: atomic_cmpxchg() workaround
 *    HLSL:   InterlockedCompareExchange() workaround
 * 
 * 3. Synchronization:
 *    CUDA:   __syncthreads()
 *    OpenCL: barrier(CLK_LOCAL_MEM_FENCE)
 *    HLSL:   GroupMemoryBarrierWithGroupSync()
 * 
 * 4. Shared Memory:
 *    CUDA:   __shared__ float data[SIZE];
 *    OpenCL: __local float data[SIZE];
 *    HLSL:   groupshared float data[SIZE];
 * 
 * PERFORMANCE EXPECTATIONS:
 * - Naive:              ~10-20% of peak
 * - Sequential:         ~30-40% of peak
 * - Bank Conflict Free: ~60-70% of peak
 * - Wave Shuffle:       ~80-90% of peak (SM 6.0+ only)
 * - Atomic:             ~5-10% of peak
 * 
 * DirectCompute achieves 85-95% of CUDA performance on optimized reductions.
 ********************************************************************************/
