/*******************************************************************************
 * FILE: vector_add.cu
 * 
 * PURPOSE:
 *   CUDA kernel for element-wise vector addition.
 *   
 *   This is the SIMPLEST possible GPU kernel - perfect for learning!
 *   
 *   Operation: C[i] = A[i] + B[i] for all i
 * 
 * WHY THIS KERNEL?
 *   - Simplest possible operation (one addition per element)
 *   - Memory-bound (bandwidth limited, not compute limited)
 *   - Perfect memory coalescing (adjacent threads access adjacent memory)
 *   - No synchronization needed
 *   - Good introduction to GPU programming
 * 
 * PERFORMANCE CHARACTERISTICS:
 *   - Memory accesses: 2 reads + 1 write = 3 × 4 bytes = 12 bytes per element
 *   - Computation: 1 floating-point add
 *   - Arithmetic intensity: 1 FLOP / 12 bytes = 0.083 FLOP/byte
 *   - Classification: MEMORY-BOUND (bottleneck is memory bandwidth)
 * 
 * EXPECTED PERFORMANCE (RTX 3050):
 *   - Theoretical bandwidth: ~192 GB/s
 *   - Practical bandwidth: ~50-60 GB/s (30-40% of peak)
 *   - Time for 1M elements: ~0.2-0.3 ms
 * 
 * WHAT YOU'LL LEARN:
 *   1. How to write a CUDA kernel
 *   2. Thread indexing (blockIdx, threadIdx, blockDim)
 *   3. Boundary checking (if idx < n)
 *   4. Memory coalescing (adjacent threads, adjacent memory)
 *   5. Kernel launch syntax (<<<grid, block>>>)
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, NVIDIA RTX 3050, CUDA 12.x
 * 
 ******************************************************************************/

// CUDA includes
#include <cuda_runtime.h>  // CUDA runtime API
#include <device_launch_parameters.h>  // blockIdx, threadIdx, etc.

// Standard library (for host code)
#include <stdio.h>   // For printf (debugging)
#include <cmath>     // For abs (verification)

/*******************************************************************************
 * KERNEL: vectorAddKernel
 * 
 * Performs element-wise addition of two vectors on the GPU.
 * 
 * PARAMETERS:
 *   a:  Input vector A (device pointer)
 *   b:  Input vector B (device pointer)
 *   c:  Output vector C (device pointer) - will contain A + B
 *   n:  Number of elements in each vector
 * 
 * THREAD ORGANIZATION:
 *   Each thread processes ONE element:
 *     Thread 0 processes element 0
 *     Thread 1 processes element 1
 *     ...
 *     Thread i processes element i
 * 
 * THREAD INDEX CALCULATION:
 *   idx = blockIdx.x * blockDim.x + threadIdx.x
 *   
 *   Explanation:
 *     - blockIdx.x:  Which block this thread belongs to
 *     - blockDim.x:  How many threads per block
 *     - threadIdx.x: Thread's position within its block
 *   
 *   Example with 256 threads per block:
 *     Block 0, Thread 0:   idx = 0 * 256 + 0 = 0
 *     Block 0, Thread 1:   idx = 0 * 256 + 1 = 1
 *     Block 0, Thread 255: idx = 0 * 256 + 255 = 255
 *     Block 1, Thread 0:   idx = 1 * 256 + 0 = 256
 *     Block 1, Thread 1:   idx = 1 * 256 + 1 = 257
 *     ...and so on
 * 
 * BOUNDARY CHECK:
 *   if (idx < n) ensures we don't access beyond array bounds.
 *   
 *   Why needed?
 *     If n = 1000 and we launch 256 threads per block:
 *       Blocks needed = ceil(1000/256) = 4 blocks
 *       Total threads = 4 * 256 = 1024 threads
 *       But we only have 1000 elements!
 *       Threads 1000-1023 must do nothing (boundary check)
 * 
 * MEMORY COALESCING:
 *   Adjacent threads access adjacent memory locations = GOOD!
 *   
 *   Thread 0 accesses a[0], b[0], c[0]
 *   Thread 1 accesses a[1], b[1], c[1]
 *   Thread 2 accesses a[2], b[2], c[2]
 *   ...
 *   
 *   GPU groups 32 threads (a "warp") together.
 *   When memory accesses are coalesced, the warp can fetch all 32
 *   values in ONE memory transaction = FAST!
 * 
 * PERFORMANCE:
 *   - Memory bandwidth limited (not compute limited)
 *   - Achieves ~30-40% of theoretical peak bandwidth
 *   - ~0.2-0.3 ms for 1 million elements on RTX 3050
 * 
 ******************************************************************************/
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    // Calculate this thread's global index
    // Each thread processes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: Make sure we don't go beyond array bounds
    // Some threads may have idx >= n if n is not divisible by block size
    if (idx < n) {
        // Perform the addition
        // This is the actual work: ONE floating-point addition
        // Thread 0 computes: c[0] = a[0] + b[0]
        // Thread 1 computes: c[1] = a[1] + b[1]
        // Thread 2 computes: c[2] = a[2] + b[2]
        // ...and so on
        c[idx] = a[idx] + b[idx];
    }
    
    // That's it! No synchronization needed because:
    //   - Each thread writes to different location (no conflicts)
    //   - No inter-thread communication required
    //   - Purely parallel operation
}

/*******************************************************************************
 * HOST FUNCTION: launchVectorAdd
 * 
 * Host-side wrapper to launch the vector addition kernel.
 * 
 * This function is called from CUDABackend.cpp to execute the kernel.
 * 
 * PARAMETERS:
 *   d_a:  Device pointer to input vector A
 *   d_b:  Device pointer to input vector B
 *   d_c:  Device pointer to output vector C
 *   n:    Number of elements
 * 
 * WORKFLOW:
 *   1. Calculate launch configuration (grid size, block size)
 *   2. Launch kernel
 *   3. Check for errors
 * 
 * LAUNCH CONFIGURATION:
 *   Block size: 256 threads per block (common choice)
 *   Grid size:  Enough blocks to cover all elements
 * 
 * WHY 256 THREADS PER BLOCK?
 *   - Multiple of 32 (warp size) = good
 *   - Not too small (underutilizes GPU)
 *   - Not too large (limits occupancy)
 *   - 256 is a "sweet spot" for many GPUs
 * 
 * ERROR CHECKING:
 *   Always check for kernel launch errors!
 *   Kernels can fail silently if you don't check.
 * 
 ******************************************************************************/
extern "C" void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int n) {
    // STEP 1: Choose block size (threads per block)
    // 256 is a good default for most kernels
    // Must be multiple of 32 (warp size)
    // Common choices: 128, 256, 512
    const int blockSize = 256;
    
    // STEP 2: Calculate grid size (number of blocks)
    // Formula: gridSize = ceil(n / blockSize)
    // 
    // Why the +blockSize-1 trick?
    //   Without it: 1000 / 256 = 3 (integer division rounds down)
    //   With it: (1000 + 255) / 256 = 1255 / 256 = 4 (rounds up)
    // 
    // This ensures we have ENOUGH blocks to cover all elements
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    // STEP 3: Launch kernel
    // Syntax: kernelName<<<gridSize, blockSize>>>(args...)
    // 
    // <<<...>>> is CUDA's way of specifying execution configuration
    // 
    // What happens:
    //   1. CUDA driver queues the kernel for execution
    //   2. GPU scheduler dispatches blocks to streaming multiprocessors (SMs)
    //   3. Each SM executes threads in warps (groups of 32)
    //   4. Function returns immediately (async!)
    // 
    // IMPORTANT: This returns immediately! GPU executes in background.
    //            Must call cudaDeviceSynchronize() to wait for completion.
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // STEP 4: Check for kernel launch errors
    // 
    // Kernel launches can fail for many reasons:
    //   - Invalid configuration (too many threads, too much shared memory)
    //   - Out of memory
    //   - GPU fault
    // 
    // cudaGetLastError() retrieves the last error without clearing it
    // You should call this after every kernel launch!
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // Kernel launch failed!
        // cudaGetErrorString() converts error code to human-readable message
        printf("CUDA Error: Vector add kernel launch failed: %s\n", 
               cudaGetErrorString(error));
    }
    
    // Note: We don't call cudaDeviceSynchronize() here because:
    //   1. The timing code needs to measure async execution
    //   2. CUDABackend will call it when needed
}

/*******************************************************************************
 * HOST FUNCTION: verifyVectorAdd (Optional Utility)
 * 
 * Verify that GPU results are correct by comparing to CPU calculation.
 * 
 * This is ESSENTIAL for development:
 *   - Easy to get fast but WRONG results
 *   - Always verify correctness before measuring performance
 *   - "If it's not tested, it doesn't work"
 * 
 * PARAMETERS:
 *   h_a:  Host vector A
 *   h_b:  Host vector B
 *   h_c:  Host vector C (GPU results copied from device)
 *   n:    Number of elements
 * 
 * RETURNS:
 *   true if all elements match (within tolerance)
 *   false if any element is incorrect
 * 
 * TOLERANCE:
 *   Floating-point arithmetic is not exact!
 *   We allow small differences (< 1e-5) due to rounding
 * 
 ******************************************************************************/
extern "C" bool verifyVectorAdd(const float* h_a, const float* h_b, const float* h_c, int n) {
    // Tolerance for floating-point comparison
    // GPUs and CPUs may compute slightly different results due to rounding
    const float epsilon = 1e-5f;
    
    // Check each element
    for (int i = 0; i < n; i++) {
        // Calculate expected value (what CPU would compute)
        float expected = h_a[i] + h_b[i];
        
        // Compare with GPU result
        float actual = h_c[i];
        
        // Check if difference is within tolerance
        float diff = fabs(actual - expected);
        if (diff > epsilon) {
            // Found an incorrect value!
            printf("Verification FAILED at element %d:\n", i);
            printf("  Expected: %f\n", expected);
            printf("  Got:      %f\n", actual);
            printf("  Error:    %f\n", diff);
            return false;  // Verification failed
        }
    }
    
    // All elements correct!
    return true;
}

/*******************************************************************************
 * END OF FILE: vector_add.cu
 * 
 * WHAT WE IMPLEMENTED:
 *   1. Simple GPU kernel for vector addition
 *   2. Host function to launch the kernel
 *   3. Verification function to check correctness
 * 
 * KEY CUDA CONCEPTS USED:
 *   - __global__ function (runs on GPU, called from CPU)
 *   - Thread indexing (blockIdx, threadIdx, blockDim)
 *   - Launch configuration (<<<grid, block>>>)
 *   - Memory coalescing (adjacent threads, adjacent memory)
 *   - Boundary checking (if idx < n)
 *   - Error checking (cudaGetLastError)
 * 
 * PERFORMANCE NOTES:
 *   - This kernel is memory-bandwidth limited
 *   - Computation (1 add) is trivial compared to memory access
 *   - Expect ~30-40% of theoretical bandwidth
 *   - Further optimization would focus on memory access patterns
 * 
 * TESTING THIS KERNEL:
 *   1. Allocate host arrays: float* h_a = new float[n];
 *   2. Initialize with test data: h_a[i] = i; h_b[i] = i*2;
 *   3. Allocate device arrays: cudaMalloc(&d_a, n*sizeof(float));
 *   4. Copy to device: cudaMemcpy(d_a, h_a, ..., cudaMemcpyHostToDevice);
 *   5. Launch kernel: launchVectorAdd(d_a, d_b, d_c, n);
 *   6. Copy results back: cudaMemcpy(h_c, d_c, ..., cudaMemcpyDeviceToHost);
 *   7. Verify: bool correct = verifyVectorAdd(h_a, h_b, h_c, n);
 * 
 * NEXT STEPS:
 *   - Implement CUDABackend class to use this kernel
 *   - Add timing measurements
 *   - Calculate bandwidth achieved
 *   - Compare with other backends
 * 
 * FOR INTERVIEWS:
 *   Explain:
 *     - Why memory-bound (arithmetic intensity analysis)
 *     - Thread indexing formula (blockIdx * blockDim + threadIdx)
 *     - Memory coalescing importance (32-thread warps)
 *     - Boundary checking necessity (non-multiple of block size)
 *     - Why async (GPU executes while CPU continues)
 * 
 ******************************************************************************/
