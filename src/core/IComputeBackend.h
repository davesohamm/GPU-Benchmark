/*******************************************************************************
 * FILE: IComputeBackend.h
 * 
 * PURPOSE:
 *   This file defines the abstract interface (pure virtual class) that ALL
 *   GPU compute backends must implement. This interface provides a uniform
 *   way to interact with different GPU APIs (CUDA, OpenCL, DirectCompute)
 *   without knowing which specific API is being used.
 * 
 * KEY CONCEPT:
 *   Polymorphism - treating different types through a common interface.
 *   
 *   Think of this like a "contract" that says: "If you want to be a GPU
 *   backend in this system, you MUST provide these capabilities."
 * 
 * WHY THIS EXISTS:
 *   Without this interface, we'd need separate code paths for each API:
 *     if (usingCUDA) { cudaDoSomething(); }
 *     else if (usingOpenCL) { clDoSomething(); }
 *     else if (usingDirectCompute) { d3dDoSomething(); }
 *   
 *   With this interface, we can write:
 *     backend->DoSomething();  // Works for all backends!
 * 
 * AUTHOR: Soham
 * DATE: January 2026
 * SYSTEM: Windows 11, Visual Studio 2022
 * 
 ******************************************************************************/

#ifndef ICOMPUTE_BACKEND_H
#define ICOMPUTE_BACKEND_H

// Standard library includes
#include <string>      // For std::string (backend names, error messages)
#include <vector>      // For std::vector (device lists, etc.)
#include <cstddef>     // For size_t (memory sizes)

/*******************************************************************************
 * NAMESPACE: GPUBenchmark
 * 
 * All code in this project lives in the GPUBenchmark namespace to avoid
 * conflicts with other libraries.
 ******************************************************************************/
namespace GPUBenchmark {

/*******************************************************************************
 * ENUM: BackendType
 * 
 * Identifies which GPU compute API is being used.
 * 
 * Used for:
 *   - Logging which backend produced results
 *   - User selection of backends
 *   - Conditional logic when needed
 ******************************************************************************/
enum class BackendType {
    CUDA,           // NVIDIA CUDA (NVIDIA GPUs only)
    OpenCL,         // OpenCL (cross-vendor: NVIDIA, AMD, Intel)
    DirectCompute,  // DirectCompute/DirectX (Windows native, all vendors)
    Unknown         // Uninitialized or error state
};

/*******************************************************************************
 * STRUCT: DeviceInfo
 * 
 * Contains information about the GPU device being used.
 * 
 * This is filled in by each backend during initialization and used for
 * logging and display to the user.
 ******************************************************************************/
struct DeviceInfo {
    std::string name;              // GPU name (e.g., "NVIDIA GeForce RTX 3050")
    size_t totalMemoryBytes;       // Total GPU memory in bytes
    size_t availableMemoryBytes;   // Available GPU memory in bytes
    std::string driverVersion;     // Driver version string
    int computeCapabilityMajor;    // Compute capability major version (CUDA) or equivalent
    int computeCapabilityMinor;    // Compute capability minor version (CUDA) or equivalent
    int maxThreadsPerBlock;        // Maximum threads per block/workgroup
    int maxBlockDimX;              // Maximum block dimension in X
    int maxBlockDimY;              // Maximum block dimension in Y
    int maxBlockDimZ;              // Maximum block dimension in Z
    
    // Constructor with default values
    DeviceInfo() 
        : totalMemoryBytes(0)
        , availableMemoryBytes(0)
        , computeCapabilityMajor(0)
        , computeCapabilityMinor(0)
        , maxThreadsPerBlock(0)
        , maxBlockDimX(0)
        , maxBlockDimY(0)
        , maxBlockDimZ(0)
    {}
};

/*******************************************************************************
 * STRUCT: KernelParams
 * 
 * Parameters passed to kernel execution functions.
 * 
 * Contains all information needed to launch a GPU kernel:
 *   - Input/output buffer pointers
 *   - Problem size
 *   - Thread configuration
 * 
 * NOTE: Buffer pointers are void* because they point to GPU memory,
 *       which has different types in different APIs:
 *         CUDA:          void* (device pointer)
 *         OpenCL:        cl_mem (buffer object)
 *         DirectCompute: ID3D11Buffer* (COM interface pointer)
 ******************************************************************************/
struct KernelParams {
    // Input/output buffer pointers (GPU memory)
    void* input1;           // First input buffer
    void* input2;           // Second input buffer (if needed)
    void* output;           // Output buffer
    
    // Problem size
    size_t numElements;     // Number of elements to process
    size_t elementSize;     // Size of each element in bytes
    
    // Thread/block configuration
    int blockSize;          // Threads per block (CUDA/HLSL) or work-group size (OpenCL)
    int gridSize;           // Number of blocks (CUDA/HLSL) or work-groups (OpenCL)
    
    // Additional parameters (benchmark-specific)
    int width;              // For 2D operations (matrix, image)
    int height;             // For 2D operations
    int depth;              // For 3D operations
    
    // Constructor with default values
    KernelParams()
        : input1(nullptr)
        , input2(nullptr)
        , output(nullptr)
        , numElements(0)
        , elementSize(0)
        , blockSize(256)    // Common default: 256 threads per block
        , gridSize(0)       // Calculated based on numElements
        , width(0)
        , height(0)
        , depth(0)
    {}
};

/*******************************************************************************
 * STRUCT: TimingResult
 * 
 * Holds timing information for a single operation.
 * 
 * GPU timing is complex because operations happen asynchronously.
 * We measure both:
 *   1. CPU-side time (includes all overhead)
 *   2. GPU-side time (actual execution time)
 ******************************************************************************/
struct TimingResult {
    double cpuTimeMS;       // CPU-side measured time (milliseconds)
    double gpuTimeMS;       // GPU-side measured time (milliseconds)
    
    // Constructor
    TimingResult() : cpuTimeMS(0.0), gpuTimeMS(0.0) {}
};

/*******************************************************************************
 * CLASS: IComputeBackend
 * 
 * ABSTRACT INTERFACE for all GPU compute backends.
 * 
 * This is a "pure virtual" class, meaning:
 *   1. Cannot be instantiated directly (must create derived class)
 *   2. All methods marked "= 0" MUST be implemented by derived classes
 *   3. Provides common interface for polymorphism
 * 
 * USAGE PATTERN:
 *   // Create concrete backend (CUDA, OpenCL, or DirectCompute)
 *   IComputeBackend* backend = new CUDABackend();
 *   
 *   // Initialize
 *   if (!backend->Initialize()) {
 *       std::cerr << "Backend initialization failed!" << std::endl;
 *       return;
 *   }
 *   
 *   // Allocate GPU memory
 *   void* deviceBuffer = backend->AllocateMemory(1024 * sizeof(float));
 *   
 *   // Copy data to GPU
 *   backend->CopyHostToDevice(deviceBuffer, hostData, 1024 * sizeof(float));
 *   
 *   // Execute kernel
 *   KernelParams params;
 *   params.input1 = deviceBuffer;
 *   params.numElements = 1024;
 *   backend->ExecuteKernel("vector_add", params);
 *   
 *   // Wait for completion
 *   backend->Synchronize();
 *   
 *   // Copy results back
 *   backend->CopyDeviceToHost(results, deviceBuffer, 1024 * sizeof(float));
 *   
 *   // Cleanup
 *   backend->FreeMemory(deviceBuffer);
 *   backend->Shutdown();
 *   delete backend;
 * 
 ******************************************************************************/
class IComputeBackend {
public:
    /**************************************************************************
     * DESTRUCTOR
     * 
     * Virtual destructor is ESSENTIAL when using polymorphism!
     * 
     * Why?
     *   When you do: delete backend;
     *   where backend is of type IComputeBackend*, but points to CUDABackend,
     *   we need to call the CUDABackend destructor, not just IComputeBackend.
     * 
     * Without "virtual", only the base class destructor would be called,
     * causing memory leaks in derived classes!
     *************************************************************************/
    virtual ~IComputeBackend() = default;
    
    /**************************************************************************
     * INITIALIZATION AND CLEANUP
     *************************************************************************/
    
    /**
     * Initialize the GPU backend.
     * 
     * This function:
     *   1. Detects available GPUs
     *   2. Initializes the API (CUDA runtime, OpenCL context, D3D device)
     *   3. Loads/compiles kernels
     *   4. Queries device capabilities
     * 
     * @return true if initialization successful, false otherwise
     * 
     * WHEN TO CALL: Once at application startup, before any other operations.
     * 
     * ERROR HANDLING:
     *   - Returns false if GPU not available
     *   - Returns false if drivers not installed
     *   - Returns false if API initialization fails
     *   
     *   Application should check return value and handle gracefully.
     */
    virtual bool Initialize() = 0;
    
    /**
     * Shutdown the GPU backend and release all resources.
     * 
     * This function:
     *   1. Releases all GPU memory
     *   2. Destroys kernels/programs
     *   3. Releases API contexts
     * 
     * WHEN TO CALL: Once at application exit, after all GPU operations complete.
     * 
     * IMPORTANT: Must call Synchronize() before Shutdown() to ensure all
     *            GPU operations are complete!
     */
    virtual void Shutdown() = 0;
    
    /**************************************************************************
     * DEVICE INFORMATION
     *************************************************************************/
    
    /**
     * Get information about the GPU device.
     * 
     * @return DeviceInfo structure with GPU details
     * 
     * WHEN TO CALL: After successful Initialize(), for logging/display.
     */
    virtual DeviceInfo GetDeviceInfo() const = 0;
    
    /**
     * Get the backend type identifier.
     * 
     * @return BackendType enum value (CUDA, OpenCL, or DirectCompute)
     */
    virtual BackendType GetBackendType() const = 0;
    
    /**
     * Get human-readable name of the backend.
     * 
     * @return String like "CUDA", "OpenCL", "DirectCompute"
     * 
     * Used for logging and display to user.
     */
    virtual std::string GetBackendName() const = 0;
    
    /**************************************************************************
     * MEMORY MANAGEMENT
     * 
     * GPU memory is separate from CPU memory. Data must be explicitly
     * transferred between CPU (host) and GPU (device).
     *************************************************************************/
    
    /**
     * Allocate memory on the GPU.
     * 
     * @param sizeBytes  Number of bytes to allocate
     * @return           Pointer to GPU memory (opaque, do not dereference on CPU!)
     *                   Returns nullptr on failure
     * 
     * IMPORTANT: The returned pointer is a GPU memory address!
     *            You CANNOT dereference it on the CPU!
     *            You can only pass it to other GPU functions.
     * 
     * CUDA:          Returns cudaMalloc() pointer
     * OpenCL:        Returns cl_mem object (wrapped as void*)
     * DirectCompute: Returns ID3D11Buffer* (wrapped as void*)
     * 
     * MEMORY MANAGEMENT:
     *   Caller is responsible for freeing memory with FreeMemory()!
     * 
     * Example:
     *   void* gpuBuffer = backend->AllocateMemory(1024 * sizeof(float));
     *   if (gpuBuffer == nullptr) {
     *       // Handle allocation failure (out of memory)
     *   }
     */
    virtual void* AllocateMemory(size_t sizeBytes) = 0;
    
    /**
     * Free GPU memory.
     * 
     * @param ptr  Pointer returned by AllocateMemory()
     * 
     * After calling this, 'ptr' is invalid and must not be used!
     * 
     * IMPORTANT: Must call Synchronize() first to ensure GPU is not
     *            still using this memory!
     */
    virtual void FreeMemory(void* ptr) = 0;
    
    /**
     * Copy data from CPU (host) to GPU (device).
     * 
     * @param devicePtr  GPU memory pointer (from AllocateMemory)
     * @param hostPtr    CPU memory pointer (regular C++ pointer)
     * @param sizeBytes  Number of bytes to copy
     * 
     * This is a blocking operation - waits until transfer completes.
     * 
     * Visual representation:
     *   CPU RAM              GPU VRAM
     *   [1,2,3,4]  ------->  [ , , , ]
     *   hostPtr              devicePtr
     * 
     * Example:
     *   float hostData[1024];
     *   // ... initialize hostData ...
     *   void* deviceData = backend->AllocateMemory(1024 * sizeof(float));
     *   backend->CopyHostToDevice(deviceData, hostData, 1024 * sizeof(float));
     */
    virtual void CopyHostToDevice(void* devicePtr, const void* hostPtr, size_t sizeBytes) = 0;
    
    /**
     * Copy data from GPU (device) to CPU (host).
     * 
     * @param hostPtr    CPU memory pointer (regular C++ pointer)
     * @param devicePtr  GPU memory pointer (from AllocateMemory)
     * @param sizeBytes  Number of bytes to copy
     * 
     * This is a blocking operation - waits until transfer completes.
     * 
     * Visual representation:
     *   CPU RAM              GPU VRAM
     *   [ , , , ]  <-------  [2,4,6,8]
     *   hostPtr              devicePtr
     * 
     * Example:
     *   float results[1024];
     *   backend->CopyDeviceToHost(results, deviceData, 1024 * sizeof(float));
     *   // Now results[] contains GPU computation output
     */
    virtual void CopyDeviceToHost(void* hostPtr, const void* devicePtr, size_t sizeBytes) = 0;
    
    /**************************************************************************
     * KERNEL EXECUTION
     *************************************************************************/
    
    /**
     * Execute a GPU kernel.
     * 
     * @param kernelName  Name of the kernel to execute (e.g., "vector_add")
     * @param params      Kernel parameters (buffers, sizes, etc.)
     * @return            true if kernel launched successfully
     * 
     * This function launches work on the GPU. The GPU executes asynchronously,
     * so this function returns immediately (non-blocking).
     * 
     * To wait for completion, call Synchronize().
     * 
     * KERNEL NAMING:
     *   - "vector_add"     : Element-wise vector addition
     *   - "matrix_mul"     : Matrix multiplication
     *   - "convolution"    : 2D image convolution
     *   - "reduction"      : Parallel sum reduction
     * 
     * Each backend translates these names to its own kernel implementation.
     * 
     * Example:
     *   KernelParams params;
     *   params.input1 = deviceInputA;
     *   params.input2 = deviceInputB;
     *   params.output = deviceOutput;
     *   params.numElements = 1024;
     *   
     *   backend->ExecuteKernel("vector_add", params);
     *   backend->Synchronize();  // Wait for completion
     */
    virtual bool ExecuteKernel(const std::string& kernelName, const KernelParams& params) = 0;
    
    /**
     * Wait for all GPU operations to complete.
     * 
     * GPU operations are asynchronous - they run in the background while
     * the CPU continues. This function blocks until ALL pending GPU
     * operations are complete.
     * 
     * WHEN TO CALL:
     *   - Before reading results (CopyDeviceToHost)
     *   - Before timing measurements (to get accurate execution time)
     *   - Before freeing memory (to ensure GPU is done using it)
     * 
     * CUDA:          cudaDeviceSynchronize()
     * OpenCL:        clFinish()
     * DirectCompute: Context->Flush() + Query
     */
    virtual void Synchronize() = 0;
    
    /**************************************************************************
     * TIMING
     * 
     * Accurate timing of GPU operations requires special APIs because
     * GPUs execute asynchronously from the CPU.
     *************************************************************************/
    
    /**
     * Start GPU timer.
     * 
     * Records a timestamp on the GPU timeline.
     * Must be called before ExecuteKernel() for timing.
     * 
     * CUDA:          cudaEventRecord(startEvent)
     * OpenCL:        Uses cl_event with profiling
     * DirectCompute: ID3D11Query with timestamp
     */
    virtual void StartTimer() = 0;
    
    /**
     * Stop GPU timer.
     * 
     * Records a timestamp on the GPU timeline.
     * Must be called after ExecuteKernel() for timing.
     * 
     * Call GetElapsedTime() to retrieve the measured time.
     */
    virtual void StopTimer() = 0;
    
    /**
     * Get elapsed time measured by GPU timer.
     * 
     * @return Time in milliseconds between StartTimer() and StopTimer()
     * 
     * Must call Synchronize() first to ensure timer events are complete!
     * 
     * Example:
     *   backend->StartTimer();
     *   backend->ExecuteKernel("vector_add", params);
     *   backend->StopTimer();
     *   backend->Synchronize();
     *   double timeMS = backend->GetElapsedTime();
     *   std::cout << "Kernel took " << timeMS << " ms" << std::endl;
     */
    virtual double GetElapsedTime() = 0;
    
    /**************************************************************************
     * UTILITY FUNCTIONS
     *************************************************************************/
    
    /**
     * Check if the backend is available on this system.
     * 
     * @return true if backend can be initialized
     * 
     * This is called before Initialize() to check if the required
     * hardware and drivers are present.
     * 
     * Examples:
     *   - CUDA: Returns false on AMD GPU
     *   - OpenCL: Returns false if no OpenCL runtime installed
     *   - DirectCompute: Returns false if DirectX < 11
     */
    virtual bool IsAvailable() const = 0;
    
    /**
     * Get last error message.
     * 
     * @return Human-readable error description
     * 
     * When a function returns false or fails, this provides details.
     */
    virtual std::string GetLastError() const = 0;
};

} // namespace GPUBenchmark

#endif // ICOMPUTE_BACKEND_H

/*******************************************************************************
 * END OF FILE: IComputeBackend.h
 * 
 * WHAT WE LEARNED:
 *   1. Abstract interfaces provide uniform access to different implementations
 *   2. Pure virtual functions (= 0) must be implemented by derived classes
 *   3. Virtual destructors are essential for polymorphism
 *   4. GPU memory is separate from CPU memory
 *   5. GPU operations are asynchronous
 *   6. Timing GPU operations requires special APIs
 * 
 * NEXT FILES TO READ:
 *   - CUDABackend.h/cpp   : CUDA implementation of this interface
 *   - OpenCLBackend.h/cpp : OpenCL implementation of this interface
 *   - DirectComputeBackend.h/cpp : DirectCompute implementation
 * 
 ******************************************************************************/
