/********************************************************************************
 * @file    OpenCLBackend.cpp
 * @brief   OpenCL Backend Implementation
 * 
 * @details Complete implementation of OpenCL backend for cross-vendor GPU compute.
 *          Supports NVIDIA, AMD, and Intel GPUs through OpenCL 1.2 API.
 * 
 * @author  GPU-Benchmark Development Team
 * @date    2026-01-09
 ********************************************************************************/

#include "OpenCLBackend.h"
#include <iostream>
#include <sstream>
#include <cstring>

namespace GPUBenchmark {

//==============================================================================
// CONSTRUCTOR & DESTRUCTOR
//==============================================================================

OpenCLBackend::OpenCLBackend()
    : m_platform(nullptr)
    , m_device(nullptr)
    , m_context(nullptr)
    , m_commandQueue(nullptr)
    , m_startEvent(nullptr)
    , m_stopEvent(nullptr)
    , m_timingActive(false)
    , m_accumulatedTime(0.0)
    , m_initialized(false)
    , m_lastError("")
    , m_logger(Logger::GetInstance())
{
    m_logger.Info("[OpenCL] Backend created");
}

OpenCLBackend::~OpenCLBackend() {
    if (m_initialized) {
        Shutdown();
    }
}

//==============================================================================
// INITIALIZATION & SHUTDOWN
//==============================================================================

bool OpenCLBackend::Initialize() {
    m_logger.Info("[OpenCL] Initializing OpenCL backend...");
    
    // Step 1: Select best platform
    if (!SelectBestPlatform()) {
        m_logger.Error("[OpenCL] Failed to select OpenCL platform");
        return false;
    }
    
    // Step 2: Select best device
    if (!SelectBestDevice()) {
        m_logger.Error("[OpenCL] Failed to select OpenCL device");
        return false;
    }
    
    // Step 3: Create context
    cl_int err;
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    if (!CheckCLError(err, "clCreateContext")) {
        return false;
    }
    
    // Step 4: Create command queue (out-of-order, profiling enabled)
    m_commandQueue = clCreateCommandQueue(m_context, m_device, 
                                          CL_QUEUE_PROFILING_ENABLE, &err);
    if (!CheckCLError(err, "clCreateCommandQueue")) {
        clReleaseContext(m_context);
        return false;
    }
    
    // Step 5: Query device info
    QueryDeviceInfo();
    
    // Step 6: Create timing events
    m_startEvent = clCreateUserEvent(m_context, &err);
    if (!CheckCLError(err, "clCreateUserEvent (start)")) {
        clReleaseCommandQueue(m_commandQueue);
        clReleaseContext(m_context);
        return false;
    }
    
    m_stopEvent = clCreateUserEvent(m_context, &err);
    if (!CheckCLError(err, "clCreateUserEvent (stop)")) {
        clReleaseEvent(m_startEvent);
        clReleaseCommandQueue(m_commandQueue);
        clReleaseContext(m_context);
        return false;
    }
    
    m_initialized = true;
    m_logger.Info("[OpenCL] Initialization complete");
    m_logger.Info("[OpenCL] Device: " + m_deviceInfo.name);
    m_logger.Info("[OpenCL] Global Memory: " + std::to_string(m_deviceInfo.totalMemoryBytes / (1024*1024)) + " MB");
    
    return true;
}

void OpenCLBackend::Shutdown() {
    if (!m_initialized) return;
    
    m_logger.Info("[OpenCL] Shutting down...");
    
    // Release kernels
    for (auto& pair : m_kernels) {
        if (pair.second) {
            clReleaseKernel(pair.second);
        }
    }
    m_kernels.clear();
    
    // Release programs
    for (auto& pair : m_programs) {
        if (pair.second) {
            clReleaseProgram(pair.second);
        }
    }
    m_programs.clear();
    
    // Release timing events
    if (m_startEvent) {
        clReleaseEvent(m_startEvent);
        m_startEvent = nullptr;
    }
    if (m_stopEvent) {
        clReleaseEvent(m_stopEvent);
        m_stopEvent = nullptr;
    }
    
    // Release command queue
    if (m_commandQueue) {
        clReleaseCommandQueue(m_commandQueue);
        m_commandQueue = nullptr;
    }
    
    // Release context
    if (m_context) {
        clReleaseContext(m_context);
        m_context = nullptr;
    }
    
    m_initialized = false;
    m_logger.Info("[OpenCL] Shutdown complete");
}

//==============================================================================
// MEMORY MANAGEMENT
//==============================================================================

void* OpenCLBackend::AllocateMemory(size_t sizeBytes) {
    if (!m_initialized) {
        m_logger.Error("[OpenCL] Cannot allocate memory - backend not initialized");
        return nullptr;
    }
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sizeBytes, nullptr, &err);
    
    if (!CheckCLError(err, "clCreateBuffer")) {
        return nullptr;
    }
    
    return (void*)buffer;
}

void OpenCLBackend::FreeMemory(void* ptr) {
    if (!ptr) return;
    
    cl_mem buffer = (cl_mem)ptr;
    cl_int err = clReleaseMemObject(buffer);
    CheckCLError(err, "clReleaseMemObject");
}

void OpenCLBackend::CopyHostToDevice(void* dst, const void* src, size_t sizeBytes) {
    if (!m_initialized) {
        m_logger.Error("[OpenCL] Cannot copy to device - backend not initialized");
        return;
    }
    
    cl_mem buffer = (cl_mem)dst;
    cl_int err = clEnqueueWriteBuffer(m_commandQueue, buffer, CL_TRUE, 
                                       0, sizeBytes, src, 0, nullptr, nullptr);
    CheckCLError(err, "clEnqueueWriteBuffer");
}

void OpenCLBackend::CopyDeviceToHost(void* dst, const void* src, size_t sizeBytes) {
    if (!m_initialized) {
        m_logger.Error("[OpenCL] Cannot copy from device - backend not initialized");
        return;
    }
    
    cl_mem buffer = (cl_mem)src;
    cl_int err = clEnqueueReadBuffer(m_commandQueue, buffer, CL_TRUE,
                                      0, sizeBytes, dst, 0, nullptr, nullptr);
    CheckCLError(err, "clEnqueueReadBuffer");
}

//==============================================================================
// SYNCHRONIZATION & TIMING
//==============================================================================

void OpenCLBackend::Synchronize() {
    if (!m_initialized) return;
    
    cl_int err = clFinish(m_commandQueue);
    CheckCLError(err, "clFinish");
}

void OpenCLBackend::StartTimer() {
    if (!m_initialized) return;
    
    // Release previous events if they exist
    if (m_startEvent && m_timingActive) {
        clReleaseEvent(m_startEvent);
    }
    if (m_stopEvent && m_timingActive) {
        clReleaseEvent(m_stopEvent);
    }
    
    // Create new timing events
    cl_int err;
    m_startEvent = clCreateUserEvent(m_context, &err);
    CheckCLError(err, "clCreateUserEvent (start timer)");
    
    // Set event to completed state (marker)
    clSetUserEventStatus(m_startEvent, CL_COMPLETE);
    
    m_timingActive = true;
    m_accumulatedTime = 0.0;  // Reset accumulated time
}

void OpenCLBackend::StopTimer() {
    if (!m_initialized || !m_timingActive) return;
    
    // Flush to ensure all commands are submitted
    clFlush(m_commandQueue);
}

double OpenCLBackend::GetElapsedTime() {
    if (!m_initialized || !m_timingActive) {
        return 0.0;
    }
    
    // Wait for queue to complete
    Synchronize();
    
    // Return accumulated time from all kernel executions
    return m_accumulatedTime;
}

bool OpenCLBackend::ExecuteKernel(const std::string& kernelName, const KernelParams& params) {
    // This is a wrapper to satisfy the IComputeBackend interface
    // OpenCL kernels are executed differently (arguments set separately)
    m_lastError = "ExecuteKernel(KernelParams) not implemented for OpenCL - use CompileKernel + SetKernelArg + ExecuteKernel";
    return false;
}

bool OpenCLBackend::IsAvailable() const {
    return m_initialized;
}

std::string OpenCLBackend::GetLastError() const {
    return m_lastError;
}

//==============================================================================
// OPENCL KERNEL MANAGEMENT
//==============================================================================

bool OpenCLBackend::CompileKernel(const std::string& kernelName,
                                   const std::string& sourceCode,
                                   const std::string& buildOptions) {
    if (!m_initialized) {
        m_logger.Error("[OpenCL] Cannot compile kernel - backend not initialized");
        return false;
    }
    
    // Check if kernel already compiled
    if (m_kernels.find(kernelName) != m_kernels.end()) {
        m_logger.Info("[OpenCL] Kernel '" + kernelName + "' already compiled");
        return true;
    }
    
    m_logger.Info("[OpenCL] Compiling kernel: " + kernelName);
    
    // Create program from source
    const char* sourcePtr = sourceCode.c_str();
    size_t sourceLen = sourceCode.length();
    
    cl_int err;
    cl_program program = clCreateProgramWithSource(m_context, 1, &sourcePtr, &sourceLen, &err);
    if (!CheckCLError(err, "clCreateProgramWithSource")) {
        return false;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &m_device, buildOptions.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        
        m_logger.Error("[OpenCL] Kernel compilation failed:");
        m_logger.Error(std::string(log.data()));
        
        clReleaseProgram(program);
        return false;
    }
    
    // Create kernel object
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
    if (!CheckCLError(err, "clCreateKernel")) {
        clReleaseProgram(program);
        return false;
    }
    
    // Cache program and kernel
    m_programs[kernelName] = program;
    m_kernels[kernelName] = kernel;
    
    m_logger.Info("[OpenCL] Kernel '" + kernelName + "' compiled successfully");
    return true;
}

bool OpenCLBackend::SetKernelArg(const std::string& kernelName,
                                  unsigned int argIndex,
                                  size_t argSize,
                                  const void* argValue) {
    if (!m_initialized) {
        m_logger.Error("[OpenCL] Cannot set kernel arg - backend not initialized");
        return false;
    }
    
    // Find kernel
    auto it = m_kernels.find(kernelName);
    if (it == m_kernels.end()) {
        m_logger.Error("[OpenCL] Kernel '" + kernelName + "' not found");
        return false;
    }
    
    cl_kernel kernel = it->second;
    cl_int err = clSetKernelArg(kernel, argIndex, argSize, argValue);
    
    return CheckCLError(err, "clSetKernelArg");
}

bool OpenCLBackend::ExecuteKernel(const std::string& kernelName,
                                   const size_t* globalWorkSize,
                                   const size_t* localWorkSize,
                                   size_t workDim) {
    if (!m_initialized) {
        m_logger.Error("[OpenCL] Cannot execute kernel - backend not initialized");
        return false;
    }
    
    // Find kernel
    auto it = m_kernels.find(kernelName);
    if (it == m_kernels.end()) {
        m_logger.Error("[OpenCL] Kernel '" + kernelName + "' not found");
        return false;
    }
    
    cl_kernel kernel = it->second;
    
    // Execute kernel
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(m_commandQueue, kernel, workDim,
                                        nullptr, globalWorkSize, localWorkSize,
                                        0, nullptr, &event);
    
    if (!CheckCLError(err, "clEnqueueNDRangeKernel")) {
        return false;
    }
    
    // Wait for completion
    clWaitForEvents(1, &event);

    // If timing is active, get execution time from event
    if (m_timingActive) {
        cl_ulong timeStart, timeEnd;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timeStart, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timeEnd, nullptr);
        
        // Convert nanoseconds to milliseconds and accumulate
        double executionTimeMS = (timeEnd - timeStart) / 1000000.0;
        m_accumulatedTime += executionTimeMS;
    }

    // Clean up event
    clReleaseEvent(event);

    return true;
}

//==============================================================================
// PRIVATE HELPER METHODS
//==============================================================================

bool OpenCLBackend::SelectBestPlatform() {
    // Get number of platforms
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (!CheckCLError(err, "clGetPlatformIDs (count)")) {
        return false;
    }
    
    if (numPlatforms == 0) {
        m_logger.Error("[OpenCL] No OpenCL platforms found");
        return false;
    }
    
    // Get platform IDs
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (!CheckCLError(err, "clGetPlatformIDs")) {
        return false;
    }
    
    m_logger.Info("[OpenCL] Found " + std::to_string(numPlatforms) + " platform(s)");
    
    // Select first platform with GPU devices
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        
        if (err == CL_SUCCESS && numDevices > 0) {
            m_platform = platforms[i];
            
            // Get platform name
            char platformName[256];
            clGetPlatformInfo(m_platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
            m_logger.Info("[OpenCL] Selected platform: " + std::string(platformName));
            
            return true;
        }
    }
    
    m_logger.Error("[OpenCL] No GPU devices found on any platform");
    return false;
}

bool OpenCLBackend::SelectBestDevice() {
    // Get number of devices
    cl_uint numDevices;
    cl_int err = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (!CheckCLError(err, "clGetDeviceIDs (count)")) {
        return false;
    }
    
    if (numDevices == 0) {
        m_logger.Error("[OpenCL] No GPU devices found");
        return false;
    }
    
    // Get device IDs
    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    if (!CheckCLError(err, "clGetDeviceIDs")) {
        return false;
    }
    
    m_logger.Info("[OpenCL] Found " + std::to_string(numDevices) + " GPU device(s)");
    
    // Select device with most compute units (simple heuristic)
    cl_uint maxComputeUnits = 0;
    for (cl_uint i = 0; i < numDevices; ++i) {
        cl_uint computeUnits;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
        
        if (computeUnits > maxComputeUnits) {
            maxComputeUnits = computeUnits;
            m_device = devices[i];
        }
    }
    
    return true;
}

void OpenCLBackend::QueryDeviceInfo() {
    // Device name
    char deviceName[256];
    clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    m_deviceInfo.name = std::string(deviceName);
    
    // Global memory (bytes)
    cl_ulong globalMem;
    clGetDeviceInfo(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, nullptr);
    m_deviceInfo.totalMemoryBytes = static_cast<size_t>(globalMem);
    m_deviceInfo.availableMemoryBytes = static_cast<size_t>(globalMem);  // Approximate
    
    // Max work-group size
    size_t maxWorkGroupSize;
    clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    m_deviceInfo.maxThreadsPerBlock = static_cast<int>(maxWorkGroupSize);
    
    // Max work-item dimensions
    size_t maxWorkItemSizes[3];
    clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, nullptr);
    m_deviceInfo.maxBlockDimX = static_cast<int>(maxWorkItemSizes[0]);
    m_deviceInfo.maxBlockDimY = static_cast<int>(maxWorkItemSizes[1]);
    m_deviceInfo.maxBlockDimZ = static_cast<int>(maxWorkItemSizes[2]);
    
    // Driver version
    char driverVersion[256];
    clGetDeviceInfo(m_device, CL_DRIVER_VERSION, sizeof(driverVersion), driverVersion, nullptr);
    m_deviceInfo.driverVersion = std::string(driverVersion);
    
    // Set compute capability to 0 (OpenCL doesn't have this concept)
    m_deviceInfo.computeCapabilityMajor = 0;
    m_deviceInfo.computeCapabilityMinor = 0;
}

bool OpenCLBackend::CheckCLError(cl_int err, const std::string& operation) {
    if (err == CL_SUCCESS) {
        return true;
    }
    
    std::string errorMsg = "[OpenCL] " + operation + " failed: " + GetCLErrorString(err);
    m_logger.Error(errorMsg);
    return false;
}

std::string OpenCLBackend::GetCLErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling info not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid MIP level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        default: return "Unknown error (" + std::to_string(err) + ")";
    }
}

} // namespace GPUBenchmark

/********************************************************************************
 * END OF FILE: OpenCLBackend.cpp
 * 
 * SUMMARY:
 * - Complete OpenCL backend implementation
 * - Platform/device enumeration and selection
 * - Runtime kernel compilation
 * - Memory management
 * - Execution and timing
 * 
 * NEXT STEPS:
 * - Create OpenCL kernel files (.cl)
 * - Port CUDA kernels to OpenCL C
 * - Test on NVIDIA GPU
 * - (Optional) Test on AMD/Intel GPUs
 ********************************************************************************/
