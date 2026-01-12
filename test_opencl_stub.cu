// Stub file to force test_opencl_backend to link with CUDA runtime
// This is needed because the test includes BenchmarkRunner which uses CUDA

__global__ void stub_kernel() {}

void stub_function() {}
