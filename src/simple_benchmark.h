/*******************************************************************************
 * FILE: simple_benchmark.h
 * 
 * PURPOSE:
 *   Simple backend-agnostic benchmark functions that work with all 3 backends
 *   (CUDA, OpenCL, DirectCompute) without calling CUDA kernels directly.
 * 
 * AUTHOR: Soham Dave
 * DATE: January 2026
 ******************************************************************************/

#ifndef SIMPLE_BENCHMARK_H
#define SIMPLE_BENCHMARK_H

#include "core/IComputeBackend.h"
#include "core/Logger.h"
#include <vector>

namespace GPUBenchmark {

// Simple vector addition benchmark that works with any backend
BenchmarkResult SimpleVectorAddBenchmark(IComputeBackend* backend, size_t numElements, int iterations = 100);

// Simple matrix multiplication benchmark
BenchmarkResult SimpleMatrixMulBenchmark(IComputeBackend* backend, int matrixSize, int iterations = 100);

} // namespace GPUBenchmark

#endif // SIMPLE_BENCHMARK_H
