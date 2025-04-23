/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#if defined(__NVCC__)

#include <cuda_runtime.h>

// Datatypes
#define hipError_t                                         cudaError_t
#define hipEvent_t                                         cudaEvent_t
#define hipStream_t                                        cudaStream_t

// Enumerations
#define hipSuccess                                         cudaSuccess

// Functions
#define hipEventCreate                                     cudaEventCreate
#define hipEventDestroy                                    cudaEventDestroy
#define hipEventElapsedTime                                cudaEventElapsedTime
#define hipGetErrorString                                  cudaGetErrorString
#define hipEventRecord                                     cudaEventRecord
#define hipStreamCreate                                    cudaStreamCreate
#define hipStreamDestroy                                   cudaStreamDestroy
#define hipStreamSynchronize                               cudaStreamSynchronize

#else

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#endif

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <numeric>

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)


__global__ void EmptyKernel(){};

float calStdDev(const std::vector<float>& allDeltaMs, float mean)
{
  std::vector<float> diff(allDeltaMs.size());
  std::transform(allDeltaMs.begin(), allDeltaMs.end(), diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / allDeltaMs.size());
  return stdev;
}

int main(int argc, char **argv)
{
  int numIterations = (argc > 1 ? atoi(argv[1]) : 10);
  int gridSize      = (argc > 2 ? atoi(argv[2]) : 1);
  int blockSize     = (argc > 3 ? atoi(argv[3]) : 1);
  int numWarmups    = 3;
  printf("Running %d iterations <<<%d,%d>>>\n", numIterations, gridSize, blockSize);

  // Create events and stream
  hipEvent_t startEvent, stopEvent;
  HIP_CALL(hipEventCreate(&startEvent));
  HIP_CALL(hipEventCreate(&stopEvent));
  hipStream_t stream;
  HIP_CALL(hipStreamCreate(&stream));

  // Run untimed warmup iterations (to cache kernel code)
  for (int iteration = 0; iteration < numWarmups; iteration++)
  {
    EmptyKernel<<<gridSize, blockSize, 0, stream>>>();
  }
  HIP_CALL(hipStreamSynchronize(stream));
  std::vector<float> allGpuDeltaMsec(numIterations);
  std::vector<float> allCpuDeltaMsec(numIterations);

  // Launch empty kernel
  // NOTE: Timing is done per-iteration, instead of batching multiple iterations
  double cpuSum = 0.0;
  double gpuSum = 0.0;
  for (int iteration = 0; iteration < numIterations; iteration++)
  {
    // Start timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    HIP_CALL(hipEventRecord(startEvent, stream));

    // Launch kernel and wait for completion
    EmptyKernel<<<gridSize, blockSize, 0, stream>>>();
    HIP_CALL(hipEventRecord(stopEvent, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    // Collect timing info
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;
    float gpuDeltaMsec;
    HIP_CALL(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));

    // Report timing
    printf("Iteration %03d Kernel Launch Time (usec) %10.5f (CPU) %10.5f (GPU)\n", iteration, cpuDeltaMsec *1000.0, gpuDeltaMsec * 1000.0);
    allGpuDeltaMsec[iteration] = gpuDeltaMsec * 1000.0;
    allCpuDeltaMsec[iteration] = cpuDeltaMsec * 1000.0;
    cpuSum += cpuDeltaMsec * 1000.0;
    gpuSum += gpuDeltaMsec * 1000.0;
  }
  printf("\n");

  // Report averages
  double avgCpuUsec = cpuSum / numIterations;
  double avgGpuUsec = gpuSum / numIterations;
  auto   minCpuUsec = std::min_element(std::begin(allCpuDeltaMsec), std::end(allCpuDeltaMsec));
  auto   minGpuUsec = std::min_element(std::begin(allGpuDeltaMsec), std::end(allGpuDeltaMsec));
  auto   maxCpuUsec = std::max_element(std::begin(allCpuDeltaMsec), std::end(allCpuDeltaMsec));
  auto   maxGpuUsec = std::max_element(std::begin(allGpuDeltaMsec), std::end(allGpuDeltaMsec));
  auto   varCpuUsec = calStdDev(allCpuDeltaMsec, avgCpuUsec);
  auto   varGpuUsec = calStdDev(allGpuDeltaMsec, avgGpuUsec);

  printf("Average       Kernel Launch time (usec) %10.5f (CPU) %10.5f (GPU)\n", avgCpuUsec, avgGpuUsec);
  printf("Minimum       Kernel Launch time (usec) %10.5f (CPU) %10.5f (GPU)\n", *minCpuUsec, *minGpuUsec);
  printf("Maximum       Kernel Launch time (usec) %10.5f (CPU) %10.5f (GPU)\n", *maxCpuUsec, *maxGpuUsec);
  printf("Stddev        Kernel Launch time (usec) %10.5f (CPU) %10.5f (GPU)\n", varCpuUsec, varGpuUsec);
  // Cleanup events and stream
  HIP_CALL(hipStreamDestroy(stream));
  HIP_CALL(hipEventDestroy(startEvent));
  HIP_CALL(hipEventDestroy(stopEvent));

  return 0;
}
