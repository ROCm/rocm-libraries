// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hip/hip_runtime.h"
#include "cuda/std/atomic"

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    if (command != hipSuccess) {                                               \
      printf("HIP ERROR: %s", hipGetErrorString(command));                     \
      std::abort();                                                            \
    }                                                                          \
  }

#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

__device__ __shared__ cuda::atomic<int> block_min, block_max, block_add;

__device__ void blockOps(int *a, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    if (i % 128 == 0) {
      printf("\e[1;32m\t\t\tThread id(%03d) performing atomic add\e[0m\n", i);
      printf("\e[1;33m\t\t\tThread id(%03d) performing atomic min\e[0m\n", i);
      printf("\e[1;34m\t\t\tThread id(%03d) performing atomic max\e[0m\n", i);
    }

    block_min.fetch_min(a[i]);
    block_max.fetch_max(a[i]);
    block_add.fetch_add(a[i]);

    if (i % 128 == 0) {
      printf("\e[1;35m\t\t\tThread id(%03d) completed atomic add\e[0m\n", i);
      printf("\e[1;36m\t\t\tThread id(%03d) completed atomic min\e[0m\n", i);
      printf("\e[1;37m\t\t\tThread id(%03d) completed atomic max\e[0m\n", i);
    }
  }

  __syncthreads();
  if (i == 0) {
    printf("\e[1;35m\n\tCheapest              : %d k$\e[0m\n", block_min.load());
    printf("\e[1;35m\tMost expensive        : %d k$\e[0m\n", block_max.load());
    printf("\e[1;35m\tTotal price           : %d k$\e[0m\n", block_add.load());
  }
}

__global__ void init(int *a, int N) {
  block_min = 10000;
  block_max = 0;
  block_add = 0;
  blockOps(a, N);
}

int main(int argc, char *argv[]) {
  printf("\e[0;97m\n\n\tA program to test the functionality of LIBHIPCXX "
         "in the context of cuDF\e[0m\n\n");
  sleep(1); // for demonstration purposes

  int deviceID;
  hipDeviceProp_t props;
  int N = 1000;
  size_t NBytes = N * sizeof(int);
  int *h_a;
  h_a = (int *)malloc(N * sizeof(int));
  int *d_a = nullptr;
  HIP_CHECK(hipGetDevice(&deviceID));
  HIP_CHECK(hipGetDeviceProperties(&props, deviceID));

  printf("\e[1;34m\tInitializing the dataframe of house prices!\e[0m\n\n");
  sleep(5); // for demonstration purposes
  for (int i = 0; i < N; i++) {
    h_a[i] = i + 100;
  }

  printf("\e[0;33m\tCopying dataframe onto GPU ...\e[0m\n\n");
  sleep(5); // for demonstration purposes
  HIP_CHECK(hipMalloc(&d_a, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(d_a, h_a, NBytes, hipMemcpyHostToDevice));

  printf("\e[0;35m\tFinding cheapest, most expensive, and total sum of the "
         "house prices\n");
  printf("\e[0;35m\t\t(Operations are running on GPU %s)\e[0m\n", props.name);
  sleep(3); // for demonstration purposes
  dim3 block(N, 1, 1);
  dim3 grid(divideAndRoundUp(N, block.x), 1, 1);
  hipLaunchKernelGGL(init, grid, block, 0, 0, d_a, N);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(d_a));
  free(h_a);
  return 0;
}

