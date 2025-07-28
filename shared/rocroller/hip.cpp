#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

// Error-checking macro
#define HIP_CHECK(cmd)                                                          \
    {                                                                           \
        hipError_t e = cmd;                                                     \
        if(e != hipSuccess)                                                     \
        {                                                                       \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(e) << " (" << e << ")\n";            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    }

#define N 1024
#define BLOCK_SIZE 256

#ifndef STRIDE
#define STRIDE 0
#endif

template <int stride = STRIDE>
__global__ void kernel()
{
    uint32_t const index = stride * threadIdx.x;
    uint32_t       r;
#pragma unroll
    for(int i = 0; i < 32; ++i)
    {
        asm volatile("ds_read_b128 v[2:5], %0" : : "v"(index));
    }
}

int main()
{
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE);
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0);

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}
