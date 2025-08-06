#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>

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

template <int byte_stride = BYTE_STRIDE, int instr = INSTR_WIDTH>
__global__ void kernel(float* reg)
{
    __shared__ float shared[1024];

    int tx = (threadIdx.x * 8) % std::size(shared);

    float temp[64];
#pragma unroll
    for(int i = 0; i < 64; ++i)
    {
        temp[i] = shared[(tx * 64) + i];
    }

    // reg[0] = shared[tx];
    // reg[1] = shared[tx + 1];

    // printf("%d -> %d\n", threadIdx.x, tx);

    __syncthreads();

#pragma unroll
    for(int i = 0; i < 64; ++i)
    {
        reg[i] = temp[i];
    }
}

int main()
{
    float* d_a;
    assert(hipMalloc(&d_a, 1024) == hipSuccess);

    dim3 threads(64);
    dim3 blocks(1024);
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, d_a);

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}