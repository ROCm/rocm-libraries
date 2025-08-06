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
    const int        BN = 64;
    const int        BK = 8;
    const int        TN = 4;
    __shared__ float shared[BK][BN];

    int tx = (threadIdx.x % (BN / TN)) * TN;

    reg[0] = shared[0][tx];

    // reg[i]     = shared[0][tx];
    // reg[i + 1] = shared[0][tx + 1];
}

int main()
{
    float* d_a;
    assert(hipMalloc(&d_a, 1024) == hipSuccess);

    dim3 threads(256);
    dim3 blocks(256);
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, d_a);

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}