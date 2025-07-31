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
__global__ void kernel()
{
    uint32_t const index = byte_stride * threadIdx.x;

    for(int i = 0; i < 1000; ++i)
    {
#pragma unroll
        for(int i = 0; i < 32; ++i)
        {
            if constexpr(instr == 32)
                asm volatile("ds_read_b32 v2, %0" : : "v"(index));
            if constexpr(instr == 128)
                asm volatile("ds_read_b128 v[2:5], %0" : : "v"(index));
            if constexpr(false)
            {
                asm volatile("s_waitcnt lgkmcnt(0)");
            }
        }
    }
}

int main()
{
    dim3 threads(256);
    dim3 blocks(1);
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0);

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}
