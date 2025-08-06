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
__global__ void kernel(int* reg)
{
    __shared__ int shared[1024];

    int tx = (threadIdx.x * 8) % std::size(shared);

    int temp[64];
#pragma unroll
    for(int i = 0; i < 64; ++i)
    {
        // temp[i] = shared[(tx * 64) + i];
        asm volatile("ds_read_b32 %0, %1"
                     : "=v"(temp[i])
                     : "v"(uint32_t(uint64_t(shared)) + (tx * 64) + i));
    }

    __syncthreads();

    // Ensure not optimized away
#pragma unroll
    for(int i = 0; i < 64; ++i)
    {
        reg[i] = temp[i];
    }
}

int main()
{
    int* d_a;
    assert(hipMalloc(&d_a, 1024) == hipSuccess);

    dim3 threads(64);
    dim3 blocks(1024);
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, d_a);

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}