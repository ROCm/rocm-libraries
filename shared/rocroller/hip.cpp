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
__global__ void kernel(int* clocks)
{
    __shared__ uint32_t shared[16128];
    const int           stride = 4 * 2;

    // clock_t start = clock();

    const int ITERS = 64;

    int temp[ITERS];

    for(int i = 0; i < ITERS; ++i)
    {
        temp[i] = uint32_t(uint64_t(shared)) + threadIdx.x * 4 * 32;
    }

#pragma unroll
    for(int i = 0; i < ITERS; ++i)
    {
        asm volatile("ds_read_b32 %0, %1 offset:0" : "=v"(temp[i]) : "v"(temp[i]));
        // temp[i] = shared[threadIdx.x];
    }

    __syncthreads();

    // clock_t end = clock();

    // if(blockIdx.x == 0)
    // {
    //     clocks[threadIdx.x] = int(end) - int(start);
    // }

    for(int i = 0; i < ITERS; ++i)
    {
        clocks[i] = temp[i];
    }
}

int main()
{
    const int    N    = 64;
    const size_t size = N * sizeof(int);
    int          h_a[N];

    int* d_a;
    assert(hipMalloc(&d_a, size) == hipSuccess);
    assert(hipMemcpy(d_a, h_a, size, hipMemcpyDefault) == hipSuccess);

    dim3 threads(N);
    dim3 blocks(256);
    hipLaunchKernelGGL(kernel, blocks, threads, 0, 0, d_a);

    assert(hipMemcpy(h_a, d_a, size, hipMemcpyDefault) == hipSuccess);

    std::cout << "clock delta: ";
    for(int i = 0; i < N; ++i)
    {
        std::cout << h_a[i] << ", ";
    }
    std::cout << std::endl;

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}