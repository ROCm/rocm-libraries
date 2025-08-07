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

    for(int i = 0; i < 16; ++i)
    {
        asm volatile("ds_read_b32 v1, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v2, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v3, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v4, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v5, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v6, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v7, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v8, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v9, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v10, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v11, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v12, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v13, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v14, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v15, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v16, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v17, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v18, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v19, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v20, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v21, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v22, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v23, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v24, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v25, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v26, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v27, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v28, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v29, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v30, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v31, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v32, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v33, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v34, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v35, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v36, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v37, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v38, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v39, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v40, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v41, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v42, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v43, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v44, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v45, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v46, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v47, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v48, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v49, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v50, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v51, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v52, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v53, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v54, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v55, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v56, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v57, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v58, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v59, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v60, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v61, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v62, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v63, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
        asm volatile("ds_read_b32 v64, %0 offset:0"
                     :
                     : "v"(uint32_t(uint64_t(shared)) + threadIdx.x * stride));
    }

    // clock_t end = clock();

    // if(blockIdx.x == 0)
    // {
    //     clocks[threadIdx.x] = int(end) - int(start);
    // }
}

int main()
{
    const int    N    = 256;
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