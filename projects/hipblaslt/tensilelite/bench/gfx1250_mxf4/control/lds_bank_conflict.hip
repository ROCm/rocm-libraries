// Positive control for the gfx1250 LDS conflict counters.
//
// lds_conflict:   every lane of a wave reads a dword 256 dwords apart, so all
//                 lanes land in the same bank on every ds_read.
// lds_noconflict: lanes read consecutive dwords, one bank each.
// Both kernels also read global memory, so TX_PERF_SEL_VMW_REQ must be nonzero.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                                   \
    do                                                                                 \
    {                                                                                  \
        hipError_t e_ = (x);                                                           \
        if(e_ != hipSuccess)                                                           \
        {                                                                              \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                              \
        }                                                                              \
    } while(0)

constexpr int kLdsWords = 32 * 256;
constexpr int kBlock    = 256;

__device__ __forceinline__ float lds_loop(const float* in, int stride, int iters)
{
    __shared__ float lds[kLdsWords];
    for(int i = threadIdx.x; i < kLdsWords; i += blockDim.x)
        lds[i] = in[i];
    __syncthreads();

    int   idx = ((threadIdx.x & 31) * stride) & (kLdsWords - 1);
    float acc = 0.f;
    for(int i = 0; i < iters; ++i)
    {
        // Keeps each read a separate ds_read_b32 instead of a merged wide load.
        asm volatile("" : "+v"(idx));
        acc += lds[idx];
        idx = (idx + 1) & (kLdsWords - 1);
    }
    return acc;
}

__global__ void lds_conflict(const float* in, float* out, int iters)
{
    out[blockIdx.x * blockDim.x + threadIdx.x] = lds_loop(in, 256, iters);
}

__global__ void lds_noconflict(const float* in, float* out, int iters)
{
    out[blockIdx.x * blockDim.x + threadIdx.x] = lds_loop(in, 1, iters);
}

template <typename K>
static float time_kernel(K kernel, const char* name, const float* in, float* out, int blocks, int iters)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(kBlock), 0, 0, in, out, iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("%-16s %8.3f ms\n", name, ms);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
}

int main(int argc, char** argv)
{
    int blocks = argc > 1 ? std::atoi(argv[1]) : 1024;
    int iters  = argc > 2 ? std::atoi(argv[2]) : 4096;

    std::vector<float> host(kLdsWords, 1.f);
    float *in = nullptr, *out = nullptr;
    HIP_CHECK(hipMalloc(&in, kLdsWords * sizeof(float)));
    HIP_CHECK(hipMalloc(&out, size_t(blocks) * kBlock * sizeof(float)));
    HIP_CHECK(hipMemcpy(in, host.data(), kLdsWords * sizeof(float), hipMemcpyHostToDevice));

    std::printf("blocks=%d iters=%d\n", blocks, iters);
    float conflict   = time_kernel(lds_conflict, "lds_conflict", in, out, blocks, iters);
    float noconflict = time_kernel(lds_noconflict, "lds_noconflict", in, out, blocks, iters);
    std::printf("slowdown %.2fx\n", conflict / noconflict);

    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
    return 0;
}
