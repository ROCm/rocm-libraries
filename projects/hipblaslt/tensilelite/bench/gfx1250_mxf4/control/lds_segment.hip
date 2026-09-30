// Probe for the gfx1250 cross-port LDS segment conflict counter.
//
// Each wave issues bank-conflict-free ds_load_b128 reads from a fixed 512B window.
// Waves with (waveId & splitMask) != 0 read at baseB, the rest at baseA.
// One dispatch per row of kCases; profile with configs/rocprof_lds_segment.yaml and join
// the counters with this program's stdout using control/seg_summary.py.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

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

// 256 KB keeps one workgroup per WGP (320 KB LDS).
constexpr int kLdsBytes = 256 * 1024;
constexpr int kBlock    = 256; // default; argv[3] overrides
constexpr int KB        = 1024;

__global__ void lds_segment(float4* out, int splitMask, int baseA, int baseB, int iters)
{
    extern __shared__ float4 lds[];
    for(int i = threadIdx.x; i < kLdsBytes / 16; i += blockDim.x)
        lds[i] = make_float4(i, i, i, i);
    __syncthreads();

    int wave = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    int base = ((wave & splitMask) ? baseB : baseA) + lane * 16;
    float4 acc = make_float4(0, 0, 0, 0);
    for(int i = 0; i < iters; ++i)
    {
        int addr = base;
        asm volatile("" : "+v"(addr));
        float4 v = lds[addr / 16];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

struct Case
{
    int mask, baseA, baseB;
};

static const Case kCases[] = {
    // Same-address baselines; the first dispatch has read low before.
    {0, 0, 0}, {0, 0, 0}, {1, 0, 0},
    // Distance sweep from 0 across the suspected 64 KB boundary.
    {1, 0, 32 * KB}, {1, 0, 48 * KB}, {1, 0, 56 * KB}, {1, 0, 60 * KB}, {1, 0, 63 * KB},
    {1, 0, 64 * KB}, {1, 0, 72 * KB}, {1, 0, 96 * KB}, {1, 0, 128 * KB}, {1, 0, 160 * KB},
    {1, 0, 192 * KB}, {1, 0, 224 * KB},
    // Aligned segments or distance: 8 KB apart across the 64 KB line, 64 KB apart off it.
    {1, 60 * KB, 68 * KB}, {1, 32 * KB, 96 * KB}, {1, 16 * KB, 80 * KB}, {1, 32 * KB, 64 * KB},
    {1, 64 * KB, 128 * KB}, {1, 128 * KB, 192 * KB}, {1, 64 * KB, 192 * KB},
    // Wave pairing: which split masks separate the two ports.
    {2, 0, 64 * KB}, {3, 0, 64 * KB}, {4, 0, 64 * KB}, {6, 0, 64 * KB},
    {0, 0, 0},
};

int main(int argc, char** argv)
{
    int blocks  = argc > 1 ? std::atoi(argv[1]) : 256;
    int iters   = argc > 2 ? std::atoi(argv[2]) : 16384;
    int threads = argc > 3 ? std::atoi(argv[3]) : kBlock;

    float4* out = nullptr;
    HIP_CHECK(hipMalloc(&out, size_t(blocks) * threads * sizeof(float4)));
    HIP_CHECK(hipFuncSetAttribute((const void*)lds_segment,
                                  hipFuncAttributeMaxDynamicSharedMemorySize, kLdsBytes));

    std::printf("dispatch,splitMask,baseA_KB,baseB_KB\n");
    int dispatch = 0;
    for(const Case& c : kCases)
    {
        hipLaunchKernelGGL(lds_segment, dim3(blocks), dim3(threads), kLdsBytes, 0,
                           out, c.mask, c.baseA, c.baseB, iters);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::printf("%d,%d,%d,%d\n", dispatch++, c.mask, c.baseA / KB, c.baseB / KB);
    }
    HIP_CHECK(hipFree(out));
    return 0;
}
