// Peak v_wmma_scale_f32_32x16x128_f4 throughput, shaped like the MT256x256x256 subtile GEMM:
// 4 waves per workgroup, one workgroup per WGP (LDS reservation), A reused across 8 accumulators.
// Prints achieved PFLOPS and the shader clock (clock64 against wall_clock64).
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

typedef float v16f __attribute__((ext_vector_type(16)));
typedef int   v16i __attribute__((ext_vector_type(16)));
typedef int   v8i  __attribute__((ext_vector_type(8)));

constexpr int    kThreads  = 128;
constexpr int    kAcc      = 8;
constexpr size_t kLdsBytes = 172032;
constexpr double kFlopsPerWmma = 2.0 * 32 * 16 * 128;

#define WMMA(acc, a, b, sa, sb)                                                          \
    asm volatile("v_wmma_scale_f32_32x16x128_f4 %0, %1, %2, %0, %3, %4 matrix_a_reuse"   \
                 : "+v"(acc)                                                             \
                 : "v"(a), "v"(b), "v"(sa), "v"(sb))

__global__ void __launch_bounds__(kThreads) __attribute__((amdgpu_waves_per_eu(1, 1)))
wmma_peak(float* out, unsigned long long* clk, int iters)
{
    extern __shared__ char lds[];
    lds[threadIdx.x] = 0;

    v16i a;
    v8i  b0, b1;
    for(int i = 0; i < 16; ++i)
        a[i] = threadIdx.x * 16 + i;
    for(int i = 0; i < 8; ++i)
    {
        b0[i] = threadIdx.x + i;
        b1[i] = threadIdx.x - i;
    }
    int sa = 0x7f7f7f7f, sb = 0x7f7f7f7f;
    v16f c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};

    unsigned long long t0 = clock64();
    unsigned long long r0 = wall_clock64();
    for(int it = 0; it < iters; ++it)
    {
        WMMA(c0, a, b0, sa, sb);
        WMMA(c1, a, b1, sa, sb);
        WMMA(c2, a, b0, sa, sb);
        WMMA(c3, a, b1, sa, sb);
        WMMA(c4, a, b0, sa, sb);
        WMMA(c5, a, b1, sa, sb);
        WMMA(c6, a, b0, sa, sb);
        WMMA(c7, a, b1, sa, sb);
    }
    unsigned long long t1 = clock64();
    unsigned long long r1 = wall_clock64();

    float s = lds[0];
    for(int i = 0; i < 16; ++i)
        s += c0[i] + c1[i] + c2[i] + c3[i] + c4[i] + c5[i] + c6[i] + c7[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        clk[0] = t1 - t0;
        clk[1] = r1 - r0;
    }
}

int main(int argc, char** argv)
{
    int blocks = argc > 1 ? std::atoi(argv[1]) : 256;
    int iters  = argc > 2 ? std::atoi(argv[2]) : 20000;

    float*              out = nullptr;
    unsigned long long* clk = nullptr;
    HIP_CHECK(hipMalloc(&out, size_t(blocks) * kThreads * sizeof(float)));
    HIP_CHECK(hipMalloc(&clk, 2 * sizeof(unsigned long long)));
    HIP_CHECK(hipFuncSetAttribute((const void*)wmma_peak,
                                  hipFuncAttributeMaxDynamicSharedMemorySize, kLdsBytes));

    hipEvent_t start, stop;
    int wallKHz = 0;
    HIP_CHECK(hipDeviceGetAttribute(&wallKHz, hipDeviceAttributeWallClockRate, 0));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for(int rep = 0; rep < 4; ++rep)
    {
        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(wmma_peak, dim3(blocks), dim3(kThreads), kLdsBytes, 0, out, clk, iters);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        unsigned long long hclk[2];
        HIP_CHECK(hipMemcpy(hclk, clk, sizeof(hclk), hipMemcpyDeviceToHost));

        double wmmas      = double(blocks) * (kThreads / 32) * iters * kAcc;
        double pflops     = wmmas * kFlopsPerWmma / (ms * 1e-3) / 1e15;
        double ghz        = double(hclk[0]) / (double(hclk[1]) / (wallKHz * 1e3)) / 1e9;
        double cycPerWmma = double(hclk[0]) / (double(iters) * kAcc);
        std::printf("rep %d: %.3f ms  %.2f PFLOPS  clock64 %.2f GHz  "
                    "%.2f cycles per WMMA per wave\n",
                    rep, ms, pflops, ghz, cycPerWmma);
    }
    HIP_CHECK(hipFree(out));
    HIP_CHECK(hipFree(clk));
    return 0;
}
