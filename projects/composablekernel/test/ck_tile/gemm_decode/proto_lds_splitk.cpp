// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// THROWAWAY PROTOTYPE (design doc gemm_decode §15.J). Standalone HIP, no CK Tile.
// Question it answers: can an intra-block LDS-reduce K-split beat wvSplitKQ
// (7.85 us) at FP8 M=1 / N=7168 / K=7168 on gfx950? If yes -> worth a real CK
// Tile B1 impl; if no -> dispatch-gate the band and drop B1.
//
// Unified split-K (both modes in one kernel):
//   total K split = k_batch (grid.z, inter-block, atomic reduce)
//                 x k_waves (intra-block, across warps, LDS reduce)
//   k_waves=1            -> today's gemm_decode behavior
//   k_batch=1,k_waves>1  -> pure intra-block LDS split-K (the wvSplitKQ mode)
//   both>1               -> hybrid (LDS reduce per block, then atomic across z)
//
// Self-contained + dependency-free numerics: A/B are raw e4m3 bytes (NaN-free by
// construction), decoded by the same routine on host (reference) and device
// (kernel), and BF16 output is bit-truncation. Timing is byte-count-bound, so
// exact FP8 rounding is irrelevant to the verdict; the relerr check only needs to
// catch real indexing/reduce bugs.
//
// Build: test/ck_tile/gemm_decode/proto_build.sh   Run: ./build/bin/proto_lds_splitk
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#define HIP_CHECK(expr)                                                              \
    do                                                                               \
    {                                                                                \
        hipError_t _e = (expr);                                                      \
        if(_e != hipSuccess)                                                         \
        {                                                                            \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),   \
                         __FILE__, __LINE__);                                        \
            std::exit(1);                                                            \
        }                                                                            \
    } while(0)

// ---- OCP e4m3 (E4M3FN) decode: 1 sign / 4 exp (bias 7) / 3 mantissa ----
// Direct FP32-bit construction (no ldexpf): the library call dominates an
// otherwise memory-bound kernel. Numerically identical to the closed form
// (normal: (8+m)*2^(e-10); subnormal: m*2^-9), so host ref and device agree.
__host__ __device__ inline float e4m3_to_f(uint8_t x)
{
    uint32_t s = static_cast<uint32_t>(x & 0x80u) << 24; // sign -> bit 31
    uint32_t e = (x >> 3) & 0xFu;
    uint32_t m = x & 0x7u;
    if(e == 0)
    {
        float v = static_cast<float>(m) * (1.0f / 512.0f); // subnormal
        return (x & 0x80u) ? -v : v;
    }
    uint32_t bits = s | ((e + 120u) << 23) | (m << 20); // exp bias 7->127 = +120
    float    f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// BF16 as bit-truncation of FP32 (round-toward-zero); keeps the file header-free.
__host__ __device__ inline uint16_t f_to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>(u >> 16);
}
__host__ __device__ inline float bf16_to_f(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float    f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Representative packed FP8 dot, mirroring gemm_decode's kUsePackedFp32 path
// (gemm_decode_numeric.hpp:97,115). The scalar per-byte e4m3 decode + 16 FP32
// FMAs is ~8x more VALU ops per 128-bit load and pins the kernel compute-bound
// (~1.5 TB/s) well below HBM; the real kernel uses one HW op to convert an FP8
// pair and one to MAC it, which is what makes it memory-bound (VALU ~10-14%).
typedef float fp32x2_t __attribute__((ext_vector_type(2)));

// 2 OCP-e4m3 bytes (sel 0 = bytes 0-1, 1 = bytes 2-3 of the word) -> 2 f32.
// sel must be a compile-time constant (HW byte selector), hence a template arg.
template <int sel>
__device__ inline fp32x2_t cvt_pair(uint32_t fp8x4)
{
#if defined(__gfx950__)
    return __builtin_amdgcn_cvt_pk_f32_fp8(fp8x4, sel);
#else
    fp32x2_t r = {0.0f, 0.0f};
    (void)fp8x4;
    return r;
#endif
}

__device__ inline fp32x2_t pk_fma(fp32x2_t acc, fp32x2_t a, fp32x2_t b)
{
#if defined(__gfx950__)
    fp32x2_t out;
    asm volatile("v_pk_fma_f32 %[out], %[a], %[b], %[acc]"
                 : [out] "=v"(out)
                 : [a] "v"(a), [b] "v"(b), [acc] "v"(acc));
    return out;
#else
    return acc + a * b;
#endif
}

// 16 e4m3 elements (two int4) accumulated into a packed FP32x2 acc: 8 pairs,
// each = cvt(A) + cvt(B) + pk_fma. Caller horizontal-adds once at the end.
__device__ inline void dot16_packed(fp32x2_t& acc, int4 a, int4 b)
{
    acc = pk_fma(acc, cvt_pair<0>(a.x), cvt_pair<0>(b.x));
    acc = pk_fma(acc, cvt_pair<1>(a.x), cvt_pair<1>(b.x));
    acc = pk_fma(acc, cvt_pair<0>(a.y), cvt_pair<0>(b.y));
    acc = pk_fma(acc, cvt_pair<1>(a.y), cvt_pair<1>(b.y));
    acc = pk_fma(acc, cvt_pair<0>(a.z), cvt_pair<0>(b.z));
    acc = pk_fma(acc, cvt_pair<1>(a.z), cvt_pair<1>(b.z));
    acc = pk_fma(acc, cvt_pair<0>(a.w), cvt_pair<0>(b.w));
    acc = pk_fma(acc, cvt_pair<1>(a.w), cvt_pair<1>(b.w));
}

// Store helper: BF16-bits path truncates; FP32 path passes through.
template <typename CT>
__device__ inline CT to_ct(float s)
{
    if constexpr(std::is_same_v<CT, uint16_t>)
        return f_to_bf16(s);
    else
        return s;
}

// COLS columns/block + PF (prefetch depth) compile-time. k_waves and k_batch are
// runtime so the geometry sweep needs no recompile. PF independent 128-bit loads
// are issued per column before any are consumed -> deeper memory pipeline (the
// lever a CK Tile pipeline supplies that a naive single-load loop lacks).
template <int COLS, int PF, typename CT>
__global__ void splitk_kernel(const uint8_t* __restrict__ A,
                              const uint8_t* __restrict__ B,
                              CT* __restrict__ C,
                              float sA,
                              float sB,
                              int   N,
                              int   K,
                              int   kbatch,
                              int   kwaves)
{
    extern __shared__ float lds[]; // [kwaves * COLS]
    const int warp = static_cast<int>(threadIdx.x) >> 6;
    const int lane = static_cast<int>(threadIdx.x) & 63;
    const int col0 = static_cast<int>(blockIdx.x) * COLS;

    // K range: grid-z shard, then this warp's intra-block slice of that shard.
    const int Kz     = K / kbatch;
    const int kz0    = static_cast<int>(blockIdx.z) * Kz;
    const int Wslice = Kz / kwaves;
    const int wbase  = kz0 + warp * Wslice;
    const int blocks = Wslice / 16; // 16 FP8 (128-bit) per coalesced lane load

    fp32x2_t acc2[COLS];
#pragma unroll
    for(int c = 0; c < COLS; ++c)
        acc2[c] = fp32x2_t{0.0f, 0.0f};

    const uint8_t* Aw = A + wbase;
    const uint8_t* Bw = B + wbase;

    // Main loop: full groups of PF blocks. Lane l owns blocks {l, l+64, ...};
    // prefetch PF of them (stride 64, still coalesced) before consuming.
    int base = lane;
    for(; base + (PF - 1) * 64 < blocks; base += 64 * PF)
    {
        int4 av[PF];
#pragma unroll
        for(int u = 0; u < PF; ++u)
            av[u] = *reinterpret_cast<const int4*>(Aw + (base + u * 64) * 16);
#pragma unroll
        for(int c = 0; c < COLS; ++c)
        {
            const uint8_t* Bc = Bw + static_cast<size_t>(col0 + c) * K;
            int4           bv[PF];
#pragma unroll
            for(int u = 0; u < PF; ++u)
                bv[u] = *reinterpret_cast<const int4*>(Bc + (base + u * 64) * 16);
#pragma unroll
            for(int u = 0; u < PF; ++u)
                dot16_packed(acc2[c], av[u], bv[u]);
        }
    }
    // Tail: leftover blocks for this lane, one at a time.
    for(int blk = base; blk < blocks; blk += 64)
    {
        int4 av0 = *reinterpret_cast<const int4*>(Aw + blk * 16);
#pragma unroll
        for(int c = 0; c < COLS; ++c)
            dot16_packed(acc2[c], av0,
                         *reinterpret_cast<const int4*>(Bw + static_cast<size_t>(col0 + c) * K +
                                                        blk * 16));
    }

    float acc[COLS];
#pragma unroll
    for(int c = 0; c < COLS; ++c)
        acc[c] = acc2[c][0] + acc2[c][1]; // horizontal add of the packed accumulator

    // Intra-wave butterfly reduce (mirrors wavefront_reduce_sum), then lane0 of
    // each warp drops its partial into LDS.
#pragma unroll
    for(int c = 0; c < COLS; ++c)
    {
        float v = acc[c];
        for(int o = 32; o > 0; o >>= 1)
            v += __shfl_down(v, o);
        if(lane == 0)
            lds[warp * COLS + c] = v;
    }
    __syncthreads();

    // Warp 0 sums the k_waves partials per column and writes once (direct if no
    // grid-z split, atomic into the shared C otherwise).
    if(warp == 0)
    {
        for(int c = lane; c < COLS; c += 64)
        {
            float s = 0.0f;
            for(int w = 0; w < kwaves; ++w)
                s += lds[w * COLS + c];
            s *= sA * sB;
            const int n = col0 + c;
            if(kbatch == 1)
            {
                C[n] = to_ct<CT>(s);
            }
            else
            {
                if constexpr(std::is_same_v<CT, float>)
                    atomicAdd(&C[n], s); // hybrid demo uses FP32 C (no BF16 atomic packing)
            }
        }
    }
}

// ---- host helpers ----
static double bytes_model(int M, int N, int K)
{
    // Mirrors bench_msweep_fp8.cpp: fp8 A,B = 1 B/elem, bf16 C = 2 B/elem.
    return static_cast<double>(N) * K + static_cast<double>(M) * N * 2.0 +
           static_cast<double>(M) * K;
}

template <int COLS, int PF, typename CT>
static float time_cfg(const uint8_t* dA,
                      const uint8_t* dB,
                      CT*            dC,
                      float          sA,
                      float          sB,
                      int            N,
                      int            K,
                      int            kbatch,
                      int            kwaves,
                      int            warmup,
                      int            repeat)
{
    dim3         grid(N / COLS, 1, kbatch);
    dim3         block(kwaves * 64);
    const size_t lds = static_cast<size_t>(kwaves) * COLS * sizeof(float);

    for(int i = 0; i < warmup; ++i)
        splitk_kernel<COLS, PF, CT><<<grid, block, lds>>>(dA, dB, dC, sA, sB, N, K, kbatch, kwaves);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(s));
    for(int i = 0; i < repeat; ++i)
        splitk_kernel<COLS, PF, CT><<<grid, block, lds>>>(dA, dB, dC, sA, sB, N, K, kbatch, kwaves);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    return ms * 1000.0f / repeat; // us/launch
}

// Dispatch runtime (cols, pf) -> the matching template instantiation. Only the
// combos the sweep actually uses are instantiated (keeps compile time sane).
template <typename CT>
static float dispatch(int  cols,
                      int  pf,
                      const uint8_t* dA,
                      const uint8_t* dB,
                      CT*            dC,
                      float          sA,
                      float          sB,
                      int            N,
                      int            K,
                      int            kbatch,
                      int            kwaves,
                      int            warmup,
                      int            repeat)
{
#define CFG(C, P) time_cfg<C, P, CT>(dA, dB, dC, sA, sB, N, K, kbatch, kwaves, warmup, repeat)
    if(pf == 1)
        switch(cols)
        {
        case 1: return CFG(1, 1);
        case 2: return CFG(2, 1);
        case 4: return CFG(4, 1);
        case 7: return CFG(7, 1);
        case 14: return CFG(14, 1);
        case 28: return CFG(28, 1);
        case 56: return CFG(56, 1);
        }
    if(pf == 2)
        switch(cols)
        {
        case 1: return CFG(1, 2);
        case 2: return CFG(2, 2);
        case 4: return CFG(4, 2);
        }
    if(pf == 4)
        switch(cols)
        {
        case 1: return CFG(1, 4);
        case 2: return CFG(2, 4);
        case 4: return CFG(4, 4);
        }
    if(pf == 8)
        switch(cols)
        {
        case 1: return CFG(1, 8);
        case 2: return CFG(2, 8);
        case 4: return CFG(4, 8);
        }
#undef CFG
    std::fprintf(stderr, "unsupported (COLS=%d, PF=%d)\n", cols, pf);
    return -1.0f;
}

int main(int argc, char** argv)
{
    int N      = (argc > 1) ? std::atoi(argv[1]) : 7168;
    int K      = (argc > 2) ? std::atoi(argv[2]) : 7168;
    int warmup = (argc > 3) ? std::atoi(argv[3]) : 25;
    int repeat = (argc > 4) ? std::atoi(argv[4]) : 200;
    const int M = 1;

    std::printf("# proto LDS-split-K  M=%d N=%d K=%d  warmup=%d repeat=%d\n", M, N, K, warmup,
                repeat);

    // Random e4m3 bytes with exp in [0..7] (|value| <= 1.875, never NaN), so the
    // K-sum stays in a sane BF16-representable range.
    std::mt19937                          rng(1234);
    std::uniform_int_distribution<int>    bits(0, 0xFF);
    auto gen = [&](size_t n) {
        std::vector<uint8_t> v(n);
        for(size_t i = 0; i < n; ++i)
        {
            int r = bits(rng);
            v[i]  = static_cast<uint8_t>((r & 0x80) | ((r & 0x7) << 3) | (r & 0x7));
        }
        return v;
    };
    std::vector<uint8_t> hA = gen(static_cast<size_t>(M) * K);
    std::vector<uint8_t> hB = gen(static_cast<size_t>(N) * K);
    const float          sA = 0.5f, sB = 0.5f;

    // FP32 reference (host), same decode as the kernel.
    std::vector<float> ref(N);
    for(int n = 0; n < N; ++n)
    {
        float r = 0.0f;
        for(int k = 0; k < K; ++k)
            r += e4m3_to_f(hA[k]) * e4m3_to_f(hB[static_cast<size_t>(n) * K + k]);
        ref[n] = r * sA * sB;
    }

    uint8_t *dA, *dB;
    HIP_CHECK(hipMalloc(&dA, hA.size()));
    HIP_CHECK(hipMalloc(&dB, hB.size()));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    uint16_t* dC16;
    HIP_CHECK(hipMalloc(&dC16, static_cast<size_t>(N) * sizeof(uint16_t)));

    auto relerr_bf16 = [&](const std::vector<uint16_t>& out) {
        double num = 0.0, den = 0.0;
        for(int n = 0; n < N; ++n)
        {
            num += std::fabs(bf16_to_f(out[n]) - ref[n]);
            den += std::fabs(ref[n]);
        }
        return num / (den + 1e-9);
    };

    std::printf("# pure LDS split-K (k_batch=1), vs wvSplitKQ = 7.85 us\n");
    std::printf("%5s %6s %9s %9s %9s\n", "COLS", "kwaves", "us", "TB/s", "relerr");
    const int   colsList[]   = {1, 2, 4, 7, 14, 28, 56};
    const int   wavesList[]  = {1, 2, 4, 8, 16};
    float       best_us      = 1e30f;
    int         best_c = 0, best_w = 0;
    std::vector<uint16_t> hC(N);
    for(int c : colsList)
        for(int w : wavesList)
        {
            float us = dispatch<uint16_t>(c, /*pf=*/1, dA, dB, dC16, sA, sB, N, K, /*kbatch=*/1, w,
                                          warmup, repeat);
            HIP_CHECK(hipMemcpy(hC.data(), dC16, static_cast<size_t>(N) * 2, hipMemcpyDeviceToHost));
            double re   = relerr_bf16(hC);
            double tbs  = bytes_model(M, N, K) / (us * 1e-6) / 1e12;
            std::printf("%5d %6d %9.2f %9.2f %9.2e%s\n", c, w, us, tbs, re,
                        us < best_us ? "  <= best" : "");
            if(us < best_us)
            {
                best_us = us;
                best_c  = c;
                best_w  = w;
            }
        }
    std::printf("# BEST pure-LDS (PF=1): COLS=%d kwaves=%d  %.2f us  (%s 7.85 us)\n", best_c, best_w,
                best_us, best_us < 7.85f ? "BEATS" : "loses to");

    // Focused prefetch sweep over the high-occupancy region: does a deeper memory
    // pipeline (what CK Tile would supply) close the gap to wvSplitKQ?
    std::printf("\n# prefetch sweep (deeper memory pipeline) over low-COLS region\n");
    std::printf("%5s %6s %4s %9s %9s %9s\n", "COLS", "kwaves", "PF", "us", "TB/s", "relerr");
    float best_pf_us = best_us;
    int   bpf_c = best_c, bpf_w = best_w, bpf_pf = 1;
    const int pfCols[]  = {1, 2, 4};
    const int pfWaves[] = {2, 4, 8};
    const int pfList[]  = {2, 4, 8};
    for(int c : pfCols)
        for(int w : pfWaves)
            for(int pf : pfList)
            {
                float us = dispatch<uint16_t>(c, pf, dA, dB, dC16, sA, sB, N, K, /*kbatch=*/1, w,
                                              warmup, repeat);
                HIP_CHECK(
                    hipMemcpy(hC.data(), dC16, static_cast<size_t>(N) * 2, hipMemcpyDeviceToHost));
                double re  = relerr_bf16(hC);
                double tbs = bytes_model(M, N, K) / (us * 1e-6) / 1e12;
                std::printf("%5d %6d %4d %9.2f %9.2f %9.2e%s\n", c, w, pf, us, tbs, re,
                            us < best_pf_us ? "  <= best" : "");
                if(us < best_pf_us)
                {
                    best_pf_us = us;
                    bpf_c = c;
                    bpf_w = w;
                    bpf_pf = pf;
                }
            }
    std::printf("# BEST overall: COLS=%d kwaves=%d PF=%d  %.2f us  (%s 7.85 us)\n", bpf_c, bpf_w,
                bpf_pf, best_pf_us, best_pf_us < 7.85f ? "BEATS" : "loses to");

    // ---- compose check: both split-K modes in one kernel (k_batch=2 x kwaves=8) ----
    float* dC32;
    HIP_CHECK(hipMalloc(&dC32, static_cast<size_t>(N) * sizeof(float)));
    HIP_CHECK(hipMemset(dC32, 0, static_cast<size_t>(N) * sizeof(float)));
    {
        dim3         grid(N / 28, 1, 2);
        dim3         block(8 * 64);
        const size_t lds = static_cast<size_t>(8) * 28 * sizeof(float);
        splitk_kernel<28, 1, float><<<grid, block, lds>>>(dA, dB, dC32, sA, sB, N, K, 2, 8);
        HIP_CHECK(hipDeviceSynchronize());
    }
    std::vector<float> hC32(N);
    HIP_CHECK(hipMemcpy(hC32.data(), dC32, static_cast<size_t>(N) * sizeof(float),
                        hipMemcpyDeviceToHost));
    double num = 0.0, den = 0.0;
    for(int n = 0; n < N; ++n)
    {
        num += std::fabs(hC32[n] - ref[n]);
        den += std::fabs(ref[n]);
    }
    std::printf("# compose (k_batch=2 x kwaves=8, COLS=28, FP32 atomic C): relerr=%.2e %s\n",
                num / (den + 1e-9), (num / (den + 1e-9) < 5e-2) ? "OK" : "FAIL");

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC16));
    HIP_CHECK(hipFree(dC32));
    return 0;
}
