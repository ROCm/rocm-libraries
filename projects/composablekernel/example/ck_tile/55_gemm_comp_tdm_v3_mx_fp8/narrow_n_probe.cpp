// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// EXPERIMENT (gfx1250, standalone): for MX FP8 GEMMs with a narrow N (e.g. 49152 x 16 x 32768,
// C = A * B with A row-major M x K, B column-major K x 16, E8M0 scales per 32 K, bf16 C):
//
//  1. read-bandwidth probe: a streaming read of a buffer the size of A (no compute), over several
//     grid sizes and unroll depths, as the practical HBM read ceiling of the device or partition;
//  2. an A-in-registers kernel: no LDS. Each wave owns 16 rows of A and loads its WMMA fragments
//     straight from global memory (128-bit loads; B and the scales likewise, from L2), then runs
//     the scaled 16x16x128 WMMA. Fragment and scale lane layout as CK's comp_tdm_v3 (16-byte K
//     chunks interleaved between the lane halves; scale word of row lane % 16 in lanes 0-15).
//
// Every GEMM variant is checked against a CPU reference (sampled rows, full K) before its time is
// reported. Effective TB/s = minimum traffic (A, B, their scales, C) / time.
//
// Build:  hipcc -O3 -std=c++17 --offload-arch=gfx1250 narrow_n_probe.cpp -o narrow_n_probe
// Run:    HIP_VISIBLE_DEVICES=<n> ./narrow_n_probe [M] [K]       (defaults 49152 32768)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x)                                                                               \
    do                                                                                         \
    {                                                                                          \
        hipError_t e_ = (x);                                                                   \
        if(e_ != hipSuccess)                                                                   \
        {                                                                                      \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(3);                                                                      \
        }                                                                                      \
    } while(0)

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

constexpr int kN = 16; // narrow N

// ---------------------------------------------------------------- 1. read bandwidth
template <int U>
__global__ void read_bw(const v4i* __restrict__ p, size_t n16, int* __restrict__ sink)
{
    const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    v4i acc             = {0, 0, 0, 0};
    for(size_t i = tid; i < n16; i += stride * U)
    {
        v4i v[U];
#pragma unroll
        for(int u = 0; u < U; ++u)
            v[u] = (i + u * stride < n16) ? __builtin_nontemporal_load(p + i + u * stride)
                                          : v4i{0, 0, 0, 0};
#pragma unroll
        for(int u = 0; u < U; ++u)
            acc ^= v[u];
    }
    if((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7fffffff) // never true for the data used
        sink[0] = 1;
}

// ---------------------------------------------------------------- 2. A-in-registers MX GEMM
// One wave per 16 rows of A. Per K atom (128): four 16-byte A loads and four B loads per lane
// (lane l: row / column l % 16, K bytes (l / 16) * 16 + 32 j + [0, 16)), one scale word each.
// U atoms per group; PIPE: the next group's loads are issued before this group's WMMAs.
template <int U, bool PIPE, bool NT>
__global__ void __launch_bounds__(256) mx_gemm_n16(const uint8_t* __restrict__ A,
                                                   const uint8_t* __restrict__ As,
                                                   const uint8_t* __restrict__ B,
                                                   const uint8_t* __restrict__ Bs,
                                                   uint16_t* __restrict__ C,
                                                   int M,
                                                   int K)
{
#if defined(__gfx1250__)
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int m0   = (blockIdx.x * (blockDim.x / 32) + wave) * 16;
    if(m0 >= M)
        return;
    const int r = lane % 16, h = lane / 16;
    const uint8_t* arow   = A + (size_t)(m0 + r) * K + h * 16;
    const uint8_t* brow   = B + (size_t)r * K + h * 16;
    const uint32_t* asrow = reinterpret_cast<const uint32_t*>(As + (size_t)(m0 + r) * (K / 32));
    const uint32_t* bsrow = reinterpret_cast<const uint32_t*>(Bs + (size_t)r * (K / 32));

    struct Grp
    {
        v4i a[U][4], b[U][4];
        uint32_t sa[U], sb[U];
    };
    auto load = [&](Grp& g, int k) {
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                const v4i* pa = reinterpret_cast<const v4i*>(arow + k + u * 128 + 32 * j);
                g.a[u][j]     = NT ? __builtin_nontemporal_load(pa) : *pa;
                g.b[u][j]     = *reinterpret_cast<const v4i*>(brow + k + u * 128 + 32 * j);
            }
            g.sa[u] = asrow[k / 128 + u];
            g.sb[u] = bsrow[k / 128 + u];
        }
    };
    v8f acc  = {0, 0, 0, 0, 0, 0, 0, 0};
    auto mma = [&](const Grp& g) {
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            const v16i a = __builtin_shufflevector(
                __builtin_shufflevector(g.a[u][0], g.a[u][1], 0, 1, 2, 3, 4, 5, 6, 7),
                __builtin_shufflevector(g.a[u][2], g.a[u][3], 0, 1, 2, 3, 4, 5, 6, 7),
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15);
            const v16i b = __builtin_shufflevector(
                __builtin_shufflevector(g.b[u][0], g.b[u][1], 0, 1, 2, 3, 4, 5, 6, 7),
                __builtin_shufflevector(g.b[u][2], g.b[u][3], 0, 1, 2, 3, 4, 5, 6, 7),
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15);
            acc = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                0, a, 0, b, 0, acc, 0, 0, (int)g.sa[u], 0, 0, (int)g.sb[u], 0, 0);
        }
    };
    const int step = 128 * U;
    if constexpr(PIPE)
    {
        Grp g0, g1;
        load(g0, 0);
        int k = 0;
        for(; k + step < K; k += step)
        {
            load(g1, k + step);
            mma(g0);
            g0 = g1;
        }
        mma(g0);
    }
    else
    {
        for(int k = 0; k < K; k += step)
        {
            Grp g;
            load(g, k);
            mma(g);
        }
    }
    // C (16 x 16 f32 per wave): VGPR i holds row i (lanes 0-15) or i + 8 (lanes 16-31), column lane
    // % 16
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        const float f = acc[i];
        uint32_t bits = __builtin_bit_cast(uint32_t, f);
        bits          = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16; // round to nearest even
        C[(size_t)(m0 + i + 8 * h) * kN + r] = (uint16_t)bits;
    }
#else
    (void)A;
    (void)As;
    (void)B;
    (void)Bs;
    (void)C;
    (void)M;
    (void)K;
#endif
}

// ---------------------------------------------------------------- host
static uint8_t e4m3_of_int(int v) // exact for |v| <= 15
{
    if(v == 0)
        return 0;
    const uint8_t s = v < 0 ? 0x80 : 0;
    int a           = std::abs(v);
    int e           = 31 - __builtin_clz((unsigned)a);
    const int mant  = ((a << 3) >> e) & 7;
    return s | (uint8_t)((e + 7) << 3) | (uint8_t)mant;
}
static float bf16_to_f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t)b << 16); }

struct Timer
{
    hipEvent_t a, b;
    Timer()
    {
        CHECK(hipEventCreate(&a));
        CHECK(hipEventCreate(&b));
    }
    template <typename F>
    double ms(F&& f, int warm, int reps, int rounds)
    {
        for(int i = 0; i < warm; ++i)
            f();
        std::vector<double> t;
        for(int r = 0; r < rounds; ++r)
        {
            CHECK(hipEventRecord(a));
            for(int i = 0; i < reps; ++i)
                f();
            CHECK(hipEventRecord(b));
            CHECK(hipEventSynchronize(b));
            float x;
            CHECK(hipEventElapsedTime(&x, a, b));
            t.push_back(x / reps);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }
};

int main(int argc, char** argv)
{
    const int M = argc > 1 ? std::atoi(argv[1]) : 49152;
    const int K = argc > 2 ? std::atoi(argv[2]) : 32768;
    if(M % 128 || K % 1024)
    {
        std::printf("M must be a multiple of 128 and K of 1024\n");
        return 2;
    }
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s, %d CUs; M=%d N=%d K=%d MX fp8, bf16 C\n",
                prop.gcnArchName,
                prop.multiProcessorCount,
                M,
                kN,
                K);

    const size_t a_bytes = (size_t)M * K, b_bytes = (size_t)kN * K;
    const size_t as_bytes = (size_t)M * (K / 32), bs_bytes = (size_t)kN * (K / 32);
    const size_t c_elems   = (size_t)M * kN;
    const double min_bytes = double(a_bytes + b_bytes + as_bytes + bs_bytes + c_elems * 2);

    // data: A, B small integers (exact in e4m3); scales 2^e, e in [-2, 2]
    std::vector<int8_t> ai(a_bytes), bi(b_bytes), ase(as_bytes), bse(bs_bytes);
    std::vector<uint8_t> a8(a_bytes), b8(b_bytes), as8(as_bytes), bs8(bs_bytes);
    std::mt19937 g(12345);
    std::uniform_int_distribution<int> dv(-3, 3), de(-2, 2);
    for(size_t i = 0; i < a_bytes; ++i)
    {
        ai[i] = (int8_t)dv(g);
        a8[i] = e4m3_of_int(ai[i]);
    }
    for(size_t i = 0; i < b_bytes; ++i)
    {
        bi[i] = (int8_t)dv(g);
        b8[i] = e4m3_of_int(bi[i]);
    }
    for(size_t i = 0; i < as_bytes; ++i)
    {
        ase[i] = (int8_t)de(g);
        as8[i] = (uint8_t)(127 + ase[i]);
    }
    for(size_t i = 0; i < bs_bytes; ++i)
    {
        bse[i] = (int8_t)de(g);
        bs8[i] = (uint8_t)(127 + bse[i]);
    }

    uint8_t *dA, *dB, *dAs, *dBs;
    uint16_t* dC;
    int* dsink;
    CHECK(hipMalloc(&dA, a_bytes));
    CHECK(hipMalloc(&dB, b_bytes));
    CHECK(hipMalloc(&dAs, as_bytes));
    CHECK(hipMalloc(&dBs, bs_bytes));
    CHECK(hipMalloc(&dC, c_elems * 2));
    CHECK(hipMalloc(&dsink, 4));
    CHECK(hipMemcpy(dA, a8.data(), a_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, b8.data(), b_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dAs, as8.data(), as_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dBs, bs8.data(), bs_bytes, hipMemcpyHostToDevice));
    Timer tm;

    // 1. read bandwidth of a buffer the size of A
    std::printf(
        "\n== read bandwidth (%.2f GB, nontemporal 128-bit loads; median of 5 x 100 after 20)\n",
        a_bytes / 1e9);
    double best_bw   = 0;
    const size_t n16 = a_bytes / 16;
    for(int blocks_per_cu : {2, 4, 8, 16, 32})
        for(int threads : {256, 1024})
        {
            const int grid = blocks_per_cu * prop.multiProcessorCount;
            auto run4      = [&] {
                read_bw<4><<<grid, threads>>>(reinterpret_cast<v4i*>(dA), n16, dsink);
            };
            auto run8 = [&] {
                read_bw<8><<<grid, threads>>>(reinterpret_cast<v4i*>(dA), n16, dsink);
            };
            const double t4 = tm.ms(run4, 20, 100, 5), t8 = tm.ms(run8, 20, 100, 5);
            const double bw4 = a_bytes / (t4 * 1e-3) / 1e12, bw8 = a_bytes / (t8 * 1e-3) / 1e12;
            best_bw = std::max(best_bw, std::max(bw4, bw8));
            std::printf("  %3d blocks/CU x %4d threads: unroll 4 %6.2f TB/s, unroll 8 %6.2f TB/s\n",
                        blocks_per_cu,
                        threads,
                        bw4,
                        bw8);
        }
    std::printf("  best read bandwidth: %.2f TB/s\n", best_bw);

    // 2. A-in-registers GEMM variants
    std::vector<int> rows; // sampled rows for the check, spread over M
    for(int i = 0; i < 512; ++i)
        rows.push_back((int)((size_t)i * (M - 1) / 511));
    std::vector<double> ref(rows.size() * kN), sabs(rows.size() * kN);
    for(size_t ri = 0; ri < rows.size(); ++ri)
        for(int n = 0; n < kN; ++n)
        {
            double s = 0, sa = 0;
            const int m = rows[ri];
            for(int kb = 0; kb < K / 32; ++kb)
            {
                long part = 0, pabs = 0;
                for(int j = 0; j < 32; ++j)
                {
                    const long p =
                        (long)ai[(size_t)m * K + kb * 32 + j] * bi[(size_t)n * K + kb * 32 + j];
                    part += p;
                    pabs += std::labs(p);
                }
                const double sc = std::ldexp(
                    1.0, ase[(size_t)m * (K / 32) + kb] + bse[(size_t)n * (K / 32) + kb]);
                s += part * sc;
                sa += pabs * sc;
            }
            ref[ri * kN + n]  = s;
            sabs[ri * kN + n] = sa;
        }

    struct V
    {
        const char* name;
        void (*k)(
            const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, uint16_t*, int, int);
    };
    const V vs[] = {
        {"U2 pipelined", mx_gemm_n16<2, true, false>},
        {"U4 pipelined", mx_gemm_n16<4, true, false>},
        {"U2 pipelined, A nontemporal", mx_gemm_n16<2, true, true>},
        {"U4 pipelined, A nontemporal", mx_gemm_n16<4, true, true>},
        {"U4 single", mx_gemm_n16<4, false, false>},
        {"U8 single", mx_gemm_n16<8, false, false>},
        {"U8 single, A nontemporal", mx_gemm_n16<8, false, true>},
    };
    std::printf(
        "\n== A-in-registers MX GEMM %dx%dx%d (median of 5 x 100 after 50; min traffic %.3f GB)\n",
        M,
        kN,
        K,
        min_bytes / 1e9);
    std::vector<uint16_t> hc(c_elems);
    for(const V& v : vs)
        for(int waves : {4, 8})
        {
            const int threads = waves * 32;
            const int grid    = M / (16 * waves);
            CHECK(hipMemset(dC, 0, c_elems * 2));
            v.k<<<grid, threads>>>(dA, dAs, dB, dBs, dC, M, K);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(hc.data(), dC, c_elems * 2, hipMemcpyDeviceToHost));
            int bad = 0, bad_t = 0;
            for(size_t ri = 0; ri < rows.size(); ++ri)
                for(int n = 0; n < kN; ++n)
                {
                    const double r   = ref[ri * kN + n];
                    const double tol = std::fabs(r) * 0x1p-8 + sabs[ri * kN + n] * 0x1p-20;
                    const double got = bf16_to_f(hc[(size_t)rows[ri] * kN + n]);
                    bad += std::fabs(got - r) > tol;
                    // diagnostic: C transposed inside the 16 x 16 block
                    const int m = rows[ri], mt = (m / 16) * 16 + n, nt = m % 16;
                    bad_t += std::fabs(bf16_to_f(hc[(size_t)mt * kN + nt]) - r) > tol;
                }
            const double t =
                tm.ms([&] { v.k<<<grid, threads>>>(dA, dAs, dB, dBs, dC, M, K); }, 50, 100, 5);
            const double tf = 2.0 * M * kN * (double)K / (t * 1e-3) / 1e12;
            std::printf(
                "  %-30s %d waves/block: %8.1f TF  %6.2f TB/s  check %s (%d bad of %zu%s)\n",
                v.name,
                waves,
                tf,
                min_bytes / (t * 1e-3) / 1e12,
                bad ? "FAIL" : "PASS",
                bad,
                rows.size() * kN,
                bad && !bad_t ? "; matches C transposed" : "");
        }
    CHECK(hipFree(dA));
    CHECK(hipFree(dB));
    CHECK(hipFree(dAs));
    CHECK(hipFree(dBs));
    CHECK(hipFree(dC));
    CHECK(hipFree(dsink));
    return 0;
}
