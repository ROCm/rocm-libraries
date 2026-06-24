// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// R1: best-config M-sweep for GemmDecodeUniversalKernel, intended for a
// head-to-head against FlyDSL's small_m_hgemm (TILE_M=16) across M=1..16.
//
// For each M in [1, Mmax] and a fixed (N, K) it sweeps five levers
//   - kMPerWarp M-tile register reuse  (mp in {1,2,4,8}, B-reuse / A4)
//   - kNPerWarp N-tile register reuse  (np in {1,2,4},   A-reuse / A1)
//   - kVector    wide vectorized loads (v  in {8,16},    A5)
//   - XCD-aware workgroup swizzle      (off, or chunk in {4,8,16,32,64})
//   - k_batch    cross-block K-split   (kb in {1,2,4,8}, runtime grid-z;
//                atomic-add epilogue, K-unaligned cells auto-skipped)
// over the (mp, np) combinations with mp*np <= 16, picks the fastest cell,
// and emits one CSV row per M:
//
//   impl,M,N,K,time_us,tflops,gbytes_s,mp,np,kv,kb,swizzle,chunk
//
// A human-readable table goes to stderr; clean CSV goes to stdout, so
//   bench_gemm_decode_msweep 10 100 8192 7168 16 > gemm_decode_msweep.csv
// produces a file the FlyDSL comparison script can join on (impl, M, N, K).
//
// Build target: bench_gemm_decode_msweep.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

using namespace ck_tile;

namespace {

template <typename T>
void fill_random(HostTensor<T>& t, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(index_t i = 0; i < static_cast<index_t>(t.get_element_space_size()); ++i)
    {
        t.mData[i] = type_convert<T>(dist(gen));
    }
}

// One timed launch for a fully resolved (kMPerWarp, kNPerWarp, kVector, swizzle,
// chunk) instance. M, N, K and k_batch are runtime, so every M reuses the same
// compiled kernels. Returns a huge sentinel time for unsupported cells (e.g. a
// k_batch that leaves K unaligned to warp_size*kVector*k_batch) so the sweep
// skips them.
template <index_t kMPerWarp, index_t kNPerWarp, index_t kVector, bool kChipletSwizzle,
          index_t kChipletNumXcds, index_t kChipletChunk>
float bench_one(index_t M, index_t N, index_t K, index_t k_batch, int warmup, int repeat,
                const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf)
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;
    using CDataType = bf16_t;
    using Problem   = GemmDecodeProblem<ADataType,
                                        BDataType,
                                        /*ComputeDataType=*/float,
                                        CDataType,
                                        /*XScaleDataType=*/float,
                                        /*WScaleDataType=*/float,
                                        /*XScaleLayout=*/void,
                                        /*WScaleLayout=*/void,
                                        kVector,
                                        /*kUseDot2=*/false,
                                        /*kUsePackedFp32=*/false,
                                        kMPerWarp,
                                        kNPerWarp,
                                        GemmDecodeOutputAxis::SmallM,
                                        /*kHasBias=*/false,
                                        /*kWarpsPerBlock=*/1,
                                        /*kBPreshuffle=*/false,
                                        kChipletSwizzle,
                                        kChipletNumXcds,
                                        kChipletChunk>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        M, N, K,
                                        /*stride_a=*/K,
                                        /*stride_b=*/K,
                                        /*stride_c=*/N,
                                        k_batch);

    // Skip K-unaligned / otherwise unsupported (kVector, k_batch) cells.
    if(!Kernel::IsSupportedArgument(kargs))
        return 1.0e30f;

    // The k_batch>1 shards atomic-add into C, so start from zero (outside timing).
    if(k_batch > 1)
        c_buf.SetZero();

    {
        const stream_config sc{nullptr, /*time_kernel=*/false};
        for(int i = 0; i < warmup; ++i)
            launch_gemm_decode_universal<Kernel>(kargs, sc);
    }
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    const stream_config sc_timed{nullptr,
                                 /*time_kernel=*/true,
                                 /*log_level=*/0,
                                 /*cold_niters=*/0,
                                 /*nrepeat=*/repeat};
    return launch_gemm_decode_universal<Kernel>(kargs, sc_timed);
}

struct Best
{
    float   t_us       = 1.0e30f;
    index_t m_per_warp = 1;
    index_t n_per_warp = 1;
    index_t vector     = 8;
    index_t k_batch    = 1;
    bool    swizzle    = false;
    index_t chunk      = 8;
};

// Sweep the chiplet chunk sizes (plus the swizzle-off point) and the runtime
// cross-block K-split (k_batch) for a fixed compile-time (kMPerWarp, kNPerWarp,
// kVector) instance, folding every cell into `best`. k_batch is a runtime knob
// (grid-z shards that atomic-add into C), so it needs no extra instantiation.
template <index_t kMPerWarp, index_t kNPerWarp, index_t kVector>
void sweep_cfg(index_t M, index_t N, index_t K, int warmup, int repeat,
               const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf, Best& best)
{
    auto consider = [&](float t, index_t kb, bool swz, index_t chunk) {
        if(t < best.t_us)
            best = Best{t, kMPerWarp, kNPerWarp, kVector, kb, swz, chunk};
    };
    for(index_t kb : {index_t{1}, index_t{2}, index_t{4}, index_t{8}})
    {
        consider(bench_one<kMPerWarp, kNPerWarp, kVector, false, 8, 8>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf),
                 kb, false, 8);
        consider(bench_one<kMPerWarp, kNPerWarp, kVector, true, 8, 4>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf),
                 kb, true, 4);
        consider(bench_one<kMPerWarp, kNPerWarp, kVector, true, 8, 8>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf),
                 kb, true, 8);
        consider(bench_one<kMPerWarp, kNPerWarp, kVector, true, 8, 16>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf),
                 kb, true, 16);
        consider(bench_one<kMPerWarp, kNPerWarp, kVector, true, 8, 32>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf),
                 kb, true, 32);
        consider(bench_one<kMPerWarp, kNPerWarp, kVector, true, 8, 64>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf),
                 kb, true, 64);
    }
}

} // namespace

int main(int argc, char** argv)
{
    int     warmup = 10;
    int     repeat = 100;
    index_t N      = 8192;
    index_t K      = 7168;
    index_t Mmax   = 16;
    if(argc > 1) warmup = std::atoi(argv[1]);
    if(argc > 2) repeat = std::atoi(argv[2]);
    if(argc > 3) N      = std::atoi(argv[3]);
    if(argc > 4) K      = std::atoi(argv[4]);
    if(argc > 5) Mmax   = std::atoi(argv[5]);

    // Allocate once for the largest M; smaller M only touch the leading rows.
    HostTensor<bf16_t> a({Mmax, K});
    HostTensor<bf16_t> b({N, K});
    HostTensor<bf16_t> c({Mmax, N});
    fill_random(a, 0xA1u);
    fill_random(b, 0xB2u);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c.get_element_space_size_in_bytes());
    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());

    const auto tflops = [&](index_t M, float t_us) {
        return 2.0f * float(M) * float(N) * float(K) / (t_us * 1.0e-6f) / 1.0e12f;
    };
    const auto gbps = [&](index_t M, float t_us) {
        const double bytes = (double(M) * K + double(N) * K + double(M) * N) * sizeof(bf16_t);
        return bytes / (t_us * 1.0e-6) / 1.0e9;
    };

    std::fprintf(stderr,
                 "--- gemm_decode M-sweep (BF16 unscaled, N=%d K=%d, num_xcds=8) ---\n"
                 "    sweeping (kMPerWarp,kNPerWarp,kVector) with mp*np<=16, v in {8,16}, "
                 "x swizzle{off,chunk 4..64}\n",
                 int(N), int(K));

    // Clean CSV to stdout for the FlyDSL join.
    std::printf("impl,M,N,K,time_us,tflops,gbytes_s,mp,np,kv,kb,swizzle,chunk\n");

    for(index_t M = 1; M <= Mmax; ++M)
    {
        const float base_us =
            bench_one<1, 1, 8, false, 8, 8>(M, N, K, /*k_batch=*/1, warmup, repeat, a_buf, b_buf, c_buf) *
            1000.0f;

        // (mp, np) register-tile grid, pruned to mp*np <= 16 to cap the
        // accumulator/register footprint. mp>1 is the B-reuse lever (A4),
        // np>1 the A-reuse lever (A1); each cell is also tried at kVector
        // 8 and 16 (A5, wider global loads).
        Best best;
        auto sweep = [&](auto mp, auto np) {
            sweep_cfg<decltype(mp)::value, decltype(np)::value, 8>(
                M, N, K, warmup, repeat, a_buf, b_buf, c_buf, best);
            sweep_cfg<decltype(mp)::value, decltype(np)::value, 16>(
                M, N, K, warmup, repeat, a_buf, b_buf, c_buf, best);
        };
        sweep(number<1>{}, number<1>{});
        sweep(number<1>{}, number<2>{});
        sweep(number<1>{}, number<4>{});
        sweep(number<2>{}, number<1>{});
        sweep(number<2>{}, number<2>{});
        sweep(number<2>{}, number<4>{});
        sweep(number<4>{}, number<1>{});
        sweep(number<4>{}, number<2>{});
        sweep(number<4>{}, number<4>{});
        sweep(number<8>{}, number<1>{});
        sweep(number<8>{}, number<2>{});
        const float best_us = best.t_us * 1000.0f;

        std::fprintf(stderr,
                     "M=%2d  base=%8.2fus  best=%8.2fus (mp%d np%d v%2d kb%d %s chunk=%2d) "
                     "spd=%5.3fx  %6.2f TF/s  %7.1f GB/s\n",
                     int(M), base_us, best_us, int(best.m_per_warp), int(best.n_per_warp),
                     int(best.vector), int(best.k_batch), best.swizzle ? "swz" : "off",
                     int(best.chunk), base_us / best_us, tflops(M, best_us), gbps(M, best_us));

        std::printf("gemm_decode_base,%d,%d,%d,%.3f,%.3f,%.2f,1,1,8,1,off,0\n",
                    int(M), int(N), int(K), base_us, tflops(M, base_us), gbps(M, base_us));
        std::printf("gemm_decode_best,%d,%d,%d,%.3f,%.3f,%.2f,%d,%d,%d,%d,%s,%d\n",
                    int(M), int(N), int(K), best_us, tflops(M, best_us), gbps(M, best_us),
                    int(best.m_per_warp), int(best.n_per_warp), int(best.vector),
                    int(best.k_batch), best.swizzle ? "on" : "off", int(best.chunk));
        std::fflush(stdout);
    }

    return 0;
}
