// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// FP8 PerTensor analog of bench_msweep.cpp: best-config M-sweep for the
// GemmDecodeUniversalKernel FP8/FP8 -> BF16 path with two FP32 PerTensor
// scalars (sA, sB) folded once in the epilogue. Intended for the FP8
// per-tensor head-to-head against:
//   - AITER wvSplitKQ            (VALU peer, M<=4)        -- wvsplitk_msweep.py --fp8
//   - CK Tile gemm_quant tensor  (MFMA M_Tile=16 ceiling) -- gemm_quant_tensor_msweep.py
//   - AITER gemm_a8w8_CK         (classic-CK MFMA)        -- wvsplitk_msweep.py --fp8
//
// For each M in [1, Mmax] and a fixed (N, K) it sweeps the same register-tile
// / swizzle / split-K levers as the BF16 sweep, except kVector is fixed at 16
// (the FP8 dot2 K-loop packs FP8x4 -> 2*BF16x2, so each lane's slice must hold
// a multiple of 4 FP8 elements; the kernel asserts kVector % 4 == 0):
//   - kMPerWarp M-tile register reuse  (mp in {1,2,4,8}, B-reuse / A4)
//   - kNPerWarp N-tile register reuse  (np in {1,2,4},   A-reuse / A1)
//   - XCD-aware workgroup swizzle      (off, or chunk in {4,8,16,32,64})
//   - k_batch    cross-block K-split   (kb in {1,2,4,8}, runtime grid-z;
//                atomic-add epilogue, K-unaligned cells auto-skipped)
// over the (mp, np) combinations with mp*np <= 16, picks the fastest cell, and
// emits one CSV row per M:
//
//   impl,M,N,K,time_us,tflops,gbytes_s,mp,np,kv,kb,swizzle,chunk
//
// A human-readable table goes to stderr; clean CSV goes to stdout so
//   bench_gemm_decode_msweep_fp8 10 100 8192 7168 8 > gemm_decode_fp8_msweep.csv
// produces a file fp8_compare.py can join on (impl, M, N, K).
//
// Build target: bench_gemm_decode_msweep_fp8 (only when CK_USE_OCP_FP8 is ON).

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

// Per-tensor scalars; identical to the single-config benchmark and the gtest
// numeric contract so a -verify run cross-checks against the same reference.
constexpr float kScaleA = 0.125f;
constexpr float kScaleB = 0.0625f;

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

// One timed launch for a fully resolved (kMPerWarp, kNPerWarp, swizzle, chunk)
// FP8 PerTensor instance. kVector is fixed at 16. M, N, K and k_batch are
// runtime, so every M reuses the same compiled kernels. Returns a huge sentinel
// time for unsupported cells (e.g. a k_batch that leaves K unaligned to
// warp_size*kVector*k_batch) so the sweep skips them.
template <index_t kMPerWarp, index_t kNPerWarp, bool kChipletSwizzle,
          index_t kChipletNumXcds, index_t kChipletChunk, index_t kWarpsPerBlock = 1,
          // P0 wvSplitKQ-recipe levers (default off so existing call sites are
          // unchanged): stage the shared A row in LDS, stream B non-temporally,
          // and use the persistent fat-WG (1 WG/CU) launch.
          bool kStageAInLds = false, bool kStreamB = false, bool kPersistent = false>
float bench_one(index_t M, index_t N, index_t K, index_t k_batch, int warmup, int repeat,
                const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf,
                const DeviceMem& sa_buf, const DeviceMem& sb_buf)
{
    using ADataType = fp8_t;
    using BDataType = fp8_t;
    using CDataType = bf16_t;
    using Problem   = GemmDecodeProblem<ADataType,
                                        BDataType,
                                        /*ComputeDataType=*/float,
                                        CDataType,
                                        /*XScaleDataType=*/float,
                                        /*WScaleDataType=*/float,
                                        GemmDecodeScaleLayout::PerTensor,
                                        GemmDecodeScaleLayout::PerTensor,
                                        /*kVector=*/16,
                                        /*kUseDot2=*/true,
                                        /*kUsePackedFp32=*/false,
                                        kMPerWarp,
                                        kNPerWarp,
                                        GemmDecodeOutputAxis::SmallM,
                                        /*kHasBias=*/false,
                                        kWarpsPerBlock,
                                        /*kBPreshuffle=*/false,
                                        kChipletSwizzle,
                                        kChipletNumXcds,
                                        kChipletChunk,
                                        /*kBias2D=*/false,
                                        kStageAInLds,
                                        kStreamB,
                                        kPersistent>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        sa_buf.GetDeviceBuffer(),
                                        sb_buf.GetDeviceBuffer(),
                                        /*p_bias=*/nullptr,
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
    float   t_us             = 1.0e30f;
    index_t m_per_warp       = 1;
    index_t n_per_warp       = 1;
    index_t k_batch          = 1;
    bool    swizzle          = false;
    index_t chunk            = 8;
    index_t warps_per_block  = 1;
    bool    a_lds            = false; // kStageAInLds
    bool    stream_b         = false; // kStreamB
    bool    persistent       = false; // kPersistent
};

// Sweep the chiplet chunk sizes (plus the swizzle-off point) and the runtime
// cross-block K-split (k_batch) for a fixed compile-time (kMPerWarp, kNPerWarp)
// instance, folding every cell into `best`. kVector is fixed at 16. k_batch is
// a runtime knob (grid-z shards that atomic-add into C), so it needs no extra
// instantiation.
template <index_t kMPerWarp, index_t kNPerWarp>
void sweep_cfg(index_t M, index_t N, index_t K, int warmup, int repeat,
               const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf,
               const DeviceMem& sa_buf, const DeviceMem& sb_buf, Best& best)
{
    auto consider = [&](float t, index_t kb, bool swz, index_t chunk) {
        if(t < best.t_us)
            best = Best{t, kMPerWarp, kNPerWarp, kb, swz, chunk};
    };
    for(index_t kb : {index_t{1}, index_t{2}, index_t{4}, index_t{8}})
    {
        consider(bench_one<kMPerWarp, kNPerWarp, false, 8, 8>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 kb, false, 8);
        consider(bench_one<kMPerWarp, kNPerWarp, true, 8, 4>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 kb, true, 4);
        consider(bench_one<kMPerWarp, kNPerWarp, true, 8, 8>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 kb, true, 8);
        consider(bench_one<kMPerWarp, kNPerWarp, true, 8, 16>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 kb, true, 16);
        consider(bench_one<kMPerWarp, kNPerWarp, true, 8, 32>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 kb, true, 32);
        consider(bench_one<kMPerWarp, kNPerWarp, true, 8, 64>(M, N, K, kb, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 kb, true, 64);
    }
}

// Multi-warp occupancy probe (design doc §15.F): for the mp=np=1 config only,
// pack kWarpsPerBlock independent warps per workgroup (swizzle off, k_batch=1)
// to raise wavefronts/CU on the small M=1 grid. The kernel auto-skips a WPB
// that does not evenly tile N (IsSupportedArgument), returning the sentinel.
// Folds every WPB into `best`.
void sweep_multiwarp(index_t M, index_t N, index_t K, int warmup, int repeat,
                     const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf,
                     const DeviceMem& sa_buf, const DeviceMem& sb_buf, Best& best)
{
    auto consider = [&](float t, index_t wpb) {
        if(t < 1.0e29f)
            std::fprintf(stderr, "        [mw] M=%2d wpb%-2d : %8.2fus\n",
                         int(M), int(wpb), t * 1000.0f);
        if(t < best.t_us)
            best = Best{t, 1, 1, 1, false, 0, wpb};
    };
    consider(bench_one<1, 1, false, 8, 8, 2>(M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf), 2);
    consider(bench_one<1, 1, false, 8, 8, 4>(M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf), 4);
    consider(bench_one<1, 1, false, 8, 8, 8>(M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf), 8);
    consider(bench_one<1, 1, false, 8, 8, 16>(M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf), 16);
}

// wvSplitKQ-recipe sweep (P0 gate): for the mp=np=1 multi-warp config and a
// given WPB, isolate the contribution of each recipe lever -- A-in-LDS staging,
// non-temporal B, and the persistent fat-WG launch -- and fold every variant
// into `best`. This is the head-to-head probe against wvSplitKQ's bespoke
// fat-WG/A-in-LDS kernel (design doc §15.J / §15.J.1).
template <index_t WPB>
void try_fatwg(index_t M, index_t N, index_t K, int warmup, int repeat, bool with_persist,
               const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf,
               const DeviceMem& sa_buf, const DeviceMem& sb_buf, Best& best)
{
    auto consider = [&](float t, const char* tag, bool al, bool sb, bool pe) {
        if(t < 1.0e29f)
            std::fprintf(stderr, "        [fatwg] M=%2d wpb%-2d %-16s : %8.2fus\n",
                         int(M), int(WPB), tag, t * 1000.0f);
        if(t < best.t_us)
            best = Best{t, 1, 1, 1, false, 0, WPB, al, sb, pe};
    };
    consider(bench_one<1, 1, false, 8, 8, WPB, true, false, false>(
                 M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
             "a_lds", true, false, false);
    consider(bench_one<1, 1, false, 8, 8, WPB, true, true, false>(
                 M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
             "a_lds+nt", true, true, false);
    // The persistent fat-WG launch is correct (gtested) but its few long-lived
    // workgroups make standalone timing unstable when the GPU is shared, so it
    // is opt-in (GD_BENCH_PERSIST=1) and excluded from the default gate sweep.
    if(with_persist)
    {
        consider(bench_one<1, 1, false, 8, 8, WPB, true, true, true>(
                     M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 "a_lds+nt+persist", true, true, true);
        consider(bench_one<1, 1, false, 8, 8, WPB, false, true, true>(
                     M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                 "nt+persist", false, true, true);
    }
}

// Drive try_fatwg over WPB in {2,4,8,16} plus the single-warp (WPB=1) main-path
// levers (non-temporal B and/or persistent, which compose with the register
// path too). Restricted to the small-M gate band where the fat-WG recipe is
// relevant.
void sweep_fatwg(index_t M, index_t N, index_t K, int warmup, int repeat, bool with_persist,
                 const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf,
                 const DeviceMem& sa_buf, const DeviceMem& sb_buf, Best& best)
{
    auto consider1 = [&](float t, const char* tag, bool sb, bool pe) {
        if(t < 1.0e29f)
            std::fprintf(stderr, "        [fatwg] M=%2d wpb1  %-16s : %8.2fus\n",
                         int(M), tag, t * 1000.0f);
        if(t < best.t_us)
            best = Best{t, 1, 1, 1, false, 0, 1, false, sb, pe};
    };
    consider1(bench_one<1, 1, false, 8, 8, 1, false, true, false>(
                  M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
              "nt", true, false);
    if(with_persist)
    {
        consider1(bench_one<1, 1, false, 8, 8, 1, false, false, true>(
                      M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                  "persist", false, true);
        consider1(bench_one<1, 1, false, 8, 8, 1, false, true, true>(
                      M, N, K, 1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf),
                  "nt+persist", true, true);
    }

    try_fatwg<2>(M, N, K, warmup, repeat, with_persist, a_buf, b_buf, c_buf, sa_buf, sb_buf, best);
    try_fatwg<4>(M, N, K, warmup, repeat, with_persist, a_buf, b_buf, c_buf, sa_buf, sb_buf, best);
    try_fatwg<8>(M, N, K, warmup, repeat, with_persist, a_buf, b_buf, c_buf, sa_buf, sb_buf, best);
    try_fatwg<16>(M, N, K, warmup, repeat, with_persist, a_buf, b_buf, c_buf, sa_buf, sb_buf, best);
}

} // namespace

int main(int argc, char** argv)
{
    int     warmup = 10;
    int     repeat = 100;
    index_t N      = 8192;
    index_t K      = 7168;
    index_t Mmax   = 8;
    if(argc > 1) warmup = std::atoi(argv[1]);
    if(argc > 2) repeat = std::atoi(argv[2]);
    if(argc > 3) N      = std::atoi(argv[3]);
    if(argc > 4) K      = std::atoi(argv[4]);
    if(argc > 5) Mmax   = std::atoi(argv[5]);

    // The persistent fat-WG launch is correct but its standalone timing is
    // unstable on a shared GPU (few long-lived workgroups); opt in explicitly.
    const char* persist_env = std::getenv("GD_BENCH_PERSIST");
    const bool  with_persist = persist_env != nullptr && std::atoi(persist_env) != 0;

    // Allocate once for the largest M; smaller M only touch the leading rows.
    HostTensor<fp8_t>  a({Mmax, K});
    HostTensor<fp8_t>  b({N, K});
    HostTensor<bf16_t> c({Mmax, N});
    fill_random(a, 0xA1u);
    fill_random(b, 0xB2u);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c.get_element_space_size_in_bytes());
    DeviceMem sa_buf(sizeof(float));
    DeviceMem sb_buf(sizeof(float));
    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    sa_buf.ToDevice(&kScaleA);
    sb_buf.ToDevice(&kScaleB);

    const auto tflops = [&](index_t M, float t_us) {
        return 2.0f * float(M) * float(N) * float(K) / (t_us * 1.0e-6f) / 1.0e12f;
    };
    const auto gbps = [&](index_t M, float t_us) {
        // A, B are FP8 (1 byte); C is BF16 (2 bytes).
        const double bytes = double(M) * K * sizeof(fp8_t) + double(N) * K * sizeof(fp8_t) +
                             double(M) * N * sizeof(bf16_t);
        return bytes / (t_us * 1.0e-6) / 1.0e9;
    };

    std::fprintf(stderr,
                 "--- gemm_decode FP8 PerTensor M-sweep (N=%d K=%d, num_xcds=8) ---\n"
                 "    sweeping (kMPerWarp,kNPerWarp) with mp*np<=16, kVector=16, "
                 "x swizzle{off,chunk 4..64} x kb{1,2,4,8}\n",
                 int(N), int(K));

    // Clean CSV to stdout for the FP8 join.
    std::printf("impl,M,N,K,time_us,tflops,gbytes_s,mp,np,kv,kb,swizzle,chunk,wpb,"
                "a_lds,streamb,persist\n");

    for(index_t M = 1; M <= Mmax; ++M)
    {
        const float base_us =
            bench_one<1, 1, false, 8, 8>(M, N, K, /*k_batch=*/1, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf) *
            1000.0f;

        // (mp, np) register-tile grid, pruned to mp*np <= 16 to cap the
        // accumulator/register footprint. mp>1 is the B-reuse lever (A4),
        // np>1 the A-reuse lever (A1). kVector is fixed at 16 for the FP8
        // dot2 path.
        Best best;
        auto sweep = [&](auto mp, auto np) {
            sweep_cfg<decltype(mp)::value, decltype(np)::value>(
                M, N, K, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf, best);
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
        // §15.F multi-warp occupancy probe (mp=np=1, WPB in {2,4,8,16}).
        sweep_multiwarp(M, N, K, warmup, repeat, a_buf, b_buf, c_buf, sa_buf, sb_buf, best);
        // P0 wvSplitKQ-recipe gate (A-in-LDS / non-temporal B / persistent
        // fat-WG), small-M band only.
        if(M <= 4)
        {
            sweep_fatwg(M, N, K, warmup, repeat, with_persist, a_buf, b_buf, c_buf, sa_buf,
                        sb_buf, best);
        }
        const float best_us = best.t_us * 1000.0f;

        std::fprintf(stderr,
                     "M=%2d  base=%8.2fus  best=%8.2fus (mp%d np%d v16 kb%d %s chunk=%2d "
                     "wpb%d a_lds=%d nt=%d persist=%d) spd=%5.3fx  %6.2f TF/s  %7.1f GB/s\n",
                     int(M), base_us, best_us, int(best.m_per_warp), int(best.n_per_warp),
                     int(best.k_batch), best.swizzle ? "swz" : "off", int(best.chunk),
                     int(best.warps_per_block), int(best.a_lds), int(best.stream_b),
                     int(best.persistent), base_us / best_us, tflops(M, best_us),
                     gbps(M, best_us));

        std::printf("gemm_decode_fp8_base,%d,%d,%d,%.3f,%.3f,%.2f,1,1,16,1,off,0,1,0,0,0\n",
                    int(M), int(N), int(K), base_us, tflops(M, base_us), gbps(M, base_us));
        std::printf("gemm_decode_fp8_best,%d,%d,%d,%.3f,%.3f,%.2f,%d,%d,16,%d,%s,%d,%d,%d,%d,%d\n",
                    int(M), int(N), int(K), best_us, tflops(M, best_us), gbps(M, best_us),
                    int(best.m_per_warp), int(best.n_per_warp),
                    int(best.k_batch), best.swizzle ? "on" : "off", int(best.chunk),
                    int(best.warps_per_block), int(best.a_lds), int(best.stream_b),
                    int(best.persistent));
        std::fflush(stdout);
    }

    return 0;
}
