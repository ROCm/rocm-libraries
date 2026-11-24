// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Shared part of the pooler / unpooler bench (see gemm_pooler_unpooler_bench.cpp): the kernel
// instances, the host data, the reference check and the timing loop. Each instances_<layout>.cpp
// instantiates the tiles for one A/B layout so the layouts compile in parallel.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace pooler_bench {

using ADataType    = ck_tile::bf16_t;
using BDataType    = ck_tile::bf16_t;
using AccDataType  = float;
using CDataType    = ck_tile::bf16_t;
using BiasDataType = ck_tile::bf16_t;
using RowMajor     = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor  = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout      = RowMajor;

constexpr ck_tile::index_t kWarpM = 16;
constexpr ck_tile::index_t kWarpN = 16;
constexpr ck_tile::index_t kWarpK = 32;

// One run-ahead GEMM. With Bias, E = A * B + bias, where bias is an N-vector broadcast over the M
// rows: a row-major D tensor with row stride 0, fused by TdmMultiDEpilogue. BiasRow reads it as
// one N-vector per wave (DsRowBroadcastMask); otherwise it is loaded as a full tile in the
// accumulator layout, one scalar load per element. Staged writes C one M iteration per wave row at
// a time; the full-tile epilogue with a full-tile bias spills at 128x128 per wave.
template <typename ALayout_,
          typename BLayout_,
          ck_tile::index_t TileM,
          ck_tile::index_t TileN,
          ck_tile::index_t TileK,
          ck_tile::index_t WarpsM,
          ck_tile::index_t WarpsN,
          ck_tile::index_t NumSlots,
          ck_tile::index_t RunAhead,
          bool Bias,
          bool Staged,
          bool BiasRow    = true,
          int StreamK     = 0, // 0: data-parallel GemmKernel, 1: Stream-K linear, 2: Stream-K tree
          bool Persistent = false>
struct Instance
{
    using ALayout                  = ALayout_;
    using BLayout                  = BLayout_;
    static constexpr bool kBias    = Bias;
    static constexpr bool kStreamK = StreamK != 0;
    // StreamKKernel takes no D tensors; TdmEpilogue overwrites C, so it pairs with the linear
    // and tree reductions only.
    static_assert(!kStreamK || (!Bias && !Staged), "Stream-K: plain GEMM, full-tile TdmEpilogue");
    static constexpr ck_tile::index_t kTileM = TileM;
    static constexpr ck_tile::index_t kTileN = TileN;
    template <int SK, bool P>
    using WithStreamK = Instance<ALayout_,
                                 BLayout_,
                                 TileM,
                                 TileN,
                                 TileK,
                                 WarpsM,
                                 WarpsN,
                                 NumSlots,
                                 RunAhead,
                                 Bias,
                                 Staged,
                                 BiasRow,
                                 SK,
                                 P>;
    using TileShape   = ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                               ck_tile::sequence<WarpsM, WarpsN, 1>,
                                               ck_tile::sequence<kWarpM, kWarpN, kWarpK>>;
    using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits      = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                         false, // kPadN
                                                         false, // kPadK
                                                         true,  // DoubleSmemBuffer
                                                         ALayout,
                                                         BLayout,
                                                         CLayout,
                                                         false, // TransposeC
                                                         false, // UseStructuredSparsity
                                                         Persistent>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              TileShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using Policy   = ck_tile::GemmPipelineAgBgCrCompTDMRunAheadPolicy<NumSlots, RunAhead>;
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompTDMRunAhead<Problem, Policy>;

    using DsDataType    = std::conditional_t<Bias, ck_tile::tuple<BiasDataType>, ck_tile::tuple<>>;
    using DsLayout      = std::conditional_t<Bias, ck_tile::tuple<RowMajor>, ck_tile::tuple<>>;
    using CDElementwise = std::
        conditional_t<Bias, ck_tile::element_wise::MultiDAdd, ck_tile::element_wise::PassThrough>;
    using EpilogueProblem                       = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                                   BDataType,
                                                                                   DsDataType,
                                                                                   AccDataType,
                                                                                   CDataType,
                                                                                   DsLayout,
                                                                                   CLayout,
                                                                                   CDElementwise,
                                                                                   TileM,
                                                                                   TileN,
                                                                                   WarpsM,
                                                                                   WarpsN,
                                                                                   kWarpM,
                                                                                   kWarpN,
                                                                                   kWarpK,
                                                                                   false, // isCTransposed
                                                                                   1, // kNumWaveGroups
                                                                                   false, // FixedVectorSize
                                                                                   1,
                                                                                   1, // BlockedXDLN_PerWarp
                                                                                   true>; // DoubleSmemBuffer
    static constexpr ck_tile::index_t kBiasMask = Bias && BiasRow ? 1 : 0;
    using Epilogue                              = std::conditional_t<
                                     Bias,
                                     std::conditional_t<
                                         Staged,
                                         ck_tile::TdmMultiDEpilogueStagedForPipeline<EpilogueProblem, 1, Pipeline, kBiasMask>,
                                         ck_tile::TdmMultiDEpilogue<EpilogueProblem, kBiasMask>>,
                                     std::conditional_t<Staged,
                                                        ck_tile::TdmEpilogueStagedForPipeline<EpilogueProblem, 1, Pipeline>,
                                                        ck_tile::TdmEpilogue<EpilogueProblem>>>;
    using StreamKPartitioner =
        ck_tile::StreamKTilePartitioner<TileShape,
                                        StreamK == 2 ? ck_tile::StreamKReductionStrategy::Tree
                                                     : ck_tile::StreamKReductionStrategy::Linear,
                                        Persistent>;
    using Kernel = std::conditional_t<
        kStreamK,
        ck_tile::StreamKKernel<StreamKPartitioner, Pipeline, Epilogue>,
        std::conditional_t<Bias,
                           ck_tile::GemmKernelMultiD<Partitioner, Pipeline, Epilogue>,
                           ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>>>;
};

#define BENCH_HIP_CHECK(expr)                                                                    \
    do                                                                                           \
    {                                                                                            \
        const hipError_t bench_err_ = (expr);                                                    \
        if(bench_err_ != hipSuccess)                                                             \
        {                                                                                        \
            throw std::runtime_error(std::string("HIP error ") + hipGetErrorString(bench_err_) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                                        \
    } while(0)

inline uint16_t bf16_rne(float f)
{
    const uint32_t u = ck_tile::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

inline float bf16_to_float(uint16_t b)
{
    return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

template <typename F>
void parallel_for(size_t n, F&& f)
{
    const unsigned hc = std::thread::hardware_concurrency();
    const size_t nt   = std::min<size_t>(hc == 0 ? 8u : std::min(hc, 64u), std::max<size_t>(n, 1));
    std::vector<std::thread> th;
    for(size_t t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for(size_t i = n * t / nt; i < n * (t + 1) / nt; ++i)
                f(i);
        });
    for(auto& x : th)
        x.join();
}

// Integers in [-4, 4] (exact in bf16 and, for K below 2^20, in every fp32 partial sum) or
// normal(0, 1) rounded to bf16. Deterministic per (seed, tensor id) whatever the thread count.
inline void fill(std::vector<uint16_t>& v, bool ints, uint64_t seed, uint64_t tensor_id)
{
    constexpr size_t kChunk = size_t{1} << 20;
    const size_t chunks     = (v.size() + kChunk - 1) / kChunk;
    parallel_for(chunks, [&](size_t c) {
        std::mt19937_64 g(seed * 0x9E3779B97F4A7C15ull + tensor_id * 0x100000000ull + c);
        std::uniform_int_distribution<int> di(-4, 4);
        std::normal_distribution<float> dn(0.f, 1.f);
        const size_t end = std::min(v.size(), (c + 1) * kChunk);
        for(size_t i = c * kChunk; i < end; ++i)
            v[i] = bf16_rne(ints ? static_cast<float>(di(g)) : dn(g));
    });
}

// The problem size and epilogue, resolved from -op and the overrides.
struct Shape
{
    ck_tile::index_t M, N, K;
    bool bias;
    ck_tile::index_t split_n; // > 0: the main kernel computes C's first split_n columns, Tail the rest
};

// The tail kernel of -split_n: 192x256 tiles with 2x2 waves, whose 88 tiles fill one round on 96
// WGPs for a 2048 x 2048 remainder.
template <typename ALayout, typename BLayout, bool Staged>
using SplitTail = Instance<ALayout, BLayout, 192, 256, 128, 2, 2, 2, 1, false, Staged>;

struct Buffers
{
    std::vector<void*> ptrs;
    void* alloc(size_t bytes)
    {
        void* p = nullptr;
        BENCH_HIP_CHECK(hipMalloc(&p, bytes));
        ptrs.push_back(p);
        return p;
    }
    ~Buffers()
    {
        for(void* p : ptrs)
            (void)hipFree(p);
    }
};

// With a Tail instance, each launch is two back-to-back kernels on C's column ranges [0, split_n)
// and [split_n, N): the tile counts of both can then be whole rounds where one tile shape over all
// of C leaves a partial last round.
template <typename Inst, typename Tail = void>
int run(const ck_tile::ArgParser& arg, const Shape& shape, const std::string& name)
{
    using Kernel             = typename Inst::Kernel;
    constexpr bool kSplit    = !std::is_void_v<Tail>;
    static_assert(!kSplit || (!Inst::kBias && !Inst::kStreamK), "-split_n: plain GEMM only");
    constexpr bool a_row     = std::is_same_v<typename Inst::ALayout, RowMajor>;
    constexpr bool b_col     = std::is_same_v<typename Inst::BLayout, ColumnMajor>;
    const ck_tile::index_t M = shape.M, N = shape.N, K = shape.K;
    const std::string mode = arg.get_str("mode");
    const uint64_t seed    = arg.get_uint64("seed");
    const bool verify      = mode == "verify-int" || mode == "verify-rand";
    if(!verify && mode != "bench")
    {
        std::printf("unknown -mode=%s (verify-int, verify-rand, bench)\n", mode.c_str());
        return 2;
    }

    int dev = 0;
    BENCH_HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    BENCH_HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf(
        "{\"type\":\"info\",\"bench\":\"%s\",\"kernel\":\"%s\",\"arch\":\"%s\",\"cus\":%d}\n",
        name.c_str(),
        Kernel::GetName().c_str(),
        prop.gcnArchName,
        prop.multiProcessorCount);

    // A is M x K (row-major, stride K) or K x M (column-major, stride M); B is N x K
    // (column-major, stride K) or K x N (row-major, stride N); C is M x N row-major.
    const size_t a_elems  = static_cast<size_t>(M) * K;
    const size_t b_elems  = static_cast<size_t>(N) * K;
    const size_t c_elems  = static_cast<size_t>(M) * N;
    const size_t stride_a = a_row ? K : M;
    const size_t stride_b = b_col ? K : N;
    const bool ints       = mode == "verify-int";
    const ck_tile::index_t N0 = kSplit ? shape.split_n : N;
    std::vector<uint16_t> a(a_elems), b(b_elems), bias(Inst::kBias ? N : 0);
    fill(a, ints, seed, 0);
    fill(b, ints, seed, 1);
    if constexpr(Inst::kBias)
        fill(bias, ints, seed, 2);

    // Timed launches rotate through enough copies of A, B and C that each one reads its inputs
    // from memory rather than from the caches, as hipBLASLt's --rotating does.
    const size_t set_bytes = (a_elems + b_elems + c_elems) * 2;
    const int copies =
        verify ? 1
               : std::max(1,
                          static_cast<int>((static_cast<size_t>(arg.get_int("rotating_mb")) << 20) /
                                           set_bytes) +
                              1);
    Buffers buf;
    void* d_bias = nullptr;
    if constexpr(Inst::kBias)
    {
        d_bias = buf.alloc(N * sizeof(BiasDataType));
        BENCH_HIP_CHECK(hipMemcpy(d_bias, bias.data(), N * 2, hipMemcpyHostToDevice));
    }
    auto make_kargs = [&](const void* d_a, const void* d_b, void* c) {
        if constexpr(Inst::kStreamK)
            return Kernel::MakeKernelArgs(
                ck_tile::StreamKHostArgs{d_a,
                                         d_b,
                                         c,
                                         M,
                                         N,
                                         K,
                                         static_cast<ck_tile::index_t>(stride_a),
                                         static_cast<ck_tile::index_t>(stride_b),
                                         N,
                                         static_cast<ck_tile::long_index_t>(a_elems),
                                         static_cast<ck_tile::long_index_t>(b_elems),
                                         static_cast<ck_tile::long_index_t>(c_elems),
                                         1});
        else if constexpr(Inst::kBias)
            return Kernel::MakeKernelArgs(
                ck_tile::GemmMultiDHostArgs<1>(d_a,
                                               d_b,
                                               {d_bias},
                                               c,
                                               1,
                                               M,
                                               N,
                                               K,
                                               static_cast<ck_tile::index_t>(stride_a),
                                               static_cast<ck_tile::index_t>(stride_b),
                                               {0}, // the bias row repeats for every M
                                               N));
        else
            return Kernel::MakeKernelArgs(
                ck_tile::GemmHostArgs(d_a,
                                      d_b,
                                      c,
                                      1,
                                      M,
                                      N0,
                                      K,
                                      static_cast<ck_tile::index_t>(stride_a),
                                      static_cast<ck_tile::index_t>(stride_b),
                                      N));
    };
    // The tail's B and C start at column N0; the strides stay those of the whole tensors.
    auto make_tail_kargs = [&](const void* d_a, const void* d_b, void* c) {
        if constexpr(kSplit)
            return Tail::Kernel::MakeKernelArgs(ck_tile::GemmHostArgs(
                d_a,
                static_cast<const BDataType*>(d_b) + (b_col ? static_cast<size_t>(N0) * K : N0),
                static_cast<CDataType*>(c) + N0,
                1,
                M,
                N - N0,
                K,
                static_cast<ck_tile::index_t>(stride_a),
                static_cast<ck_tile::index_t>(stride_b),
                N));
        else
            return 0;
    };
    std::vector<void*> d_c;
    std::vector<decltype(make_kargs(nullptr, nullptr, nullptr))> kargs;
    std::vector<decltype(make_tail_kargs(nullptr, nullptr, nullptr))> tail_kargs;
    for(int r = 0; r < copies; ++r)
    {
        void* d_a = buf.alloc(a_elems * 2);
        void* d_b = buf.alloc(b_elems * 2);
        d_c.push_back(buf.alloc(c_elems * 2));
        BENCH_HIP_CHECK(hipMemcpy(d_a, a.data(), a_elems * 2, hipMemcpyHostToDevice));
        BENCH_HIP_CHECK(hipMemcpy(d_b, b.data(), b_elems * 2, hipMemcpyHostToDevice));
        BENCH_HIP_CHECK(hipMemset(d_c.back(), 0, c_elems * 2));
        kargs.push_back(make_kargs(d_a, d_b, d_c.back()));
        tail_kargs.push_back(make_tail_kargs(d_a, d_b, d_c.back()));
    }
    bool supported = Kernel::IsSupportedArgument(kargs[0]);
    if constexpr(kSplit)
        supported = supported && Tail::Kernel::IsSupportedArgument(tail_kargs[0]);
    if(!supported)
    {
        std::printf("{\"type\":\"reject\",\"bench\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d}\n",
                    name.c_str(),
                    M,
                    N,
                    K);
        return 2;
    }
    // Stream-K: the partial tiles and their ready flags share one workspace. The kernel never
    // clears a flag, so every launch first zeroes the flags (inside the timed loop).
    size_t sk_flag_bytes = 0;
    dim3 grids;
    if constexpr(Inst::kStreamK)
    {
        void* ws      = buf.alloc(Kernel::GetWorkSpaceSize(kargs[0]));
        sk_flag_bytes = kargs[0].tile_partitioner.get_flags_buffer_size();
        for(auto& k : kargs)
            k.workspace_ptr = ws;
        grids = Kernel::GridSize(kargs[0].tile_partitioner);
        std::printf("{\"type\":\"streamk\",\"grid\":%u,\"dp_tiles\":%d,\"sk_tiles\":%d,"
                    "\"sk_ctas\":%d,\"iters_per_sk_cta\":%d}\n",
                    grids.x,
                    static_cast<int>(kargs[0].tile_partitioner.get_dp_tiles()),
                    static_cast<int>(kargs[0].tile_partitioner.get_sk_tiles()),
                    static_cast<int>(kargs[0].tile_partitioner.get_sk_ctas()),
                    static_cast<int>(kargs[0].tile_partitioner.get_iters_per_sk_cta()));
    }
    else
        grids = Kernel::GridSize(M, N0, 1);
    const dim3 blocks = Kernel::BlockSize();
    auto launch       = [&](int r) {
        if constexpr(Inst::kStreamK)
            BENCH_HIP_CHECK(hipMemsetAsync(kargs[r].workspace_ptr, 0, sk_flag_bytes, nullptr));
        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false},
                               ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs[r]));
        if constexpr(kSplit)
        {
            using TailKernel = typename Tail::Kernel;
            ck_tile::launch_kernel(
                ck_tile::stream_config{nullptr, false},
                ck_tile::make_kernel<1>(TailKernel{},
                                        TailKernel::GridSize(M, N - N0, 1),
                                        TailKernel::BlockSize(),
                                        0,
                                        tail_kargs[r]));
        }
    };

    if(verify)
    {
        launch(0);
        BENCH_HIP_CHECK(hipGetLastError());
        BENCH_HIP_CHECK(hipDeviceSynchronize());
        std::vector<uint16_t> out(c_elems);
        BENCH_HIP_CHECK(hipMemcpy(out.data(), d_c[0], c_elems * 2, hipMemcpyDeviceToHost));

        // Every output tile's corners plus random points, against a double reference.
        std::vector<std::pair<ck_tile::index_t, ck_tile::index_t>> pts;
        std::mt19937_64 g(seed);
        const ck_tile::index_t per_tile = arg.get_int("check_per_tile");
        for(ck_tile::index_t tm = 0; tm < M; tm += 128)
            for(ck_tile::index_t tn = 0; tn < N; tn += 128)
            {
                const ck_tile::index_t m1 = std::min(M, tm + 128) - 1;
                const ck_tile::index_t n1 = std::min(N, tn + 128) - 1;
                pts.insert(pts.end(), {{tm, tn}, {tm, n1}, {m1, tn}, {m1, n1}});
                std::uniform_int_distribution<ck_tile::index_t> dm(tm, m1), dn(tn, n1);
                for(ck_tile::index_t i = 0; i < per_tile; ++i)
                    pts.push_back({dm(g), dn(g)});
            }
        const size_t a_m = a_row ? stride_a : 1, a_k = a_row ? 1 : stride_a;
        const size_t b_n = b_col ? stride_b : 1, b_k = b_col ? 1 : stride_b;
        std::vector<double> refs(pts.size()), sabs(pts.size());
        parallel_for(pts.size(), [&](size_t i) {
            const auto [m, n] = pts[i];
            double s = 0, sa = 0;
            for(size_t k = 0; k < static_cast<size_t>(K); ++k)
            {
                const double prod = static_cast<double>(bf16_to_float(a[m * a_m + k * a_k])) *
                                    bf16_to_float(b[n * b_n + k * b_k]);
                s += prod;
                sa += std::fabs(prod);
            }
            if constexpr(Inst::kBias)
            {
                s += bf16_to_float(bias[n]);
                sa += std::fabs(bf16_to_float(bias[n]));
            }
            refs[i] = s;
            sabs[i] = sa;
        });
        size_t bad     = 0;
        double max_abs = 0;
        for(size_t i = 0; i < pts.size(); ++i)
        {
            const auto [m, n]  = pts[i];
            const uint16_t got = out[static_cast<size_t>(m) * N + n];
            const double err   = std::fabs(bf16_to_float(got) - refs[i]);
            // Integer data: every fp32 sum is exact, so C is the exact result rounded once.
            const bool ok = ints ? got == bf16_rne(static_cast<float>(refs[i]))
                                 : err <= std::fabs(refs[i]) * 0x1p-7 + sabs[i] * 0x1p-16;
            max_abs       = std::max(max_abs, err);
            if(!ok && bad++ < 5)
                std::printf("mismatch m=%d n=%d ref=%.9g got=%.9g\n",
                            m,
                            n,
                            refs[i],
                            static_cast<double>(bf16_to_float(got)));
        }
        std::printf(
            "{\"type\":\"verify\",\"bench\":\"%s\",\"mode\":\"%s\",\"M\":%d,\"N\":%d,"
            "\"K\":%d,\"checked\":%zu,\"bad\":%zu,\"max_abs_err\":%.6g,\"result\":\"%s\"}\n",
            name.c_str(),
            mode.c_str(),
            M,
            N,
            K,
            pts.size(),
            bad,
            max_abs,
            bad == 0 ? "PASS" : "FAIL");
        return bad == 0 ? 0 : 1;
    }

    const int warmup = arg.get_int("warmup"), repeat = arg.get_int("repeat");
    const int rounds = arg.get_int("rounds");
    hipEvent_t start, stop;
    BENCH_HIP_CHECK(hipEventCreate(&start));
    BENCH_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    int it = 0;
    for(int r = 0; r < rounds; ++r)
    {
        for(int w = 0; w < warmup; ++w)
            launch(it++ % copies);
        BENCH_HIP_CHECK(hipEventRecord(start, nullptr));
        for(int i = 0; i < repeat; ++i)
            launch(it++ % copies);
        BENCH_HIP_CHECK(hipEventRecord(stop, nullptr));
        BENCH_HIP_CHECK(hipEventSynchronize(stop));
        BENCH_HIP_CHECK(hipGetLastError());
        float total_ms = 0;
        BENCH_HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        samples.push_back(static_cast<double>(total_ms) / repeat);
    }
    BENCH_HIP_CHECK(hipEventDestroy(start));
    BENCH_HIP_CHECK(hipEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double flop   = 2.0 * M * N * static_cast<double>(K);
    std::printf("{\"type\":\"summary\",\"bench\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"copies\":%d,"
                "\"median_us\":%.2f,\"best_us\":%.2f,\"median_tflops\":%.1f}\n",
                name.c_str(),
                M,
                N,
                K,
                copies,
                median * 1e3,
                samples.front() * 1e3,
                flop / (median * 1e-3) / 1e12);
    return 0;
}

// The Stream-K variants of a data-parallel plain-GEMM instance; built for the 256x256 tiles only
// (the unpooler's), to bound the build.
template <typename ALayout, typename BLayout, typename Plain>
int sk_dispatch(const ck_tile::ArgParser& arg,
                const Shape& shape,
                const std::string& name,
                bool tree,
                bool persistent)
{
    if constexpr(Plain::kTileM == 256 && Plain::kTileN == 256)
    {
        const std::string sk_name =
            name + (tree ? "_sk_tree" : "_sk_linear") + (persistent ? "_persistent" : "");
        if(tree)
            return persistent
                       ? run<typename Plain::template WithStreamK<2, true>>(arg, shape, sk_name)
                       : run<typename Plain::template WithStreamK<2, false>>(arg, shape, sk_name);
        return persistent
                   ? run<typename Plain::template WithStreamK<1, true>>(arg, shape, sk_name)
                   : run<typename Plain::template WithStreamK<1, false>>(arg, shape, sk_name);
    }
    else
    {
        std::printf("-streamk is built for the 256x256 tiles only\n");
        return 2;
    }
}

// Run time -> instance for one A/B layout and epilogue: -tile picks the tile, warps and ring;
// -bias the D tensor. hipBLASLt is column-major, so its MT<a>x<b> is a <b>x<a> tile here, with its
// wave grid transposed too.
template <typename ALayout, typename BLayout, bool Staged>
int dispatch_layout(const ck_tile::ArgParser& arg, const Shape& shape, const std::string& name)
{
    const std::string tile = arg.get_str("tile");
    const bool bias_row    = arg.get_str("bias_load") == "row";
    const std::string sk   = arg.get_str("streamk");
    const bool persistent  = arg.get_int("persistent") != 0;
    auto go                = [&](auto inst_bias_row, auto inst_bias_tile, auto inst_plain) {
        using Plain = decltype(inst_plain);
        if(sk != "none")
        {
            // Stream-K: plain GEMM with the full-tile TDM epilogue only.
            if constexpr(!Staged)
            {
                if(!shape.bias)
                {
                    using L = typename Plain::ALayout;
                    using R = typename Plain::BLayout;
                    return sk_dispatch<L, R, Plain>(arg, shape, name, sk == "tree", persistent);
                }
            }
            std::printf("-streamk needs -bias=0 and -epilogue=tdm\n");
            return 2;
        }
        if(!shape.bias)
        {
            if(shape.split_n > 0)
                return run<Plain, SplitTail<ALayout, BLayout, Staged>>(arg, shape, name);
            return run<Plain>(arg, shape, name);
        }
        return bias_row ? run<decltype(inst_bias_row)>(arg, shape, name)
                                       : run<decltype(inst_bias_tile)>(arg, shape, name);
    };
#define POOLER_BENCH_TILE(TNAME, TM, TN, TK, WM, WN, S, R)                                     \
    if(tile == TNAME)                                                                          \
        return go(Instance<ALayout, BLayout, TM, TN, TK, WM, WN, S, R, true, Staged, true>{},  \
                  Instance<ALayout, BLayout, TM, TN, TK, WM, WN, S, R, true, Staged, false>{}, \
                  Instance<ALayout, BLayout, TM, TN, TK, WM, WN, S, R, false, Staged>{});
    // hipBLASLt MT256x256x128, MIWaveGroup [2,2]: the unpooler's tile.
    POOLER_BENCH_TILE("256x256x64_s3r2", 256, 256, 64, 2, 2, 3, 2)
    POOLER_BENCH_TILE("256x256x128_s2r1", 256, 256, 128, 2, 2, 2, 1)
    // hipBLASLt MT256x176x128, MIWaveGroup [4,1]: the pooler's tile.
    POOLER_BENCH_TILE("176x256x64_s4r3", 176, 256, 64, 1, 4, 4, 3)
    POOLER_BENCH_TILE("176x256x128_s2r1", 176, 256, 128, 1, 4, 2, 1)
    // Half tiles.
    POOLER_BENCH_TILE("128x256x128_s3r2", 128, 256, 128, 2, 2, 3, 2)
    // The pooler's C is 176 of these, two rounds on 96 WGPs like hipBLASLt's MT256x176, with
    // 2x2 waves instead of 1x4.
    POOLER_BENCH_TILE("192x256x64_s4r3", 192, 256, 64, 2, 2, 4, 3)
    POOLER_BENCH_TILE("192x256x128_s2r1", 192, 256, 128, 2, 2, 2, 1)
#undef POOLER_BENCH_TILE
    std::printf("unknown -tile=%s\n", tile.c_str());
    return 2;
}

// One per instances_<layout>_<epilogue>.cpp.
#define POOLER_BENCH_DECLARE(L, E) \
    int dispatch_##L##_##E(        \
        const ck_tile::ArgParser& arg, const Shape& shape, const std::string& name);
POOLER_BENCH_DECLARE(rcr, tdm)
POOLER_BENCH_DECLARE(rcr, tdmstaged)
POOLER_BENCH_DECLARE(rrr, tdm)
POOLER_BENCH_DECLARE(rrr, tdmstaged)
POOLER_BENCH_DECLARE(crr, tdm)
POOLER_BENCH_DECLARE(crr, tdmstaged)
#undef POOLER_BENCH_DECLARE

} // namespace pooler_bench
