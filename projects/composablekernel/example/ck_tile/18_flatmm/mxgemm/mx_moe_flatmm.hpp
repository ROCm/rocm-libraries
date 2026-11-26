// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MX (microscaling) MoE GEMM on gfx1250 (shared by tile_example_mx_moe_flatmm and
// test_tile_mx_moe_flatmm): B is MXFP4 (pk_fp4_t), A is MXFP4 (a4w4) or MX FP8 (fp8_t, a8w4),
// both with one e8m0 scale per
// 32 K values. One launch runs one FFN GEMM of an MoE layer:
//
//   gemm1_gate_up  A = tokens          [num_tokens, K]          (gathered by sorted token id)
//                  B = gate|up weights [experts, N, K]          (N = 2 * intermediate)
//                  C = [num_tokens * topk, N / 2] = silu(A B_gate) * (A B_up)
//   gemm2          A = intermediate    [num_tokens * topk, K]
//                  B = down weights    [experts, N, K]
//                  C = [num_tokens, N] += routing weight * (A B), accumulated with atomics
//   gemm2_per_slot as gemm2, but row (token, slot) of A B is stored to P[num_tokens * topk, N]
//                  and MoeTopkCombineKernel sums each token's rows with their weights:
//                  C = [num_tokens, N], bf16, exact in fp32 and deterministic
//
// The routing is random (top-k distinct experts per token) and sorted on the host into expert
// tiles of M_Tile rows, padded with a sentinel token id, as moe_sorting does on the device.
// A scales are passed in sorted-row order, B scales per expert, both in the gfx1250 MX layout.
// The result is checked against a CPU reference computed from the dequantized operands.

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

// Block tile 32 x 256 x 256, four waves along N, 16x16x128 scaled WMMA. 32-row tiles keep the
// padding of each expert's rows small.
struct MXMoeFlatmmConfig
{
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;

    // C tile layout of the MX pipeline: pairs of 16-column blocks per wave (NXdlPack = 2).
    static constexpr int BlockedXDLN_PerWarp = 2;
};

static constexpr int kScaleBlockK = 32; // K values per e8m0 scale
static constexpr int kScalePack   = 4;  // scales per int32 in the gfx1250 layout
static constexpr int kScaleRows   = 32; // rows per block in the gfx1250 layout

// One problem for run_mx_moe_case().
struct MxMoeCase
{
    int experts    = 8;
    int num_tokens = 512;
    int topk       = 2;
    int N          = 1024; // gemm1_gate_up: 2 x intermediate; gemm2: hidden
    int K          = 1024; // gemm1_gate_up: hidden; gemm2: intermediate
    int init       = 0;    // 0 random, 1 constant, 2 row r of A = v(r) (row-mapping check)
    unsigned seed  = 11939;
    bool validate  = true;
    int warmup     = 0;
    int repeat     = 0; // timed launches; 0 = no timing
    bool verbose   = true;
};

struct MxMoeResult
{
    bool pass;               // true if not validated
    float ms;                // mean time of a timed launch, 0 if not timed
    int max_rows_per_expert; // most (token, slot) rows routed to one expert
    int sorted_rows;         // rows of the sorted token list, padding included
};

// Offset of scale (mn, kb) in the gfx1250 MX scale layout [MN / 32][K / 128][32][4].
inline std::size_t mx_scale_offset(std::size_t mn, std::size_t kb, std::size_t K)
{
    const std::size_t k_packs = K / (kScaleBlockK * kScalePack);
    return (mn / kScaleRows) * k_packs * kScaleRows * kScalePack +
           (kb / kScalePack) * kScaleRows * kScalePack + (mn % kScaleRows) * kScalePack +
           kb % kScalePack;
}

// Column of B as the kernel sees it. gemm1_gate_up interleaves gate and up in blocks of 16
// columns (gate block j -> 2j, up block j -> 2j + 1), so that a wave's pair of 16-column blocks
// holds matching gate and up columns.
inline int b_column(int n, int N, bool gate_up)
{
    if(!gate_up)
        return n;
    constexpr int NLane = 16;
    const int n0        = n / NLane;
    const int n1        = n % NLane;
    const int n0_il     = n >= N / 2 ? (n0 - N / 2 / NLane) * 2 + 1 : n0 * 2;
    return n0_il * NLane + n1;
}

// Preshuffle the fp4 weights of every expert into the flat layout of the MX pipeline
// (16-column lanes, 16-byte K packs; same as preShuffleWeight in mx_flatmm.cpp), with the
// gate/up interleave for gemm1_gate_up.
inline void shuffle_fp4_weight(
    const ck_tile::pk_fp4_t* src, ck_tile::pk_fp4_t* dst, int experts, int N, int K, bool gate_up)
{
    constexpr int KPack = 16; // bytes
    constexpr int NLane = 16;
    const int KLane     = ck_tile::get_warp_size() / NLane;
    const int K_pk      = K / 2;
    const int K0        = K_pk / (KLane * KPack);
    for(long e = 0; e < experts; ++e)
        for(int n = 0; n < N; ++n)
        {
            const int nb = b_column(n, N, gate_up);
            const int n0 = nb / NLane;
            const int n1 = nb % NLane;
            for(int k = 0; k < K_pk; ++k)
            {
                const int k0   = k / (KLane * KPack);
                const int k1   = (k % (KLane * KPack)) / KPack;
                const int k2   = k % KPack;
                const long out = e * N * K_pk + long(n0) * KPack * NLane * KLane * K0 +
                                 k0 * KPack * NLane * KLane + k1 * KPack * NLane + n1 * KPack + k2;
                dst[out] = src[e * N * K_pk + long(n) * K_pk + k];
            }
        }
}

// Value of element (r, k) of a row-major packed fp4 tensor.
inline float fp4_at(const ck_tile::HostTensor<ck_tile::pk_fp4_t>& t, std::size_t r, std::size_t k)
{
    const auto pk                 = t(r, k);
    const std::size_t flat_offset = t.mDesc.GetOffsetFromMultiIndex(r, k);
    return flat_offset % 2 == 0
               ? ck_tile::type_convert<float>(pk.template unpack<>(ck_tile::number<0>{}))
               : ck_tile::type_convert<float>(pk.template unpack<>(ck_tile::number<1>{}));
}

// MoeFlatmmKernel on the gfx1250 MX pipeline with this example's tile, for one MoE kind.
template <ck_tile::MoeFlatmmKind kind, typename ADataType, typename BDataType, typename CDataType>
struct MxMoeKernelTypes
{
    using Config  = MXMoeFlatmmConfig;
    using Row     = ck_tile::tensor_layout::gemm::RowMajor;
    using Col     = ck_tile::tensor_layout::gemm::ColumnMajor;
    using AccType = float;

    using Shape = ck_tile::TileGemmShape<
        ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
        ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
        ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<Shape,
                                                   Config::TileParitionerGroupNum,
                                                   Config::TileParitionerM01>;
    using Traits = ck_tile::TileGemmUniversalTraits<Config::kPadM,
                                                    Config::kPadN,
                                                    Config::kPadK,
                                                    Config::DoubleSmemBuffer,
                                                    Row,
                                                    Col,
                                                    Row,
                                                    Config::TransposeC,
                                                    Config::UseStructuredSparsity,
                                                    false, // persistent
                                                    Config::NumWaveGroups,
                                                    true>; // preshuffled B
    // The pipeline picks its hot-loop / tail variant at run time from the loop count.
    using Problem = ck_tile::
        MXFlatmmPipelineProblem<ADataType, BDataType, AccType, Shape, Traits, Config::Scheduler>;
    using Pipeline = ck_tile::MXFlatmmPipelineAGmemBGmemCRegV1<Problem>;
    using Epilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         ADataType,
                                         ck_tile::tuple<>,
                                         AccType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         Row,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         Config::M_Warp,
                                         Config::N_Warp,
                                         Config::M_Warp_Tile,
                                         Config::N_Warp_Tile,
                                         Config::K_Warp_Tile,
                                         Problem::TransposeC,
                                         Config::NumWaveGroups,
                                         false, // FixedVectorSize
                                         1,     // VectorSizeC
                                         Config::BlockedXDLN_PerWarp>>;
    using Kernel =
        ck_tile::MoeFlatmmKernel<TilePartitioner, Pipeline, Epilogue, kind, ck_tile::moe::MoeSilu>;
};

template <ck_tile::MoeFlatmmKind kind,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename HostArgs>
float launch_mx_moe_gemm(const HostArgs& args,
                         const ck_tile::stream_config& s,
                         bool clear_c,
                         const float* topk_weights = nullptr, // gemm2_per_slot: [tokens, topk]
                         void* combined            = nullptr)            // gemm2_per_slot: bf16 [tokens, N]
{
    using Kernel      = typename MxMoeKernelTypes<kind, ADataType, BDataType, CDataType>::Kernel;
    using Config      = MXMoeFlatmmConfig;
    const auto kargs  = Kernel::MakeKernelArgs(args);
    const dim3 grids  = Kernel::GridSize(kargs);
    const dim3 blocks = Kernel::BlockSize();
    if(!Kernel::IsSupportedArgument(kargs))
        throw std::runtime_error("Arguments not supported by the MX MoE kernel");

    if(s.log_level_ > 0)
        std::cout << "kernel: " << Kernel::GetName() << "\ngrid: {" << grids.x << ", " << grids.y
                  << ", " << grids.z << "}, block: {" << blocks.x << "}" << std::endl;

    // gemm2 accumulates into C with atomics: clear it before every launch.
    const std::size_t c_bytes = (kind == ck_tile::MoeFlatmmKind::kFFN_gemm2
                                     ? std::size_t(args.NumTokens) * args.N
                                     : std::size_t(args.NumTokens) * args.TopK * (args.N / 2)) *
                                sizeof(CDataType);
    auto clear = [&]() {
        if(clear_c)
            HIP_CHECK_ERROR(hipMemsetAsync(args.e_ptr, 0, c_bytes, s.stream_id_));
    };
    const auto moe_kernel =
        ck_tile::make_kernel<Config::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs);
    if constexpr(kind == ck_tile::MoeFlatmmKind::kFFN_gemm2_per_slot)
    {
        // P = this kernel's output; the combine reads it and the router's top-k weights.
        using Combine     = ck_tile::MoeTopkCombineKernel<CDataType, ck_tile::bf16_t>;
        const auto ckargs = Combine::MakeKargs(
            args.e_ptr, topk_weights, combined, args.NumTokens, args.TopK, args.N, args.N, args.N);
        if(!Combine::IsSupportedArgument(ckargs))
            throw std::runtime_error("Arguments not supported by the top-k combine kernel");
        return ck_tile::launch_kernel_time_mask(
            s,
            clear,
            moe_kernel,
            ck_tile::make_kernel<1>(
                Combine{}, Combine::GridSize(ckargs), Combine::BlockSize(), 0, ckargs));
    }
    else
        return ck_tile::launch_kernel_time_mask(s, clear, moe_kernel);
}

// Random routing (top-k distinct experts per token, weights uniform in [0, 1), or 1 when init != 0)
// sorted on the host into expert tiles of m_tile rows, padded with the sentinel token id
// num_tokens, as moe_sorting does on the device.
struct MxMoeRouting
{
    std::vector<int> expert;   // [num_tokens * topk]
    std::vector<float> weight; // [num_tokens * topk], token order
    std::vector<ck_tile::index_t> sorted_ids, sorted_expert_ids;
    std::vector<float> sorted_weights;
    int max_rows_per_expert = 0;
    int M                   = 0; // sorted rows, padding included
};

inline MxMoeRouting
make_mx_moe_routing(int experts, int num_tokens, int topk, int init, unsigned seed, int m_tile)
{
    MxMoeRouting r;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wdist(0.f, 1.f);
    r.expert.resize(std::size_t(num_tokens) * topk);
    r.weight.resize(std::size_t(num_tokens) * topk);
    std::vector<int> perm(experts);
    for(int t = 0; t < num_tokens; ++t)
    {
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for(int s = 0; s < topk; ++s)
        {
            r.expert[t * topk + s] = perm[s];
            r.weight[t * topk + s] = init == 0 ? wdist(rng) : 1.f;
        }
    }
    for(int e = 0; e < experts; ++e)
    {
        int rows = 0;
        for(int t = 0; t < num_tokens; ++t)
            for(int s = 0; s < topk; ++s)
                if(r.expert[t * topk + s] == e)
                {
                    r.sorted_ids.push_back((s << 24) | t);
                    r.sorted_weights.push_back(r.weight[t * topk + s]);
                    ++rows;
                }
        r.max_rows_per_expert = std::max(r.max_rows_per_expert, rows);
        const int padded      = ck_tile::integer_least_multiple(rows, m_tile);
        for(int i = rows; i < padded; ++i)
        {
            r.sorted_ids.push_back(num_tokens); // sentinel: no token
            r.sorted_weights.push_back(0.f);
        }
        for(int tile = 0; tile < padded / m_tile; ++tile)
            r.sorted_expert_ids.push_back(e);
    }
    r.M = static_cast<int>(r.sorted_ids.size());
    return r;
}

// Value of element (r, k) of a row-major A or B tensor (packed fp4 or one byte per element).
template <typename T>
float elem_at(const ck_tile::HostTensor<T>& t, std::size_t r, std::size_t k)
{
    if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
        return fp4_at(t, r, k);
    else
        return ck_tile::type_convert<float>(t(r, k));
}

// ADataType: ck_tile::pk_fp4_t (a4w4) or ck_tile::fp8_t (a8w4).
template <ck_tile::MoeFlatmmKind kind, typename ADataType_ = ck_tile::pk_fp4_t>
MxMoeResult run_mx_moe_case(const MxMoeCase& c)
{
    using ADataType = ADataType_;
    static_assert(std::is_same_v<ADataType, ck_tile::pk_fp4_t> ||
                      std::is_same_v<ADataType, ck_tile::fp8_t>,
                  "A must be MXFP4 or MX FP8");
    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    using BDataType           = ck_tile::pk_fp4_t;
    using ScaleType           = ck_tile::e8m0_t;
    constexpr bool kGateUp    = kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up;
    constexpr bool kPerSlot   = kind == ck_tile::MoeFlatmmKind::kFFN_gemm2_per_slot;
    // A rows per (token, slot): gemm2 and gemm2_per_slot.
    constexpr bool kASlot = kind == ck_tile::MoeFlatmmKind::kFFN_gemm2 || kPerSlot;
    // gemm2 accumulates the top-k contributions of a token with fp32 atomics; gemm2_per_slot
    // stores them in fp32 and the combine writes bf16.
    using CDataType   = std::conditional_t<kGateUp, ck_tile::bf16_t, float>;
    using OutDataType = std::conditional_t<kPerSlot, ck_tile::bf16_t, CDataType>;

    const int experts    = c.experts;
    const int num_tokens = c.num_tokens;
    const int topk       = c.topk;
    const int N          = c.N;
    const int K          = c.K;
    const int init       = c.init;
    const unsigned seed  = c.seed;
    const int MTile      = MXMoeFlatmmConfig::M_Tile;

    if(topk > experts)
        throw std::runtime_error("TopK must not exceed the number of experts");
    if(K % MXMoeFlatmmConfig::K_Tile != 0 || N % MXMoeFlatmmConfig::N_Tile != 0)
        throw std::runtime_error("K must be a multiple of K_Tile (256) and N of N_Tile (256)");

    // ---- routing: top-k distinct experts per token, sorted into expert tiles ----
    const MxMoeRouting rt    = make_mx_moe_routing(experts, num_tokens, topk, init, seed, MTile);
    const auto& route_expert = rt.expert;
    const auto& route_weight = rt.weight;
    const auto& sorted_ids   = rt.sorted_ids;
    const auto& sorted_expert_ids = rt.sorted_expert_ids;
    const auto& sorted_weights    = rt.sorted_weights;
    const int max_rows_per_expert = rt.max_rows_per_expert;
    const int M                   = rt.M; // sorted rows, multiple of M_Tile

    const int rows_a  = kASlot ? num_tokens * topk : num_tokens;
    const int rows_c  = kASlot ? num_tokens : num_tokens * topk; // final output rows
    const int rows_p  = kPerSlot ? num_tokens * topk : rows_c;   // rows the MoE kernel writes
    const int outputN = kGateUp ? N / 2 : N;

    if(c.verbose)
        std::cout << "mx_moe_flatmm "
                  << (kGateUp    ? "gemm1_gate_up"
                      : kPerSlot ? "gemm2_per_slot"
                                 : "gemm2")
                  << ": experts " << experts << ", tokens " << num_tokens << ", topk " << topk
                  << ", N " << N << ", K " << K << ", sorted rows " << M
                  << ", most rows of one expert " << max_rows_per_expert << " (M_Tile " << MTile
                  << ")" << std::endl;
    if(c.verbose && max_rows_per_expert <= MTile)
        std::cout << "note: no expert spans more than one M tile; multi-tile experts are not "
                     "exercised"
                  << std::endl;

    // ---- operands ----
    ck_tile::HostTensor<ADataType> a_host({std::size_t(rows_a), std::size_t(K)},
                                          {std::size_t(K), std::size_t(1)});
    ck_tile::HostTensor<BDataType> b_host({std::size_t(experts) * N, std::size_t(K)},
                                          {std::size_t(K), std::size_t(1)});
    ck_tile::HostTensor<ScaleType> a_scale({std::size_t(rows_a), std::size_t(K / kScaleBlockK)},
                                           {std::size_t(K / kScaleBlockK), std::size_t(1)});
    ck_tile::HostTensor<ScaleType> b_scale(
        {std::size_t(experts) * N, std::size_t(K / kScaleBlockK)},
        {std::size_t(K / kScaleBlockK), std::size_t(1)});
    if(init == 0)
    {
        ck_tile::FillUniformDistribution<>{-1.f, 1.f, seed + 1}(a_host);
        ck_tile::FillUniformDistribution<>{-1.f, 1.f, seed + 2}(b_host);
        ck_tile::FillUniformDistribution<>{0.5f, 2.f, seed + 3}(a_scale);
        ck_tile::FillUniformDistribution<>{0.5f, 2.f, seed + 4}(b_scale);
    }
    else if(init == 2)
    {
        // Debug: row r of A holds one value v(r) (exact in fp4), B and scales are 1, so each
        // output row shows which A row was read.
        const float vals[7] = {0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        for(int r = 0; r < rows_a; ++r)
        {
            ck_tile::HostTensor<ADataType> row({std::size_t(1), std::size_t(K)},
                                               {std::size_t(K), std::size_t(1)});
            ck_tile::FillUniformDistribution<>{vals[r % 7], vals[r % 7]}(row);
            std::copy(row.begin(), row.end(), a_host.begin() + std::size_t(r) * K / APackedSize);
        }
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(b_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(a_scale);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(b_scale);
    }
    else
    {
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(b_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(a_scale);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(b_scale);
    }

    // Device layouts: preshuffled B; A scales in sorted-row order and B scales per expert, both
    // in the gfx1250 MX scale layout.
    std::vector<BDataType> b_shuffled(std::size_t(experts) * N * K / 2);
    shuffle_fp4_weight(b_host.data(), b_shuffled.data(), experts, N, K, kGateUp);

    const int KB = K / kScaleBlockK;
    std::vector<ScaleType> a_scale_sorted(std::size_t(M) * KB,
                                          ck_tile::type_convert<ScaleType>(1.f));
    for(int r = 0; r < M; ++r)
    {
        const int t = sorted_ids[r] & 0xffffff;
        const int s = sorted_ids[r] >> 24;
        if(t >= num_tokens)
            continue;
        const int ra = kASlot ? t * topk + s : t;
        for(int kb = 0; kb < KB; ++kb)
            a_scale_sorted[mx_scale_offset(r, kb, K)] = a_scale(ra, kb);
    }
    std::vector<ScaleType> b_scale_shuffled(std::size_t(experts) * N * KB);
    for(int e = 0; e < experts; ++e)
        for(int n = 0; n < N; ++n)
            for(int kb = 0; kb < KB; ++kb)
                b_scale_shuffled[std::size_t(e) * N * KB +
                                 mx_scale_offset(b_column(n, N, kGateUp), kb, K)] =
                    b_scale(std::size_t(e) * N + n, kb);

    ck_tile::DeviceMem a_dev(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev(b_shuffled.size() * sizeof(BDataType));
    ck_tile::DeviceMem a_scale_dev(a_scale_sorted.size() * sizeof(ScaleType));
    ck_tile::DeviceMem b_scale_dev(b_scale_shuffled.size() * sizeof(ScaleType));
    ck_tile::DeviceMem c_dev(std::size_t(rows_p) * outputN * sizeof(CDataType));
    // gemm2_per_slot: the router's top-k weights in token order and the combined output.
    ck_tile::DeviceMem topk_w_dev(kPerSlot ? route_weight.size() * sizeof(float) : 0);
    ck_tile::DeviceMem out_dev(kPerSlot ? std::size_t(rows_c) * outputN * sizeof(OutDataType) : 0);
    if constexpr(kPerSlot)
        topk_w_dev.ToDevice(route_weight.data());
    ck_tile::DeviceMem sorted_ids_dev(sorted_ids.size() * sizeof(ck_tile::index_t));
    ck_tile::DeviceMem sorted_eids_dev(sorted_expert_ids.size() * sizeof(ck_tile::index_t));
    ck_tile::DeviceMem sorted_w_dev(sorted_weights.size() * sizeof(float));
    ck_tile::DeviceMem max_token_dev(sizeof(ck_tile::index_t));
    const ck_tile::index_t max_token_id = M;

    a_dev.ToDevice(a_host.data());
    b_dev.ToDevice(b_shuffled.data());
    a_scale_dev.ToDevice(a_scale_sorted.data());
    b_scale_dev.ToDevice(b_scale_shuffled.data());
    sorted_ids_dev.ToDevice(sorted_ids.data());
    sorted_eids_dev.ToDevice(sorted_expert_ids.data());
    sorted_w_dev.ToDevice(sorted_weights.data());
    max_token_dev.ToDevice(&max_token_id);
    c_dev.SetZero();

    using ScaleA = ck_tile::FlatmmScalePointer<1, kScaleBlockK, ScaleType>;
    using ScaleB = ck_tile::FlatmmScalePointer<1, kScaleBlockK, ScaleType>;
    using Args   = ck_tile::MoeFlatmmHostArgs<ScaleA, ScaleB, ck_tile::FlatmmScalePointer<-1>>;
    Args args{static_cast<const ck_tile::index_t*>(sorted_ids_dev.GetDeviceBuffer()),
              sorted_w_dev.GetDeviceBuffer(),
              static_cast<const ck_tile::index_t*>(sorted_eids_dev.GetDeviceBuffer()),
              static_cast<const ck_tile::index_t*>(max_token_dev.GetDeviceBuffer()),
              a_dev.GetDeviceBuffer(),
              b_dev.GetDeviceBuffer(),
              c_dev.GetDeviceBuffer(),
              num_tokens,
              experts,
              topk,
              1, // k_batch
              M,
              N,
              K,
              K,       // stride_A
              K,       // stride_B
              outputN, // stride_C
              ScaleA{static_cast<const ScaleType*>(a_scale_dev.GetDeviceBuffer()), M},
              ScaleB{static_cast<const ScaleType*>(b_scale_dev.GetDeviceBuffer()), N},
              {}};

    // One untimed launch for the result.
    const auto* topk_w = static_cast<const float*>(topk_w_dev.GetDeviceBuffer());
    launch_mx_moe_gemm<kind, ADataType, BDataType, CDataType>(
        args,
        ck_tile::stream_config{nullptr, false, c.verbose ? 1 : 0},
        !kPerSlot,
        topk_w,
        out_dev.GetDeviceBuffer());
    std::vector<OutDataType> c_out(std::size_t(rows_c) * outputN);
    if constexpr(kPerSlot)
        out_dev.FromDevice(c_out.data());
    else
        c_dev.FromDevice(c_out.data());

    MxMoeResult result{true, 0.f, max_rows_per_expert, M};
    if(c.repeat > 0)
    {
        result.ms = launch_mx_moe_gemm<kind, ADataType, BDataType, CDataType>(
            args,
            ck_tile::stream_config{nullptr, true, 0, c.warmup, c.repeat, true},
            false, // a clear would be timed too; gemm2's output is not checked after this
            topk_w,
            out_dev.GetDeviceBuffer());
        const double flop = 2.0 * (std::size_t(num_tokens) * topk) * N * K;
        if(c.verbose)
            std::cout << "time " << result.ms << " ms, " << flop / 1e9 / result.ms
                      << " TFlop/s (valid rows only)" << std::endl;
    }

    if(c.validate)
    {
        // Dequantized operands.
        std::vector<float> af(std::size_t(rows_a) * K), bf(std::size_t(experts) * N * K);
        auto dequant = [&](auto& out, const auto& data, const auto& scale, std::size_t rows) {
            ck_tile::make_ParallelTensorFunctor(
                [&](std::size_t r) {
                    for(int k = 0; k < K; ++k)
                        out[r * K + k] = elem_at(data, r, k) *
                                         ck_tile::type_convert<float>(scale(r, k / kScaleBlockK));
                },
                rows)(std::thread::hardware_concurrency());
        };
        dequant(af, a_host, a_scale, rows_a);
        dequant(bf, b_host, b_scale, std::size_t(experts) * N);
        auto dot = [&](int ra, int e, int n) {
            const float* pa = &af[std::size_t(ra) * K];
            const float* pb = &bf[(std::size_t(e) * N + n) * K];
            float acc       = 0.f;
            for(int k = 0; k < K; ++k)
                acc += pa[k] * pb[k];
            return acc;
        };

        std::vector<float> c_ref(std::size_t(rows_c) * outputN);
        ck_tile::make_ParallelTensorFunctor(
            [&](std::size_t rc) {
                for(int n = 0; n < outputN; ++n)
                {
                    float v = 0.f;
                    if constexpr(kGateUp)
                    {
                        const int t = rc / topk, e = route_expert[rc];
                        v = ck_tile::moe::MoeSilu{}(dot(t, e, n), dot(t, e, n + N / 2));
                    }
                    else
                    {
                        for(int s = 0; s < topk; ++s)
                            v += route_weight[rc * topk + s] *
                                 dot(rc * topk + s, route_expert[rc * topk + s], n);
                    }
                    c_ref[rc * outputN + n] = v;
                }
            },
            rows_c)(std::thread::hardware_concurrency());

        std::vector<OutDataType> c_ref_typed(c_ref.size());
        std::transform(c_ref.begin(), c_ref.end(), c_ref_typed.begin(), [](float v) {
            return ck_tile::type_convert<OutDataType>(v);
        });
        const float rtol = 1e-2f, atol = 1e-2f;
        result.pass =
            ck_tile::check_err(c_out, c_ref_typed, "Error: incorrect results!", rtol, atol);
        if(c.verbose)
            std::cout << "CPU reference check (rtol " << rtol << ", atol " << atol
                      << "): " << (result.pass ? "correct" : "fail") << std::endl;
    }
    return result;
}

// ---- A whole MoE layer, a4w4 ----
//
//   1. gemm1_gate_up_mx_quant: X [tokens, hidden] x W_gate|up [experts, 2 * inter, hidden]
//      -> SiLU(gate) * up, quantized to MXFP4 in the epilogue: the intermediate
//      [tokens * topk, inter] (packed fp4) and its e8m0 scales in sorted-row order.
//   2. down projection, reading that intermediate as is: gemm2 (fp32 atomic combine, fp32
//      output) or gemm2_per_slot + MoeTopkCombineKernel (bf16 output).
//
// The check runs in two parts: the intermediate against a CPU quantization of the CPU
// SiLU(gate) * up (with the same scale rule and CK's host fp4 conversion; values whose
// fp32 input lands within rounding noise of a decision point may round the other way, so a
// small fraction is allowed to differ), and the layer output against a CPU down projection of
// the GPU intermediate.
struct MxMoeLayerCase
{
    int experts    = 8;
    int num_tokens = 512;
    int topk       = 2;
    int hidden     = 1024;
    int inter      = 512;
    bool per_slot  = true; // down projection: per-slot + combine, or the atomic gemm2
    // Run the layer as one persistent task-queue launch (MoeMxPersistentKernel): gate_up and
    // down tiles as tasks with per-row-tile dependencies. With per_slot, the combine kernel
    // follows as a second launch. false: two GEMM kernels (gate_up, then the down projection).
    bool persistent = false;
    // Persistent with the atomic combine: the most K slices the down tasks may take (the kernel
    // picks how many it uses). 0: one per 256 intermediate columns.
    int max_slices = 0;
    int init       = 0;
    unsigned seed  = 11939;
    bool validate  = true;
    int warmup     = 0;
    int repeat     = 0;
    bool verbose   = true;
};

// How to run a layer: combine (per-slot + MoeTopkCombineKernel, or fp32 atomics into Y) and
// launch (MoeMxPersistentKernel, or gate_up and down kernels).
struct MxMoeLayerPlan
{
    bool per_slot;
    bool persistent;
};

// Picks the faster plan for a layer from crossovers measured on gfx1250 (1 to 1024 tokens; row
// tiles are estimated as one per active expert plus one per 32 pairs):
// - Atomic combine for up to 128 (token, slot) pairs; past that its atomics cost more than the
//   per-slot store and the combine launch.
// - Persistent with the atomic combine when its tasks fit in part of the resident grid (its
//   synchronization then costs next to nothing, a launch is saved, and idle workgroups take K
//   slices of the down projection), or with at most 8 experts, whose long tasks gain the most
//   from overlapping the two projections.
// - Persistent with the per-slot combine from some number of pairs, where the overlap of the two
//   projections outweighs the task synchronization.
// The thresholds depend on the activation type (AType):
// - a4w4 (pk_fp4_t; 7 shapes: 8 to 256 experts, top-k 2 to 8, hidden 2048 to 7168, intermediate
//   768 to 7168; worst pick 6% slower than the best plan, 0.4% on average): atomic tasks within
//   half the grid; per-slot persistent from 1024 pairs with at most 16 experts or an intermediate
//   of 4096 or more.
// - a8w4 (fp8_t; 6 shapes: 8 to 128 experts, same ranges otherwise; worst pick 7% slower, 1.0%
//   on average): the heavier MX FP8 intermediate makes the overlap pay sooner: atomic tasks
//   within three quarters of the grid; per-slot persistent from 256 pairs with at most 128
//   experts.
template <typename AType = ck_tile::pk_fp4_t>
MxMoeLayerPlan
choose_mx_moe_layer_plan(int experts, int num_tokens, int topk, int hidden, int inter)
{
    using K            = ck_tile::MoeFlatmmKind;
    using F4           = ck_tile::pk_fp4_t;
    constexpr bool Fp8 = std::is_same_v<AType, ck_tile::fp8_t>;
    static_assert(Fp8 || std::is_same_v<AType, F4>, "AType is ck_tile::pk_fp4_t or ck_tile::fp8_t");
    constexpr K QuantKind =
        Fp8 ? K::kFFN_gemm1_gate_up_mxfp8_quant : K::kFFN_gemm1_gate_up_mx_quant;
    using GateUp    = typename MxMoeKernelTypes<QuantKind, AType, F4, float>::Kernel;
    using Down      = typename MxMoeKernelTypes<K::kFFN_gemm2, AType, F4, float>::Kernel;
    using P         = ck_tile::MoeMxPersistentKernel<GateUp, Down>;
    using SP        = ck_tile::FlatmmScalePointer<1, kScaleBlockK, ck_tile::e8m0_t>;
    using Args      = ck_tile::MoeFlatmmHostArgs<SP, SP, ck_tile::FlatmmScalePointer<-1>>;
    using PKargs    = decltype(P::MakeKargs(std::declval<Args>(), std::declval<Args>(), nullptr));
    const int pairs = num_tokens * topk;
    if(pairs <= 128)
    {
        const int row_tiles = std::min(experts, pairs) + pairs / MXMoeFlatmmConfig::M_Tile;
        const int tasks     = row_tiles * (2 * inter / MXMoeFlatmmConfig::N_Tile +
                                       hidden / MXMoeFlatmmConfig::N_Tile);
        const int grid      = P::template MaxResidentWorkgroups<PKargs>();
        // Atomic tasks within half (a4w4) or three quarters (a8w4) of the grid.
        const bool one_wave = Fp8 ? 4 * tasks <= 3 * grid : 2 * tasks <= grid;
        return {false, experts <= 8 || one_wave};
    }
    if constexpr(Fp8)
        return {true, pairs >= 256 && experts <= 128};
    else
        return {true, pairs >= 1024 && (experts <= 16 || inter >= 4096)};
}

struct MxMoeLayerResult
{
    bool pass;                    // intermediate and output checks (true if not validated)
    double intermediate_mismatch; // fraction of intermediate values that differ from the CPU
    float ms_gate_up;             // timed launches, 0 if not timed
    float ms_down;                // down projection (and combine for per-slot)
    int sorted_rows;
};

// AType: the activations' element type, ck_tile::pk_fp4_t (a4w4: MXFP4 X and intermediate) or
// ck_tile::fp8_t (a8w4: MX FP8 X and intermediate). The weights are MXFP4 either way.
template <typename AType = ck_tile::pk_fp4_t>
MxMoeLayerResult run_mx_moe_layer(const MxMoeLayerCase& c)
{
    using F4 = ck_tile::pk_fp4_t;
    using A  = AType;
    static_assert(std::is_same_v<A, F4> || std::is_same_v<A, ck_tile::fp8_t>,
                  "AType is ck_tile::pk_fp4_t or ck_tile::fp8_t");
    constexpr bool Fp8 = std::is_same_v<A, ck_tile::fp8_t>;
    using K            = ck_tile::MoeFlatmmKind;
    // gate_up with an epilogue that quantizes the intermediate to A's MX format.
    constexpr K QuantKind =
        Fp8 ? K::kFFN_gemm1_gate_up_mxfp8_quant : K::kFFN_gemm1_gate_up_mx_quant;
    using ScaleType         = ck_tile::e8m0_t;
    const int E             = c.experts;
    const int T             = c.num_tokens;
    const int topk          = c.topk;
    const int H             = c.hidden;
    const int I             = c.inter;
    const int MTile         = MXMoeFlatmmConfig::M_Tile;
    constexpr int kTokenBit = 24;
    if(H % MXMoeFlatmmConfig::K_Tile != 0 || I % MXMoeFlatmmConfig::K_Tile != 0 ||
       (2 * I) % MXMoeFlatmmConfig::N_Tile != 0 || H % MXMoeFlatmmConfig::N_Tile != 0)
        throw std::runtime_error("hidden and inter must be multiples of 256");

    const MxMoeRouting rt = make_mx_moe_routing(E, T, topk, c.init, c.seed, MTile);
    const int M           = rt.M;

    // Operands: X, W1 (gate rows [0, I), up rows [I, 2I) per expert), W2, with MX scales.
    auto tensor2d = [](std::size_t rows, std::size_t cols) {
        return std::pair<std::vector<std::size_t>, std::vector<std::size_t>>{{rows, cols},
                                                                             {cols, 1}};
    };
    auto [xl, xs]   = tensor2d(T, H);
    auto [w1l, w1s] = tensor2d(std::size_t(E) * 2 * I, H);
    auto [w2l, w2s] = tensor2d(std::size_t(E) * H, I);
    ck_tile::HostTensor<A> x(xl, xs);
    ck_tile::HostTensor<F4> w1(w1l, w1s), w2(w2l, w2s);
    auto [xsl, xss]   = tensor2d(T, H / kScaleBlockK);
    auto [w1sl, w1ss] = tensor2d(std::size_t(E) * 2 * I, H / kScaleBlockK);
    auto [w2sl, w2ss] = tensor2d(std::size_t(E) * H, I / kScaleBlockK);
    ck_tile::HostTensor<ScaleType> xsc(xsl, xss), w1sc(w1sl, w1ss), w2sc(w2sl, w2ss);
    const float lo = c.init == 0 ? -1.f : 1.f, hi = 1.f;
    const float slo = c.init == 0 ? 0.5f : 1.f, shi = c.init == 0 ? 2.f : 1.f;
    ck_tile::FillUniformDistribution<>{lo, hi, c.seed + 1}(x);
    ck_tile::FillUniformDistribution<>{lo, hi, c.seed + 2}(w1);
    ck_tile::FillUniformDistribution<>{lo, hi, c.seed + 3}(w2);
    ck_tile::FillUniformDistribution<>{slo, shi, c.seed + 4}(xsc);
    ck_tile::FillUniformDistribution<>{slo, shi, c.seed + 5}(w1sc);
    ck_tile::FillUniformDistribution<>{slo, shi, c.seed + 6}(w2sc);

    // Device layouts.
    std::vector<F4> w1_shuf(w1.mData.size()), w2_shuf(w2.mData.size());
    shuffle_fp4_weight(w1.data(), w1_shuf.data(), E, 2 * I, H, true);
    shuffle_fp4_weight(w2.data(), w2_shuf.data(), E, H, I, false);
    const int HB = H / kScaleBlockK, IB = I / kScaleBlockK;
    std::vector<ScaleType> x_scale_sorted(std::size_t(M) * HB,
                                          ck_tile::type_convert<ScaleType>(1.f));
    for(int r = 0; r < M; ++r)
    {
        const int t = rt.sorted_ids[r] & 0xffffff;
        if(t < T)
            for(int kb = 0; kb < HB; ++kb)
                x_scale_sorted[mx_scale_offset(r, kb, H)] = xsc(t, kb);
    }
    std::vector<ScaleType> w1_scale(std::size_t(E) * 2 * I * HB), w2_scale(std::size_t(E) * H * IB);
    for(int e = 0; e < E; ++e)
    {
        for(int n = 0; n < 2 * I; ++n)
            for(int kb = 0; kb < HB; ++kb)
                w1_scale[std::size_t(e) * 2 * I * HB +
                         mx_scale_offset(b_column(n, 2 * I, true), kb, H)] =
                    w1sc(std::size_t(e) * 2 * I + n, kb);
        for(int n = 0; n < H; ++n)
            for(int kb = 0; kb < IB; ++kb)
                w2_scale[std::size_t(e) * H * IB + mx_scale_offset(n, kb, I)] =
                    w2sc(std::size_t(e) * H + n, kb);
    }

    using OutDataType = float; // the atomic gemm2 output; per-slot writes bf16
    // Row stride of the intermediate, in values: the persistent kernel needs whole 256-byte rows
    // (512 fp4 or 256 fp8 values; see MoeMxPersistentKernel::IsSupportedArgument).
    const int I_stride = c.persistent ? ck_tile::integer_least_multiple(I, Fp8 ? 256 : 512) : I;
    const std::size_t mid_bytes = std::size_t(T) * topk * I_stride / (Fp8 ? 1 : 2);
    ck_tile::DeviceMem x_dev(x.get_element_space_size_in_bytes()), w1_dev(w1_shuf.size()),
        w2_dev(w2_shuf.size()), xs_dev(x_scale_sorted.size()), w1s_dev(w1_scale.size()),
        w2s_dev(w2_scale.size()), mid_dev(mid_bytes), mids_dev(std::size_t(M) * IB),
        p_dev(std::size_t(T) * topk * H * sizeof(float)),
        out_dev(std::size_t(T) * H * sizeof(OutDataType)),
        out16_dev(std::size_t(T) * H * sizeof(ck_tile::bf16_t)), // per-slot combine output
        ids_dev(rt.sorted_ids.size() * sizeof(ck_tile::index_t)),
        eids_dev(rt.sorted_expert_ids.size() * sizeof(ck_tile::index_t)),
        sw_dev(rt.sorted_weights.size() * sizeof(float)), tw_dev(rt.weight.size() * sizeof(float)),
        mt_dev(sizeof(ck_tile::index_t));
    x_dev.ToDevice(x.data());
    w1_dev.ToDevice(w1_shuf.data());
    w2_dev.ToDevice(w2_shuf.data());
    xs_dev.ToDevice(x_scale_sorted.data());
    w1s_dev.ToDevice(w1_scale.data());
    w2s_dev.ToDevice(w2_scale.data());
    ids_dev.ToDevice(rt.sorted_ids.data());
    eids_dev.ToDevice(rt.sorted_expert_ids.data());
    sw_dev.ToDevice(rt.sorted_weights.data());
    tw_dev.ToDevice(rt.weight.data());
    const ck_tile::index_t max_token_id = M;
    mt_dev.ToDevice(&max_token_id);
    mid_dev.SetZero();
    mids_dev.SetZero();
    out_dev.SetZero();

    using SP   = ck_tile::FlatmmScalePointer<1, kScaleBlockK, ScaleType>;
    using Args = ck_tile::MoeFlatmmHostArgs<SP, SP, ck_tile::FlatmmScalePointer<-1>>;
    auto ids   = static_cast<const ck_tile::index_t*>(ids_dev.GetDeviceBuffer());
    auto eids  = static_cast<const ck_tile::index_t*>(eids_dev.GetDeviceBuffer());
    auto mtok  = static_cast<const ck_tile::index_t*>(mt_dev.GetDeviceBuffer());
    Args gate_up{ids,
                 sw_dev.GetDeviceBuffer(),
                 eids,
                 mtok,
                 x_dev.GetDeviceBuffer(),
                 w1_dev.GetDeviceBuffer(),
                 mid_dev.GetDeviceBuffer(),
                 T,
                 E,
                 topk,
                 1,
                 M,
                 2 * I,
                 H,
                 H,
                 H,
                 I_stride, // stride_C: the intermediate's row stride, in values
                 SP{static_cast<const ScaleType*>(xs_dev.GetDeviceBuffer()), M},
                 SP{static_cast<const ScaleType*>(w1s_dev.GetDeviceBuffer()), 2 * I},
                 {}};
    gate_up.e_scale_ptr     = mids_dev.GetDeviceBuffer();
    const bool per_slot_out = c.per_slot;
    void* down_out          = per_slot_out ? p_dev.GetDeviceBuffer() : out_dev.GetDeviceBuffer();
    Args down{ids,
              sw_dev.GetDeviceBuffer(),
              eids,
              mtok,
              mid_dev.GetDeviceBuffer(),
              w2_dev.GetDeviceBuffer(),
              down_out,
              T,
              E,
              topk,
              1,
              M,
              H,
              I,
              I_stride,
              I,
              H,
              SP{static_cast<const ScaleType*>(mids_dev.GetDeviceBuffer()), M},
              SP{static_cast<const ScaleType*>(w2s_dev.GetDeviceBuffer()), H},
              {}};
    const auto* tw = static_cast<const float*>(tw_dev.GetDeviceBuffer());

    auto run1 = [&](const ck_tile::stream_config& s) {
        return launch_mx_moe_gemm<QuantKind, A, F4, float>(gate_up, s, false);
    };
    using GateUpKernel = typename MxMoeKernelTypes<QuantKind, A, F4, float>::Kernel;
    using DownKernel   = typename MxMoeKernelTypes<K::kFFN_gemm2, A, F4, float>::Kernel;
    // Persistent task-queue launch; its counters live in a workspace zeroed once here.
    using DownSlotKernel = typename MxMoeKernelTypes<K::kFFN_gemm2_per_slot, A, F4, float>::Kernel;
    using PersistentAtomic  = ck_tile::MoeMxPersistentKernel<GateUpKernel, DownKernel>;
    using PersistentPerSlot = ck_tile::MoeMxPersistentKernel<GateUpKernel, DownSlotKernel>;
    // Most K slices of the persistent down tasks (the kernel picks how many it uses): max_slices
    // if given, else one per 256 columns with the atomic combine; per-slot output takes one.
    const int persistent_slices =
        per_slot_out ? 1 : (c.max_slices > 0 ? c.max_slices : PersistentAtomic::MaxUsefulSlices(I));
    ck_tile::DeviceMem ws_dev(PersistentAtomic::WorkspaceWords(M / MTile, persistent_slices) *
                              sizeof(int32_t));
    ws_dev.SetZero();
    auto run_persistent = [&](const ck_tile::stream_config& s, bool clear) {
        auto launch = [&](auto kernel_type) {
            using P = decltype(kernel_type);
            const auto pk =
                P::MakeKargs(gate_up, down, ws_dev.GetDeviceBuffer(), persistent_slices);
            if(!P::IsSupportedArgument(pk))
                throw std::runtime_error("Arguments not supported by the persistent MX MoE kernel");
            return ck_tile::make_kernel<1>(P{}, P::GridSize(pk), P::BlockSize(), 0, pk);
        };
        auto clear_out = [&]() {
            if(clear && !per_slot_out)
                HIP_CHECK_ERROR(hipMemsetAsync(
                    out_dev.GetDeviceBuffer(), 0, out_dev.GetBufferSize(), s.stream_id_));
        };
        if(per_slot_out)
        {
            using Combine     = ck_tile::MoeTopkCombineKernel<float, ck_tile::bf16_t>;
            const auto ckargs = Combine::MakeKargs(
                p_dev.GetDeviceBuffer(), tw, out16_dev.GetDeviceBuffer(), T, topk, H, H, H);
            return ck_tile::launch_kernel_time_mask(
                s,
                clear_out,
                launch(PersistentPerSlot{}),
                ck_tile::make_kernel<1>(
                    Combine{}, Combine::GridSize(ckargs), Combine::BlockSize(), 0, ckargs));
        }
        return ck_tile::launch_kernel_time_mask(s, clear_out, launch(PersistentAtomic{}));
    };
    auto run2 = [&](const ck_tile::stream_config& s, bool clear) {
        return per_slot_out ? launch_mx_moe_gemm<K::kFFN_gemm2_per_slot, A, F4, float>(
                                  down, s, false, tw, out16_dev.GetDeviceBuffer())
                            : launch_mx_moe_gemm<K::kFFN_gemm2, A, F4, float>(down, s, clear);
    };
    if(c.persistent)
    {
        // Twice: the checked launch runs on the counters the first one reset.
        run_persistent(ck_tile::stream_config{nullptr, false}, true);
        run_persistent(ck_tile::stream_config{nullptr, false}, true);
    }
    else
    {
        run1(ck_tile::stream_config{nullptr, false});
        run2(ck_tile::stream_config{nullptr, false}, true);
    }

    std::vector<uint8_t> mid(mid_bytes), mid_scale(std::size_t(M) * IB);
    mid_dev.FromDevice(mid.data());
    mids_dev.FromDevice(mid_scale.data());
    std::vector<float> out(std::size_t(T) * H);
    if(per_slot_out)
    {
        std::vector<ck_tile::bf16_t> ob(out.size());
        out16_dev.FromDevice(ob.data());
        for(std::size_t i = 0; i < out.size(); ++i)
            out[i] = ck_tile::type_convert<float>(ob[i]);
    }
    else
        out_dev.FromDevice(out.data());

    MxMoeLayerResult res{true, 0.0, 0.f, 0.f, M};
    if(c.repeat > 0)
    {
        const ck_tile::stream_config s{nullptr, true, 0, c.warmup, c.repeat, true};
        if(c.persistent)
            res.ms_gate_up = run_persistent(s, false); // the whole layer; ms_down stays 0
        else
        {
            res.ms_gate_up = run1(s);
            res.ms_down    = run2(s, false);
        }
    }
    if(!c.validate)
        return res;

    // ---- reference ----
    std::vector<float> xf(std::size_t(T) * H), w1f(std::size_t(E) * 2 * I * H),
        w2f(std::size_t(E) * H * I);
    auto dequant =
        [&](std::vector<float>& dst, const auto& v, const auto& sc, std::size_t rows, int K) {
            ck_tile::make_ParallelTensorFunctor(
                [&](std::size_t r) {
                    for(int k = 0; k < K; ++k)
                        dst[r * K + k] = elem_at(v, r, k) *
                                         ck_tile::type_convert<float>(sc(r, k / kScaleBlockK));
                },
                rows)(std::thread::hardware_concurrency());
        };
    dequant(xf, x, xsc, T, H);
    dequant(w1f, w1, w1sc, std::size_t(E) * 2 * I, H);
    dequant(w2f, w2, w2sc, std::size_t(E) * H, I);
    auto dot = [](const float* a, const float* b, int K) {
        float acc = 0.f;
        for(int k = 0; k < K; ++k)
            acc += a[k] * b[k];
        return acc;
    };
    // Sorted row of each (token, slot), for the intermediate's scales.
    std::vector<int> sorted_row(std::size_t(T) * topk, -1);
    for(int r = 0; r < M; ++r)
    {
        const int t = rt.sorted_ids[r] & 0xffffff, s = rt.sorted_ids[r] >> kTokenBit;
        if(t < T)
            sorted_row[std::size_t(t) * topk + s] = r;
    }
    // Intermediate: CPU SiLU(gate) * up quantized to MXFP4 / MX FP8 vs the GPU's, both dequantized.
    std::vector<float> mid_gpu(std::size_t(T) * topk * I);
    std::vector<long> mismatches(std::size_t(T) * topk, 0);
    ck_tile::make_ParallelTensorFunctor(
        [&](std::size_t rc) {
            const int t = rc / topk, e = rt.expert[rc], r = sorted_row[rc];
            std::vector<float> h(I);
            for(int n = 0; n < I; ++n)
                h[n] = ck_tile::moe::MoeSilu{}(
                    dot(&xf[std::size_t(t) * H], &w1f[(std::size_t(e) * 2 * I + n) * H], H),
                    dot(&xf[std::size_t(t) * H], &w1f[(std::size_t(e) * 2 * I + I + n) * H], H));
            for(int kb = 0; kb < IB; ++kb)
            {
                float amax = 0.f;
                for(int j = 0; j < kScaleBlockK; ++j)
                    amax = std::max(amax, std::abs(h[kb * kScaleBlockK + j]));
                const int e_amax =
                    static_cast<int>((ck_tile::bit_cast<uint32_t>(amax) >> 23) & 0xff) - 127;
                // Scale exponent: floor(log2(amax)) - largest element exponent (fp4 2, fp8 8).
                const int shared      = std::max(-127, std::min(127, e_amax - (Fp8 ? 8 : 2)));
                const float scale_ref = std::ldexp(1.f, shared);
                const int code = mid_scale[std::size_t(r / 32) * (I / 128) * 128 + (kb / 4) * 128 +
                                           (r % 32) * 4 + kb % 4];
                const float scale_gpu = std::ldexp(1.f, code - 127);
                for(int j = 0; j < kScaleBlockK; ++j)
                {
                    const int n = kb * kScaleBlockK + j;
                    float gpu;
                    // CPU quantization of a value: clamp to the element range after scaling (CK's
                    // host conversions do not saturate correctly: fp4 maps every value beyond 6
                    // to +6, dropping the sign; the kernel clamps fp8 to 448 itself), then round
                    // to nearest even.
                    auto quantize = [&](float val) {
                        if constexpr(Fp8)
                        {
                            const float xc =
                                std::max(-448.f * scale_ref, std::min(448.f * scale_ref, val));
                            return ck_tile::type_convert<float>(
                                       ck_tile::type_convert<ck_tile::fp8_t>(xc / scale_ref)) *
                                   scale_ref;
                        }
                        else
                        {
                            const float xc =
                                std::max(-6.f * scale_ref, std::min(6.f * scale_ref, val));
                            return ck_tile::type_convert<float>(
                                       ck_tile::float_to_pk_fp4(xc, scale_ref)
                                           .template unpack<>(ck_tile::number<0>{})) *
                                   scale_ref;
                        }
                    };
                    if constexpr(Fp8)
                        gpu = ck_tile::type_convert<float>(
                                  ck_tile::bit_cast<ck_tile::fp8_t>(mid[rc * I_stride + n])) *
                              scale_gpu;
                    else
                    {
                        const uint8_t byte = mid[rc * (I_stride / 2) + n / 2];
                        const uint8_t nib  = n % 2 == 0 ? (byte & 0xf) : (byte >> 4);
                        gpu                = ck_tile::type_convert<float>(
                                  ck_tile::pk_fp4_t(ck_tile::pk_fp4_raw_t(nib))
                                      .template unpack<>(ck_tile::number<0>{})) *
                              scale_gpu;
                    }
                    const float ref = quantize(h[n]);
                    // The CPU sums the dot products in another order than the GPU, so a value on
                    // a rounding midpoint can round either way (seen for fp8, about 1 value in
                    // millions). Count only differences that a 1e-5 relative nudge of the CPU
                    // value does not explain.
                    // NaN compares false either way, so test it explicitly.
                    const bool differs = std::isnan(gpu) || gpu < ref || gpu > ref;
                    const bool tie     = differs && !std::isnan(gpu) &&
                                     (!(gpu < quantize(h[n] * (1.f + 1e-5f)) ||
                                        gpu > quantize(h[n] * (1.f + 1e-5f))) ||
                                      !(gpu < quantize(h[n] * (1.f - 1e-5f)) ||
                                        gpu > quantize(h[n] * (1.f - 1e-5f))));
                    mid_gpu[rc * I + n] = gpu;
                    mismatches[rc] += differs && !tie ? 1 : 0;
                }
            }
        },
        std::size_t(T) * topk)(std::thread::hardware_concurrency());
    long total_mismatch = 0;
    for(auto m : mismatches)
        total_mismatch += m;
    res.intermediate_mismatch = double(total_mismatch) / (double(T) * topk * I);
    const bool mid_ok         = res.intermediate_mismatch < 1e-3;

    // Output: CPU down projection and weighted combine of the GPU intermediate.
    std::vector<float> ref(std::size_t(T) * H, 0.f);
    ck_tile::make_ParallelTensorFunctor(
        [&](std::size_t t) {
            for(int n = 0; n < H; ++n)
            {
                float v = 0.f;
                for(int s = 0; s < topk; ++s)
                {
                    const std::size_t rc = t * topk + s;
                    v += rt.weight[rc] *
                         dot(&mid_gpu[rc * I], &w2f[(std::size_t(rt.expert[rc]) * H + n) * I], I);
                }
                ref[t * H + n] =
                    per_slot_out
                        ? ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::bf16_t>(v))
                        : v;
            }
        },
        T)(std::thread::hardware_concurrency());
    float ref_max = 0.f;
    for(float v : ref)
        ref_max = std::max(ref_max, std::abs(v));
    const bool out_ok =
        ck_tile::check_err(out, ref, "layer output: incorrect results!", 1e-2, 1e-3 * ref_max);
    res.pass = mid_ok && out_ok;
    if(c.verbose)
        std::cout << "mx_moe layer " << (Fp8 ? "a8w4" : "a4w4") << " ("
                  << (c.persistent
                          ? std::string("persistent, ") + (per_slot_out ? "per-slot" : "atomic")
                      : per_slot_out ? std::string("per-slot")
                                     : std::string("atomic"))
                  << "): experts " << E << ", tokens " << T << ", topk " << topk << ", hidden " << H
                  << ", inter " << I
                  << "; intermediate values differing from the CPU quantization: "
                  << res.intermediate_mismatch * 100 << "%; output "
                  << (out_ok ? "correct" : "fail") << std::endl;
    return res;
}
