// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MX (microscaling) MoE GEMM on gfx1250: A and B are MXFP4 (pk_fp4_t) with one e8m0 scale per
// 32 K values. One launch runs one FFN GEMM of an MoE layer:
//
//   gemm1_gate_up  A = tokens          [num_tokens, K]          (gathered by sorted token id)
//                  B = gate|up weights [experts, N, K]          (N = 2 * intermediate)
//                  C = [num_tokens * topk, N / 2] = silu(A B_gate) * (A B_up)
//   gemm2          A = intermediate    [num_tokens * topk, K]
//                  B = down weights    [experts, N, K]
//                  C = [num_tokens, N] += routing weight * (A B), accumulated with atomics
//
// The routing is random (top-k distinct experts per token) and sorted on the host into expert
// tiles of M_Tile rows, padded with a sentinel token id, as moe_sorting does on the device.
// A scales are passed in sorted-row order, B scales per expert, both in the gfx1250 MX layout.
// The result is checked against a CPU reference computed from the dequantized operands.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"

// Block tile 128 x 512 x 256, four waves along N, 16x16x128 scaled WMMA: the tile the MX
// pipeline is tested with in tile_example_mx_flatmm (fp4). Smaller tiles, which suit MoE better,
// do not work with the pipeline yet (32-row tiles give wrong results, 256-column tiles race).
struct MXMoeFlatmmConfig
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 512;
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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("experts", "4", "number of experts")
        .insert("NumTokens", "512", "number of tokens")
        .insert("TopK", "2", "experts per token")
        .insert("N",
                "1024",
                "N of the GEMM, a multiple of 512 (gemm1_gate_up: 2 x intermediate; gemm2: hidden)")
        .insert("K", "1024", "K of the GEMM (gemm1_gate_up: hidden; gemm2: intermediate)")
        .insert("gemm_kind", "gemm1_gate_up", "gemm1_gate_up | gemm2")
        .insert("mx_prec", "fp4xfp4", "A x B data types; supported: fp4xfp4")
        .insert("init",
                "0",
                "0: random, 1: constant (all 1), 2: row r of A = v(r), rest 1 (row-mapping check)")
        .insert("validate", "1", "0: no validation, 1: validate against a CPU reference")
        .insert("warmup", "0", "warm-up launches before timing (0 with repeat 0: no timing)")
        .insert("repeat", "0", "timed launches")
        .insert("seed", "11939", "seed of the routing and data");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Offset of scale (mn, kb) in the gfx1250 MX scale layout [MN / 32][K / 128][32][4].
static std::size_t mx_scale_offset(std::size_t mn, std::size_t kb, std::size_t K)
{
    const std::size_t k_packs = K / (kScaleBlockK * kScalePack);
    return (mn / kScaleRows) * k_packs * kScaleRows * kScalePack +
           (kb / kScalePack) * kScaleRows * kScalePack + (mn % kScaleRows) * kScalePack +
           kb % kScalePack;
}

// Column of B as the kernel sees it. gemm1_gate_up interleaves gate and up in blocks of 16
// columns (gate block j -> 2j, up block j -> 2j + 1), so that a wave's pair of 16-column blocks
// holds matching gate and up columns.
static int b_column(int n, int N, bool gate_up)
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
static void shuffle_fp4_weight(
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
static float fp4_at(const ck_tile::HostTensor<ck_tile::pk_fp4_t>& t, std::size_t r, std::size_t k)
{
    const auto pk                 = t(r, k);
    const std::size_t flat_offset = t.mDesc.GetOffsetFromMultiIndex(r, k);
    return flat_offset % 2 == 0
               ? ck_tile::type_convert<float>(pk.template unpack<>(ck_tile::number<0>{}))
               : ck_tile::type_convert<float>(pk.template unpack<>(ck_tile::number<1>{}));
}

template <ck_tile::MoeFlatmmKind kind,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename HostArgs>
float launch_mx_moe_gemm(const HostArgs& args, const ck_tile::stream_config& s, bool clear_c)
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
    return ck_tile::launch_kernel_time_mask(
        s, clear, ck_tile::make_kernel<Config::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

template <ck_tile::MoeFlatmmKind kind>
int run(const ck_tile::ArgParser& arg_parser)
{
    using ADataType        = ck_tile::pk_fp4_t;
    using BDataType        = ck_tile::pk_fp4_t;
    using ScaleType        = ck_tile::e8m0_t;
    constexpr bool kGateUp = kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up;
    // gemm2 accumulates the top-k contributions of a token with atomics; fp32 keeps that exact
    // up to the accumulation order.
    using CDataType = std::conditional_t<kGateUp, ck_tile::bf16_t, float>;

    const int experts    = arg_parser.get_int("experts");
    const int num_tokens = arg_parser.get_int("NumTokens");
    const int topk       = arg_parser.get_int("TopK");
    const int N          = arg_parser.get_int("N");
    const int K          = arg_parser.get_int("K");
    const int init       = arg_parser.get_int("init");
    const int warmup     = arg_parser.get_int("warmup");
    const int repeat     = arg_parser.get_int("repeat");
    const unsigned seed  = arg_parser.get_uint32("seed");
    const int MTile      = MXMoeFlatmmConfig::M_Tile;

    if(topk > experts)
        throw std::runtime_error("TopK must not exceed the number of experts");
    if(K % MXMoeFlatmmConfig::K_Tile != 0 || N % MXMoeFlatmmConfig::N_Tile != 0)
        throw std::runtime_error("K must be a multiple of K_Tile (256) and N of N_Tile (512)");

    // ---- routing: top-k distinct experts per token, sorted into expert tiles ----
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wdist(0.f, 1.f);
    std::vector<int> route_expert(std::size_t(num_tokens) * topk);
    std::vector<float> route_weight(std::size_t(num_tokens) * topk);
    std::vector<int> perm(experts);
    for(int t = 0; t < num_tokens; ++t)
    {
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for(int s = 0; s < topk; ++s)
        {
            route_expert[t * topk + s] = perm[s];
            route_weight[t * topk + s] = init == 0 ? wdist(rng) : 1.f;
        }
    }

    std::vector<ck_tile::index_t> sorted_ids, sorted_expert_ids;
    std::vector<float> sorted_weights;
    int max_rows_per_expert = 0;
    for(int e = 0; e < experts; ++e)
    {
        int rows = 0;
        for(int t = 0; t < num_tokens; ++t)
            for(int s = 0; s < topk; ++s)
                if(route_expert[t * topk + s] == e)
                {
                    sorted_ids.push_back((s << 24) | t);
                    sorted_weights.push_back(route_weight[t * topk + s]);
                    ++rows;
                }
        max_rows_per_expert = std::max(max_rows_per_expert, rows);
        const int padded    = ck_tile::integer_least_multiple(rows, MTile);
        for(int r = rows; r < padded; ++r)
        {
            sorted_ids.push_back(num_tokens); // sentinel: no token
            sorted_weights.push_back(0.f);
        }
        for(int tile = 0; tile < padded / MTile; ++tile)
            sorted_expert_ids.push_back(e);
    }
    const int M = static_cast<int>(sorted_ids.size()); // sorted rows, multiple of M_Tile

    const int rows_a  = kGateUp ? num_tokens : num_tokens * topk;
    const int rows_c  = kGateUp ? num_tokens * topk : num_tokens;
    const int outputN = kGateUp ? N / 2 : N;

    std::cout << "mx_moe_flatmm " << (kGateUp ? "gemm1_gate_up" : "gemm2") << ": experts "
              << experts << ", tokens " << num_tokens << ", topk " << topk << ", N " << N << ", K "
              << K << ", sorted rows " << M << ", most rows of one expert " << max_rows_per_expert
              << " (M_Tile " << MTile << ")" << std::endl;
    if(max_rows_per_expert <= MTile)
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
            std::copy(row.begin(), row.end(), a_host.begin() + std::size_t(r) * K / 2);
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
        const int ra = kGateUp ? t : t * topk + s;
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
    ck_tile::DeviceMem c_dev(std::size_t(rows_c) * outputN * sizeof(CDataType));
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
    launch_mx_moe_gemm<kind, ADataType, BDataType, CDataType>(
        args, ck_tile::stream_config{nullptr, false, 1}, true);
    std::vector<CDataType> c_out(std::size_t(rows_c) * outputN);
    c_dev.FromDevice(c_out.data());

    if(repeat > 0)
    {
        const float ms = launch_mx_moe_gemm<kind, ADataType, BDataType, CDataType>(
            args, ck_tile::stream_config{nullptr, true, 0, warmup, repeat, true}, true);
        const double flop = 2.0 * (std::size_t(num_tokens) * topk) * N * K;
        std::cout << "time " << ms << " ms, " << flop / 1e9 / ms << " TFlop/s (valid rows only)"
                  << std::endl;
    }

    bool pass = true;
    if(arg_parser.get_int("validate"))
    {
        // Dequantized operands.
        std::vector<float> af(std::size_t(rows_a) * K), bf(std::size_t(experts) * N * K);
        auto dequant = [&](auto& out, const auto& data, const auto& scale, std::size_t rows) {
            ck_tile::make_ParallelTensorFunctor(
                [&](std::size_t r) {
                    for(int k = 0; k < K; ++k)
                        out[r * K + k] = fp4_at(data, r, k) *
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

        std::vector<CDataType> c_ref_typed(c_ref.size());
        std::transform(c_ref.begin(), c_ref.end(), c_ref_typed.begin(), [](float v) {
            return ck_tile::type_convert<CDataType>(v);
        });
        const float rtol = 1e-2f, atol = 1e-2f;
        pass = ck_tile::check_err(c_out, c_ref_typed, "Error: incorrect results!", rtol, atol);
        std::cout << "CPU reference check (rtol " << rtol << ", atol " << atol
                  << "): " << (pass ? "correct" : "fail") << std::endl;
    }
    return pass ? 0 : 1;
}

int main(int argc, char* argv[])
{
    auto [ok, arg_parser] = create_args(argc, argv);
    if(!ok)
        return EXIT_FAILURE;
    try
    {
        if(arg_parser.get_str("mx_prec") != "fp4xfp4")
            throw std::runtime_error("only mx_prec=fp4xfp4 is supported");
        const std::string kind = arg_parser.get_str("gemm_kind");
        if(kind == "gemm1_gate_up")
            return run<ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(arg_parser);
        if(kind == "gemm2")
            return run<ck_tile::MoeFlatmmKind::kFFN_gemm2>(arg_parser);
        throw std::runtime_error("gemm_kind must be gemm1_gate_up or gemm2");
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
