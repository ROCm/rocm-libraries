// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// FP8 batch-prefill page tile (fp8bf16): the data path of its policy
// (block_fmha_batch_prefill_pipeline_qr_ks_vs_tdm_sched_fp8_page_tile_policy.hpp) on one 128-key
// tile of a D64 P64 vectorized (x = 16) pool:
//  - the page-tile window writes the K and V tiles through a page table, one TDM box per wave;
//  - Q is loaded with the policy's Q distribution; K and V are read from the LDS tiles with the
//    policy's b64 reads; S = Q K^T is computed stage by stage; P = clamp(S, -16, 16) is converted
//    to FP8 and re-packed by the policy; O = P V.
// Integer-valued FP8 inputs keep every product and sum exact, so S and O must equal the host values
// bit for bit. Warp 0 records the LDS addresses of its reads: each wave-wide read must cover one
// contiguous 256-byte span of its tile (each of the 64 four-byte LDS banks once, no padding).
// Controls read K or V as whole 16-byte chunks, in the b128 order of AttrNumAccess::Double, which
// the Default-order WMMAs mis-pair: the K control must change S, the V control O alone. A further
// kernel runs the forward FP8 policy's split softmax on the page-tile problem.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_batch_prefill_qr_tdm_page_tile_window.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_tdm_sched_fp8_page_tile_policy.hpp"
#include "fmha_test_common.hpp"

namespace {

using namespace ck_tile;

constexpr index_t kPage    = 64;
constexpr index_t kHeadDim = 64;
constexpr index_t kTileM   = 128;
constexpr index_t kTileN   = 128;
using TypeConfig           = FmhaFwdTypeConfig<FmhaFwdFp8Bf16>;

template <index_t Page, BlockAttentionKVCacheMemoryLayoutEnum Layout>
struct ProblemFor
{
    using Shape = TileFmhaShape<sequence<kTileM, kTileN, kHeadDim, kHeadDim, kTileN, kHeadDim>,
                                sequence<4, 1, 1>,
                                sequence<16, 16, 64>,
                                sequence<4, 1, 1>,
                                sequence<16, 16, 128>,
                                true,
                                false>;
    using Traits =
        TileFmhaBatchPrefillTraits<true,
                                   true,
                                   false,
                                   false,
                                   false,
                                   BlockAttentionBiasEnum::NO_BIAS,
                                   false,
                                   false,
                                   false,
                                   BlockAttentionQuantScaleEnum::PERTENSOR,
                                   1,
                                   false,
                                   false,
                                   Page,
                                   Layout,
                                   BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D,
                                   BlockAttentionKVCacheLoadModeEnum::BUFFER_LOAD>;
    using type =
        BlockFmhaBatchPrefillPipelineProblem<typename TypeConfig::QDataType,
                                             typename TypeConfig::KDataType,
                                             typename TypeConfig::VDataType,
                                             typename TypeConfig::SaccDataType,
                                             typename TypeConfig::SMPLComputeDataType,
                                             typename TypeConfig::BiasDataType,
                                             typename TypeConfig::RandValOutputDataType,
                                             typename TypeConfig::LSEDataType,
                                             typename TypeConfig::PDataType,
                                             typename TypeConfig::OaccDataType,
                                             typename TypeConfig::ODataType,
                                             Shape,
                                             true,
                                             ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                             SimplifiedGenericAttentionMask<false>,
                                             false,
                                             Page,
                                             Traits>;
};

using Problem =
    typename ProblemFor<kPage, BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT>::type;
using Policy   = FmhaBatchPrefillFp8PageTilePolicyFor<Problem>;
using Geometry = typename Policy::Geometry;

// The policy's scope: the vectorized x = 16 pool with 32-row window boxes, not a linear pool or
// P16 pages (16-row boxes).
static_assert(Policy::template IsSupportedProblem<Problem>());
static_assert(
    !Policy::template IsSupportedProblem<
        typename ProblemFor<kPage, BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT>::type>());
static_assert(
    !Policy::template IsSupportedProblem<
        typename ProblemFor<16, BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT>::type>());
static_assert(Problem::kVectorSize == 16 && Policy::kVectorSize == 16);
// Arena: four unpadded 8 KB tiles; reads per stage and lane from the geometry (D9).
static_assert(Policy::kLdsOffsetK0 == 0 && Policy::kLdsOffsetK1 == 8192 &&
              Policy::kLdsOffsetV0 == 16384 && Policy::kLdsOffsetV1 == 24576 &&
              Policy::kLdsArenaSize == 32768);
static_assert(Geometry::kQkStages == 4 && Geometry::kQkSuColumns == 32 &&
              Geometry::kPvStages == 1 && Geometry::kKSuLoadCount == 8 &&
              Geometry::kVStageLoadCount == 32);
static_assert(Policy::Schedule::kNumQkStages == 4 && Policy::Schedule::kNumPvStages == 1);
static_assert(Policy::kBlockPerCu == 1 && Policy::Options::kScalarRowClamp == 1 &&
              Policy::Options::kWmmaOperandGuard && !Policy::kDrainLdsBeforeTdmIssue);
static_assert(Policy::template MakeQLdsBlockDescriptor<Problem>().get_element_space_size() <=
              Policy::kV1FootprintBytes);

struct Args
{
    const fp8_t* q;          // [kTileM][kHeadDim]
    const fp8_t* k_pool;     // [pages][1][D/16][P][16]
    const fp8_t* v_pool;     // [pages][1][P/16][D][16]
    const int32_t* page_row; // physical pages of logical pages 0 and 1
    long_index_t page_stride;
    float* s;           // [kTileM][kTileN]
    float* o;           // [kTileM][kHeadDim]
    uint32_t* k_reads;  // LDS byte address of each K read of warp 0: [stage][access][lane]
    uint32_t* v_reads;  // of each V read of warp 0: [access][lane]
    uint32_t* lds_base; // LDS byte address of the arena
};

// Contraction index of the run a whole-chunk (b128, AttrNumAccess::Double) read places where the
// Default order expects 16 k0 + 8 h: a lane's two runs of one chunk-pair become chunk 2 (k0 / 2) +
// h.
[[maybe_unused]] CK_TILE_HOST_DEVICE constexpr index_t WholeChunkOrder(index_t c)
{
    const index_t k0 = c / 16;
    const index_t h  = (c / 8) % 2;
    return 16 * (2 * (k0 / 2) + h) + 8 * (k0 % 2) + c % 8;
}

template <bool ControlK, bool ControlV>
struct OperandProbe
{
    static constexpr index_t kBlockSize = Geometry::kWaves * Geometry::kWaveSize;

    // The policy's reads; warp 0 also records the LDS byte address of each read in `addresses`.
    template <typename Reader, typename Tile, typename Window>
    CK_TILE_DEVICE static void ReadOperand(Tile& tile, const Window& window, uint32_t* addresses)
    {
        const uint32_t base = Reader::LaneBase(window);
        static_for<0, Reader::kNumAccess, 1>{}([&](auto a) {
            Reader::template Load<decltype(a)::value>(tile, base);
            if(get_warp_id() == 0)
                addresses[decltype(a)::value * get_warp_size() + get_lane_id()] =
                    base + static_cast<uint32_t>(Reader::template ByteDelta<decltype(a)::value>());
        });
    }

    // Control: each access at the whole-chunk position of its contraction index (the column of a
    // K read, the row of a V read).
    template <typename Reader, index_t ContractionDim, typename Tile, typename Window>
    CK_TILE_DEVICE static void ReadWholeChunks(Tile& tile, const Window& window)
    {
        const auto& view   = window.get_bottom_tensor_view();
        const uint32_t lds = fmha_tdm_sched_fp8_lds_offset(view.get_buffer_view().p_data_);
        static_for<0, Reader::kNumAccess, 1>{}([&](auto a) {
            auto index = Reader::template LaneBottomIndex<decltype(a)::value>();
            index(number<ContractionDim>{}) = WholeChunkOrder(index[number<ContractionDim>{}]);
            const auto coord                = make_tensor_coordinate(view.get_tensor_descriptor(),
                                                      window.get_window_origin() + index);
            Reader::template LoadAt<decltype(a)::value>(
                tile, lds + static_cast<uint32_t>(coord.get_offset()));
        });
    }

    // A [Rows][Cols] fp32 tile to row-major global memory.
    template <index_t Rows, index_t Cols, typename Tile>
    CK_TILE_DEVICE static void Store(float* out, const Tile& tile)
    {
        auto view = make_naive_tensor_view<address_space_enum::global>(
            out,
            make_tuple(number<Rows>{}, number<Cols>{}),
            make_tuple(number<Cols>{}, number<1>{}),
            number<1>{},
            number<1>{});
        auto window = make_tile_window(
            view, make_tuple(number<Rows>{}, number<Cols>{}), {0, 0}, tile.get_tile_distribution());
        store_tile(window, tile);
    }

    // The host pass dispatches the warp GEMMs to other shapes: the probes are device-only.
    CK_TILE_DEVICE void operator()(Args args) const
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        __shared__ char arena[Policy::kLdsArenaSize];
        if(threadIdx.x == 0)
            *args.lds_base = fmha_tdm_sched_fp8_lds_offset(arena);
        // An LDS byte that no TDM box wrote reads as an e4m3 NaN.
        for(index_t i = threadIdx.x; i < Policy::kLdsArenaSize; i += kBlockSize)
            arena[i] = static_cast<char>(0x7f);
        block_sync_lds<0>();

        // ---- K and V tiles, written by the page-tile window as the kernel creates it ----
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<fp8_t*>(arena + Policy::kLdsOffsetK0),
            Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<fp8_t*>(arena + Policy::kLdsOffsetV0),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        using KLengths = decltype(make_tuple(number<kTileN>{}, number<kHeadDim>{}));
        using VLengths = decltype(make_tuple(number<kHeadDim>{}, number<kTileN>{}));
        const auto k_source =
            FmhaBatchPrefillQrTdmPageTileWindow<fp8_t,
                                                KLengths,
                                                void,
                                                Policy::template GetAlignmentK<Problem>(),
                                                kPage,
                                                kHeadDim,
                                                Problem::kVectorSize,
                                                FmhaPageTileKVKind::kVectorizedK,
                                                Policy::Options::kScalarRowClamp,
                                                Policy::kDrainLdsBeforeTdmIssue>{
                args.k_pool, kTileN, kHeadDim, args.page_row, args.page_stride, {}, {0, 0}};
        const auto v_source =
            FmhaBatchPrefillQrTdmPageTileWindow<fp8_t,
                                                VLengths,
                                                void,
                                                Policy::template GetAlignmentV<Problem>(),
                                                kPage,
                                                kHeadDim,
                                                Problem::kVectorSize,
                                                FmhaPageTileKVKind::kVectorizedV,
                                                Policy::Options::kScalarRowClamp,
                                                Policy::kDrainLdsBeforeTdmIssue>{
                args.v_pool, kTileN, kHeadDim, args.page_row, args.page_stride, {}, {0, 0}};
        auto k_window = make_tile_window(
            k_source, {0, 0}, Policy::template MakeKDramTileDistribution<Problem>());
        auto v_window = make_tile_window(
            v_source, {0, 0}, Policy::template MakeVDramTileDistribution<Problem>());
        typename decltype(k_window)::PageIds page_ids{};
        page_ids.owned[0] = k_window.load_fragment_id(0, get_warp_id(), 0);
        k_window.set_page_id(page_ids);
        v_window.set_page_id(page_ids);
        const auto config_of = [](auto padding) {
            TDMConfig config{};
            config.pad_enable              = padding[number<0>{}];
            config.pad_config.pad_amount   = padding[number<1>{}];
            config.pad_config.pad_interval = padding[number<2>{}];
            return config;
        };
        auto k_write = make_tile_window(
            k_lds, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto v_write = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        load_tile_tdm(
            config_of(Policy::template GetLdsPaddingConfigK<Problem>()), k_write, k_window);
        load_tile_tdm(
            config_of(Policy::template GetLdsPaddingConfigV<Problem>()), v_write, v_window);
        s_wait_tensorcnt_barrier<0>();

        // ---- S = Q K^T, stage by stage ----
        auto q_view = make_naive_tensor_view<address_space_enum::global>(
            args.q,
            make_tuple(number<kTileM>{}, number<kHeadDim>{}),
            make_tuple(number<kHeadDim>{}, number<1>{}),
            number<8>{},
            number<1>{});
        const auto q_tile =
            load_tile(make_tile_window(q_view,
                                       make_tuple(number<kTileM>{}, number<kHeadDim>{}),
                                       {0, 0},
                                       Policy::template MakeQRegTileDistribution<Problem>()));
        constexpr auto gemm_0    = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_0_su = Policy::template GetQKBlockGemmSu<Problem>();
        using Qk                 = remove_cvref_t<decltype(gemm_0_su)>;
        static_assert(std::is_same_v<typename decltype(q_tile)::StaticTileDistribution::DstrEncode,
                                     remove_cvref_t<decltype(Qk::MakeABlockDistributionEncode())>>);
        auto s_acc = decltype(gemm_0.MakeCBlockTile()){};
        static_for<0, Geometry::kQkStages, 1>{}([&](auto stage) {
            constexpr index_t kStage = decltype(stage)::value;
            auto window =
                make_tile_window(k_lds,
                                 make_tuple(number<Geometry::kQkSuColumns>{}, number<kHeadDim>{}),
                                 {kStage * Geometry::kQkSuColumns, 0},
                                 Policy::template MakeKSuRegTileDistribution<Problem>());
            using KRead = typename Policy::template KReadFor<decltype(window)>;
            static_assert(KRead::kNumAccess == Geometry::kKSuLoadCount);
            auto k_tile = make_static_distributed_tensor<fp8_t>(
                Policy::template MakeKSuRegTileDistribution<Problem>());
            if constexpr(ControlK)
                ReadWholeChunks<KRead, 1>(k_tile, window);
            else
                ReadOperand<KRead>(
                    k_tile, window, args.k_reads + kStage * KRead::kNumAccess * get_warp_size());
            auto s_su = gemm_0_su.MakeCBlockTile();
            clear_tile(s_su);
            gemm_0_su(s_su, q_tile, k_tile);
            set_slice_tile(s_acc,
                           s_su,
                           sequence<0, kStage * Geometry::kQkSuColumns>{},
                           sequence<kTileM, (kStage + 1) * Geometry::kQkSuColumns>{});
        });
        Store<kTileM, kTileN>(args.s, s_acc);

        // ---- P = clamp(S, -16, 16) in FP8 (exact for these integers), re-packed by the policy
        // ----
        auto p_compute = s_acc;
        tile_elementwise_inout([](auto& x) { x = min(max(x, -16.0f), 16.0f); }, p_compute);
        const auto p_fp8      = cast_tile<fp8_t>(p_compute);
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();
        using Pv              = remove_cvref_t<decltype(gemm_1)>;
        const auto p_tile     = Policy::template RepackPForGemm1<Problem, Pv>(p_fp8);
        static_assert(std::is_same_v<typename decltype(p_tile)::StaticTileDistribution::DstrEncode,
                                     remove_cvref_t<decltype(Pv::MakeABlockDistributionEncode())>>);

        // ---- O = P V ----
        auto v_read_window = make_tile_window(v_lds,
                                              make_tuple(number<kTileN>{}, number<kHeadDim>{}),
                                              {0, 0},
                                              Policy::template MakeVRegTileDistribution<Problem>());
        using VRead        = typename Policy::template VReadFor<decltype(v_read_window)>;
        static_assert(VRead::kNumAccess == Geometry::kVStageLoadCount);
        auto v_tile = make_static_distributed_tensor<fp8_t>(typename VRead::OwnerDistribution{});
        static_assert(std::is_same_v<typename decltype(v_tile)::StaticTileDistribution::DstrEncode,
                                     remove_cvref_t<decltype(Pv::MakeBBlockDistributionEncode())>>);
        if constexpr(ControlV)
            ReadWholeChunks<VRead, 0>(v_tile, v_read_window);
        else
            ReadOperand<VRead>(v_tile, v_read_window, args.v_reads);
        auto o_acc = gemm_1.MakeCBlockTile();
        clear_tile(o_acc);
        gemm_1(o_acc, p_tile, v_tile);
        Store<kTileM, kHeadDim>(args.o, o_acc);
#else
        ignore = args;
#endif
    }
};

// The forward FP8 policy's split softmax (the FP8 page tile's, design.md "Reused without copying")
// on the page-tile problem's score tile: scores of 0 give P = 1 everywhere.
struct SoftmaxProbe
{
    static constexpr index_t kBlockSize = Geometry::kWaves * Geometry::kWaveSize;

    [[maybe_unused]] CK_TILE_DEVICE void operator()(float* out) const
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using ForwardFp8Policy =
            BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<Geometry, typename Policy::Tuning>;
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();
        auto score            = gemm_0.MakeCBlockTile();
        clear_tile(score);
        using OTile      = decltype(gemm_1.MakeCBlockTile());
        using STile      = decltype(cast_tile<float>(OTile{}));
        const auto f_max = [](auto a, auto b) { return max(a, b); };
        using RowTile = decltype(block_tile_reduce<float>(STile{}, sequence<1>{}, f_max, float{0}));
        RowTile row_max{};
        RowTile row_sum{};
        set_tile(row_max, -numeric<float>::infinity());
        clear_tile(row_sum);
        auto output    = ForwardFp8Policy::OutputFragments::MakeZero();
        float delta_m0 = 0, delta_m1 = 0, scale_m0 = 0, scale_m1 = 0;
        ForwardFp8Policy::template RunSplitSoftmaxPart01<Problem>(
            score, row_max, 1.0f, delta_m0, delta_m1);
        ForwardFp8Policy::template RunSplitSoftmaxPart2AndGetScale<Problem, false>(
            score, row_max, row_sum, 1.0f, delta_m0, delta_m1, scale_m0, scale_m1, output);
        OperandProbe<false, false>::Store<kTileM, kTileN>(out, score);
#else
        ignore = out;
#endif
    }
};

int Clamp16(int x) { return x < -16 ? -16 : (x > 16 ? 16 : x); }

struct Mismatches
{
    std::size_t s = 0;
    std::size_t o = 0;
    // Reads of warp 0 whose 32 lanes do not cover one contiguous, 8-byte aligned 256-byte span of
    // their tile (the bank model: each of 64 four-byte banks once).
    std::size_t scattered_reads = 0;
};

// The 32 lane addresses of one wave-wide b64 read: one contiguous 256-byte span inside
// [tile, tile + 8 KB).
bool ContiguousRead(std::vector<uint32_t> lanes, uint32_t tile)
{
    std::sort(lanes.begin(), lanes.end());
    for(std::size_t i = 0; i < lanes.size(); ++i)
        if(lanes[i] != lanes[0] + 8 * i)
            return false;
    return lanes[0] % 8 == 0 && lanes[0] >= tile && lanes.back() + 8 <= tile + 8192;
}

// Host data of one tile: Q, K in {-1, 0, 1} and V in {-2, ..., 2} (exact in e4m3); three physical
// pages, logical pages 0 and 1 at physical pages 2 and 0, physical page 1 holding other values.
struct Tile
{
    static constexpr index_t kPhysicalPages = 3;
    static constexpr index_t kPageElements  = kPage * kHeadDim; // one KV head
    std::vector<int> q, k, v;                                   // q [m][d], k / v [physical key][d]
    std::array<int32_t, 2> page_row{2, 0};

    explicit Tile(unsigned seed)
        : q(kTileM * kHeadDim), k(kPhysicalPages * kPage * kHeadDim), v(k.size())
    {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> qk(-1, 1), value(-2, 2);
        for(auto& x : q)
            x = qk(generator);
        for(auto& x : k)
            x = qk(generator);
        for(auto& x : v)
            x = value(generator);
    }

    // Logical key n of the tile -> physical key of the pool.
    int PhysicalKey(int n) const { return page_row[n / kPage] * kPage + n % kPage; }

    // Vectorized pools: K [page][D/16][P][16], V [page][P/16][D][16].
    std::vector<fp8_t> KPool() const
    {
        std::vector<fp8_t> pool(kPhysicalPages * kPageElements);
        for(int key = 0; key < kPhysicalPages * kPage; ++key)
            for(int d = 0; d < kHeadDim; ++d)
                pool[(key / kPage) * kPageElements + (d / 16) * kPage * 16 + (key % kPage) * 16 +
                     d % 16] = type_convert<fp8_t>(static_cast<float>(k[key * kHeadDim + d]));
        return pool;
    }

    std::vector<fp8_t> VPool() const
    {
        std::vector<fp8_t> pool(kPhysicalPages * kPageElements);
        for(int key = 0; key < kPhysicalPages * kPage; ++key)
            for(int d = 0; d < kHeadDim; ++d)
                pool[(key / kPage) * kPageElements + ((key % kPage) / 16) * kHeadDim * 16 + d * 16 +
                     key % 16] = type_convert<fp8_t>(static_cast<float>(v[key * kHeadDim + d]));
        return pool;
    }

    std::vector<float> ReferenceS() const
    {
        std::vector<float> s(kTileM * kTileN);
        for(int m = 0; m < kTileM; ++m)
            for(int n = 0; n < kTileN; ++n)
            {
                int dot = 0;
                for(int d = 0; d < kHeadDim; ++d)
                    dot += q[m * kHeadDim + d] * k[PhysicalKey(n) * kHeadDim + d];
                s[m * kTileN + n] = static_cast<float>(dot);
            }
        return s;
    }

    std::vector<float> ReferenceO(const std::vector<float>& s) const
    {
        std::vector<float> o(kTileM * kHeadDim);
        for(int m = 0; m < kTileM; ++m)
            for(int d = 0; d < kHeadDim; ++d)
            {
                int sum = 0;
                for(int n = 0; n < kTileN; ++n)
                    sum += Clamp16(static_cast<int>(s[m * kTileN + n])) *
                           v[PhysicalKey(n) * kHeadDim + d];
                o[m * kHeadDim + d] = static_cast<float>(sum);
            }
        return o;
    }
};

template <bool ControlK, bool ControlV>
Mismatches Run(const Tile& tile, const char* name)
{
    const auto k_pool = tile.KPool();
    const auto v_pool = tile.VPool();
    std::vector<fp8_t> q(tile.q.size());
    for(std::size_t i = 0; i < q.size(); ++i)
        q[i] = type_convert<fp8_t>(static_cast<float>(tile.q[i]));
    constexpr index_t kKReads = Geometry::kQkStages * Geometry::kKSuLoadCount * 32;
    constexpr index_t kVReads = Geometry::kVStageLoadCount * 32;
    DeviceMem dq(q.size()), dk(k_pool.size()), dv(v_pool.size()),
        dpages(tile.page_row.size() * sizeof(int32_t)), ds(kTileM * kTileN * sizeof(float)),
        dout(kTileM * kHeadDim * sizeof(float)), dk_reads(kKReads * sizeof(uint32_t)),
        dv_reads(kVReads * sizeof(uint32_t)), dbase(sizeof(uint32_t));
    dq.ToDevice(q.data());
    dk.ToDevice(k_pool.data());
    dv.ToDevice(v_pool.data());
    dpages.ToDevice(tile.page_row.data());
    const Args args{static_cast<const fp8_t*>(dq.GetDeviceBuffer()),
                    static_cast<const fp8_t*>(dk.GetDeviceBuffer()),
                    static_cast<const fp8_t*>(dv.GetDeviceBuffer()),
                    static_cast<const int32_t*>(dpages.GetDeviceBuffer()),
                    Tile::kPageElements,
                    static_cast<float*>(ds.GetDeviceBuffer()),
                    static_cast<float*>(dout.GetDeviceBuffer()),
                    static_cast<uint32_t*>(dk_reads.GetDeviceBuffer()),
                    static_cast<uint32_t*>(dv_reads.GetDeviceBuffer()),
                    static_cast<uint32_t*>(dbase.GetDeviceBuffer())};
    using Probe = OperandProbe<ControlK, ControlV>;
    launch_kernel(stream_config{nullptr, false, 0},
                  make_kernel<1, gfx125_t>(Probe{}, dim3(1), dim3(Probe::kBlockSize), 0, args));
    hip_check_error(hipDeviceSynchronize());
    std::vector<float> s(kTileM * kTileN), o(kTileM * kHeadDim);
    ds.FromDevice(s.data());
    dout.FromDevice(o.data());
    const auto s_ref = tile.ReferenceS();
    const auto o_ref = tile.ReferenceO(s_ref);
    Mismatches result;
    for(std::size_t i = 0; i < s.size(); ++i)
        result.s += s[i] != s_ref[i];
    for(std::size_t i = 0; i < o.size(); ++i)
        result.o += o[i] != o_ref[i];
    std::cout << name << ": S mismatches " << result.s << " / " << s.size() << ", O mismatches "
              << result.o << " / " << o.size();
    if constexpr(!ControlK && !ControlV)
    {
        std::vector<uint32_t> k_reads(kKReads), v_reads(kVReads);
        uint32_t base = 0;
        dk_reads.FromDevice(k_reads.data());
        dv_reads.FromDevice(v_reads.data());
        dbase.FromDevice(&base);
        for(index_t read = 0; read < kKReads / 32; ++read)
            result.scattered_reads +=
                !ContiguousRead({k_reads.begin() + read * 32, k_reads.begin() + (read + 1) * 32},
                                base + Policy::kLdsOffsetK0);
        for(index_t read = 0; read < kVReads / 32; ++read)
            result.scattered_reads +=
                !ContiguousRead({v_reads.begin() + read * 32, v_reads.begin() + (read + 1) * 32},
                                base + Policy::kLdsOffsetV0);
        std::cout << ", reads not one contiguous 256-byte span " << result.scattered_reads << " / "
                  << (kKReads + kVReads) / 32;
    }
    std::cout << '\n';
    return result;
}

} // namespace

// The K and V LDS tiles are the 5D layouts of the window's boxes (host check of the descriptors).
TEST(Fp8PageTileLayout, LdsTilesAre5D)
{
    constexpr auto k  = Policy::MakeKLdsBlockDescriptor<Problem>();
    constexpr auto v  = Policy::MakeVLdsBlockDescriptor<Problem>();
    std::size_t k_bad = 0, v_bad = 0;
    for(int n = 0; n < kTileN; ++n)
        for(int d = 0; d < kHeadDim; ++d)
        {
            k_bad += k.calculate_offset(make_multi_index(n, d)) !=
                     (n / 32) * 32 * kHeadDim + (d / 16) * 32 * 16 + (n % 32) * 16 + d % 16;
            v_bad += v.calculate_offset(make_multi_index(n, d)) !=
                     (n / 16) * kHeadDim * 16 + d * 16 + n % 16;
        }
    EXPECT_EQ(k_bad, 0u);
    EXPECT_EQ(v_bad, 0u);
    EXPECT_EQ(k.get_element_space_size(), kTileN * kHeadDim);
    EXPECT_EQ(v.get_element_space_size(), kTileN * kHeadDim);
}

using Fp8PageTileOperands = ck_tile::test::Gfx125FmhaTest;

TEST_F(Fp8PageTileOperands, PairExactly)
{
    for(unsigned seed : {1u, 2u})
    {
        const auto result = Run<false, false>(Tile(seed), "b64 reads");
        EXPECT_EQ(result.s, 0u);
        EXPECT_EQ(result.o, 0u);
        EXPECT_EQ(result.scattered_reads, 0u);
    }
}

TEST_F(Fp8PageTileOperands, WholeChunkKReadsMisPair)
{
    const auto result = Run<true, false>(Tile(1), "whole-chunk K reads (control)");
    EXPECT_GT(result.s, 0u);
}

TEST_F(Fp8PageTileOperands, WholeChunkVReadsMisPair)
{
    const auto result = Run<false, true>(Tile(1), "whole-chunk V reads (control)");
    EXPECT_EQ(result.s, 0u);
    EXPECT_GT(result.o, 0u);
}

TEST_F(Fp8PageTileOperands, ForwardSplitSoftmaxRuns)
{
    DeviceMem dout(kTileM * kTileN * sizeof(float));
    launch_kernel(stream_config{nullptr, false, 0},
                  make_kernel<1, gfx125_t>(SoftmaxProbe{},
                                           dim3(1),
                                           dim3(SoftmaxProbe::kBlockSize),
                                           0,
                                           static_cast<float*>(dout.GetDeviceBuffer())));
    hip_check_error(hipDeviceSynchronize());
    std::vector<float> p(kTileM * kTileN);
    dout.FromDevice(p.data());
    std::size_t bad = 0;
    for(float x : p)
        bad += x != 1.0f;
    std::cout << "forward FP8 split softmax on the page-tile problem: P != 1 in " << bad << " / "
              << p.size() << '\n';
    EXPECT_EQ(bad, 0u);
}
