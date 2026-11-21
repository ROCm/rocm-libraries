// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_batch_prefill_page_tile.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_tdm_sched_page_tile_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_fp8_policy.hpp"

namespace ck_tile {

namespace detail {

// LDS operand reads of the FP8 page tile. The vectorized KV layout keeps the 16-byte chunks of
// the pages in LDS as the page-tile window's TDM boxes write them: K [N/32][D/16][32][16] (a chunk
// holds 16 columns of one key), V [N/16][D][16] (a chunk holds 16 keys of one column). The FP8
// WMMAs take K (QK B operand) and V (PV B operand) in 8-element runs: with the Default
// AttrNumAccess, lane m + 16 h holds the contraction indices 16 k0 + 8 h + [0, 8) of its row. Each
// operand read is therefore one ds_load_b64 of half a chunk:
//  - K (Transposed = false): the window's own access, 8 consecutive columns of one key;
//  - V (Transposed = true): the 8 consecutive keys of one column that a transposed read
//    (ds_load_tr8_b64) of a row-major [key][column] tile leaves in the lane. In the V layout these
//    keys are adjacent, so a plain load of the chunk half, scattered as the transposed read
//    scatters, fills the same registers (the FP8 form of FmhaTdmVecVRead).
// A b128 read of a whole chunk would pair keys 16 k0 + [0, 16) with the WMMA's 16 k0 + 8 h + [0, 8)
// and mis-pair the operands. Every read of a window is a compile-time byte offset from one
// per-lane base (the lane's access 0), the same in every lane, so a window needs one pinned base.
// ValidateLanes checks this at compile time for the 32 lanes of warp 0 and for lane 0 of the other
// warps; the operand-layout device test checks every lane of every warp. The offsets are taken at
// the tile origin, so they hold for windows whose origin is a whole number of row groups (32 keys
// of the K tile, 16 of the V tile) at column 0, as the QK stage windows and the V window are.

// Lane ownership of the elements a read leaves in the lanes: the window's distribution, or, for a
// transposed read, the transposed (output) distribution, which only a transposed read window has.
template <typename TileWindow, bool Transposed>
struct FmhaBatchPrefillFp8PageTileOwner
{
    using type = remove_cvref_t<typename TileWindow::Base::TileDstr>;
};
template <typename TileWindow>
struct FmhaBatchPrefillFp8PageTileOwner<TileWindow, true>
{
    using type = remove_cvref_t<decltype(make_static_tile_distribution(
        typename OutputTileDistributionTraits<
            typename TileWindow::Base::TileDstr::DstrEncode,
            remove_cvref_t<typename TileWindow::Base::DataType>>::TransposedDstrEncode{}))>;
};

// Bytes of one operand read (ds_load_b64 of 8 FP8 elements): half of a 16-byte chunk of the
// vectorized layout.
inline constexpr index_t kFmhaBatchPrefillFp8ReadBytes = 8;

template <typename TileWindow, bool Transposed>
struct FmhaBatchPrefillFp8PageTileRead
{
    using WindowBase = typename TileWindow::Base;
    using Traits     = typename WindowBase::Traits;
    using DataType   = remove_cvref_t<typename WindowBase::DataType>;
    using VectorType = typename Traits::vector_t;
    using Descriptor = remove_cvref_t<typename remove_cvref_t<
        decltype(std::declval<const TileWindow&>().get_bottom_tensor_view())>::TensorDesc>;
    // The V operand tile uses the transposed distribution.
    using OwnerDistribution =
        typename FmhaBatchPrefillFp8PageTileOwner<TileWindow, Transposed>::type;

    static constexpr index_t kReadBytes = kFmhaBatchPrefillFp8ReadBytes;
    static constexpr index_t kNumAccess = TileWindow::NumAccessPerCoord;
    static constexpr index_t kElements  = Traits::ScalarPerVector;
    static_assert(sizeof(DataType) == 1 && Traits::PackedSize == 1 && kElements == kReadBytes &&
                      sizeof(VectorType) == kReadBytes,
                  "an FP8 page-tile operand read is one 8-element ds_load_b64");
    static_assert(Descriptor::is_static(), "the read offsets come from a static LDS descriptor");

    // Y index, in OwnerDistribution, of element J of access IAccess.
    template <index_t IAccess, index_t J>
    CK_TILE_HOST_DEVICE static constexpr auto OwnerYIndex()
    {
        constexpr auto input_index  = Traits::SFC_Ys::get_index(number<IAccess>{});
        constexpr auto source_index = generate_tuple(
            [&](auto dim) {
                return dim == Traits::VectorDimY ? (input_index[dim] + J) : input_index[dim];
            },
            number<WindowBase::NDimY>{});
        if constexpr(Transposed)
            return DefaultTranspose<DataType>::group_func(source_index);
        else
            return source_index;
    }

    // As an array, evaluated once per (access, element) so that the per-lane checks below stay
    // within the constant-evaluation budget.
    template <index_t IAccess, index_t J>
    static constexpr auto kOwnerYIndex =
        to_array<index_t, OwnerDistribution::NDimY>(OwnerYIndex<IAccess, J>());

    // Window (row, column) of element J of access IAccess for one partition (warp, lane). The
    // transposed distribution's X dimensions are (column, row).
    template <index_t IAccess, index_t J, typename Partition>
    CK_TILE_HOST_DEVICE static constexpr auto BottomIndex(const Partition& partition)
    {
        const auto x =
            make_tensor_adaptor_coordinate(OwnerDistribution{}.get_ps_ys_to_xs_adaptor(),
                                           container_concat(partition, kOwnerYIndex<IAccess, J>))
                .get_bottom_index();
        if constexpr(Transposed)
            return make_multi_index(x[number<1>{}], x[number<0>{}]);
        else
            return make_multi_index(x[number<0>{}], x[number<1>{}]);
    }

    template <index_t IAccess, index_t J, typename Partition>
    CK_TILE_HOST_DEVICE static constexpr index_t ElementOffset(const Partition& partition)
    {
        return Descriptor{}.calculate_offset(BottomIndex<IAccess, J>(partition));
    }

    // Byte offset of access IAccess from access 0 in lane 0 of warp 0.
    template <index_t IAccess>
    CK_TILE_HOST_DEVICE static constexpr index_t LaneZeroDelta()
    {
        constexpr array<index_t, 2> lane0{0, 0};
        return (ElementOffset<IAccess, 0>(lane0) - ElementOffset<0, 0>(lane0)) *
               static_cast<index_t>(sizeof(DataType));
    }

    // In each of warp 0's 32 lanes: the access's 8 elements are adjacent and 8-byte aligned (one
    // ds_load_b64), and the access is LaneZeroDelta bytes from the lane's access 0. The operand is
    // replicated over the warps: lane 0 of warps 1-3 reads what lane 0 of warp 0 reads. (The 4
    // warps and 32 lanes are the geometry gate's kWaves and the wave32 width.)
    template <index_t IAccess, index_t... J>
    CK_TILE_HOST_DEVICE static constexpr bool ValidateLanes(std::integer_sequence<index_t, J...>)
    {
        constexpr index_t delta = LaneZeroDelta<IAccess>();
        for(index_t warp = 1; warp < 4; ++warp)
            if(ElementOffset<IAccess, 0>(array<index_t, 2>{warp, 0}) !=
               ElementOffset<IAccess, 0>(array<index_t, 2>{0, 0}))
                return false;
        for(index_t lane = 0; lane < 32; ++lane)
        {
            const array<index_t, 2> partition{0, lane};
            const index_t first = ElementOffset<IAccess, 0>(partition);
            if(first % kReadBytes != 0 ||
               (first - ElementOffset<0, 0>(partition)) * static_cast<index_t>(sizeof(DataType)) !=
                   delta)
                return false;
            if(!((ElementOffset<IAccess, J>(partition) == first + J) && ...))
                return false;
        }
        return true;
    }

    // Byte offset of access IAccess from the lane's base; it fits the 16-bit DS offset field.
    template <index_t IAccess>
    CK_TILE_HOST_DEVICE static constexpr index_t ByteDelta()
    {
        static_assert(IAccess >= 0 && IAccess < kNumAccess);
        static_assert(ValidateLanes<IAccess>(std::make_integer_sequence<index_t, kElements>{}),
                      "an FP8 page-tile operand read is not 8 adjacent elements at a "
                      "lane-independent offset from the lane's base");
        constexpr index_t delta = LaneZeroDelta<IAccess>();
        static_assert(delta >= 0 && delta + kReadBytes <= 65536,
                      "the read offset must fit the DS instruction offset");
        return delta;
    }

    // Window (row, column) of element 0 of access IAccess in this lane.
    template <index_t IAccess>
    CK_TILE_DEVICE static auto LaneBottomIndex()
    {
        return BottomIndex<IAccess, 0>(get_partition_index(OwnerDistribution{}));
    }

    // LDS byte address of element 0 of access 0 in this lane: the window's base.
    CK_TILE_DEVICE static uint32_t LaneBase(const TileWindow& window)
    {
        const auto& view = window.get_bottom_tensor_view();
        const auto coord = make_tensor_coordinate(
            view.get_tensor_descriptor(), window.get_window_origin() + LaneBottomIndex<0>());
        const index_t off = coord.get_offset() * static_cast<index_t>(sizeof(DataType));
        return fmha_tdm_sched_fp8_lds_offset(view.get_buffer_view().p_data_) +
               static_cast<uint32_t>(off);
    }

    // One ds_load_b64 at the LDS byte address into the registers of access IAccess.
    template <index_t IAccess, typename DistributedTensor>
    CK_TILE_DEVICE static void LoadAt(DistributedTensor& dst, uint32_t address)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using DstDistribution = remove_cvref_t<typename DistributedTensor::StaticTileDistribution>;
        static_assert(IAccess >= 0 && IAccess < kNumAccess);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);
        if constexpr(Transposed)
            static_assert(detail::is_transpose_output_compatible_v<OwnerDistribution,
                                                                   DstDistribution,
                                                                   DataType>);
        else
            static_assert(std::is_same_v<DstDistribution, OwnerDistribution>);
        using raw_t = int32_t __attribute__((ext_vector_type(kReadBytes / 4)));
        const VectorType value =
            bit_cast<VectorType>(*fmha_tdm_sched_fp8_lds_ptr<const raw_t>(address));
        static_for<0, kElements, 1>{}([&](auto j) {
            constexpr index_t offset = DstDistribution{}.get_ys_to_d_descriptor().calculate_offset(
                kOwnerYIndex<IAccess, decltype(j)::value>);
            dst.get_thread_buffer().template at<offset>() = value.template get_as<DataType>()[j];
        });
#else
        ignore = dst;
        ignore = address;
#endif
    }

    // Access IAccess from the window's base (LaneBase).
    template <index_t IAccess, typename DistributedTensor>
    CK_TILE_DEVICE static void Load(DistributedTensor& dst, uint32_t base)
    {
        LoadAt<IAccess>(dst, base + static_cast<uint32_t>(ByteDelta<IAccess>()));
    }
};

} // namespace detail

// Geometries of the FP8 page tile: the M128 tiles of the shared FP8 list
// (FmhaTdmSchedFp8SupportedGeometry), and the D64 M64 tile, which the forward FP8 pipeline does not
// have. Its four waves hold one 16-row WMMA tile each and walk QK in two 64-key stages, as the BF16
// D64 M64 page tile does (kFmhaBatchPrefillPageTileQkSuColumnsFor).
template <typename Geometry>
struct FmhaBatchPrefillFp8PageTileM64Geometry
{
    using Q = typename Geometry::QDataType;
    static_assert(std::is_same_v<Q, fp8_t> && std::is_same_v<typename Geometry::KDataType, Q> &&
                      std::is_same_v<typename Geometry::VDataType, Q> &&
                      std::is_same_v<typename Geometry::PDataType, Q>,
                  "the FP8 page tile requires homogeneous FP8 Q/K/V/P");
    static_assert(Geometry::kM == 64 && Geometry::kN == 128 && Geometry::kHeadDimQK == 64 &&
                      Geometry::kDv == 64,
                  "the FP8 page tile supports the shared FP8 M128 tiles and M64 D64/V64 N128");
    static_assert(Geometry::kWaves == 4 && Geometry::kWaveSize == 32,
                  "the FP8 M64 page tile requires four wave32 waves");
    static_assert(Geometry::kQkWmmaK == 64 && Geometry::kPvWmmaK == 128,
                  "the FP8 M64 page tile needs a K64 QK WMMA and a K128 PV WMMA");
    static_assert(Geometry::kQkStages == Geometry::kN / 64 &&
                      Geometry::kPvStages == Geometry::kN / 128,
                  "the FP8 M64 page tile requires 64-key QK stages and one 128-key PV stage");
    static_assert(Geometry::PRepackLayout::kInThread,
                  "gemm_0 C^T and gemm_1 A lane layouts differ; P cannot be re-packed in-thread");
    static constexpr bool value = true;
};

template <typename Geometry>
inline constexpr bool kFmhaBatchPrefillFp8PageTileGeometryOk =
    std::conditional_t<Geometry::kM == 128,
                       FmhaTdmSchedFp8SupportedGeometry<Geometry>,
                       FmhaBatchPrefillFp8PageTileM64Geometry<Geometry>>::value;

// The split softmax of the FP8 M64 page tile: the two entry points that the page-tile pipeline
// calls on the forward FP8 policy (BlockFmhaPipelineQRKSVSTdmSchedFp8Policy::RunSplitSoftmaxPart01
// and RunSplitSoftmaxPart2AndGetScale), adapted to the one 16-row M iteration of an M64 wave. That
// policy asserts the M128 geometries, keeps two row halves (m0 / m1) and sizes its exp stream for
// four MSBs, so these are adapted copies, not calls. The score groups stay 32 columns wide whatever
// the QK stage width, as the 16-bit policy keeps them (kScoreGroupColumns).
template <typename Geometry>
struct FmhaBatchPrefillFp8PageTileM64Softmax
{
    static_assert(Geometry::kQkMIter == 1 && Geometry::kPvMIter == 1,
                  "the M64 softmax handles one M iteration per wave");
    static constexpr index_t kScoreGroupColumns = 32;
    using ScoreSoftmax =
        FmhaTdmSchedSplitSoftmaxFor<Geometry::kN / kScoreGroupColumns, Geometry::kQkMIter>;
    using ScoreMapping = typename ScoreSoftmax::Mapping;
    using OutputFragments =
        FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2, Geometry::kPvMIter>;
    static_assert(ScoreMapping::kNumMIter == Geometry::kQkMIter &&
                      ScoreMapping::kNumNIter == Geometry::kN / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumSu == Geometry::kN / kScoreGroupColumns &&
                      ScoreMapping::kNIterPerSu == kScoreGroupColumns / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumMsb ==
                          Geometry::kQkMIter * (kScoreGroupColumns / Geometry::kQkWmmaN) &&
                      ScoreMapping::kElementsPerWmma ==
                          Geometry::QkWmma::kCM0PerLane * Geometry::QkWmma::kCM1PerLane,
                  "split-softmax score mapping does not match the QK geometry");

    // Max of N values by max3 levels; a two-value remainder carries to the next level (copied from
    // BlockFmhaPipelineQRKSVSTdmSchedFp8Policy::TreeMax).
    template <index_t N>
    CK_TILE_DEVICE static float TreeMax(const array<float, N>& v)
    {
        if constexpr(N == 1)
            return v[number<0>{}];
        else if constexpr(N == 2)
            return max(v[number<0>{}], v[number<1>{}]);
        else
        {
            constexpr index_t kTriples = N / 3;
            constexpr index_t kRest    = N % 3;
            array<float, kTriples + kRest> next{};
            static_for<0, kTriples, 1>{}([&](auto i) {
                constexpr index_t k = 3 * decltype(i)::value;
                next(i) = max(max(v[number<k>{}], v[number<k + 1>{}]), v[number<k + 2>{}]);
            });
            static_for<0, kRest, 1>{}([&](auto i) {
                constexpr index_t k          = decltype(i)::value;
                next(number<kTriples + k>{}) = v[number<3 * kTriples + k>{}];
            });
            return TreeMax<kTriples + kRest>(next);
        }
    }

    // The forward RunSplitSoftmaxPart01 for its one row half: one max3 tree over the old max and
    // the lane's scores of the row (both MSBs, in GatherRowPairs order), then a single lane
    // exchange. delta_m1 is 0: there is no second row half.
    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = ScoreMapping;
        static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 1);

        // An elementwise bias may carry -inf, like a mask.
        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking ||
                                      Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
        const float old_max = row_max.get_thread_buffer()[number<0>{}];
        array<float, 2 * Mapping::kElementsPerMsb + 1> values{};
        values(number<0>{}) = old_max;
        static_for<0, Mapping::kNumSu, 1>{}([&](auto su) {
            static_for<0, 2, 1>{}([&](auto h) {
                static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto p) {
                    constexpr index_t at = 1 + ((su * 2 + h) * Mapping::kPairsPerFragment + p) * 2;
                    const auto x =
                        Mapping::template LoadPair<h, su * Mapping::kPairsPerFragment + p>(score);
                    values(number<at>{})     = x[0];
                    values(number<at + 1>{}) = x[1];
                });
            });
        });
        const float lane_max = TreeMax(values);
        const auto p1        = Softmax::template RunPart1<kValidateMax>(
            lane_max, Softmax::PermuteLaneX16(lane_max), old_max * log2e_scale, log2e_scale);
        row_max.get_thread_buffer()[number<0>{}] = p1.row_max;
        delta_m0                                 = p1.delta;
        delta_m1                                 = 0.0f;
    }

    // The forward RunSplitSoftmaxPart2AndGetScale with SumRows false for its one row half: the exp
    // stream of RunInterleavedLocalSums (each slot issues exp, fma, exp and, every second slot, the
    // in-place rescale of one output pair, so no exp pairs up) without the row-sum add tree, whose
    // sum the caller forms from the quantized P (AccumulateQuantizedMass). output_scale_m1 is 1:
    // there is no second row half.
    template <typename Problem,
              bool SumRows,
              typename ScoreTensor,
              typename RowTensor,
              typename Fragments>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2AndGetScale(ScoreTensor& score,
                                                               const RowTensor& row_max,
                                                               RowTensor& row_sum,
                                                               float log2e_scale,
                                                               float delta_m0,
                                                               float delta_m1,
                                                               float& output_scale_m0,
                                                               float& output_scale_m1,
                                                               Fragments& output)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = ScoreMapping;
        using Pair    = typename Softmax::Pair;
        static_assert(!SumRows, "the FP8 M64 page tile sums the quantized P, not the exp stream");
        static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 1);
        ignore = row_sum;
        ignore = delta_m1;

        // An elementwise bias may carry -inf, like a mask.
        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking ||
                                      Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
        const float raw_max = row_max.get_thread_buffer()[number<0>{}];
        const float exponent_max =
            kValidateMax && raw_max <= -numeric<float>::infinity() ? 0.0f : raw_max;

        constexpr index_t kPairs               = Mapping::kNumMsb * Mapping::kPairsPerMsb;
        constexpr index_t kOutPairsPerFragment = OutputFragments::kElementsPerFragment / 2;
        constexpr index_t kOutPairs = OutputFragments::kNumFragments * kOutPairsPerFragment;
        static_assert(kPairs % kOutPairs == 0);
        constexpr index_t kSlotsPerOutPair = kPairs / kOutPairs;
        // SALU, WMMA, VMEM and LDS may cross; VALU and TRANS keep the source order.
        constexpr int32_t kValuFence = LLVMSchedGroupMask::SALU | LLVMSchedGroupMask::MFMA |
                                       LLVMSchedGroupMask::VMEM | LLVMSchedGroupMask::VMEM_READ |
                                       LLVMSchedGroupMask::VMEM_WRITE | LLVMSchedGroupMask::DS |
                                       LLVMSchedGroupMask::DS_READ | LLVMSchedGroupMask::DS_WRITE;

        array<Pair, kPairs> leaf{};
        auto make_delta = [&](auto q) {
            constexpr index_t msb  = decltype(q)::value / Mapping::kPairsPerMsb;
            constexpr index_t pair = decltype(q)::value % Mapping::kPairsPerMsb;
            leaf(q) = __builtin_elementwise_fma(Mapping::template LoadPair<msb, pair>(score),
                                                Softmax::Broadcast(log2e_scale),
                                                -Softmax::Broadcast(exponent_max * log2e_scale));
        };
        auto exp_element = [&](auto q, auto e) {
            leaf(q)[decltype(e)::value] = Softmax::Exp2(leaf[q][decltype(e)::value]);
            if constexpr(decltype(e)::value == 1)
                Mapping::template StorePair<decltype(q)::value / Mapping::kPairsPerMsb,
                                            decltype(q)::value % Mapping::kPairsPerMsb>(score,
                                                                                        leaf[q]);
        };
        auto rescale_pair = [&](auto o) {
            constexpr index_t ordinal = decltype(o)::value / kOutPairsPerFragment;
            constexpr index_t e       = decltype(o)::value % kOutPairsPerFragment * 2;
            auto& fragment            = output.at(number<ordinal>{});
            const Pair v = Pair{fragment[e], fragment[e + 1]} * Softmax::Broadcast(output_scale_m0);
            fragment[e]  = v[0];
            fragment[e + 1] = v[1];
        };

        output_scale_m0 = Softmax::Exp2(delta_m0);
        output_scale_m1 = 1.0f;
        __builtin_amdgcn_sched_barrier(kValuFence);
        make_delta(number<0>{});
        __builtin_amdgcn_sched_barrier(kValuFence);
        make_delta(number<1>{});
        __builtin_amdgcn_sched_barrier(kValuFence);
        static_for<0, kPairs, 1>{}([&](auto q) {
            exp_element(q, number<0>{});
            __builtin_amdgcn_sched_barrier(kValuFence);
            if constexpr(decltype(q)::value + 2 < kPairs)
            {
                make_delta(number<decltype(q)::value + 2>{});
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
            exp_element(q, number<1>{});
            __builtin_amdgcn_sched_barrier(kValuFence);
            if constexpr(decltype(q)::value % kSlotsPerOutPair == 0)
            {
                rescale_pair(number<decltype(q)::value / kSlotsPerOutPair>{});
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
        });
    }
};

// The softmax of the M128 page tile: the forward FP8 policy's, except that the row maxima reduce
// score pairs (c, c + 1) as two-wide vectors before one horizontal max per row. On scalars, the
// clang 24 SLP vectorizer pairs the two row halves' max trees into maxnum.v2f32 (gfx1250 has no
// packed FP32 max, so the ISA keeps the same v_max3 count) and carries that row pairing into the
// score PHIs and the mask selects, whose lane masks then spill: 346 VGPRs and 28 bytes of scratch
// instead of 330 and 0. Score pairs are adjacent WMMA registers, as in the exp stream.
template <typename Geometry, typename Tuning>
struct FmhaBatchPrefillFp8PageTileM128Softmax
    : BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<Geometry, Tuning>
{
    using Base         = BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<Geometry, Tuning>;
    using ScoreSoftmax = typename Base::ScoreSoftmax;
    using ScorePair    = typename ScoreSoftmax::Mapping::Pair;

    // Base::TreeMax on two-wide vectors: max3 levels, lane by lane.
    template <index_t N>
    CK_TILE_DEVICE static ScorePair PairTreeMax(const array<ScorePair, N>& v)
    {
        if constexpr(N == 1)
            return v[number<0>{}];
        else if constexpr(N == 2)
            return __builtin_elementwise_maxnum(v[number<0>{}], v[number<1>{}]);
        else
        {
            constexpr index_t kTriples = N / 3;
            constexpr index_t kRest    = N % 3;
            array<ScorePair, kTriples + kRest> next{};
            static_for<0, kTriples, 1>{}([&](auto i) {
                constexpr index_t k = 3 * decltype(i)::value;
                next(i)             = __builtin_elementwise_maxnum(
                    __builtin_elementwise_maxnum(v[number<k>{}], v[number<k + 1>{}]),
                    v[number<k + 2>{}]);
            });
            static_for<0, kRest, 1>{}([&](auto i) {
                constexpr index_t k          = decltype(i)::value;
                next(number<kTriples + k>{}) = v[number<3 * kTriples + k>{}];
            });
            return PairTreeMax<kTriples + kRest>(next);
        }
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        // Sink tiles keep the forward policy's row max (its chained form).
        if constexpr(Problem::kHasSink)
        {
            Base::template RunSplitSoftmaxPart01<Problem>(
                score, row_max, log2e_scale, delta_m0, delta_m1);
        }
        else
        {
            using Softmax = ScoreSoftmax;
            using Mapping = typename Softmax::Mapping;
            static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
            static_assert(!Problem::kHasLogitsSoftCap);
            static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
            static_assert(RowTensor::get_thread_buffer_size() == 2);

            // An elementwise bias may carry -inf, like a mask.
            constexpr bool kValidateMax =
                Problem::FmhaMask::IsMasking ||
                Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
            const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
            const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];
            auto row_part          = [&](auto half, float old_max) {
                constexpr index_t kHalf = decltype(half)::value;
                array<ScorePair, Mapping::kNumSu * 2 * Mapping::kPairsPerFragment> pairs{};
                static_for<0, Mapping::kNumSu, 1>{}([&](auto su) {
                    static_for<0, 2, 1>{}([&](auto h) {
                        static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto p) {
                            constexpr index_t at   = (su * 2 + h) * Mapping::kPairsPerFragment + p;
                            constexpr index_t msb  = kHalf * 2 + h;
                            constexpr index_t pair = su * Mapping::kPairsPerFragment + p;
                            pairs(number<at>{})    = Mapping::template LoadPair<msb, pair>(score);
                        });
                    });
                });
                const ScorePair pair_max = PairTreeMax(pairs);
                const float lane_max     = max(max(pair_max[0], pair_max[1]), old_max);
                return Softmax::template RunPart1<kValidateMax>(lane_max,
                                                                Softmax::PermuteLaneX16(lane_max),
                                                                old_max * log2e_scale,
                                                                log2e_scale);
            };
            const auto p1_m0 = row_part(number<0>{}, old_max_m0);
            const auto p1_m1 = row_part(number<1>{}, old_max_m1);

            row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
            row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
            delta_m0                                 = p1_m0.delta;
            delta_m1                                 = p1_m1.delta;
        }
    }
};

// Policy of the gfx1250 FP8 (fp8bf16) batch-prefill page tile: FP8 Q / K / V / P with the FP8
// WMMAs (16x16x64 QK, 16x16x128 PV), the vectorized KV layout with x = 16 and the page-tile window.
// The data path: the LDS arena, the K / V LDS layouts the window writes, the operand reads
// (FmhaBatchPrefillFp8PageTileRead), the GEMMs and the P repack between them. The tile loop of
// BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedFp8PageTile: its loop-shape predicates (those of the
// BF16 page tile), the WMMA operand guard, the QK / PV stage runners and the members the schedule
// layer reads. It derives from the default TDM policy (AsyncCopy, QLoadOnce) as the forward FP8
// policy does (block_fmha_pipeline_qr_ks_vs_tdm_sched_fp8_policy.hpp), not from that policy, whose
// LDS reads assume its padded row-major tiles.
template <typename Geometry_, typename Tuning_, typename TileSchedule_>
struct BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedFp8PageTilePolicy
    : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Options    = typename FmhaTdmSchedTuningOptionsOf<Tuning>::type;
    using Schedule   = TileSchedule_;
    static_assert(kFmhaBatchPrefillFp8PageTileGeometryOk<Geometry> && Geometry::kWaves == 4);
    static_assert(Schedule::kNumQkStages == Geometry::kQkStages &&
                  Schedule::kNumPvStages == Geometry::kPvStages &&
                  Schedule::kQkWmmasPerStage == Geometry::kQkWmmasPerStage &&
                  Schedule::kPvWmmasPerStage == Geometry::kPvWmmasPerStage);
    // The FP8 stages run in the fixed (n, k, m) order (BlockFmhaPipelineQRKSVSTdmSchedFp8Policy::
    // RunQkSuWmma), read K in access order and take the row max by TreeMax: these options have no
    // reader here, and a set one would pass silently.
    static_assert(!Options::kQkKOuter && !Options::kKReadKMajor && !Options::kRowMaxMaximum,
                  "QkKOuter, KReadKMajor and RowMaxMaximum are not implemented for FP8");
    // One workgroup per CU: occupancy two (the forward FP8 policy's) is a later tuning step.
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;
    static_assert(kBlockPerCu == 1);
    // Read by the page-tile kernel for the window's TDM issue.
    static constexpr bool kDrainLdsBeforeTdmIssue = Options::kDrainLdsBeforeTdmIssue;

    using QDataType = typename Geometry::QDataType;
    using KDataType = typename Geometry::KDataType;
    using VDataType = typename Geometry::VDataType;
    // x of the vectorized layout is the problem's kVectorSize (one 16-byte chunk); the policy
    // requires a chunk to be two operand reads (IsSupportedProblem).
    static constexpr index_t kLdsRows       = Geometry::kN;
    static constexpr index_t kKValidWidth   = Geometry::kHeadDimQK;
    static constexpr index_t kVLogicalWidth = Geometry::kDv;
    // Rows of the window's K box (a wave slice, FmhaBatchPrefillPageTileTraits::kFragmentRows for
    // P >= 64): the K tile is [N/kKGroupRows][D/x][kKGroupRows][x].
    static constexpr index_t kKGroupRows = 32;
    static_assert(kLdsRows % kKGroupRows == 0);

    // Arena: K0, K1, V0, V1 back to back, unpadded (one wave-wide b64 read covers 16 adjacent
    // chunks, 256 contiguous bytes: each of the 64 banks once). Q is staged in V1.
    static constexpr index_t kKFootprintBytes  = kLdsRows * kKValidWidth * sizeof(KDataType);
    static constexpr index_t kVFootprintBytes  = kLdsRows * kVLogicalWidth * sizeof(VDataType);
    static constexpr index_t kQFootprintBytes  = Geometry::kM * kKValidWidth * sizeof(QDataType);
    static constexpr index_t kV1FootprintBytes = max(kVFootprintBytes, kQFootprintBytes);
    static constexpr index_t kLdsOffsetK0      = 0;
    static constexpr index_t kLdsOffsetK1      = kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0      = kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1      = kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize     = kLdsOffsetV1 + kV1FootprintBytes;
    static_assert(kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 && kLdsOffsetV1 % 16 == 0 &&
                  kLdsArenaSize % 16 == 0);

    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK0() { return kLdsOffsetK0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK1() { return kLdsOffsetK1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV0() { return kLdsOffsetV0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV1() { return kLdsOffsetV1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsArenaSize() { return kLdsArenaSize; }

    // Problems whose data path this policy implements: the FP8 geometry above (FP8 Q / K / V / P,
    // fp32 accumulation, one wave layout for both GEMMs, a full-head K load), the vectorized KV
    // layout whose chunk (the problem's kVectorSize elements, 16 bytes) is two operand reads and
    // divides the tile, pages of 64 tokens or more (one 32-row K box per wave slice), no bias and
    // no sink. The scope of the first instances (D64, P64, PERTENSOR, ...) is the pipeline's.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == Geometry::kWaveSize;
#else
        constexpr bool is_wave32 = true;
#endif
        return std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kK0 == Geometry::kHeadDimQK && Shape::kSubQKHeaddim == Geometry::kHeadDimQK &&
               Shape::IsVLayoutRowMajor && Shape::NumWarps == Geometry::kWaves && is_wave32 &&
               detail::fmha_tdm_is_vectorized_kv<Problem>::value &&
               Problem::kVectorSize * sizeof(KDataType) ==
                   2 * detail::kFmhaBatchPrefillFp8ReadBytes &&
               kLdsRows % Problem::kVectorSize == 0 && kKValidWidth % Problem::kVectorSize == 0 &&
               Problem::kPageBlockSize >= 64 &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !Problem::kHasSink;
    }

    // ---- TDM writes (the page-tile window) ----

    // DRAM distributions of the window: one box per wave slice. The vectorized issue takes only the
    // box width from them: K columns a multiple of x, V columns the head width.
    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>() && LoadOnce);
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kKValidWidth>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>());
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kVLogicalWidth>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    // K tile [N/32][D/x][32][x]: each wave's TDM box (32 keys of a page, all columns) lands at row
    // 32 w as the page stores it, [D/x][32][x], so a chunk is x columns of one key.
    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>() && LoadOnce);
        static_assert(FmhaBatchPrefillPageTileTraits<Problem::kPageBlockSize,
                                                     Geometry::kHeadDimQK>::kFragmentRows ==
                          kKGroupRows,
                      "the K tile groups the rows of one window box");
        constexpr index_t kX   = Problem::kVectorSize;
        constexpr index_t kG   = kKGroupRows;
        constexpr auto grouped = make_naive_tensor_descriptor(
            make_tuple(
                number<kLdsRows / kG>{}, number<kKValidWidth / kX>{}, number<kG>{}, number<kX>{}),
            make_tuple(number<kG * kKValidWidth>{}, number<kG * kX>{}, number<kX>{}, number<1>{}),
            number<kX>{},
            number<1>{});
        return transform_tensor_descriptor(
            grouped,
            make_tuple(make_merge_transform(make_tuple(number<kLdsRows / kG>{}, number<kG>{})),
                       make_merge_transform(make_tuple(number<kKValidWidth / kX>{}, number<kX>{}))),
            make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // V tile [N/x][Dv][x]: the window copies whole groups of x keys as the page stores them, so a
    // chunk is x keys of one column. The declared column vector of 8 is the transposed-read
    // geometry of the reads (8 elements per access, MakeVRegTileDistribution), not a property of
    // this layout: FmhaBatchPrefillFp8PageTileRead addresses the tile itself.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>());
        constexpr index_t kX   = Problem::kVectorSize;
        constexpr auto grouped = make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows / kX>{}, number<kX>{}, number<kVLogicalWidth>{}),
            make_tuple(number<kX * kVLogicalWidth>{}, number<1>{}, number<kX>{}),
            number<8>{},
            number<1>{});
        return transform_tensor_descriptor(
            grouped,
            make_tuple(make_merge_transform(make_tuple(number<kLdsRows / kX>{}, number<kX>{})),
                       make_pass_through_transform(number<kVLogicalWidth>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // No TDM padding of the K / V tiles (see the arena).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedProblem<Problem>());
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedProblem<Problem>());
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    // Q is staged in V1 as the default TDM policy lays it out (FP8 rows are not padded).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>());
        constexpr auto desc = BasePolicy::template MakeQLdsBlockDescriptor<Problem>();
        static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                          kV1FootprintBytes,
                      "Q LDS descriptor overflows the V1 buffer");
        return desc;
    }

    // The Q TDM box pads as the Q LDS descriptor does (both from LdsPaddingConfigQ).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(IsSupportedProblem<Problem>());
        using Padding = detail::EncodedTdmPadding<LdsPaddingConfigQ<Problem>>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    // ---- operand reads and GEMMs ----

    // gemm_0 of one QK stage: the stage's kQkSuColumns keys (as BlockFmhaPipelineQRKSVSTdmSchedFp8
    // Policy::GetQKBlockGemmSu). Q uses the full-tile gemm_0's A distribution
    // (MakeQRegTileDistribution), the same warp GEMM.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemmSu()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Geometry::kQkSuColumns,
                                                    Problem::BlockFmhaShape::kQKHeaddim>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    // K operand of one QK stage, read from the K tile with FmhaBatchPrefillFp8PageTileRead (as
    // BlockFmhaPipelineQRKSVSTdmSchedFp8Policy::MakeKSuRegTileDistribution).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKSuRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemmSu<Problem>())>;
        static_assert(BlockGemm::NIterPerWarp == Geometry::kQkSuNIter &&
                      BlockGemm::KIterPerWarp == Geometry::kQkKIter);
        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    // Read distribution of the V operand: the input distribution of a transposed read whose output
    // is gemm_1's B distribution (as the forward FP8 policy's), so that the V tile has the operand
    // layout; FmhaBatchPrefillFp8PageTileRead<..., true> fills it from the V tile.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>());
        using BlockGemm = remove_cvref_t<decltype(BasePolicy::template GetPVBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t kMWarp        = Geometry::kWaves;
        constexpr index_t kNWarp        = 1;
        constexpr index_t kNIterPerWarp = Geometry::kPvNIter;
        constexpr index_t kKIterPerWarp = Geometry::kPvStageKIter;
        static_assert(kNIterPerWarp == kVLogicalWidth / (kNWarp * WarpGemm::kN) &&
                      kKIterPerWarp == Geometry::kPvStageK / WarpGemm::kK);

        constexpr auto outer_encoding = tile_distribution_encoding<
            sequence<kMWarp>,
            tuple<sequence<kNIterPerWarp, kNWarp>, sequence<kKIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<2, 1>,
            sequence<0, 0>>{};
        constexpr auto block_encoding = detail::make_embed_tile_distribution_encoding(
            outer_encoding, typename WarpGemm::BWarpDstrEncoding{});
        using ReadEncoding =
            typename InputTileDistributionTraits<decltype(block_encoding),
                                                 typename Problem::VDataType>::TransposedDstrEncode;

        return make_static_tile_distribution(ReadEncoding{});
    }

    // Reads of a K or V read window.
    template <typename KWindow>
    using KReadFor = detail::FmhaBatchPrefillFp8PageTileRead<KWindow, false>;
    template <typename VWindow>
    using VReadFor = detail::FmhaBatchPrefillFp8PageTileRead<VWindow, true>;

    // P re-packed in-thread from gemm_0's C layout (M outer) into gemm_1's A layout (K outer): the
    // lanes already hold the same keys (Geometry::PRepackLayout). PTile is P already converted to
    // FP8 in gemm_0's C distribution; the copy is the re-pack of
    // BlockFmhaPipelineQRKSVSTdmSchedFp8::MakePForGemm1, whose conversion stays with the caller.
    template <typename Problem, typename Gemm1, typename PTile>
    CK_TILE_DEVICE static auto RepackPForGemm1(const PTile& p_src)
    {
        using PDataType = remove_cvref_t<typename Problem::PDataType>;
        static_assert(std::is_same_v<remove_cvref_t<typename PTile::DataType>, PDataType>);
        static_assert(Geometry::PRepackLayout::kInThread &&
                          Gemm1::MIterPerWarp == Geometry::kPvMIter,
                      "P re-pack requires gemm_0 C^T and gemm_1 A to share lanes");
        auto p_tile =
            make_static_distributed_tensor<PDataType>(MakePRegTileDistribution<Problem>());
        constexpr index_t kPBuf = PTile::get_thread_buffer_size();
        constexpr index_t kPMI  = Gemm1::MIterPerWarp;
        constexpr index_t kPKI  = Geometry::kN / Gemm1::WarpGemm::kK;
        constexpr index_t kPSub = kPBuf / (kPMI * kPKI);
        static_assert(kPSub * kPMI * kPKI == kPBuf &&
                      decltype(p_tile)::get_thread_buffer_size() == kPBuf);
        using p_bulk_t = array<PDataType, kPSub>;
        static_for<0, kPMI, 1>{}([&](auto mi) {
            static_for<0, kPKI, 1>{}([&](auto ki) {
                p_tile.get_thread_buffer().template set_as<p_bulk_t>(
                    number<ki * kPMI + mi>{},
                    p_src.get_thread_buffer().template get_as<p_bulk_t>(number<mi * kPKI + ki>{}));
            });
        });
        return p_tile;
    }

    // Where SumQuantizedPRows builds its FP8 ones operand (16 VGPRs). As a constant, the compiler
    // builds it once before the KV loop and keeps it live through the loop: the M128 tile then
    // holds 348 VGPRs (occupancy 2) instead of 334 (occupancy 3). Built per call, it lives only
    // around the row-sum WMMAs. The M64 tile is at occupancy 4 either way and keeps the constant.
    static constexpr bool kRowSumOnesPerCall = Geometry::kM == 128;

    // The row sum of the FP8 P that gemm_1 consumes (the pipeline's quantized mass), on the WMMA
    // unit: per M iteration of gemm_1, the sum of the lane's row of the P tile (gemm_1's A layout,
    // as RepackPForGemm1 returns it). Each is the PV WMMA of that iteration's P chunk with V's
    // operand replaced by FP8 ones: C = P * ones(128 x 16). Every column of C's row m is the sum
    // of P's row m, and C has O's layout, so the lane holds the row of its l (the WMMA sums both
    // lane halves). The WMMA reads the FP8 values PV reads; products by 1.0 are exact and it
    // accumulates in fp32. The chunks are sliced as the pipeline's PV emitter slices them.
    template <typename Gemm1, typename PTile>
    CK_TILE_DEVICE static auto SumQuantizedPRows(const PTile& p_tile)
    {
        using WarpGemm    = typename Gemm1::WarpGemm;
        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;
        using BBuffer     = remove_cvref_t<decltype(BWarpTensor{}.get_thread_buffer())>;
        static_assert(WarpGemm::kK == Geometry::kN &&
                          std::is_same_v<typename WarpGemm::BDataType, fp8_t>,
                      "one P chunk per M iteration covers the KV tile; the ones are FP8");
        constexpr index_t kPMI = Gemm1::MIterPerWarp;
        static_assert(kPMI == Geometry::kPvMIter);
        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};

        BWarpTensor ones;
        if constexpr(kRowSumOnesPerCall)
        {
            // Four FP8 ones in a word that an empty volatile asm takes and returns: the compiler
            // cannot treat the word as loop invariant, so it builds the operand from it here.
            // The word is a constant (1.0 = exponent bias, mantissa 0), not a converted value, so
            // the allocator rematerializes it at the asm instead of keeping it through the loop.
            using Word = uint32_t;
            static_assert(sizeof(fp8_t) == 1 && sizeof(BBuffer) % sizeof(Word) == 0);
            constexpr Word kOneBits = static_cast<Word>(numeric_traits<fp8_t>::bias)
                                      << numeric_traits<fp8_t>::mant;
            Word word = 0x01010101u * kOneBits;
            asm volatile("" : "+v"(word));
            array<Word, sizeof(BBuffer) / sizeof(Word)> words;
            static_for<0, sizeof(BBuffer) / sizeof(Word), 1>{}([&](auto w) { words(w) = word; });
            ones.get_thread_buffer() = bit_cast<BBuffer>(words);
        }
        else
        {
            static_for<0, BBuffer::size(), 1>{}(
                [&](auto e) { ones.get_thread_buffer()(e) = type_convert<fp8_t>(1.0f); });
        }
        array<AWarpTensor, kPMI> p_rows;
        array<CWarpTensor, kPMI> sums;
        static_for<0, kPMI, 1>{}([&](auto m) {
            constexpr index_t kM          = decltype(m)::value;
            p_rows(m).get_thread_buffer() = p_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<0, kM>{}, a_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
            sums(m).get_thread_buffer().template set_as<fp32x8_t>(number<0>{}, fp32x8_t{});
            WarpGemm{}(sums(m), p_rows[m], ones);
        });
        // The guard: the RAW fence on the last result, then a full drain, since under the hazard
        // note's model (WMMAs do not retire in issue order) the fence does not cover the earlier
        // row-sum WMMAs; the operands stay allocated until the drain.
        if constexpr(kWmmaOperandGuard)
        {
            WaitWmmaOperandsRead(sums(number<kPMI - 1>{}));
            asm volatile("s_wait_alu depctr_va_vdst(0)" ::: "memory");
            KeepOperandLive(ones);
            static_for<0, kPMI, 1>{}([&](auto m) { KeepOperandLive(p_rows[m]); });
        }
        array<float, kPMI> rows;
        static_for<0, kPMI, 1>{}(
            [&](auto m) { rows(m) = sums[m].get_thread_buffer()[number<0>{}]; });
        return rows;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>());
        return GetLdsArenaSize();
    }

    // ---- the tile loop: members the schedule layer reads ----

    // FmhaTdmSchedMode and FmhaTdmSchedSequentialTileSchedule read these. The output rescale is
    // fused into the forward FP8 softmax's part 2 (the pipeline passes an empty rescale callback),
    // so the early slot skips the post pass. No lead read; the tensor waits and stage tails of the
    // derived tuning.
    static constexpr bool kEarlyOutputRescale      = true;
    static constexpr bool kLeadFirstLoad           = false;
    static constexpr index_t kKPrefetchTensorCount = Tuning::kKWait;
    static constexpr index_t kVPrefetchTensorCount = Tuning::kVWait;
    static constexpr bool kPrefetchTailDrain       = Tuning::kTailDrain;
    static_assert(kKPrefetchTensorCount == 0 || kKPrefetchTensorCount == 1);
    static_assert(kVPrefetchTensorCount == 0 || kVPrefetchTensorCount == 1);

    // fp32 O fragments of the PV WMMAs, as the forward FP8 policy's, over the geometry's M
    // iterations (one at M64). One PV stage covers the KV tile, so every PV row is an output
    // fragment.
    using OutputFragments =
        FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2, Geometry::kPvMIter>;
    static_assert(OutputFragments::kNumFragments == Geometry::kPvMIter * Geometry::kPvNIter &&
                      OutputFragments::kNumDmsb == 2 * Geometry::kPvMIter &&
                      OutputFragments::kElementsPerFragment ==
                          Geometry::PvWmma::kCM0PerLane * Geometry::PvWmma::kCM1PerLane,
                  "output fragments do not match the PV geometry");
    static_assert(Geometry::kPvStages == 1 &&
                  Geometry::kPvWmmasPerStage == OutputFragments::kNumFragments);

    // The softmax of the M128 tiles is the forward FP8 pipeline's: the static entry points
    // RunSplitSoftmaxPart01 and RunSplitSoftmaxPart2AndGetScale of its policy on this geometry,
    // with the pair row maxima of FmhaBatchPrefillFp8PageTileM128Softmax.
    // They read the geometry and the problem, no LDS layout and no tuning member. The M64 tile,
    // which that policy does not take, has the same entry points for its one M iteration.
    using SoftmaxPolicy =
        std::conditional_t<Geometry::kQkMIter == 2,
                           FmhaBatchPrefillFp8PageTileM128Softmax<Geometry, Tuning>,
                           FmhaBatchPrefillFp8PageTileM64Softmax<Geometry>>;
    using ScoreSoftmax = typename SoftmaxPolicy::ScoreSoftmax;
    static_assert(std::is_same_v<typename SoftmaxPolicy::OutputFragments, OutputFragments>);

    // K(i+1) is waited at NextK (WaitTensorReadyAtLoadNextK), not in the softmax: the lean D64
    // loop's three barriers per tile. The forward FP8 policy defers it into the softmax.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool DeferTensorReady()
    {
        return false;
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQkTailDsCount()
    {
        return Tuning::template GetQkTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPvTailDsCount()
    {
        return Tuning::template GetPvTailDsCount<Stage>();
    }

    // Stage tail waits, with the tail counts above, copied from BlockFmhaPipelineQRKSVSTdmSchedFp8
    // Policy: a count past the DS counter field still waits. The counts are b64 reads, while
    // s_wait_dscnt counts DS instructions and the compiler pairs these reads into
    // ds_load_2addr(_stride64)_b64 (O13), so a wait admits up to twice as many reads. Buffer reuse
    // does not rest on these counts: every read of a buffer that a TDM transfer overwrites feeds a
    // WMMA issued before the barrier that precedes the transfer (the pipeline's Begin and NextK).
    template <index_t Stage>
    CK_TILE_DEVICE static void WaitQkStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<GetQkTailDsCount<Stage>()>>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<GetPvTailDsCount<Stage>()>>();
    }

    // ---- loop shape, copied from BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTilePolicy ----

    // The lean D64 loop of the page tile: the countdown loop, the tile's LDS read bases formed
    // once ahead of every WMMA (MakeTileBases), and K(i+1) waited at the barrier in load_next_k,
    // which publishes it before its first reader, the PV stage. D64/V64 N128, one page per wave
    // slice (P >= 64), no sink and no bias. The FP8 pipeline implements this loop only.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseLeanD64Loop()
    {
        return Geometry::kHeadDimQK == 64 && Geometry::kDv == 64 && Geometry::kN == 128 &&
               Problem::kPageBlockSize >= 64 && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    // The countdown loop keeps no tile index, which ALiBi and the sink tiles read.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        constexpr bool kCountdown = UseLeanD64Loop<Problem>();
        static_assert(!kCountdown ||
                      (!Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS));
        return kCountdown;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool WaitTensorReadyAtLoadNextK()
    {
        return UseLeanD64Loop<Problem>() && !DeferTensorReady<Problem>();
    }

    // Lean TDM issue (per-wave descriptor inputs derived once, make_lean_issue_state) of the lean
    // D64 loop.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseLeanTdmIssue()
    {
        return UseLeanD64Loop<Problem>();
    }

    // Mask predicate of a score at (row, col): the unsigned forms make one unsigned compare per
    // element. The causal mask also skips the row-validity test: rows at or past seqlen_q are never
    // written out (the O view is bounded by seqlen_q), so their scores may be masked arbitrarily.
    // This is the BF16 page tile's predicate for the problems this policy supports (P >= 64, no
    // sink tokens).
    template <typename Problem, typename Mask>
    CK_TILE_DEVICE static bool IsMaskedOut(const Mask& mask, index_t row, index_t col)
    {
        static_assert(IsSupportedProblem<Problem>() && !Problem::kHasSink &&
                      Problem::kPageBlockSize >= 64);
        if constexpr(std::is_same_v<Mask, SimplifiedGenericAttentionMask<true>>)
            return mask.IsOutOfBoundUnsignedStoredRow(row, col);
        else
            return mask.IsOutOfBound(row, col);
    }

    // P >= 64 page ids in VGPR lanes (load_page_id_batch of the page-tile window: one vector load
    // per 32 tiles instead of one LDS publication per tile), without sink tokens, KV split or
    // logits soft cap.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UsePageIdBatch()
    {
        return Problem::kPageBlockSize >= 64 && !Problem::kHasSink &&
               !detail::fmha_tdm_is_split_partial_v<Problem> && !Problem::kHasLogitsSoftCap;
    }

    // Masked tiles whose KV loop forms the mask's interior key-tile range once per query tile and
    // drops the split tail predicate: without sink tokens, KV split or logits soft cap.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseMaskLoopTrims()
    {
        return !Problem::kHasSink && !detail::fmha_tdm_is_split_partial_v<Problem> &&
               !Problem::kHasLogitsSoftCap;
    }

    // ---- WMMA operand guard ----
    // The guard of the hazard note in block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp, copied
    // from BlockFmhaPipelineQRKSVSTdmSchedPolicy, which does not take the FP8 geometry: grouped
    // full drains, operands kept allocated until the drain that covers them, and a RAW fence on
    // each stage's last row.

    static constexpr bool kWmmaOperandGuard = Options::kWmmaOperandGuard;

    // RAW fence: waits until the WMMA that produced c has completed, and so has read its A/B
    // operands. Reading the result makes the compiler insert the WMMA result wait before the
    // instruction that follows.
    template <typename CWarpTensor>
    CK_TILE_DEVICE static void WaitWmmaOperandsRead(CWarpTensor& c)
    {
        auto x = c.get_thread_buffer()[number<0>{}];
        asm volatile("v_mov_b32 %0, %0" : "+v"(x)::"memory");
        c.get_thread_buffer()(number<0>{}) = x;
    }

    // Empty use: keeps an operand's registers live so no later load is allocated onto them. The
    // asm operand is allocated in v0-v255. The FP8 buffer is bit-cast to 16-byte chunks (an 8-bit
    // thread buffer has no get_as of another scalar type).
    template <typename Tensor>
    CK_TILE_DEVICE static void KeepOperandLive(const Tensor& t)
    {
        using Buffer            = remove_cvref_t<decltype(t.get_thread_buffer())>;
        constexpr index_t kSize = sizeof(Buffer);
        static_assert(kSize % 16 == 0);
        const auto chunks = bit_cast<array<int32x4_t, kSize / 16>>(t.get_thread_buffer());
#pragma unroll
        for(index_t i = 0; i < kSize / 16; ++i)
            asm volatile("" ::"v"(chunks[i]) : "memory");
    }

    // The Drain or KeepPending point of QK row Row of a grouped-drain schedule: the full drain,
    // then the keep-alive of the last-M K slices of the pairs it covers. KeepPending, after the
    // RAW fence of a QK stage's last row, drains too: the fence waits for the last WMMA's result
    // only, and under the hazard note's model (WMMAs do not retire in issue order) the earlier
    // pairs of the group may still read their K slices. The BF16 / FP16 page tile places a Drain
    // point there instead (FmhaBatchPrefillPageTileGroupedDrainPlacement), except in the D128 M128
    // tiles without prepared tile bases, which keep the shared KeepPending release
    // (fmha_page_tile_d128_m128_spills_v).
    template <typename Placement, typename BlockGemm, index_t Row, bool Drain, typename KeepPair>
    CK_TILE_DEVICE static void EmitQkGroupedDrain(KeepPair& keep_k_of_pair)
    {
        static_assert(Placement::kQkMIter == BlockGemm::MIterPerWarp);
        constexpr auto kPairs = Placement::QkKeptPairs(Row, Drain);
        __builtin_amdgcn_sched_barrier(0);
        // Full drain: a counted va_vdst(k >= 1) does not cover the WAR.
        asm volatile("s_wait_alu depctr_va_vdst(0)" ::: "memory");
        static_for<kPairs.begin, kPairs.end, 1>{}([&](auto q) { keep_k_of_pair(q); });
        __builtin_amdgcn_sched_barrier(0);
    }

    // Grouped waits: the schedule runs FmhaTdmSchedGroupedDrainPlacement. The other placements
    // keep a RAW fence after every last reader.
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr bool GroupedWait()
    {
        constexpr bool kGrouped =
            kIsFmhaTdmSchedGroupedDrain<typename Mode::SelectedPolicy::Schedule::LocalPlacement>;
        static_assert(!kGrouped || kWmmaOperandGuard,
                      "the grouped drains belong to the WMMA operand guard");
        return kGrouped;
    }

    // PV rows [kPvLastMRow, kPvWmmasPerStage) of the stage are its last M iteration.
    static constexpr index_t kPvLastMRow = kFmhaTdmSchedPvLastMRow<Geometry>;

    // The PV WMMAs are issued by the pipeline's WMMA emitter, which guards PV row r as follows.
    // RAW fence after its WMMA: the last M iteration's rows, with grouped waits only the stage's
    // last row.
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr bool PvRawFence(index_t r)
    {
        return kWmmaOperandGuard && r >= kPvLastMRow &&
               (!GroupedWait<Mode>() || r == Geometry::kPvWmmasPerStage - 1);
    }

    // M iterations [0, n) whose P chunk (the PV A operand) the emitter keeps allocated after the
    // RAW fence of row r: with grouped waits, every M iteration but the last at the stage's last
    // row. P(m0) is last read in rows [0, kPvLastMRow), and the next-K reads of those rows and of
    // the later ones are issued without a drain, so they could be allocated onto it while its
    // WMMAs are in flight. (BlockFmhaPipelineQRKSVSTdmSchedPolicy keeps it only without QK tail
    // reads; with them its static scan found no read placed there. Here P(m0) would otherwise stay
    // allocated only because the quantized row sum reads P after the stage.)
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr index_t PvKeptPIters(index_t r)
    {
        if constexpr(GroupedWait<Mode>() && Geometry::kPvStageKIter == 1)
            return r == Geometry::kPvWmmasPerStage - 1 ? Geometry::kPvMIter - 1 : 0;
        else
            return 0;
    }

    // Rows [begin, end) whose operands stay allocated (KeepOperandLive) at row r: with grouped
    // waits, the placement's PvKeptRows; otherwise none.
    template <typename Mode>
    CK_TILE_HOST_DEVICE static constexpr FmhaTdmSchedRowRange PvKeptRows(index_t r,
                                                                         bool after_fence)
    {
        if constexpr(GroupedWait<Mode>())
        {
            using Placement = typename Mode::SelectedPolicy::Schedule::LocalPlacement;
            return Placement::PvKeptRows(r, after_fence);
        }
        else
            return {r, r};
    }

    // ---- the tile's reads and stage runners ----

    // LDS read bases of one tile (FmhaBatchPrefillFp8PageTileRead::LaneBase): row group 0 of the K
    // tile that QK reads (its stages read row groups 1-3 at constant offsets), the V tile, and row
    // group 0 of the next K tile (the PV stage's reads). Formed once per tile, ahead of every
    // WMMA, and kept in VGPRs: formed next to their first reads instead, each base would be a VALU
    // write between WMMAs, and the read that uses it would wait for the WMMA pipe (as the BF16
    // page tile's PrepareTileBases). Each base is one dword of the low 256 VGPRs.
    struct TileBases
    {
        uint32_t k;
        uint32_t v;
        uint32_t next_k;
    };

    template <typename KWindow, typename VWindow>
    CK_TILE_DEVICE static TileBases
    MakeTileBases(const KWindow& k_window, const VWindow& v_window, const KWindow& next_k_window)
    {
        TileBases bases{KReadFor<KWindow>::LaneBase(k_window),
                        VReadFor<VWindow>::LaneBase(v_window),
                        KReadFor<KWindow>::LaneBase(next_k_window)};
        asm volatile("" : "+v"(bases.k));
        asm volatile("" : "+v"(bases.v));
        asm volatile("" : "+v"(bases.next_k));
        return bases;
    }

    // QK WMMA WmmaOrdinal of a stage in the (n, k, m) order of the forward FP8 policy's
    // RunQkSuWmma. With FenceLastMIter the last M iteration's WMMA, the last reader of its K
    // slice, ends with the RAW fence.
    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              bool FenceLastMIter>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor,
                                           bool_constant<FenceLastMIter>)
    {
        using WarpGemm    = typename BlockGemm::WarpGemm;
        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using BWarpDstr   = typename WarpGemm::BWarpDstr;
        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        constexpr index_t kKIter = BlockGemm::KIterPerWarp;
        constexpr index_t kMIter = BlockGemm::MIterPerWarp;
        static_assert(WmmaOrdinal >= 0 && WmmaOrdinal < BlockGemm::NIterPerWarp * kKIter * kMIter);
        constexpr index_t n_iter = WmmaOrdinal / (kKIter * kMIter);
        constexpr index_t k_iter = (WmmaOrdinal / kMIter) % kKIter;
        constexpr index_t m_iter = WmmaOrdinal % kMIter;

        AWarpTensor a_warp_tensor;
        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

        BWarpTensor b_warp_tensor;
        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<n_iter, k_iter>{}, b_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

        CWarpTensor c_warp_tensor;
        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        if constexpr(FenceLastMIter && m_iter == kMIter - 1)
            WaitWmmaOperandsRead(c_warp_tensor);
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    // QK stage Stage of a tile: its WMMAs in the placement's rows, the reads of the next stage's
    // operand between them through Read from the tile's bases (stages before the last: K row group
    // Stage + 1, a constant offset from row group 0 of bases.k; the last stage: the V tile from
    // bases.v), and the guard's points. Adapted from BlockFmhaPipelineQRKSVSTdmSchedPolicy::
    // RunQkScheduledStage (no lead read, no k-major reads, FP8 reads); the kept K slices follow
    // the (n, k, m) WMMA order.
    template <index_t Stage,
              typename Mode,
              typename Read,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor>
    CK_TILE_DEVICE static void RunQkStage(const BlockGemm&,
                                          CBlockTensor& c_block_tensor,
                                          const ABlockTensor& a_block_tensor,
                                          const BBlockTensor& b_block_tensor,
                                          NextBBlockTensor& next_b_block_tensor,
                                          const TileBases& bases)
    {
        using Problem          = typename Mode::SelectedProblem;
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        constexpr bool kReadsV = Stage == Geometry::kQkStages - 1;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages);
        static_assert(Read::kNumAccess ==
                      (kReadsV ? Geometry::kVStageLoadCount : Geometry::kKSuLoadCount));
        static_assert(!Mode::SelectedPolicy::kLeadFirstLoad);
        // The read offsets of FmhaBatchPrefillFp8PageTileRead hold for whole K row groups only.
        static_assert(Geometry::kQkSuColumns % kKGroupRows == 0,
                      "a QK stage reads whole row groups of the K tile");
        const uint32_t base = kReadsV ? bases.v : bases.k;
        // Byte offset of K row group Stage + 1 (the next stage's keys) from row group 0.
        constexpr index_t kOffset = [] {
            if constexpr(kReadsV)
                return index_t{0};
            else
                return MakeKLdsBlockDescriptor<Problem>().calculate_offset(
                           make_multi_index((Stage + 1) * Geometry::kQkSuColumns, 0)) *
                       static_cast<index_t>(sizeof(KDataType));
        }();

        auto emit_token = [&](auto kind, auto access) {
            static_assert(decltype(kind)::value == (kReadsV ? Kind::VRead : Kind::KRead));
            Read::template Load<decltype(access)::value>(next_b_block_tensor,
                                                         base + static_cast<uint32_t>(kOffset));
        };
        // Grouped drains (FmhaTdmSchedGroupedDrainPlacement): the K slice of a pair's last-M WMMA
        // stays allocated (an empty asm use) until after the drain that covers it, so the reads
        // issued in between cannot be allocated onto it while that WMMA is in flight.
        auto keep_k_of_pair = [&](auto pair) {
            constexpr index_t kMIter = BlockGemm::MIterPerWarp;
            constexpr index_t kKIter = BlockGemm::KIterPerWarp;
            using KWarpDstr          = typename BlockGemm::WarpGemm::BWarpDstr;
            constexpr auto k_lengths =
                to_sequence(KWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto k_zeros = uniform_sequence_gen_t<KWarpDstr::NDimY, 0>{};
            constexpr index_t kO   = kMIter * decltype(pair)::value + kMIter - 1; // last M reader
            typename BlockGemm::WarpGemm::BWarpTensor kept;
            kept.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<kO / (kKIter * kMIter), (kO / kMIter) % kKIter>{},
                                k_zeros),
                merge_sequences(sequence<1, 1>{}, k_lengths));
            KeepOperandLive(kept);
        };
        auto emit_point = [&](auto, auto row, auto point) {
            using Point            = FmhaTdmSchedSchedulePoint;
            constexpr Point kPoint = decltype(point)::value;
            if constexpr(kPoint == Point::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else if constexpr(kPoint == Point::Drain || kPoint == Point::KeepPending)
                EmitQkGroupedDrain<typename SelectedSchedule::LocalPlacement,
                                   BlockGemm,
                                   decltype(row)::value,
                                   kPoint == Point::Drain>(keep_k_of_pair);
            else
                __builtin_amdgcn_sched_barrier(LLVMSchedGroupMask::VALU |
                                               LLVMSchedGroupMask::TRANS);
        };
        // With the guard, the last M iteration's WMMAs keep the RAW fence: with grouped waits only
        // on the stage's last row, otherwise on every one.
        auto emit_wmma = [&](auto, auto wmma) {
            constexpr index_t kOrd  = decltype(wmma)::value;
            constexpr bool kLastRow = kOrd == Geometry::kQkWmmasPerStage - 1;
            RunQkSuWmma<kOrd, BlockGemm>(c_block_tensor,
                                         a_block_tensor,
                                         b_block_tensor,
                                         bool_constant < kWmmaOperandGuard &&
                                             (kLastRow || !GroupedWait<Mode>()) > {});
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    // The PV stage of a tile: the WMMAs of the pipeline's emitter (which applies the guard's kept
    // rows, RAW fence and kept P chunks), the reads of the next K tile's row group 0 from
    // bases.next_k between them, and the full drains of the grouped-drain points. The output
    // rescale is fused into the softmax, so the rows emit none. Adapted from
    // BlockFmhaPipelineQRKSVSTdmSchedPolicy::RunPvScheduledStageWithWmma (one stage, no lead read,
    // FP8 reads).
    template <index_t Stage,
              typename Mode,
              typename Read,
              typename NextKTensor,
              typename WmmaEmitter>
    CK_TILE_DEVICE static void
    RunPvStage(NextKTensor& next_k_tensor, const TileBases& bases, WmmaEmitter& emit_wmma)
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage == SelectedSchedule::kNumPvStages - 1 &&
                      Read::kNumAccess == Geometry::kKSuLoadCount);
        static_assert(!Mode::SelectedPolicy::kLeadFirstLoad);
        const uint32_t base = bases.next_k;
        auto emit_token     = [&](auto kind, auto access) {
            static_assert(decltype(kind)::value == Kind::KRead);
            Read::template Load<decltype(access)::value>(next_k_tensor, base);
        };
        auto emit_point = [](auto, auto, auto point) {
            using Point            = FmhaTdmSchedSchedulePoint;
            constexpr Point kPoint = decltype(point)::value;
            if constexpr(kPoint == Point::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else if constexpr(kPoint == Point::Drain)
            {
                // The WMMA emitter keeps the operands of the drained rows allocated until the
                // stage's last row (PvKeptRows).
                __builtin_amdgcn_sched_barrier(0);
                // Full drain only: a counted va_vdst(k >= 1) does not cover the WAR.
                asm volatile("s_wait_alu depctr_va_vdst(0)" ::: "memory");
                __builtin_amdgcn_sched_barrier(0);
            }
            else
                __builtin_amdgcn_sched_barrier(LLVMSchedGroupMask::VALU |
                                               LLVMSchedGroupMask::TRANS);
        };
        auto no_rescale = [](auto) {};
        Executor::template ExecutePvStage<Stage>(emit_wmma, emit_token, emit_point, no_rescale);
        WaitPvStageTail<Stage>();
    }
};

// QK stage width of an FP8 page-tile problem: 64 keys for M64 tiles, whose waves have one 16-row
// WMMA tile per column tile (as the BF16 D64 M64 page tile,
// kFmhaBatchPrefillPageTileQkSuColumnsFor), else two WMMA column tiles (32 keys).
template <typename Problem, typename Shape = remove_cvref_t<typename Problem::BlockFmhaShape>>
inline constexpr index_t kFmhaBatchPrefillFp8PageTileQkSuColumnsFor =
    Shape::kM0 == 64 ? 64 : kFmhaTdmSchedDefaultQkSuColumns<typename Shape::Gemm0WarpTile>;

// Geometry of an FP8 page-tile problem: the QK stages above, one 128-deep PV stage.
template <typename Problem>
using FmhaBatchPrefillFp8PageTileGeometryFor =
    FmhaTdmSchedGeometryFor<Problem, kFmhaBatchPrefillFp8PageTileQkSuColumnsFor<Problem>>;

// Code-generation options of the FP8 M128 page tile: the M128 page tile's scalar row clamp 1 and
// the WMMA operand guard (the vectorized KV layout requires it); the LDS drain before the TDM issue
// stays off as in the BF16 M128 tile (the FP8 race screen found no whole-tile event with it off).
// QkKOuter, KReadKMajor and RowMaxMaximum have no FP8 implementation.
using FmhaBatchPrefillFp8PageTileOptions =
    FmhaTdmSchedTuningOptions<false, false, false, 1, true, false>;

// Code-generation options of the FP8 M64 page tile: those of the BF16 P64 D64 M64 page tile, the
// WMMA operand guard with scalar row clamp 0, and the LDS drain before every K / V TDM issue, which
// every 16-bit D64 M64 tile runs (kDrainLdsBeforeTdmIssue of the scheduled-TDM policy: a
// stale-fragment event at P64 without it).
using FmhaBatchPrefillFp8PageTileM64Options =
    FmhaTdmSchedTuningOptions<false, false, false, 0, true, true>;

template <typename Problem, typename Geometry = FmhaBatchPrefillFp8PageTileGeometryFor<Problem>>
using FmhaBatchPrefillFp8PageTileOptionsFor =
    std::conditional_t<Geometry::kM == 64,
                       FmhaBatchPrefillFp8PageTileM64Options,
                       FmhaBatchPrefillFp8PageTileOptions>;

template <typename Problem, typename Geometry = FmhaBatchPrefillFp8PageTileGeometryFor<Problem>>
using FmhaBatchPrefillFp8PageTileTuningFor =
    FmhaBatchPrefillPageTileTuning<FmhaTdmSchedDerivedTuning<Geometry, 1>,
                                   FmhaBatchPrefillFp8PageTileOptionsFor<Problem, Geometry>>;

// Schedule: the WMMA-first rows with grouped drains, from the shared templates. M128: groups of
// four QK m0/m1 pairs and four PV rows, QK tail reads on, and the FP8 reads evenly interleaved
// (FmhaTdmSchedFp8EvenlyInterleavedReadSchedule, whose packed next-K reads take more PV rows than
// the four of M64). M64: the placement's defaults, with the shared evenly interleaved reads, as the
// BF16 D64 M64 page tile.
template <typename Problem, typename Geometry = FmhaBatchPrefillFp8PageTileGeometryFor<Problem>>
using FmhaBatchPrefillFp8PageTileScheduleFor =
    FmhaTdmSchedSequentialTileSchedule<std::conditional_t<
        Geometry::kM == 64,
        FmhaTdmSchedGroupedDrainPlacement<Geometry>,
        FmhaTdmSchedGroupedDrainPlacement<Geometry,
                                          4,
                                          4,
                                          true,
                                          FmhaTdmSchedFp8EvenlyInterleavedReadSchedule<Geometry>>>>;

template <typename Problem>
using FmhaBatchPrefillFp8PageTilePolicyFor =
    BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedFp8PageTilePolicy<
        FmhaBatchPrefillFp8PageTileGeometryFor<Problem>,
        FmhaBatchPrefillFp8PageTileTuningFor<Problem>,
        FmhaBatchPrefillFp8PageTileScheduleFor<Problem>>;

} // namespace ck_tile
