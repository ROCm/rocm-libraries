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
// per-lane base (the lane's access 0), the same in every lane (checked below), so a window needs
// one pinned base. The offsets are taken at the tile origin, so they hold for windows whose origin
// is a whole number of row groups (32 keys of the K tile, 16 of the V tile) at column 0, as the
// QK stage windows and the V window are.

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

    static constexpr index_t kReadBytes = 8;
    static constexpr index_t kNumAccess = TileWindow::NumAccessPerCoord;
    static constexpr index_t kElements  = Traits::ScalarPerVector;
    static_assert(sizeof(DataType) == 1 && Traits::PackedSize == 1 && kElements == kReadBytes &&
                      sizeof(VectorType) == kReadBytes,
                  "an FP8 page-tile operand read is one 8-element ds_load_b64");
    static_assert(Descriptor::is_static(), "the read offsets come from a static LDS descriptor");
    // Every read of a window is one base and a constant offset.
    static constexpr index_t kNumBases = 1;

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

    // In every lane: the access's 8 elements are adjacent and 8-byte aligned (one ds_load_b64), and
    // the access is LaneZeroDelta bytes from the lane's access 0. The operand is replicated over
    // the warps: warps 1-3 read what warp 0 reads.
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

// Policy of the gfx1250 FP8 (fp8bf16) batch-prefill page tile: FP8 Q / K / V / P with the FP8
// WMMAs (16x16x64 QK, 16x16x128 PV), the vectorized KV layout with x = 16 and the page-tile window.
// This part is the data path: the LDS arena, the K / V LDS layouts the window writes, the operand
// reads (FmhaBatchPrefillFp8PageTileRead), the GEMMs and the P repack between them. It derives
// from the default TDM policy (AsyncCopy, QLoadOnce) as the forward FP8 policy does
// (block_fmha_pipeline_qr_ks_vs_tdm_sched_fp8_policy.hpp), not from that policy, whose LDS reads
// assume its padded row-major tiles.
template <typename Geometry_, typename Tuning_, typename TileSchedule_>
struct BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedFp8PageTilePolicy
    : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Options    = typename FmhaTdmSchedTuningOptionsOf<Tuning>::type;
    using Schedule   = TileSchedule_;
    static_assert(kFmhaTdmSchedFp8GeometryOk<Geometry> && Geometry::kWaves == 4);
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

    using KDataType = typename Geometry::KDataType;
    using VDataType = typename Geometry::VDataType;
    // x of the vectorized layout: 16 FP8 elements per 16-byte vector (the problem's kVectorSize).
    static constexpr index_t kVectorSize    = 16 / sizeof(KDataType);
    static constexpr index_t kLdsRows       = Geometry::kN;
    static constexpr index_t kKValidWidth   = Geometry::kHeadDimQK;
    static constexpr index_t kVLogicalWidth = Geometry::kDv;
    // Rows of the window's K box (a wave slice, FmhaBatchPrefillPageTileTraits::kFragmentRows for
    // P >= 64): the K tile is [N/kKGroupRows][D/x][kKGroupRows][x].
    static constexpr index_t kKGroupRows = 32;
    static_assert(kLdsRows % kKGroupRows == 0 && kLdsRows % kVectorSize == 0 &&
                  kKValidWidth % kVectorSize == 0);

    // Arena: K0, K1, V0, V1 back to back, unpadded (one wave-wide b64 read covers 16 adjacent
    // chunks, 256 contiguous bytes: each of the 64 banks once). Q is staged in V1.
    static constexpr index_t kKFootprintBytes  = kLdsRows * kKValidWidth * sizeof(KDataType);
    static constexpr index_t kVFootprintBytes  = kLdsRows * kVLogicalWidth * sizeof(VDataType);
    static constexpr index_t kQFootprintBytes  = Geometry::kM * kKValidWidth;
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
    // layout with x = 16, pages of 64 tokens or more (one 32-row K box per wave slice), no bias
    // and no sink. The scope of the first instances (D64, P64, PERTENSOR, ...) is the pipeline's.
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
               Problem::kVectorSize == kVectorSize && Problem::kPageBlockSize >= 64 &&
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
        constexpr index_t kX   = kVectorSize;
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
        constexpr index_t kX   = kVectorSize;
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

    // K operand of one QK stage, read from the K tile with FmhaBatchPrefillFp8PageTileRead.
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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>());
        return GetLdsArenaSize();
    }
};

// Geometry of an FP8 page-tile problem: 32-column QK stages, one 128-deep PV stage.
template <typename Problem>
using FmhaBatchPrefillFp8PageTileGeometryFor = FmhaTdmSchedGeometryFor<Problem>;

// Code-generation options of the FP8 page tile: the M128 page tile's scalar row clamp 1 and the
// WMMA operand guard (the vectorized KV layout requires it); the LDS drain before the TDM issue
// stays off as in the BF16 M128 tile (no FP8 evidence yet). QkKOuter, KReadKMajor and
// RowMaxMaximum have no FP8 implementation.
using FmhaBatchPrefillFp8PageTileOptions =
    FmhaTdmSchedTuningOptions<false, false, false, 1, true, false>;

template <typename Problem, typename Geometry = FmhaBatchPrefillFp8PageTileGeometryFor<Problem>>
using FmhaBatchPrefillFp8PageTileTuningFor =
    FmhaBatchPrefillPageTileTuning<FmhaTdmSchedDerivedTuning<Geometry, 1>,
                                   FmhaBatchPrefillFp8PageTileOptions>;

// Schedule: the WMMA-first rows with grouped drains (groups of four QK m0/m1 pairs and four PV
// rows, QK tail reads on) and the FP8 reads evenly interleaved, from the shared templates.
template <typename Problem, typename Geometry = FmhaBatchPrefillFp8PageTileGeometryFor<Problem>>
using FmhaBatchPrefillFp8PageTileScheduleFor = FmhaTdmSchedSequentialTileSchedule<
    FmhaTdmSchedGroupedDrainPlacement<Geometry,
                                      4,
                                      4,
                                      true,
                                      FmhaTdmSchedFp8EvenlyInterleavedReadSchedule<Geometry>>>;

template <typename Problem>
using FmhaBatchPrefillFp8PageTilePolicyFor =
    BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedFp8PageTilePolicy<
        FmhaBatchPrefillFp8PageTileGeometryFor<Problem>,
        FmhaBatchPrefillFp8PageTileTuningFor<Problem>,
        FmhaBatchPrefillFp8PageTileScheduleFor<Problem>>;

} // namespace ck_tile
