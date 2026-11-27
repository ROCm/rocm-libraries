// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"

namespace ck_tile {

// Policy parts shared by the 16-bit and FP8 qr_tdm sched pipelines; Derived adds the layout.
template <typename Derived,
          typename Geometry_,
          typename Tuning_,
          typename TileSchedule_,
          typename Layout_>
struct BlockFmhaPipelineQRKSVSTdmSchedCommonPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy                     = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry                       = Geometry_;
    using Tuning                         = Tuning_;
    using Schedule                       = TileSchedule_;
    using Layout                         = Layout_;
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;

    static constexpr index_t kLdsRows            = Geometry::kN;
    static constexpr index_t kKValidWidth        = Geometry::kHeadDimQK;
    static constexpr index_t kKPhysicalStride    = Layout::kKPhysicalStride;
    static constexpr index_t kVLogicalWidth      = Geometry::kDv;
    static constexpr index_t kVPhysicalStride    = Layout::kVPhysicalStride;
    static constexpr index_t kKFootprintElements = kLdsRows * kKPhysicalStride;
    static constexpr index_t kVFootprintElements = kLdsRows * kVPhysicalStride;
    static constexpr index_t kKFootprintBytes =
        kKFootprintElements * sizeof(typename Geometry::KDataType);
    static constexpr index_t kVFootprintBytes =
        kVFootprintElements * sizeof(typename Geometry::VDataType);
    static constexpr index_t kKPrefetchTensorCount = Tuning::kKWait;
    static constexpr index_t kVPrefetchTensorCount = Tuning::kVWait;
    static constexpr bool kPrefetchTailDrain       = Tuning::kTailDrain;
    static_assert(kKPrefetchTensorCount == 0 || kKPrefetchTensorCount == 1);
    static_assert(kVPrefetchTensorCount == 0 || kVPrefetchTensorCount == 1);

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

    // fp32 S/O fragments are input-type independent, so one softmax/output map serves both.
    using OutputFragments = FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2>;
    using ScoreSoftmax    = FmhaTdmSchedSplitSoftmaxFor<Geometry::kQkStages>;
    using ScoreMapping    = typename ScoreSoftmax::Mapping;
    // The hand-written score mapping and output fragments must match the geometry.
    static_assert(ScoreMapping::kNumMIter == Geometry::kQkMIter &&
                      ScoreMapping::kNumNIter == Geometry::kN / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumSu == Geometry::kQkStages &&
                      ScoreMapping::kNIterPerSu == Geometry::kQkSuNIter &&
                      ScoreMapping::kNumMsb == Geometry::kQkMIter * Geometry::kQkSuNIter &&
                      ScoreMapping::kElementsPerWmma ==
                          Geometry::QkWmma::kCM0PerLane * Geometry::QkWmma::kCM1PerLane,
                  "split-softmax score mapping does not match the QK geometry");
    static_assert(OutputFragments::kNumFragments == Geometry::kPvMIter * Geometry::kPvNIter &&
                      OutputFragments::kNumDmsb % Geometry::kPvMIter == 0 &&
                      OutputFragments::kElementsPerFragment ==
                          Geometry::PvWmma::kCM0PerLane * Geometry::PvWmma::kCM1PerLane,
                  "output fragments do not match the PV geometry");

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

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKSuRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemmSu<Problem>())>;
        static_assert(BlockGemm::NIterPerWarp == Geometry::kQkSuNIter &&
                      BlockGemm::KIterPerWarp == Geometry::kQkKIter);
        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    // Marks a GEMM whose WMMA scale operands stay neutral.
    struct NoScale
    {
    };

    template <FmhaTdmSchedSchedulePoint Point>
    CK_TILE_DEVICE static void EmitSchedulePoint()
    {
        if constexpr(Point == FmhaTdmSchedSchedulePoint::AfterTokens)
            __builtin_amdgcn_sched_barrier(0);
        else
            __builtin_amdgcn_sched_barrier(LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::TRANS);
    }

    // Fence emitter of the placement executor: (stage, wmma, point).
    struct SchedulePointEmitter
    {
        template <typename Stage, typename Wmma, typename Point>
        CK_TILE_DEVICE void operator()(Stage, Wmma, Point) const
        {
            EmitSchedulePoint<Point::value>();
        }
    };

    // Scaled D192 runs K64 steps 0+1 as one K128 WMMA; step 2 stays zero-padded.
    template <typename BlockGemm>
    static constexpr bool kPairsScaledK64 =
        BlockGemm::WarpGemm::kK != 128 && BlockGemm::KIterPerWarp == 3;

    // One QK WMMA of a K group; with scales, K64 runs as Derived's paired or zero-padded K128.
    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor,
                                           const AScaleFn& a_scale = {},
                                           const BScaleFn& b_scale = {})
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

        static_assert(WmmaOrdinal >= 0 && WmmaOrdinal < BlockGemm::NIterPerWarp *
                                                            BlockGemm::KIterPerWarp *
                                                            BlockGemm::MIterPerWarp);
        constexpr index_t n_iter =
            WmmaOrdinal / (BlockGemm::KIterPerWarp * BlockGemm::MIterPerWarp);
        constexpr index_t k_iter =
            (WmmaOrdinal / BlockGemm::MIterPerWarp) % BlockGemm::KIterPerWarp;
        constexpr index_t m_iter = WmmaOrdinal % BlockGemm::MIterPerWarp;

        constexpr bool kPairedK64 =
            !std::is_same_v<AScaleFn, NoScale> && kPairsScaledK64<BlockGemm>;
        if constexpr(kPairedK64 && k_iter == 0)
            return;

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

        if constexpr(std::is_same_v<AScaleFn, NoScale>)
            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        else if constexpr(WarpGemm::kK == 128)
            WarpGemm{}(c_warp_tensor,
                       a_warp_tensor,
                       b_warp_tensor,
                       a_scale(number<m_iter>{}, number<k_iter>{}),
                       b_scale(number<n_iter>{}, number<k_iter>{}));
        else if constexpr(kPairedK64 && k_iter == 1)
        {
            AWarpTensor a_lo_tensor;
            a_lo_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<m_iter, 0>{}, a_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
            BWarpTensor b_lo_tensor;
            b_lo_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<n_iter, 0>{}, b_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
            Derived::RunPairedScaledWmma(c_warp_tensor,
                                         a_lo_tensor,
                                         a_warp_tensor,
                                         b_lo_tensor,
                                         b_warp_tensor,
                                         a_scale(number<m_iter>{}, number<k_iter>{}),
                                         b_scale(number<n_iter>{}, number<k_iter>{}));
        }
        else
            Derived::RunPaddedScaledWmma(c_warp_tensor,
                                         a_warp_tensor,
                                         b_warp_tensor,
                                         a_scale(number<m_iter>{}, number<k_iter>{}),
                                         b_scale(number<n_iter>{}, number<k_iter>{}));
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    // Problem conditions both pipelines require besides bias and the QK K-tile width.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedSchedProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == Geometry::kWaveSize;
#else
        constexpr bool is_wave32 = true;
#endif
        // Exactly this policy's geometry; both GEMMs share one wave layout so P stays in S's lanes.
        return std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kSubQKHeaddim == Geometry::kHeadDimQK && Shape::IsVLayoutRowMajor &&
               Shape::NumWarps == Geometry::kWaves && is_wave32;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK0() { return Derived::kLdsOffsetK0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK1() { return Derived::kLdsOffsetK1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV0() { return Derived::kLdsOffsetV0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV1() { return Derived::kLdsOffsetV1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsArenaSize()
    {
        return Derived::kLdsArenaSize;
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm sched requires a full-head K TDM load");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<Derived::kKDramBoxWidth>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<Derived::kKLdsWidth>{}),
            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "qr_tdm sched requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackV<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kVLogicalWidth>{}),
            make_tuple(number<kVPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(!Xor, "the V TDM writer requires an affine V LDS descriptor");
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
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

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm sched requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        return GetLdsArenaSize();
    }

    template <typename Padding>
    CK_TILE_HOST_DEVICE static constexpr auto EncodeTdmPadding()
    {
        using Encoded = detail::EncodedTdmPadding<Padding>;
        return make_tuple(number<Encoded::kEnabled>{},
                          number<Encoded::kPadAmount>{},
                          number<Encoded::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        return EncodeTdmPadding<typename Derived::template QPadding<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        return EncodeTdmPadding<typename Derived::KPadding>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(Derived::template IsSupportedProblem<Problem>(),
                      "qr_tdm sched policy geometry does not match the problem");
        return EncodeTdmPadding<typename Derived::VPadding>();
    }
};

} // namespace ck_tile
