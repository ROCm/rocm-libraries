// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/fmha/detail/fmha_fwd_cr_args.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_cr.hpp"

namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdKernel;

// Host selection is independent of the compact and shared device entries.
template <typename Pipeline_, typename Epilogue_, typename Tuning_ = FmhaTdmCrTuning<>>
struct FmhaFwdTdmCrSelector
{
    using FmhaPipeline = Pipeline_;
    using Tuning       = Tuning_;
    static constexpr bool kHeadPadded =
        Pipeline_::Problem::kPadHeadDimQ || Pipeline_::Problem::kPadHeadDimV;
    static constexpr bool kGeneral = std::is_same_v<Tuning, FmhaTdmCrGeneralTuning> ||
                                     Pipeline_::kGeneralArguments || Tuning::kVirtualSink;
    using StridedKernel = FmhaFwdKernel<Pipeline_, Epilogue_>;
    using GeneralKernel =
        FmhaFwdKernel<FmhaTdmCrPipelineVariant<Pipeline_, FmhaTdmCrGeneralTuning>, Epilogue_>;
    using SinkKernel =
        FmhaFwdKernel<FmhaTdmCrPipelineVariant<Pipeline_, FmhaTdmCrSinkTuning>, Epilogue_>;

    // Choose a native specialization; argument construction, validation and
    // launch remain the caller's responsibility. Preserve this order when
    // adding tuning variants, since several supported ranges overlap.
    template <typename Args, typename Visitor>
    CK_TILE_HOST static float VisitKernels(const Args& a, Visitor&& visitor)
    {
        using Kernel   = FmhaFwdTdmCrSelector<Pipeline_, Epilogue_>;
        using Pipeline = FmhaPipeline;
        using Problem  = typename Pipeline::Problem;
        if(ck_tile::get_device_name().compare(0, 7, "gfx1250") != 0)
            return -1;
        if(a.sink_ptr)
        {
            using SinkSelector = FmhaFwdTdmCrSelector<Pipeline_, Epilogue_, FmhaTdmCrSinkTuning>;
            if(!SinkSelector::IsSupportedInput(a))
                return -1;
            if constexpr(Pipeline::kHeadDim == 192 && Pipeline::kM0 == 64 && Pipeline::kN0 == 64 &&
                         Pipeline::kCausal && !Pipeline::kStoreLSE && !kHeadPadded &&
                         !Pipeline::kGeneralArguments && !Problem::kSkipMinSeqlenQ)
            {
                if(a.window_size_left < 0 && a.window_size_right == 0)
                {
                    using CausalSinkKernel = FmhaFwdKernel<
                        FmhaTdmCrPipelineVariant<Pipeline_, FmhaTdmCrCausalSinkTuning>,
                        Epilogue_>;
                    return visitor(CausalSinkKernel{});
                }
            }
            return visitor(SinkKernel{});
        }
        if(!Kernel::IsSupportedInput(a))
            return -1;

        if constexpr(kHeadPadded || Pipeline::kGeneralArguments)
            return visitor(StridedKernel{});
        else
        {
            if(a.scale_s <= 0 || !detail::fmha_fwd_cr_vector_output(a))
                return visitor(GeneralKernel{});

            auto launch_strided = [&] { return visitor(typename Kernel::StridedKernel{}); };
            if constexpr(!Pipeline::kCompact)
                return launch_strided();
            else
            {
                if(!detail::fmha_fwd_cr_compact_args(a))
                    return launch_strided();
                auto launch = [&](auto tuning) {
                    using K =
                        FmhaFwdKernel<FmhaTdmCrCompactPipelineVariant<Pipeline_, decltype(tuning)>,
                                      Epilogue_>;
                    return visitor(K{});
                };
                if constexpr(Pipeline::kM0 == 64)
                    return launch(ck_tile::FmhaTdmCrTuning<0, true, true>{});
                else
                {
                    if constexpr(Pipeline::kM0 == 128 && !Problem::kPadSeqLenQ &&
                                 !Problem::kPadSeqLenK)
                    {
                        if constexpr(Pipeline::kCausal)
                        {
                            // A single query tile has no traversal to group across heads.
                            if(a.seqlen_q == 128 && a.seqlen_k == 128)
                                return launch(ck_tile::FmhaTdmCrTuning<0, true, false, 0>{});
                            // Four causal query tiles benefit from larger head groups once
                            // every XCD owns at least two complete groups of 32 rows.
                            if(a.nhead_q == 8 && a.seqlen_q == 512 && a.seqlen_k == 512 &&
                               a.batch >= 64 && a.batch % 32 == 0)
                                return launch(ck_tile::FmhaTdmCrTileTuning<4>{});
                        }
                        if(detail::fmha_fwd_cr_has_static_tuning(a))
                        {
                            switch(a.seqlen_q)
                            {
                            case 512: return launch(ck_tile::FmhaTdmCrTuning<512>{});
                            case 1024: return launch(ck_tile::FmhaTdmCrTuning<1024>{});
                            case 2048: return launch(ck_tile::FmhaTdmCrTuning<2048>{});
                            case 4096: return launch(ck_tile::FmhaTdmCrTuning<4096>{});
                            case 8192: return launch(ck_tile::FmhaTdmCrTuning<8192>{});
                            case 16384: return launch(ck_tile::FmhaTdmCrTuning<16384>{});
                            case 32768: return launch(ck_tile::FmhaTdmCrTuning<32768>{});
                            }
                        }
                        if constexpr(!Pipeline::kCausal)
                        {
                            if(a.nhead_q == 8)
                            {
                                if(a.seqlen_q == a.seqlen_k)
                                {
                                    switch(a.seqlen_q)
                                    {
                                    case 128: return launch(ck_tile::FmhaTdmCrTileTuning<1>{});
                                    case 256: return launch(ck_tile::FmhaTdmCrTileTuning<2>{});
                                    case 512: return launch(ck_tile::FmhaTdmCrTileTuning<4>{});
                                    case 640: return launch(ck_tile::FmhaTdmCrTileTuning<5>{});
                                    case 1024: return launch(ck_tile::FmhaTdmCrTileTuning<8>{});
                                    case 4096:
                                        if(a.batch <= 4)
                                            return launch(ck_tile::FmhaTdmCrTileTuning<32>{});
                                        break;
                                    }
                                }
                                if(a.seqlen_q == 1024 && a.seqlen_k == 2048)
                                    return launch(ck_tile::FmhaTdmCrTileTuning<8, 16>{});
                                if(a.seqlen_q == 2048 && a.seqlen_k == 1024)
                                    return launch(ck_tile::FmhaTdmCrTileTuning<16, 8>{});
                            }
                        }
                    }
                    if constexpr(Pipeline::kM0 == 128 && !Pipeline::kCausal &&
                                 Problem::kPadSeqLenQ && Problem::kPadSeqLenK)
                    {
                        if(a.nhead_q == 8 && a.seqlen_q > 512 && a.seqlen_q < 640 &&
                           a.seqlen_k > 512 && a.seqlen_k < 640)
                        {
                            // Nearly full tails have a fixed K tile count and benefit from
                            // fragment masks for their partial query/key tiles.
                            if(a.seqlen_q % 16 == 15 && a.seqlen_k % 128 > 112)
                                return launch(ck_tile::FmhaTdmCrTileTuning<5, 0, true>{});
                            return launch(ck_tile::FmhaTdmCrTileTuning<5, 0>{});
                        }
                    }
                    if(a.seqlen_q <= 512)
                        return launch(ck_tile::FmhaTdmCrTuning<0, true, false>{});
                    if(a.seqlen_q <= 2048)
                        return launch(ck_tile::FmhaTdmCrTuning<0, true, true>{});
                    return launch(ck_tile::FmhaTdmCrTuning<>{});
                }
            }
        }
    }

    // Validate the complete input before compact conversion discards its layout.
    // Input exposes the named fields of CK's forward tensor/sequence view; it
    // need not be the example API's fmha_fwd_args type.
    template <typename Input>
    CK_TILE_HOST static bool IsSupportedInput(const Input& a)
    {
        using Problem = typename FmhaPipeline::Problem;
        return detail::fmha_fwd_cr_supported_args(a) &&
               (Tuning::kVirtualSink == (a.sink_ptr != nullptr)) &&
               (!Problem::kHasSink || detail::fmha_fwd_cr_sink_args(a)) &&
               detail::fmha_fwd_cr_score_args<Problem::BiasEnum, Problem::kHasLogitsSoftCap>(a) &&
               Problem::kIsGroupMode == (a.seqstart_q_ptr != nullptr) &&
               FmhaPipeline::kCausal == (a.mask_type != 0) &&
               (!FmhaPipeline::kStoreLSE || a.lse_ptr) &&
               (Problem::kPadHeadDimQ ? a.hdim_q <= FmhaPipeline::kHeadDim
                                      : a.hdim_q == FmhaPipeline::kHeadDim) &&
               (Problem::kPadHeadDimV ? a.hdim_v <= FmhaPipeline::kValueDim
                                      : a.hdim_v == FmhaPipeline::kValueDim) &&
               (Problem::kIsGroupMode || Problem::kPadSeqLenQ ||
                a.seqlen_q % FmhaPipeline::kM0 == 0) &&
               (Problem::kIsGroupMode || Problem::kPadSeqLenK ||
                a.seqlen_k % FmhaPipeline::kN0 == 0);
    }
};

// Optional host selection for CR; ordinary pipelines use the empty policy.
template <typename Pipeline, typename Epilogue, typename>
struct FmhaFwdKernelLaunchPolicy;

template <typename Pipeline, typename Epilogue>
struct FmhaFwdKernelLaunchPolicy<Pipeline, Epilogue, std::void_t<typename Pipeline::CrBasePipeline>>
{
    static constexpr bool kUsesPreparedArguments = true;
    static constexpr bool kUsesCompactArguments  = Pipeline::kUsesCompactArguments;
    template <typename DefaultArguments>
    using DeviceArguments =
        std::conditional_t<kUsesCompactArguments, typename Pipeline::Arguments, DefaultArguments>;
    using Base          = typename Pipeline::CrBasePipeline;
    using Tuning        = typename Pipeline::Tuning;
    using Selector      = FmhaFwdTdmCrSelector<Base, Epilogue, Tuning>;
    using StridedKernel = FmhaFwdKernel<Pipeline, Epilogue>;
    template <typename OtherTuning>
    using WithTuning = std::conditional_t<
        kUsesCompactArguments,
        FmhaFwdKernel<FmhaTdmCrCompactPipelineVariant<Base, OtherTuning>, Epilogue>,
        std::conditional_t<std::is_same_v<OtherTuning, FmhaTdmCrTuning<>>,
                           FmhaFwdKernel<Base, Epilogue>,
                           FmhaFwdKernel<FmhaTdmCrPipelineVariant<Base, OtherTuning>, Epilogue>>>;
    static constexpr bool kGeneral = Selector::kGeneral;
    // Resolve offsets first where retaining tensor strides in scalar registers
    // improves the group schedule.
    static constexpr bool kLoadsGroupLengthsFirst =
        !(Pipeline::kIsGroupMode && Pipeline::kHeadDim == 192 && Pipeline::kM0 == 64 &&
          Pipeline::kN0 == 64 && Pipeline::kCausal && !Pipeline::Problem::kPadHeadDimQ &&
          !Pipeline::Problem::kPadHeadDimV &&
          Pipeline::BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !Pipeline::kHasLogitsSoftCap &&
          !Pipeline::kHasSink && !Pipeline::Problem::kSkipMinSeqlenQ &&
          !std::is_same_v<Tuning, FmhaTdmCrCausalSinkTuning>) &&
        !(Pipeline::kIsGroupMode && Pipeline::kCausal && !Pipeline::kStoreLSE &&
          !Pipeline::kHasSink && !Pipeline::Problem::kSkipMinSeqlenQ && !Tuning::kVirtualSink &&
          ((Pipeline::kHeadDim == 128 && Pipeline::kM0 == 64 && !Pipeline::Problem::kPadHeadDimQ &&
            !Pipeline::Problem::kPadHeadDimV &&
            (Pipeline::BiasEnum != BlockAttentionBiasEnum::NO_BIAS ||
             Pipeline::kHasLogitsSoftCap)) ||
           (std::is_same_v<typename Pipeline::QDataType, half_t> && Pipeline::kHeadDim == 32 &&
            Pipeline::kM0 == 128 && Pipeline::Problem::kPadHeadDimQ &&
            Pipeline::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
            !Pipeline::kHasLogitsSoftCap)));

    template <typename Input>
    CK_TILE_HOST static bool IsSupportedInput(const Input& a)
    {
        return Selector::IsSupportedInput(a);
    }

    template <typename Input, typename Visitor>
    CK_TILE_HOST static float VisitKernels(const Input& a, Visitor&& visitor)
    {
        return Selector::VisitKernels(a, static_cast<Visitor&&>(visitor));
    }

    // Validate the input before construction. Compact arguments discard layout
    // metadata and additionally require fmha_fwd_cr_compact_args(a).
    template <typename Input>
    CK_TILE_HOST static auto MakeKargs(const Input& a)
    {
        if constexpr(kUsesCompactArguments)
        {
            return typename Pipeline::Arguments{a.q_ptr,
                                                a.k_ptr,
                                                a.v_ptr,
                                                a.o_ptr,
                                                static_cast<float*>(a.lse_ptr),
                                                a.batch,
                                                a.nhead_q,
                                                a.seqlen_q,
                                                a.seqlen_k,
                                                a.scale_s};
        }
        else
        {
            typename StridedKernel::Kargs x{};
            x.q_ptr          = a.q_ptr;
            x.k_ptr          = a.k_ptr;
            x.v_ptr          = a.v_ptr;
            x.o_ptr          = a.o_ptr;
            x.sink_ptr       = a.sink_ptr;
            x.seqlen_q       = Base::kIsGroupMode ? a.max_seqlen_q : a.seqlen_q;
            x.seqlen_k       = a.seqlen_k;
            x.hdim_q         = a.hdim_q;
            x.hdim_v         = a.hdim_v;
            x.num_head_q     = a.nhead_q;
            x.nhead_ratio_qk = a.nhead_q / a.nhead_k;
            if constexpr(detail::FmhaFwdCrHasHeadRange<Input>::value)
            {
                x.num_head_q_total = a.num_head_q_total;
                x.head_start       = a.head_start;
            }
            x.scale_s         = a.scale_s;
            x.stride_q        = a.stride_q;
            x.stride_k        = a.stride_k;
            x.stride_v        = a.stride_v;
            x.stride_o        = a.stride_o;
            x.nhead_stride_q  = a.nhead_stride_q;
            x.nhead_stride_k  = a.nhead_stride_k;
            x.nhead_stride_v  = a.nhead_stride_v;
            x.nhead_stride_o  = a.nhead_stride_o;
            x.cu_seqlen_q_ptr = static_cast<const int32_t*>(a.cu_seqlen_q_ptr);
            x.cu_seqlen_k_ptr = static_cast<const int32_t*>(a.cu_seqlen_k_ptr);
            if constexpr(Base::kIsGroupMode)
            {
                x.seqstart_q_ptr = static_cast<const int32_t*>(a.seqstart_q_ptr);
                x.seqstart_k_ptr = static_cast<const int32_t*>(a.seqstart_k_ptr);
                x.seqlen_q_ptr   = static_cast<const int32_t*>(a.seqlen_q_ptr);
                x.seqlen_k_ptr   = static_cast<const int32_t*>(a.seqlen_k_ptr);
            }
            else
            {
                x.batch_stride_q = a.batch_stride_q;
                x.batch_stride_k = a.batch_stride_k;
                x.batch_stride_v = a.batch_stride_v;
                x.batch_stride_o = a.batch_stride_o;
            }
            if constexpr(Base::kStoreLSE)
            {
                x.lse_ptr          = a.lse_ptr;
                x.nhead_stride_lse = a.nhead_stride_lse;
                x.batch_stride_lse = a.batch_stride_lse;
            }
            if constexpr(Base::kCausal)
            {
                x.mask_type         = static_cast<GenericAttentionMaskEnum>(a.mask_type);
                x.window_size_left  = a.window_size_left;
                x.window_size_right = a.window_size_right;
                x.sink_size         = a.sink_size;
            }
            if constexpr(Base::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                x.bias_ptr          = a.bias_ptr;
                x.stride_bias       = a.stride_bias;
                x.nhead_stride_bias = a.nhead_stride_bias;
                if constexpr(!Base::kIsGroupMode)
                    x.batch_stride_bias = a.batch_stride_bias;
            }
            else if constexpr(Base::BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                x.alibi_slope_ptr    = a.bias_ptr;
                x.alibi_slope_stride = a.stride_bias;
            }
            if constexpr(Base::kHasLogitsSoftCap)
                x.init_logits_soft_cap(a.logits_soft_cap);
            if constexpr(Base::Problem::kSkipMinSeqlenQ)
                x.min_seqlen_q = a.min_seqlen_q;
            return x;
        }
    }

    template <typename Kargs>
    CK_TILE_HOST static bool IsSupportedKargs(const Kargs& x)
    {
        if constexpr(kUsesCompactArguments)
        {
            const auto& a = x;
            detail::FmhaFwdCrInput view{};
            view.hdim_q   = Base::kHeadDim;
            view.hdim_v   = Base::kValueDim;
            view.seqlen_q = view.max_seqlen_q = a.sq;
            // Bound products before reconstructing the implicit compact strides.
            if(a.heads <= 0 || a.heads > std::numeric_limits<int32_t>::max() / 128)
                return false;
            view.q_ptr   = a.q;
            view.k_ptr   = a.k;
            view.v_ptr   = a.v;
            view.o_ptr   = a.o;
            view.lse_ptr = a.lse;
            view.batch   = a.batch;
            view.nhead_q = view.nhead_k = a.heads;
            view.seqlen_k               = a.sk;
            view.scale_s                = a.scale;
            view.stride_q = view.stride_k = view.stride_v = view.stride_o = int64_t{a.heads} * 128;
            view.nhead_stride_q = view.nhead_stride_k = view.nhead_stride_v = view.nhead_stride_o =
                128;
            view.batch_stride_q = view.batch_stride_o = view.stride_q * a.sq;
            view.batch_stride_k = view.batch_stride_v = view.stride_k * a.sk;
            view.nhead_stride_lse                     = a.sq;
            view.batch_stride_lse                     = int64_t{a.heads} * a.sq;
            view.mask_type                            = Base::kCausal ? 1 : 0;
            view.window_size_left                     = -1;
            view.window_size_right                    = Base::kCausal ? 0 : -1;
            if(!Base::kCompact || !detail::fmha_fwd_cr_compact_args(view))
                return false;
            if constexpr(Tuning::kSequence != 0)
                if(a.batch != 32 || a.heads != 8 || a.sq != Tuning::kSequence ||
                   a.sk != Tuning::kSequence)
                    return false;
            if constexpr(Tuning::kQueryTiles != 0)
            {
                if(a.heads != 8 ||
                   (int64_t{a.sq} + Base::kM0 - 1) / Base::kM0 != Tuning::kQueryTiles)
                    return false;
                if constexpr(Tuning::kKeyTiles != 0)
                    if(a.sq != Tuning::kQueryTiles * Base::kM0 ||
                       a.sk != Tuning::kKeyTiles * Base::kN0)
                        return false;
            }
            if constexpr(Tuning::kTailFragments)
                if(a.sq % 16 != 15 || a.sk <= 512 || a.sk >= 640 || a.sk % 128 <= 112)
                    return false;
            return Selector::IsSupportedInput(view) && view.scale_s > 0 &&
                   detail::fmha_fwd_cr_vector_output(view);
        }
        else
        {
            if(x.nhead_ratio_qk <= 0 || x.num_head_q <= 0 || x.num_head_q % x.nhead_ratio_qk ||
               x.head_start != 0 || (x.num_head_q_total != 0 && x.num_head_q_total != x.num_head_q))
                return false;
            detail::FmhaFwdCrInput a{};
            a.q_ptr    = x.q_ptr;
            a.k_ptr    = x.k_ptr;
            a.v_ptr    = x.v_ptr;
            a.o_ptr    = x.o_ptr;
            a.sink_ptr = x.sink_ptr;
            // Batch count belongs to the launch grid, as for the other forward pipelines.
            a.batch    = 1;
            a.seqlen_q = a.max_seqlen_q = x.seqlen_q;
            a.seqlen_k                  = x.seqlen_k;
            a.hdim_q                    = x.hdim_q;
            a.hdim_v                    = x.hdim_v;
            a.nhead_q                   = x.num_head_q;
            a.nhead_k                   = x.num_head_q / x.nhead_ratio_qk;
            a.scale_s                   = x.scale_s;
            a.stride_q                  = x.stride_q;
            a.stride_k                  = x.stride_k;
            a.stride_v                  = x.stride_v;
            a.stride_o                  = x.stride_o;
            a.nhead_stride_q            = x.nhead_stride_q;
            a.nhead_stride_k            = x.nhead_stride_k;
            a.nhead_stride_v            = x.nhead_stride_v;
            a.nhead_stride_o            = x.nhead_stride_o;
            a.cu_seqlen_q_ptr           = x.cu_seqlen_q_ptr;
            a.cu_seqlen_k_ptr           = x.cu_seqlen_k_ptr;
            if constexpr(Base::kIsGroupMode)
            {
                a.seqstart_q_ptr = x.seqstart_q_ptr;
                a.seqstart_k_ptr = x.seqstart_k_ptr;
                a.seqlen_q_ptr   = x.seqlen_q_ptr;
                a.seqlen_k_ptr   = x.seqlen_k_ptr;
            }
            else
            {
                a.batch_stride_q = x.batch_stride_q;
                a.batch_stride_k = x.batch_stride_k;
                a.batch_stride_v = x.batch_stride_v;
                a.batch_stride_o = x.batch_stride_o;
            }
            if constexpr(Base::kStoreLSE)
            {
                a.lse_ptr          = x.lse_ptr;
                a.nhead_stride_lse = x.nhead_stride_lse;
                a.batch_stride_lse = x.batch_stride_lse;
            }
            if constexpr(Base::kCausal)
            {
                a.mask_type         = static_cast<int>(x.mask_type);
                a.window_size_left  = x.window_size_left;
                a.window_size_right = x.window_size_right;
                a.sink_size         = x.sink_size;
            }
            if constexpr(Base::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                a.bias_ptr          = x.bias_ptr;
                a.stride_bias       = x.stride_bias;
                a.nhead_stride_bias = x.nhead_stride_bias;
                if constexpr(!Base::kIsGroupMode)
                    a.batch_stride_bias = x.batch_stride_bias;
            }
            else if constexpr(Base::BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                a.bias_ptr    = x.alibi_slope_ptr;
                a.stride_bias = x.alibi_slope_stride;
            }
            if constexpr(Base::kHasLogitsSoftCap)
                a.logits_soft_cap = x.logits_soft_cap;
            if constexpr(Base::Problem::kSkipMinSeqlenQ)
                a.min_seqlen_q = x.min_seqlen_q;
            return IsSupportedInput(a) && (Base::kPadHeadDimQ || Base::kPadHeadDimV || kGeneral ||
                                           (a.scale_s > 0 && detail::fmha_fwd_cr_vector_output(a)));
        }
    }
};

} // namespace ck_tile
