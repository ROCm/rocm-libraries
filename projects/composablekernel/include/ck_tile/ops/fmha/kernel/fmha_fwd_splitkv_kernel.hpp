// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/detail/fmha_fwd_kernel_path_markers.hpp"

#include <string>
#include <type_traits>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

namespace detail {
template <typename T>
using has_max_waves_per_eu = decltype(T::kMaxWavesPerEu);
} // namespace detail

// With a non-void CombineODataType_ the split kernel also merges the splits: the last
// workgroup to finish an output tile (counted in split_counter_ptr, which must start zeroed and
// is left zeroed) reads every split's o_acc/lse_acc back and writes O and LSE, so no combine
// kernel runs.
template <typename FmhaPipeline_, typename EpiloguePipeline_, typename CombineODataType_ = void>
struct FmhaFwdSplitKVKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;
    static constexpr ck_tile::index_t kMaxWavesPerEu   = []() {
        if constexpr(is_detected<detail::has_max_waves_per_eu, FmhaPipeline>::value)
            return FmhaPipeline::kMaxWavesPerEu;
        else
            return 0;
    }();

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;
    using OaccDataType = remove_cvref_t<typename FmhaPipeline::OaccDataType>;
    using ODataType    = remove_cvref_t<typename FmhaPipeline::ODataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = FmhaPipeline::Problem::kDoFp8StaticQuant;
    static constexpr auto QScaleEnum        = FmhaPipeline::Problem::QScaleEnum;
    static constexpr bool kPerTensorQScale  = QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR;
    static constexpr bool kIsPagedKV        = FmhaPipeline::Problem::kIsPagedKV;
    static constexpr bool kHasSink          = FmhaPipeline::Problem::kHasSink;
    static constexpr bool kMergeNumHeadGroupsSeqLenQ =
        FmhaPipeline::Problem::kMergeNumHeadGroupsSeqLenQ;
    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    using CombineODataType             = CombineODataType_;
    static constexpr bool kFuseCombine = !std::is_void_v<CombineODataType>;
    // weights of every split for each row of the tile live in LDS
    static constexpr index_t kMaxFusedSplits = 128;
    // the largest workgroup cluster
    static constexpr index_t kMaxClusterSplits = 16;
    static_assert(!kFuseCombine || (!kIsGroupMode && !kDoFp8StaticQuant),
                  "the fused combine supports batch mode without the static-quant O scale");

    // Markers are described in fmha_fwd_kernel_path_markers.hpp. Unlike
    // fmha_fwd_kernel.hpp there is no name-based fallback: a pipeline that
    // wants these paths must declare them.
    static constexpr bool kUsesUntransposedVKernelPath =
        detail::uses_untransposed_v_kernel_path_v<FmhaPipeline>;
    static constexpr bool kUsesTdmAffineDramPath =
        detail::uses_tdm_affine_dram_path_v<FmhaPipeline>;

    // PageBlockNavigator's VirtualDim names the dimension that is paged, i.e.
    // the one holding seqlen_k. The generic path transposes V to
    // (hdim_v, seqlen_k) and so pages dim 1, while the untransposed path keeps
    // the native (seqlen_k, hdim_v) and pages dim 0. K is always (seqlen_k, hdim_q).
    static constexpr index_t kVPagedDim = kUsesUntransposedVKernelPath ? 0 : 1;

    // make_v_dram leaves untransposed V unpadded, which is only safe when the
    // TDM box clamp bounds the reads past seqlen_k.
    static_assert(!kUsesUntransposedVKernelPath || kUsesTdmAffineDramPath,
                  "untransposed V requires the TDM affine DRAM path");

    // see FmhaFwdKernel::kKVCoherence
#if defined(__gfx125__)
    static constexpr auto kKVCoherence = detail::uses_kv_nt_ht_hint_v<FmhaPipeline>
                                             ? amd_buffer_coherence_enum::NT_HT
                                             : amd_buffer_coherence_enum::coherence_default;
#else
    static constexpr auto kKVCoherence = amd_buffer_coherence_enum::coherence_default;
#endif

    // With the fused combine, the splits' o_acc/lse_acc go to and come back from the device
    // scope directly, so the hand-off needs no cache writeback or invalidate (see FuseCombine).
#if defined(__gfx12__)
    static constexpr auto kAccCoherence = kFuseCombine
                                              ? amd_buffer_coherence_enum::DEVICE
                                              : amd_buffer_coherence_enum::coherence_default;
#else
    static constexpr auto kAccCoherence = amd_buffer_coherence_enum::coherence_default;
#endif

    // with a mask or a sink, only the TDM pipelines merge: their merged rows take the mask of
    // row 0, and the sink logit of their own Q head
    static_assert(!kMergeNumHeadGroupsSeqLenQ ||
                  (kMergeNumHeadGroupsSeqLenQ && BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                   ((!kHasMask && !kHasSink) || kUsesTdmAffineDramPath)));

    // fp8 takes either the static-quant P scale or per-tensor q/k/v descales, not both
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
                      (kPerTensorQScale && !kDoFp8StaticQuant),
                  "split-kv supports per-tensor descales only");

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs = typename FmhaPipeline::BlockFmhaShape;
        using g0br = typename bfs::Gemm0BlockWarps;
        using g1br = typename bfs::Gemm1BlockWarps;
        using g0wt = typename bfs::Gemm0WarpTile;
        using g1wt = typename bfs::Gemm1WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_fwd_splitkv_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_"
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" +
                    _TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kQKHeaddim) + "_" +
            "r" + _TS_(g0br::at(ck_tile::number<0>{})) + "x" + _TS_(g0br::at(ck_tile::number<1>{})) + "x" + _TS_(g0br::at(ck_tile::number<2>{})) + "_" +
            "r" + _TS_(g1br::at(ck_tile::number<0>{})) + "x" + _TS_(g1br::at(ck_tile::number<1>{})) + "x" + _TS_(g1br::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(g0wt::at(ck_tile::number<0>{})) + "x" + _TS_(g0wt::at(ck_tile::number<1>{})) + "x" + _TS_(g0wt::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(g1wt::at(ck_tile::number<0>{})) + "x" + _TS_(g1wt::at(ck_tile::number<1>{})) + "x" + _TS_(g1wt::at(ck_tile::number<2>{})) + "_" +
            (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) + _SS_(FmhaPipeline::name) + "_" +
            "v" + (std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor> ? "r" : "c") + (pn.empty() ? "_npad" : "_" + pn) +
            (kHasLogitsSoftCap ? "_logits" : "_nlogits" ) + (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("_nbias") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) +
            (kHasMask ? "_" + _SS_(FmhaMask::name) : "_nmask") + (kStoreLSE ? "_lse" : "_nlse" ) +
            (kDoFp8StaticQuant ? "_squant" : "_nsquant") + (kIsPagedKV ? "_pagedkv" : "_npagedkv" ) + (kHasSink ? "_sink" : "_nsink" ) +
            (kPerTensorQScale ? _SS_("_") + BlockAttentionQuantScaleEnumToStr<QScaleEnum>::name : _SS_("")) +
            (kFuseCombine ? "_fcomb" : "");
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct EmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct CommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* lse_acc_ptr;
        void* o_acc_ptr;
        const void* sink_ptr;

        ck_tile::index_t batch;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
        ck_tile::index_t num_splits;

        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o_acc;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_lse_acc;
        ck_tile::index_t nhead_stride_o_acc;

        ck_tile::index_t split_stride_lse_acc;
        ck_tile::index_t split_stride_o_acc;
    };

    struct LogitsSoftCapKargs
    {
        LogitsSoftCapKargs() = default;

        void init_logits_soft_cap(float logits_soft_cap_)
        {
            if(0 < logits_soft_cap_)
            {
                logits_soft_cap     = logits_soft_cap_;
                logits_soft_cap_rcp = 1.f / logits_soft_cap;
            }
            else
            {
                logits_soft_cap     = 0.f;
                logits_soft_cap_rcp = 0.f;
            }
        }

        float logits_soft_cap;
        float logits_soft_cap_rcp;
    };

    struct CommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct BatchModeBiasKargs : CommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct AlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct MaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right, sink_size;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct Fp8StaticQuantKargs
    {
        float scale_p;
    };

    // final O/LSE and the per-tile split counter of the fused combine
    struct FusedCombineKargs
    {
        void* o_ptr                       = nullptr;
        void* lse_ptr                     = nullptr; // LSE is stored only when not null
        int32_t* split_counter_ptr        = nullptr;
        ck_tile::index_t stride_o         = 0;
        ck_tile::index_t nhead_stride_o   = 0;
        ck_tile::index_t batch_stride_o   = 0;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
        // the splits of one tile form a workgroup cluster and merge after a cluster barrier
        int32_t cluster = 0;
    };

    // one fp32 descale each, read on the device
    struct PerTensorQScaleKargs
    {
        const void* q_descale_ptr = nullptr;
        const void* k_descale_ptr = nullptr;
        const void* v_descale_ptr = nullptr;
    };

    struct CommonPageBlockTableKargs
    {
        const int32_t* block_table_ptr;
        ck_tile::index_t batch_stride_block_table;
        ck_tile::index_t page_block_size;
    };

    struct GroupModePageBlockTableKargs : CommonPageBlockTableKargs
    {
        bool is_gappy = false;
    };

    struct CacheBatchIdxKargs
    {
        const int32_t* cache_batch_idx;
    };

    struct BatchModeKargs
        : CommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             BatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                AlibiKargs,
                                                EmptyKargs<0>>>,
          std::conditional_t<kHasMask, MaskKargs, EmptyKargs<1>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<2>>,
          std::conditional_t<kIsPagedKV, CommonPageBlockTableKargs, CacheBatchIdxKargs>,
          std::conditional_t<kHasLogitsSoftCap, LogitsSoftCapKargs, EmptyKargs<3>>,
          std::conditional_t<kPerTensorQScale, PerTensorQScaleKargs, EmptyKargs<5>>,
          std::conditional_t<kFuseCombine, FusedCombineKargs, EmptyKargs<6>>
    {
        const int32_t* seqlen_k_ptr;

        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k; // when using paged-kvcache, this will be stride/size for
                                         // single kcache page-block
        ck_tile::index_t batch_stride_v; // when using paged-kvcache, this will be stride/size for
                                         // single vcache page-block
        ck_tile::index_t batch_stride_lse_acc;
        ck_tile::index_t batch_stride_o_acc;
    };

    struct GroupModeKargs
        : CommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             CommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                AlibiKargs,
                                                EmptyKargs<0>>>,
          std::conditional_t<kHasMask, MaskKargs, EmptyKargs<1>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<2>>,
          std::conditional_t<kIsPagedKV, GroupModePageBlockTableKargs, EmptyKargs<3>>,
          std::conditional_t<kHasLogitsSoftCap, LogitsSoftCapKargs, EmptyKargs<4>>,
          std::conditional_t<kPerTensorQScale, PerTensorQScaleKargs, EmptyKargs<5>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;

        ck_tile::index_t batch_stride_k; // only used for paged-kvcache, this will be stride/size
                                         // for single kcache page-block
        ck_tile::index_t batch_stride_v; // only used for paged-kvcache, this will be stride/size
                                         // for single vcache page-block
    };

    using Kargs = std::conditional_t<kIsGroupMode, GroupModeKargs, BatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* lse_acc_ptr, /* workspace for lse accumulation when num_splits > 1, otherwise
                                    final lse */
              void* o_acc_ptr, /* workspace for o accumulation when num_splits > 1, otherwise final
                                  o */
              ck_tile::index_t batch,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k, // only used if 'seqlen_k_ptr' is not specified
              const void* seqlen_k_ptr,  // only used for (paged-) kvcache
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              ck_tile::index_t num_splits,
              const void* block_table_ptr,
              ck_tile::index_t batch_stride_block_table,
              ck_tile::index_t page_block_size,
              const void* cache_batch_idx,
              float scale_s,
              float scale_p,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_o_acc,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_lse_acc,
              ck_tile::index_t batch_stride_o_acc,
              ck_tile::index_t split_stride_lse_acc,
              ck_tile::index_t split_stride_o_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              const void* sink_ptr              = nullptr,
              const void* q_descale_ptr         = nullptr,
              const void* k_descale_ptr         = nullptr,
              const void* v_descale_ptr         = nullptr,
              void* o_ptr                       = nullptr,
              void* lse_ptr                     = nullptr,
              void* split_counter_ptr           = nullptr,
              ck_tile::index_t stride_o         = 0,
              ck_tile::index_t nhead_stride_o   = 0,
              ck_tile::index_t batch_stride_o   = 0,
              ck_tile::index_t nhead_stride_lse = 0,
              ck_tile::index_t batch_stride_lse = 0,
              bool cluster                      = false)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_acc_ptr,
                     o_acc_ptr,
                     sink_ptr,
                     batch,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     num_splits,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o_acc,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_lse_acc,
                     nhead_stride_o_acc,
                     split_stride_lse_acc,
                     split_stride_o_acc}, // args for common karg
                    {},                   // placeholder for bias
                    {},                   // placeholder for mask
                    {},                   // placeholder for fp8_static_quant args
                    {},                   // placeholder for paged-block table or cache_batch_idx
                    {},                   // placeholder for logits_soft_cap
                    {},                   // placeholder for per-tensor descales
                    {},                   // placeholder for the fused combine
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_lse_acc,
                    batch_stride_o_acc};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.sink_size         = sink_size;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
        }
        if constexpr(kIsPagedKV)
        {
            kargs.block_table_ptr          = reinterpret_cast<const int32_t*>(block_table_ptr);
            kargs.batch_stride_block_table = batch_stride_block_table;
            kargs.page_block_size          = page_block_size;
        }
        else
        {
            kargs.cache_batch_idx = reinterpret_cast<const int32_t*>(cache_batch_idx);
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }
        if constexpr(kPerTensorQScale)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        else
        {
            ignore = q_descale_ptr;
            ignore = k_descale_ptr;
            ignore = v_descale_ptr;
        }
        if constexpr(kFuseCombine)
        {
            kargs.o_ptr             = o_ptr;
            kargs.lse_ptr           = lse_ptr;
            kargs.split_counter_ptr = static_cast<int32_t*>(split_counter_ptr);
            kargs.stride_o          = stride_o;
            kargs.nhead_stride_o    = nhead_stride_o;
            kargs.batch_stride_o    = batch_stride_o;
            kargs.nhead_stride_lse  = nhead_stride_lse;
            kargs.batch_stride_lse  = batch_stride_lse;
            kargs.cluster           = cluster ? 1 : 0;
        }
        else
        {
            ignore = o_ptr;
            ignore = lse_ptr;
            ignore = split_counter_ptr;
            ignore = cluster;
            ignore = stride_o;
            ignore = nhead_stride_o;
            ignore = batch_stride_o;
            ignore = nhead_stride_lse;
            ignore = batch_stride_lse;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              void* lse_acc_ptr, /* workspace for lse accumulation when num_splits > 1, otherwise
                                    final lse */
              void* o_acc_ptr, /* workspace for o accumulation when num_splits > 1, otherwise final
                                  o */
              ck_tile::index_t batch,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              ck_tile::index_t num_splits,
              const void* block_table_ptr,
              ck_tile::index_t batch_stride_block_table,
              ck_tile::index_t page_block_size,
              bool is_gappy,
              float scale_s,
              float scale_p,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_o_acc,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t batch_stride_k, // only used for paged-kvcache
              ck_tile::index_t batch_stride_v, // only used for paged-kvcache
              ck_tile::index_t split_stride_lse_acc,
              ck_tile::index_t split_stride_o_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              const void* sink_ptr      = nullptr,
              const void* q_descale_ptr = nullptr,
              const void* k_descale_ptr = nullptr,
              const void* v_descale_ptr = nullptr)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_acc_ptr,
                     o_acc_ptr,
                     sink_ptr,
                     batch,
                     -1, // seqlen_q will be updated by another pointer
                     -1, // seqlen_k will be updated by another pointer
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     num_splits,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o_acc,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_lse_acc,
                     nhead_stride_o_acc,
                     split_stride_lse_acc,
                     split_stride_o_acc}, // args for common karg
                    {},                   // placeholder for bias
                    {},                   // placeholder for mask
                    {},                   // placeholder for fp8_static_quant args
                    {},                   // placeholder for paged-block table
                    {},                   // placeholder for logits_soft_cap
                    {},                   // placeholder for per-tensor descales
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_k,
                    batch_stride_v};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.sink_size         = sink_size;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
        }
        if constexpr(kIsPagedKV)
        {
            kargs.block_table_ptr          = reinterpret_cast<const int32_t*>(block_table_ptr);
            kargs.batch_stride_block_table = batch_stride_block_table;
            kargs.page_block_size          = page_block_size;
            kargs.is_gappy                 = is_gappy;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }
        if constexpr(kPerTensorQScale)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        else
        {
            ignore = q_descale_ptr;
            ignore = k_descale_ptr;
            ignore = v_descale_ptr;
        }
        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
                                                ck_tile::index_t nhead_q,
                                                ck_tile::index_t nhead_kv,
                                                ck_tile::index_t max_seqlen_q,
                                                ck_tile::index_t hdim_v,
                                                ck_tile::index_t num_splits)
    {
        ck_tile::index_t nhead_ = kMergeNumHeadGroupsSeqLenQ ? nhead_kv : nhead_q;
        ck_tile::index_t max_seqlen_q_ =
            max_seqlen_q * (kMergeNumHeadGroupsSeqLenQ ? nhead_q / nhead_kv : 1);

        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(max_seqlen_q_, FmhaPipeline::kM0) *
                        ck_tile::integer_divide_ceil(hdim_v, FmhaPipeline::kN1) * num_splits,
                    nhead_,
                    batch_size);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [mn, i_split]        = f(blockIdx.x, kargs.num_splits);
        const auto [i_tile_m, i_tile_n] = f(mn, num_tile_n1);
        const index_t i_nhead           = blockIdx.y;
        const index_t i_batch           = blockIdx.z;

        if constexpr(kHasMask)
        {
            // assume that num_tile_n1 is always 1
            return ck_tile::make_tuple(
                (gridDim.x / kargs.num_splits) - 1 - i_tile_m, i_tile_n, i_split, i_nhead, i_batch);
        }
        else
        {
            return ck_tile::make_tuple(i_tile_m, i_tile_n, i_split, i_nhead, i_batch);
        }
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t kFusedCombineBytes =
            kFuseCombine
                ? 16 + kMaxFusedSplits * FmhaPipeline::kM0 * static_cast<index_t>(sizeof(float))
                : 0;
        return ck_tile::max(
            FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize(), kFusedCombineBytes);
    }

    // The combine of FmhaFwdSplitKVCombineKernel for one output tile, run by the last of its
    // num_splits workgroups after all of them stored o_acc/lse_acc. Same math: logsum of the
    // split LSEs, O = sum_i exp(lse_i - logsum) * o_acc_i, and 0 / -inf for a row no split saw.
    CK_TILE_DEVICE static void FuseCombine(const Kargs& kargs,
                                           index_t i_m0,
                                           index_t i_n1,
                                           index_t i_split,
                                           index_t i_nhead,
                                           index_t i_batch,
                                           void* smem)
    {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        if(kargs.cluster)
        {
            FuseCombineInCluster(kargs, i_m0, i_n1, i_split, i_nhead, i_batch);
            return;
        }
#else
        ignore = i_split;
#endif

        constexpr index_t kM0 = FmhaPipeline::kM0;
        constexpr index_t kN1 = FmhaPipeline::kN1;

        auto* is_last = static_cast<int32_t*>(smem);
        auto* weights = reinterpret_cast<float*>(static_cast<char*>(smem) + 16); // [split][row]

        // o_acc/lse_acc are stored and loaded at device scope (kAccCoherence), so once every
        // wave's stores are acknowledged a relaxed count hands them over. A release/acquire
        // pair would instead write back and invalidate the whole cache on each split.
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            const index_t num_mn = static_cast<index_t>(gridDim.x) / kargs.num_splits;
            const long_index_t tile =
                (static_cast<long_index_t>(blockIdx.z) * gridDim.y + blockIdx.y) * num_mn +
                static_cast<index_t>(blockIdx.x) / kargs.num_splits;
            int32_t* counter = kargs.split_counter_ptr + tile;
            const bool last =
                __scoped_atomic_fetch_add(counter, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) ==
                kargs.num_splits - 1;
            if(last) // zeroed again for the next launch; every split has counted
                __scoped_atomic_store_n(counter, 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            *is_last = last ? 1 : 0;
        }
        __syncthreads();
        if(*is_last == 0)
            return;

        // tile row r is merged row g = i_m0 + r: q head (g / seqlen_q) of the group, query
        // (g % seqlen_q); without merging g < seqlen_q, so the head offset is 0
        const index_t ratio           = kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1;
        const index_t total_rows      = kargs.seqlen_q * ratio;
        const index_t num_splits      = kargs.num_splits;
        const long_index_t i_nhead_q0 = static_cast<long_index_t>(i_nhead) * ratio;
        const auto row_offset         = [&](index_t g, index_t nhead_stride, index_t row_stride) {
            return static_cast<long_index_t>(g / kargs.seqlen_q) * nhead_stride +
                   static_cast<long_index_t>(g % kargs.seqlen_q) * row_stride;
        };

        const auto* lse_acc = static_cast<const LSEDataType*>(kargs.lse_acc_ptr) +
                              i_nhead_q0 * kargs.nhead_stride_lse_acc +
                              static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse_acc;
        const auto* o_acc = static_cast<const OaccDataType*>(kargs.o_acc_ptr) +
                            i_nhead_q0 * kargs.nhead_stride_o_acc +
                            static_cast<long_index_t>(i_batch) * kargs.batch_stride_o_acc;
        // byte offsets from these bases fit in 32 bits for the workspace of one batch/head
        const auto lse_acc_res = make_wave_buffer_resource(lse_acc, 0x7ffffffe);
        const auto o_acc_res   = make_wave_buffer_resource(o_acc, 0x7ffffffe);
        auto* o = static_cast<CombineODataType*>(kargs.o_ptr) + i_nhead_q0 * kargs.nhead_stride_o +
                  static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        auto* lse = static_cast<LSEDataType*>(kargs.lse_ptr);

        // The partial results are read in as few dependent rounds as possible: one round
        // carries this lane's LSEs and the o_acc of up to kChunk splits, and each further
        // kChunk splits take one more. A lane walking the splits one load at a time is
        // latency bound and costs more than the combine kernel this replaces.
        constexpr index_t kVec   = 4; // contiguous columns per access
        constexpr index_t kIters = kM0 * kN1 / (kVec * FmhaPipeline::kBlockSize);
        constexpr index_t kChunk = 8;
        static_assert(kN1 % kVec == 0 && kIters * kVec * FmhaPipeline::kBlockSize == kM0 * kN1,
                      "the fused combine splits the tile evenly over the lanes");
        static_assert(kMaxFusedSplits * kM0 % FmhaPipeline::kBlockSize == 0);
        using vec_t = ext_vector_t<float, kVec>;

        long_index_t src[kIters];
        long_index_t dst[kIters];
        index_t wrow[kIters];
        bool valid[kIters];
        static_for<0, kIters, 1>{}([&](auto it) {
            const index_t idx = threadIdx.x + it * FmhaPipeline::kBlockSize;
            const index_t r   = idx / (kN1 / kVec);
            const index_t g   = i_m0 + r;
            const index_t col = i_n1 + idx % (kN1 / kVec) * kVec;
            valid[it]         = g < total_rows && col < kargs.hdim_v;
            // lanes past the tile still load, from a clamped in-bounds address
            const index_t g_in   = min(g, total_rows - 1);
            const index_t col_in = min(col, kargs.hdim_v - kVec);
            src[it]  = row_offset(g_in, kargs.nhead_stride_o_acc, kargs.stride_o_acc) + col_in;
            dst[it]  = row_offset(g_in, kargs.nhead_stride_o, kargs.stride_o) + col;
            wrow[it] = r;
        });
        // loads of kChunk splits from i0 on; splits past num_splits reread the last one and
        // are given weight 0 below
        const auto load_chunk = [&](index_t i0, vec_t(&part)[kChunk][kIters]) {
            static_for<0, kChunk, 1>{}([&](auto j) {
                const long_index_t split = static_cast<long_index_t>(min(i0 + j, num_splits - 1)) *
                                           kargs.split_stride_o_acc;
                static_for<0, kIters, 1>{}([&](auto it) {
                    part[j][it] =
                        bit_cast<vec_t>(amd_buffer_load_impl<OaccDataType, kVec, kAccCoherence>(
                            o_acc_res,
                            static_cast<index_t>((split + src[it]) * sizeof(OaccDataType)),
                            0));
                });
            });
        };

        vec_t part[kChunk][kIters];
        load_chunk(0, part);
        constexpr index_t kLsePerLane = kMaxFusedSplits * kM0 / FmhaPipeline::kBlockSize;
        LSEDataType lse_in[kLsePerLane];
        static_for<0, kLsePerLane, 1>{}([&](auto j) {
            const index_t idx   = threadIdx.x + j * FmhaPipeline::kBlockSize;
            const index_t g     = i_m0 + idx % kM0;
            const LSEDataType v = amd_buffer_load_impl<LSEDataType, 1, kAccCoherence>(
                lse_acc_res,
                static_cast<index_t>(
                    (static_cast<long_index_t>(min(idx / kM0, num_splits - 1)) *
                         kargs.split_stride_lse_acc +
                     row_offset(min(g, total_rows - 1), kargs.nhead_stride_lse_acc, 1)) *
                    sizeof(LSEDataType)),
                0)[number<0>{}];
            lse_in[j] =
                idx < num_splits * kM0 && g < total_rows ? v : -numeric<LSEDataType>::infinity();
        });
        static_for<0, kLsePerLane, 1>{}([&](auto j) {
            const index_t idx = threadIdx.x + j * FmhaPipeline::kBlockSize;
            if(idx < num_splits * kM0)
                weights[idx] = lse_in[j];
        });
        __syncthreads();

        for(index_t r = threadIdx.x; r < kM0; r += blockDim.x)
        {
            const index_t g = i_m0 + r;
            if(total_rows <= g)
                continue;
            LSEDataType lse_max = -numeric<LSEDataType>::infinity();
            for(index_t i = 0; i < num_splits; ++i)
                lse_max = max(lse_max, weights[i * kM0 + r]);
            LSEDataType lse_sum = 0;
            if(lse_max != -numeric<LSEDataType>::infinity())
                for(index_t i = 0; i < num_splits; ++i)
                    lse_sum += ck_tile::exp(weights[i * kM0 + r] - lse_max);
            const LSEDataType logsum =
                lse_sum == 0 ? -numeric<LSEDataType>::infinity() : ck_tile::log(lse_sum) + lse_max;
            for(index_t i = 0; i < num_splits; ++i)
                weights[i * kM0 + r] = logsum == -numeric<LSEDataType>::infinity()
                                           ? 0.f
                                           : ck_tile::exp(weights[i * kM0 + r] - logsum);
            if(lse != nullptr && i_n1 == 0)
                lse[static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse +
                    i_nhead_q0 * kargs.nhead_stride_lse +
                    row_offset(g, kargs.nhead_stride_lse, 1)] = logsum;
        }
        __syncthreads();

        vec_t acc[kIters];
        static_for<0, kIters, 1>{}([&](auto it) { acc[it] = vec_t{0.f, 0.f, 0.f, 0.f}; });
        for(index_t i0 = 0; i0 < num_splits; i0 += kChunk)
        {
            if(i0 > 0)
                load_chunk(i0, part);
            static_for<0, kChunk, 1>{}([&](auto j) {
                if(i0 + j < num_splits)
                    static_for<0, kIters, 1>{}([&](auto it) {
                        acc[it] += weights[(i0 + j) * kM0 + wrow[it]] * part[j][it];
                    });
            });
        }
        static_for<0, kIters, 1>{}([&](auto it) {
            if(valid[it])
                static_for<0, kVec, 1>{}([&](auto v) {
                    o[dst[it] + v] = type_convert<CombineODataType>(acc[it][v.value]);
                });
        });
    }

#if CK_TILE_ENABLE_CLUSTER_LAUNCH
    // The fused combine with the num_splits workgroups of a tile launched as one cluster: after a
    // cluster barrier each of them merges its own share of the tile's elements, every lane
    // straight from the partials of its row (no counter, no LDS).
    CK_TILE_DEVICE static void FuseCombineInCluster(const Kargs& kargs,
                                                    index_t i_m0,
                                                    index_t i_n1,
                                                    index_t i_split,
                                                    index_t i_nhead,
                                                    index_t i_batch)
    {
        constexpr index_t kM0   = FmhaPipeline::kM0;
        constexpr index_t kN1   = FmhaPipeline::kN1;
        constexpr index_t kVec  = 4;
        constexpr index_t kVecs = kM0 * kN1 / kVec;
        static_assert(kN1 % kVec == 0);
        using vec_t = ext_vector_t<float, kVec>;

        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __builtin_amdgcn_s_cluster_barrier();

        const index_t ratio           = kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1;
        const index_t total_rows      = kargs.seqlen_q * ratio;
        const index_t num_splits      = kargs.num_splits;
        const long_index_t i_nhead_q0 = static_cast<long_index_t>(i_nhead) * ratio;
        const auto row_offset         = [&](index_t g, index_t nhead_stride, index_t row_stride) {
            return static_cast<long_index_t>(g / kargs.seqlen_q) * nhead_stride +
                   static_cast<long_index_t>(g % kargs.seqlen_q) * row_stride;
        };
        const auto* lse_acc = static_cast<const LSEDataType*>(kargs.lse_acc_ptr) +
                              i_nhead_q0 * kargs.nhead_stride_lse_acc +
                              static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse_acc;
        const auto* o_acc = static_cast<const OaccDataType*>(kargs.o_acc_ptr) +
                            i_nhead_q0 * kargs.nhead_stride_o_acc +
                            static_cast<long_index_t>(i_batch) * kargs.batch_stride_o_acc;
        auto* o = static_cast<CombineODataType*>(kargs.o_ptr) + i_nhead_q0 * kargs.nhead_stride_o +
                  static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        auto* lse              = static_cast<LSEDataType*>(kargs.lse_ptr);
        const auto lse_acc_res = make_wave_buffer_resource(lse_acc, 0x7ffffffe);
        const auto o_acc_res   = make_wave_buffer_resource(o_acc, 0x7ffffffe);

        const index_t per   = integer_divide_ceil(kVecs, num_splits);
        const index_t v_end = min((i_split + 1) * per, kVecs);
        for(index_t vid = i_split * per + static_cast<index_t>(threadIdx.x); vid < v_end;
            vid += FmhaPipeline::kBlockSize)
        {
            const index_t r   = vid / (kN1 / kVec);
            const index_t g   = i_m0 + r;
            const index_t col = i_n1 + vid % (kN1 / kVec) * kVec;
            if(total_rows <= g || kargs.hdim_v <= col)
                continue;
            const long_index_t src =
                row_offset(g, kargs.nhead_stride_o_acc, kargs.stride_o_acc) + col;
            const long_index_t lse_src = row_offset(g, kargs.nhead_stride_lse_acc, 1);

            LSEDataType lse_in[kMaxClusterSplits];
            vec_t part[kMaxClusterSplits];
            static_for<0, kMaxClusterSplits, 1>{}([&](auto i) {
                if(num_splits <= i)
                    return;
                const long_index_t s = i;
                lse_in[i]            = amd_buffer_load_impl<LSEDataType, 1, kAccCoherence>(
                    lse_acc_res,
                    static_cast<index_t>((s * kargs.split_stride_lse_acc + lse_src) *
                                         sizeof(LSEDataType)),
                    0)[number<0>{}];
                part[i] = bit_cast<vec_t>(amd_buffer_load_impl<OaccDataType, kVec, kAccCoherence>(
                    o_acc_res,
                    static_cast<index_t>((s * kargs.split_stride_o_acc + src) *
                                         sizeof(OaccDataType)),
                    0));
            });
            LSEDataType lse_max = -numeric<LSEDataType>::infinity();
            static_for<0, kMaxClusterSplits, 1>{}([&](auto i) {
                if(i < num_splits)
                    lse_max = max(lse_max, lse_in[i]);
            });
            LSEDataType lse_sum = 0;
            if(lse_max != -numeric<LSEDataType>::infinity())
                static_for<0, kMaxClusterSplits, 1>{}([&](auto i) {
                    if(i < num_splits)
                        lse_sum += ck_tile::exp(lse_in[i] - lse_max);
                });
            const LSEDataType logsum =
                lse_sum == 0 ? -numeric<LSEDataType>::infinity() : ck_tile::log(lse_sum) + lse_max;
            vec_t acc{0.f, 0.f, 0.f, 0.f};
            if(logsum != -numeric<LSEDataType>::infinity())
                static_for<0, kMaxClusterSplits, 1>{}([&](auto i) {
                    if(i < num_splits)
                        acc += ck_tile::exp(lse_in[i] - logsum) * part[i];
                });
            const long_index_t dst = row_offset(g, kargs.nhead_stride_o, kargs.stride_o) + col;
            static_for<0, kVec, 1>{}([&](auto v) {
                if(col + v < kargs.hdim_v)
                    o[dst + v] = type_convert<CombineODataType>(acc[v.value]);
            });
            if(lse != nullptr && col == 0)
                lse[static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse +
                    i_nhead_q0 * kargs.nhead_stride_lse +
                    row_offset(g, kargs.nhead_stride_lse, 1)] = logsum;
        }
    }
#endif

    // With one split there is nothing to merge: O goes out in its final type, and the LSE window
    // already points at the final LSE.
    template <typename OaccTile>
    CK_TILE_DEVICE static void StoreFinalO(const Kargs& kargs,
                                           const OaccTile& o_acc_tile,
                                           index_t i_m0,
                                           index_t i_n1,
                                           index_t i_nhead,
                                           index_t i_batch)
    {
        const index_t ratio = kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1;
        auto* o_ptr         = static_cast<CombineODataType*>(kargs.o_ptr) +
                      static_cast<long_index_t>(i_nhead) * ratio * kargs.nhead_stride_o +
                      static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        const auto o_dram_naive = [&] {
            if constexpr(kMergeNumHeadGroupsSeqLenQ)
            {
                const auto view = make_naive_tensor_view<address_space_enum::global>(
                    o_ptr,
                    make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q, kargs.hdim_v),
                    make_tuple(kargs.nhead_stride_o, kargs.stride_o, 1),
                    number<1>{},
                    number<1>{});
                return transform_tensor_view(view,
                                             make_tuple(make_merge_transform(make_tuple(
                                                            kargs.nhead_ratio_qk, kargs.seqlen_q)),
                                                        make_pass_through_transform(kargs.hdim_v)),
                                             make_tuple(sequence<0, 1>{}, sequence<2>{}),
                                             make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    o_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_v),
                    make_tuple(kargs.stride_o, 1),
                    number<1>{},
                    number<1>{});
            }
        }();
        const auto o_dram =
            pad_tensor_view(o_dram_naive,
                            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                            sequence<kPadSeqLenQ, kPadHeadDimV>{});
        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});
        store_tile(o_dram_window, cast_tile<CombineODataType>(o_acc_tile));
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        // The qr_tdm arena slices itself at 256 B boundaries and its offset
        // asserts are relative, so they cannot catch a misaligned base. Today
        // this is the only __shared__ here and lands at offset 0 anyway.
        alignas(256) __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_tile_n, i_split, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_k       = 0; // unused for paged-kvcache
        long_index_t batch_offset_v       = 0; // unused for paged-kvcache
        long_index_t batch_offset_bias    = 0;
        long_index_t batch_offset_lse_acc = 0;
        long_index_t batch_offset_o_acc   = 0;
        index_t kv_l2p_offset =
            0; // logical-to-physical offset of seqlen_k coordinate. only used for paged-kvcache

        // Per-tensor q/k descales fold into scale_s before anything reads it (the sink
        // logit and LSE share the folded scale, as in FmhaFwdKernel::GetSoftmaxScale).
        [[maybe_unused]] float v_descale = 1.0f;
        if constexpr(kPerTensorQScale)
        {
            kargs.scale_s *= *reinterpret_cast<const float*>(kargs.q_descale_ptr) *
                             *reinterpret_cast<const float*>(kargs.k_descale_ptr);
            v_descale = *reinterpret_cast<const float*>(kargs.v_descale_ptr);
        }

        const float sink_value =
            kargs.sink_ptr != nullptr
                ? (*(static_cast<const float*>(kargs.sink_ptr) + i_nhead)) / kargs.scale_s
                : -numeric<float>::infinity();

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                batch_offset_v = key_start * kargs.stride_v;
            }
            else
            {
                batch_offset_v = key_start;
            }
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = query_start * kargs.stride_bias;
            }

            batch_offset_lse_acc = query_start;
            batch_offset_o_acc   = query_start * kargs.stride_o_acc;

            // get real # queries & # keys under group mode
            kargs.seqlen_q = kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q * (kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1) <= i_m0)
            {
                return;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                kargs.seqlen_k = kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch];
            }

            if constexpr(kIsPagedKV)
            {
                if(kargs.is_gappy)
                {
                    // seqstart_k_ptr has different meaning in this case
                    kv_l2p_offset = kargs.seqstart_k_ptr[i_batch];
                }
            }
        }
        else
        {
            const index_t i_cache_batch = [&, i_batch_ = i_batch] {
                if constexpr(kIsPagedKV)
                {
                    return i_batch_;
                }
                else
                {
                    return (kargs.cache_batch_idx != nullptr ? kargs.cache_batch_idx[i_batch_]
                                                             : i_batch_);
                }
            }();

            batch_offset_q       = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k       = static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_k;
            batch_offset_v       = static_cast<long_index_t>(i_cache_batch) * kargs.batch_stride_v;
            batch_offset_lse_acc = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse_acc;
            batch_offset_o_acc   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o_acc;

            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
        }
        // for simplicity, batch stride we just modify the pointer
        const index_t i_nhead_k =
            (kMergeNumHeadGroupsSeqLenQ ? i_nhead : i_nhead / kargs.nhead_ratio_qk);

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) *
                                     (kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1) *
                                     kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                                 static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k +
                                 batch_offset_k;
        const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                                 static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v +
                                 batch_offset_v;

        ODataType* o_acc_ptr = reinterpret_cast<ODataType*>(kargs.o_acc_ptr) +
                               static_cast<long_index_t>(i_nhead) *
                                   (kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1) *
                                   kargs.nhead_stride_o_acc +
                               batch_offset_o_acc + i_split * kargs.split_stride_o_acc;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&] {
            const auto q_dram_naive = [&] {
                if constexpr(kMergeNumHeadGroupsSeqLenQ)
                {
                    // reshape: (nhead_ratio_qk, seqlen_q, hdim_q) -> (nhead_ratio_qk * seqlen_q,
                    // hdim_q)
                    const auto view = make_naive_tensor_view<address_space_enum::global>(
                        q_ptr,
                        make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q, kargs.hdim_q),
                        make_tuple(kargs.nhead_stride_q, kargs.stride_q, 1),
                        number<FmhaPipeline::kAlignmentQ>{},
                        number<1>{});

                    return transform_tensor_view(
                        view,
                        make_tuple(
                            make_merge_transform(make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q)),
                            make_pass_through_transform(kargs.hdim_q)),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        q_ptr,
                        make_tuple(kargs.seqlen_q, kargs.hdim_q),
                        make_tuple(kargs.stride_q, 1),
                        number<FmhaPipeline::kAlignmentQ>{},
                        number<1>{});
                }
            }();

            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                    sequence<false, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<false, kPadHeadDimQ>{});
            }
        }();

        const auto make_k_dram = [&](const KDataType* data, index_t height) {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global,
                                                             memory_operation_enum::set,
                                                             kKVCoherence>(
                data, // will update this pointer if using paged-kvcache
                make_tuple(height, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            // TDM pipelines take the unpadded view; the TDM box clamp already
            // zero-fills the OOB head-dim tail, so a software pad adds nothing.
            if constexpr(kUsesTdmAffineDramPath)
            {
                return k_dram_naive;
            }
            else
            {
                return pad_tensor_view(
                    k_dram_naive,
                    make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<false, kPadHeadDimQ>{});
            }
        };
        const auto k_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_k_dram(nullptr, kargs.page_block_size);
            }
            else
            {
                return make_k_dram(k_ptr, kargs.seqlen_k);
            }
        }();

        const auto make_v_dram = [&](const VDataType* data, index_t length) {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global,
                                                                 memory_operation_enum::set,
                                                                 kKVCoherence>(
                    data, // will update this pointer if using paged-kvcache
                    make_tuple(length, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                // Pipelines with native row-major V consume (seqlen, hdim_v)
                // windows directly. They are on kUsesTdmAffineDramPath (asserted
                // next to kVPagedDim), so V stays unpadded like K in make_k_dram.
                if constexpr(kUsesUntransposedVKernelPath)
                {
                    return v_dram_naive;
                }
                else
                {
                    const auto v_dram_transposed =
                        transform_tensor_view(v_dram_naive,
                                              make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                         make_pass_through_transform(length)),
                                              make_tuple(sequence<1>{}, sequence<0>{}),
                                              make_tuple(sequence<0>{}, sequence<1>{}));

                    return pad_tensor_view(
                        v_dram_transposed,
                        make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                        sequence<kPadHeadDimV, kPadSeqLenK>{});
                }
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    data, // will update this pointer if using paged-kvcache
                    make_tuple(kargs.hdim_v, length),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<false, kPadSeqLenK>{});
            }
        };
        const auto v_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_v_dram(nullptr, kargs.page_block_size);
            }
            else
            {
                return make_v_dram(v_ptr, kargs.seqlen_k);
            }
        }();

        auto k_page_block_navigator = [&, i_batch_ = i_batch]() {
            if constexpr(kIsPagedKV)
            {
                const auto* block_indices =
                    reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                    i_batch_ * kargs.batch_stride_block_table;
                const index_t num_blocks =
                    integer_divide_ceil(kv_l2p_offset + kargs.seqlen_k, kargs.page_block_size);

                const long_index_t fixed_offset =
                    static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k;

                return make_page_block_navigator<const KDataType, 0>(
                    kargs.k_ptr,
                    kargs.batch_stride_k, // kcache page-block stride/size
                    fixed_offset,
                    block_indices,
                    num_blocks,
                    kargs.page_block_size,
                    k_dram,
                    make_k_dram(nullptr,
                                (kv_l2p_offset + kargs.seqlen_k) -
                                    (num_blocks - 1) * kargs.page_block_size));
            }
            else
            {
                return make_page_block_navigator(k_dram);
            }
        }();

        auto v_page_block_navigator = [&, i_batch_ = i_batch]() {
            if constexpr(kIsPagedKV)
            {
                const auto* block_indices =
                    reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                    i_batch_ * kargs.batch_stride_block_table;
                const index_t num_blocks =
                    integer_divide_ceil(kv_l2p_offset + kargs.seqlen_k, kargs.page_block_size);

                const long_index_t fixed_offset =
                    static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v;

                return make_page_block_navigator<const VDataType, kVPagedDim>(
                    kargs.v_ptr,
                    kargs.batch_stride_v, // vcache page-block stride/size
                    fixed_offset,
                    block_indices,
                    num_blocks,
                    kargs.page_block_size,
                    v_dram,
                    make_v_dram(nullptr,
                                (kv_l2p_offset + kargs.seqlen_k) -
                                    (num_blocks - 1) * kargs.page_block_size));
            }
            else
            {
                return make_page_block_navigator(v_dram);
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(number<FmhaPipeline::kM0>{},
                                      number<FmhaPipeline::kSubQKHeaddim>{});
                else
                    return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{});
        auto v_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{});

        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto bias_dram_window_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        number<FmhaPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(
                        bias_dram_naive, bias_dram_window_lengths, sequence<false, kPadSeqLenK>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // lse acc
        auto lse_acc_dram_window = [&,
                                    i_nhead_ = i_nhead,
                                    i_split_ = i_split,
                                    i_batch_ = i_batch]() {
            constexpr auto lse_acc_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
            LSEDataType* lse_acc_ptr = reinterpret_cast<LSEDataType*>(kargs.lse_acc_ptr) +
                                       static_cast<long_index_t>(i_nhead_) *
                                           (kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1) *
                                           kargs.nhead_stride_lse_acc +
                                       batch_offset_lse_acc + i_split_ * kargs.split_stride_lse_acc;
            index_t nhead_stride_lse_acc = kargs.nhead_stride_lse_acc;
            // a single split writes the final LSE itself (see StoreFinalO)
            if constexpr(kFuseCombine)
            {
                if(kargs.num_splits == 1 && kargs.lse_ptr != nullptr)
                {
                    lse_acc_ptr = static_cast<LSEDataType*>(kargs.lse_ptr) +
                                  static_cast<long_index_t>(i_nhead_) *
                                      (kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1) *
                                      kargs.nhead_stride_lse +
                                  static_cast<long_index_t>(i_batch_) * kargs.batch_stride_lse;
                    nhead_stride_lse_acc = kargs.nhead_stride_lse;
                }
            }
            else
            {
                ignore = i_batch_;
            }

            const auto lse_acc_dram = [&] {
                const auto lse_acc_dram_naive = [&] {
                    if constexpr(kMergeNumHeadGroupsSeqLenQ)
                    {
                        // reshape: (nhead_ratio_qk, seqlen_q) -> (nhead_ratio_qk * seqlen_q)
                        const auto view = make_naive_tensor_view<address_space_enum::global,
                                                                 memory_operation_enum::set,
                                                                 kAccCoherence>(
                            lse_acc_ptr,
                            make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q),
                            make_tuple(nhead_stride_lse_acc, 1),
                            number<1>{},
                            number<1>{});

                        return transform_tensor_view(view,
                                                     make_tuple(make_merge_transform(make_tuple(
                                                         kargs.nhead_ratio_qk, kargs.seqlen_q))),
                                                     make_tuple(sequence<0, 1>{}),
                                                     make_tuple(sequence<0>{}));
                    }
                    else
                    {
                        return make_naive_tensor_view<address_space_enum::global,
                                                      memory_operation_enum::set,
                                                      kAccCoherence>(lse_acc_ptr,
                                                                     make_tuple(kargs.seqlen_q),
                                                                     make_tuple(1),
                                                                     number<1>{},
                                                                     number<1>{});
                    }
                }();
                return pad_tensor_view(
                    lse_acc_dram_naive, lse_acc_dram_window_lengths, sequence<kPadSeqLenQ>{});
            }();

            return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {i_m0});
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.sink_size,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                // data loading, shared by entire wg
                // TODO: how to use s_read?
                SaccDataType slope =
                    *(reinterpret_cast<const SaccDataType*>(kargs.alibi_slope_ptr) +
                      i_batch_ * kargs.alibi_slope_stride + i_nhead_);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                slope *= ck_tile::log2e_v<>;
#endif
                if constexpr(kHasMask)
                {
                    return make_alibi_from_lr_mask<SaccDataType, true, 32>(slope,
                                                                           kargs.window_size_left,
                                                                           kargs.window_size_right,
                                                                           kargs.seqlen_q,
                                                                           kargs.seqlen_k,
                                                                           kargs.mask_type);
                }
                else
                {
                    return Alibi<SaccDataType, true, 32>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }
            else
            {
                return EmptyPositionEncoding<SaccDataType>{};
            }
        }();

        AttentionVariant variant;
        const auto variant_params = [&] {
            if constexpr(kHasLogitsSoftCap)
            {
                return ck_tile::LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2>{
                    mask, kargs.scale_s, kargs.logits_soft_cap, kargs.logits_soft_cap_rcp};
            }
            else
            {
                return ck_tile::StandardAttentionParams<FmhaMask>{mask, kargs.scale_s};
            }
        }();

        BlockIndices block_indices{i_batch, i_nhead, i_nhead_k};

        // merged rows are (Q head of the group, seqlen_q) pairs, each Q head with its own sink
        const auto sink_v = [&, i_nhead_ = i_nhead]() {
            if constexpr(kMergeNumHeadGroupsSeqLenQ && kHasSink)
                return [sink_ptr    = static_cast<const float*>(kargs.sink_ptr),
                        i_nhead_q   = i_nhead_ * kargs.nhead_ratio_qk,
                        nhead_ratio = kargs.nhead_ratio_qk,
                        seqlen_q    = kargs.seqlen_q,
                        scale_s     = kargs.scale_s](index_t i_row) {
                    // padding rows past the last head are never stored
                    return sink_ptr != nullptr
                               ? sink_ptr[i_nhead_q + min(i_row / seqlen_q, nhead_ratio - 1)] /
                                     scale_s
                               : -numeric<float>::infinity();
                };
            else
                return sink_value;
        }();

        auto o_acc_tile = [&, i_split_ = i_split]() {
            if constexpr(kDoFp8StaticQuant)
            {
                return FmhaPipeline{}(q_dram_window,
                                      identity{}, // q_element_func
                                      k_dram_window_lengths,
                                      k_page_block_navigator,
                                      identity{}, // k_element_func
                                      v_dram_window_lengths,
                                      v_page_block_navigator,
                                      identity{}, // v_element_func
                                      bias_dram_window,
                                      identity{}, // bias_element_func
                                      lse_acc_dram_window,
                                      identity{}, // lse_element_func
                                      identity{}, // s_acc_element_func
                                      scales<remove_cvref_t<decltype(kargs.scale_p)>>{
                                          kargs.scale_p}, // p_compute_element_func
                                      identity{},         // o_acc_element_func
                                      kargs.num_splits,
                                      i_split_,
                                      mask,
                                      position_encoding,
                                      kargs.scale_s,
                                      variant,
                                      variant_params,
                                      block_indices,
                                      kv_l2p_offset,
                                      smem_ptr,
                                      sink_v);
            }
            else if constexpr(kPerTensorQScale)
            {
                return FmhaPipeline{}(q_dram_window,
                                      k_dram_window_lengths,
                                      k_page_block_navigator,
                                      v_dram_window_lengths,
                                      v_page_block_navigator,
                                      bias_dram_window,
                                      lse_acc_dram_window,
                                      kargs.num_splits,
                                      i_split_,
                                      mask,
                                      position_encoding,
                                      kargs.scale_s,
                                      variant,
                                      variant_params,
                                      block_indices,
                                      kv_l2p_offset,
                                      smem_ptr,
                                      sink_v,
                                      v_descale);
            }
            else
            {
                return FmhaPipeline{}(q_dram_window,
                                      k_dram_window_lengths,
                                      k_page_block_navigator,
                                      v_dram_window_lengths,
                                      v_page_block_navigator,
                                      bias_dram_window,
                                      lse_acc_dram_window,
                                      kargs.num_splits,
                                      i_split_,
                                      mask,
                                      position_encoding,
                                      kargs.scale_s,
                                      variant,
                                      variant_params,
                                      block_indices,
                                      kv_l2p_offset,
                                      smem_ptr,
                                      sink_v);
            }
        }();

        if constexpr(kFuseCombine)
        {
            if(kargs.num_splits == 1)
            {
                StoreFinalO(kargs, o_acc_tile, i_m0, i_n1, i_nhead, i_batch);
                return;
            }
        }

        // Oacc DRAM and Oacc DRAM window
        auto o_acc_dram = [&] {
            const auto o_acc_dram_naive = [&] {
                if constexpr(kMergeNumHeadGroupsSeqLenQ)
                {
                    // reshape: (nhead_ratio_qk, seqlen_q, hdim_v) -> (nhead_ratio_qk * seqlen_q,
                    // hdim_v)
                    const auto view = make_naive_tensor_view<address_space_enum::global,
                                                             memory_operation_enum::set,
                                                             kAccCoherence>(
                        o_acc_ptr,
                        make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q, kargs.hdim_v),
                        make_tuple(kargs.nhead_stride_o_acc, kargs.stride_o_acc, 1),
                        number<FmhaPipeline::kAlignmentOacc>{},
                        number<1>{});

                    return transform_tensor_view(
                        view,
                        make_tuple(
                            make_merge_transform(make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q)),
                            make_pass_through_transform(kargs.hdim_v)),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::set,
                                                  kAccCoherence>(
                        o_acc_ptr,
                        make_tuple(kargs.seqlen_q, kargs.hdim_v),
                        make_tuple(kargs.stride_o_acc, 1),
                        number<FmhaPipeline::kAlignmentOacc>{},
                        number<1>{});
                }
            }();

            return pad_tensor_view(
                o_acc_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_acc_dram_window =
            make_tile_window(o_acc_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_acc_dram_window, o_acc_tile, nullptr);

        if constexpr(kFuseCombine)
            FuseCombine(kargs, i_m0, i_n1, i_split, i_nhead, i_batch, smem_ptr);
    }
};

} // namespace ck_tile
