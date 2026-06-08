// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_fwd_setting.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_pipeline_problem.hpp"
#include "hstu_attention_traits.hpp"
#include "hstu_attention_with_softmax_fwd_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_qr_async_pipeline.hpp"
#include "hstu_attention_fwd_kernel.hpp"
#include "hstu_attention_epilogue.hpp"
#include "hstu_attention_fwd_pipeline_policy_agpr.hpp"

#ifndef HSTU_COMPILE_NO_SPLITKV
#include "hstu_attention_jagged_forward_splitkv_dispatch.hpp"
#endif

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK,
          ck_tile::index_t MTile,
          bool kUseAsyncPipeline   = false,
          ck_tile::index_t WarpK   = 16,
          // kUseAgpr=true selects the AGPR-pinned pipeline policy
          // (HipKittens Principle 2 — `WGAttrCtlEnum::Raw_vaa`). Only
          // wired through the synchronous std (no-softmax) branch — the
          // softmax / qr_async / trload branches keep the default
          // policy and must reject `kUseAgpr=true` at compile time.
          bool kUseAgpr            = false,
          // kUsePingPong=true threads `s_setprio` ping-pong intrinsics
          // into the std no-softmax pipeline (Round 2 #2, HipKittens
          // Principle 1 — 4-wave interleave). Same constraints as
          // kUseAgpr: std-only, no softmax, no qr_async, no trload.
          bool kUsePingPong        = false,
          bool kUseSchedGroup      = false,
          // kUseTrLoad=true forces the JIT to the TrLoad
          // (xor-swizzled LDS + ds_read_tr* + buffer_load) pipeline
          // headers (`HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad` /
          // `HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad`),
          // overriding the `BUILD_HSTU_FOR_GFX95_ONLY` static gate.
          // NOTE: the underlying `ds_read_tr16_b64_v4bf16` /
          // `ds_read_tr16_b64_v4f16` intrinsics in
          // `amd_buffer_addressing.hpp::amd_transpose_load_to_vgpr`
          // are `#if defined(__gfx950__)` only — on gfx942 the
          // matching `buffer_view::transpose_get` branch returns
          // `X{numeric<T>::zero()}` (`buffer_view.hpp` LDS branch),
          // so the kernel will compile but read zeros for V. This
          // is enforced as a runtime numerical check, not a static
          // assert (Round 3 #1 falsification path).
          bool kUseTrLoad          = false,
          // Per-CU block budget (Round 3 #2 sweep axis). -1 keeps
          // the pipeline's compile-time heuristic for `kBlockPerCu`
          // (current default → existing builds byte-identical);
          // a positive value forces `__launch_bounds__(_, n)` on
          // the kernel via `HstuAttentionFwdTraits::kBlockPerCu`.
          // Lower values free up VGPR/LDS budget per block (useful
          // for register-pressure-bound shapes); higher values
          // raise theoretical occupancy (useful when LDS-light).
          ck_tile::index_t Occupancy = -1>
struct jagged_forward_causal_softmax_bias_dropout_dispatch
{
    static_assert(!(kUseAgpr && kUseSoftmax),
                  "AGPR pipeline policy is no-softmax only");
    static_assert(!(kUseAgpr && kUseAsyncPipeline),
                  "AGPR pipeline policy currently incompatible with qr_async");
    static_assert(!(kUsePingPong && kUseSoftmax),
                  "PingPong intrinsics live in the no-softmax pipeline only");
    static_assert(!(kUsePingPong && kUseAsyncPipeline),
                  "PingPong intrinsics not wired through qr_async pipeline");
    static_assert(!(kUsePingPong && kUseAgpr),
                  "kUsePingPong and kUseAgpr are independent variants — pick one");
    static_assert(!(kUseTrLoad && kUseAsyncPipeline),
                  "TrLoad and qr_async are mutually exclusive pipeline choices");
    static_assert(!(kUseTrLoad && kUseAgpr),
                  "TrLoad path is not wired through the AGPR policy");
    static_assert(!(kUseTrLoad && kUsePingPong),
                  "TrLoad path is not wired through the PingPong intrinsics");
    static_assert(!(kUseSchedGroup && kUseSoftmax),
                  "Sched-group hints live in the no-softmax pipeline only");
    static_assert(!(kUseSchedGroup && kUseAsyncPipeline),
                  "Sched-group hints not wired through qr_async pipeline");
    static_assert(!(kUseSchedGroup && kUsePingPong),
                  "Sched-group and PingPong are independent — pick one");
    static_assert(!(kUseSchedGroup && kUseAgpr),
                  "Sched-group is not wired through the AGPR policy");
    static_assert(!(kUseTrLoad && kUseSchedGroup),
                  "TrLoad path is not wired through sched-group hints");
    // For the no-softmax (HSTU deployment) path we use the parametric
    // ...TileSettingW form so the JIT can A/B WarpK=16 vs WarpK=32.  The
    // softmax path stays on the original 2-arg tile setting (WarpK=16
    // implicit) since the deployment kernels don't go through it.
    using HstuAttentionTileSetting =
        typename std::conditional_t<kUseSoftmax,
                                    HstuAttentionWithSoftmaxFwdTileSetting<MaxK, MTile>,
                                    HstuAttentionNoSoftmaxFwdTileSettingW<MaxK, MTile, WarpK>>::Type;

    // Round-3 #1 (TrLoad enable): allow the JIT to opt in to the
    // TrLoad pipeline via `kUseTrLoad`, overriding the original
    // gfx95-only gate. Existing `BUILD_HSTU_FOR_GFX95_ONLY` builds
    // (which force TrLoad unconditionally) stay byte-identical
    // because the `||` short-circuits with the template default
    // (`kUseTrLoad=false`) keeping the wire untouched there.
#ifdef BUILD_HSTU_FOR_GFX95_ONLY
    static constexpr bool use_trload_pipeline = true;
#else
    static constexpr bool use_trload_pipeline = kUseTrLoad;
#endif

    template <bool kIsCrossAttention>
    using HstuPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
        InOutDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
        typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
        kIsCrossAttention,
        false, // kUseGroup
        true,  // kIsJagged
        kHasBias,
        kHasDropout,
        kUseCausal,
        kUseSoftmax,
        HstuAttentionTileSetting>;

    static void Run(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
    {
        constexpr ck_tile::index_t occupancy = Occupancy;

        const bool pad_headdim_qk = !(param.hdim_qk % HstuAttentionTileSetting::kQKHeaddim == 0);
        const bool pad_headdim_v  = !(param.hdim_v % HstuAttentionTileSetting::kN1 == 0);

        // no need to check seqlen_q since it is not used as fastest dim,
        // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
        constexpr bool kPadSeqLenQ = false;

        constexpr bool kPadSeqLenK = true;

        BOOL_SWITCH_2(pad_headdim_qk, kPadHeadDimQK, pad_headdim_v, kPadHeadDimV, [&] {
            using HstuTraits = ck_tile::HstuAttentionFwdTraits<kPadSeqLenQ,
                                                               kPadSeqLenK,
                                                               kPadHeadDimQK,
                                                               kPadHeadDimV,
                                                               occupancy>;

            using HstuEpilogue = ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename HstuAttentionFwdTypeConfig<InOutDataType>::OaccDataType,
                typename HstuAttentionFwdTypeConfig<InOutDataType>::ODataType,
                kPadSeqLenQ,
                kPadHeadDimV>>;

            BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
                using HstuPipelineProblem = HstuPipelineProblemTemp<kIsCrossAttention>;

                if constexpr(!use_trload_pipeline)
                {
                    // qr_async pipeline is no-softmax only (mirrors FMHA's
                    // qr_async constraint). For kUseSoftmax=true we always
                    // use the synchronous with-softmax pipeline.
                    using HstuPipeline = std::conditional_t<
                        kUseSoftmax,
                        ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVS<HstuPipelineProblem,
                                                                           HstuTraits>,
                        std::conditional_t<
                            kUseAsyncPipeline,
                            ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSAsync<
                                HstuPipelineProblem,
                                HstuTraits>,
                            std::conditional_t<
                                kUseAgpr,
                                ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<
                                    HstuPipelineProblem,
                                    HstuTraits,
                                    ck_tile::HstuAttentionFwdPipelineQRKSVSPolicyAGPR,
                                    /*kUsePingPong=*/false,
                                    /*kUseSchedGroup=*/false>,
                                ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<
                                    HstuPipelineProblem,
                                    HstuTraits,
                                    ck_tile::HstuAttentionFwdPipelineQRKSVSPolicy,
                                    kUsePingPong,
                                    kUseSchedGroup>>>>;

                    using HstuKernel = ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                    RunWithKernel<HstuKernel>(param, stream);
                }
                else
                {
                    using HstuPipeline = std::conditional_t<
                        kUseSoftmax,
                        ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad<
                            HstuPipelineProblem,
                            HstuTraits>,
                        ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad<HstuPipelineProblem,
                                                                               HstuTraits>>;

                    using HstuKernel = ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                    RunWithKernel<HstuKernel>(param, stream);
                };
            });
        });
    };

    template <typename HstuKernel>
    static void RunWithKernel(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
    {
        const auto kargs = [&] {
            return HstuKernel::MakeKargs(param.q_ptr,
                                         param.k_ptr,
                                         param.v_ptr,
                                         param.bias_ptr,
                                         param.o_ptr,
                                         param.seq_q_offsets_ptr,
                                         param.is_cross_attention ? param.seq_kv_offsets_ptr
                                                                  : param.seq_q_offsets_ptr,
                                         param.max_seqlen_q,
                                         param.hdim_qk,
                                         param.hdim_v,
                                         param.num_head,
                                         param.scale_s,
                                         param.attn_scale,
                                         param.seq_stride_q,
                                         param.seq_stride_k,
                                         param.seq_stride_v,
                                         param.seq_stride_bias,
                                         param.seq_stride_o,
                                         param.nhead_stride_q,
                                         param.nhead_stride_k,
                                         param.nhead_stride_v,
                                         param.nhead_stride_bias,
                                         param.nhead_stride_o,
                                         param.num_targets_ptr,
                                         param.contextual_seqlen,
                                         param.window_size,
                                         param.min_full_attn_seqlen,
                                         param.p_drop,
                                         param.philox_seed,
                                         param.philox_offset);
        }();

        bool has_minfull_attn_seqlen           = (param.min_full_attn_seqlen > 0);
        dim3 kGridSize                         = HstuKernel::GridSize(param.num_batch,
                                              param.num_head,
                                              param.max_seqlen_q,
                                              param.hdim_v,
                                              has_minfull_attn_seqlen);
        dim3 kBlockSize                        = HstuKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(
            ck_tile::stream_config{stream, false},
            ck_tile::make_kernel<kBlockPerCu>(HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
    };
};

template <typename InOutDataType,
          bool kUseCausal,
          bool kUseSoftmax,
          bool kHasBias,
          bool kHasDropout,
          ck_tile::index_t MaxK>
void run_jagged_forward_causal_softmax_bias_dropout_dispatch(HstuAttentionNoGroupFwdParams& param,
                                                             hipStream_t stream)
{
    auto effective_mtile = [&]() {
        const char* env_p = std::getenv("HSTU_FORCE_MTILE");
        if(env_p != nullptr)
            return std::atoi(env_p);
        return get_hstu_attention_fwd_mtile(param.num_batch, param.num_head, param.max_seqlen_q);
    }();
    if(effective_mtile == 128)
        jagged_forward_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                            kUseCausal,
                                                            kUseSoftmax,
                                                            kHasBias,
                                                            kHasDropout,
                                                            MaxK,
                                                            128>::Run(param, stream);
    else
    {
#if defined(HSTU_COMPILE_NO_SPLITKV)
        jagged_forward_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                            kUseCausal,
                                                            kUseSoftmax,
                                                            kHasBias,
                                                            kHasDropout,
                                                            MaxK,
                                                            64>::Run(param, stream);
#else
        const bool disable_fwd_splitkv = []() {
            const char* env_p = std::getenv("HSTU_DISABLE_SPLITKV");
            if(env_p == nullptr)
                return false;
            return static_cast<bool>(atoi(env_p));
        }();

        const bool force_splitkv = []() {
            const char* env_p = std::getenv("HSTU_FORCE_SPLITKV");
            return env_p != nullptr && std::atoi(env_p) != 0;
        }();
        if(!disable_fwd_splitkv &&
           (force_splitkv ||
            shall_use_splitkv(param.num_batch, param.num_head, param.max_seqlen_q)))
        {
            jagged_forward_splitkv_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                                        kUseCausal,
                                                                        kUseSoftmax,
                                                                        kHasBias,
                                                                        kHasDropout,
                                                                        MaxK,
                                                                        64>::Run(param, stream);
        }
        else
            jagged_forward_causal_softmax_bias_dropout_dispatch<InOutDataType,
                                                                kUseCausal,
                                                                kUseSoftmax,
                                                                kHasBias,
                                                                kHasDropout,
                                                                MaxK,
                                                                64>::Run(param, stream);
#endif
    };
};
