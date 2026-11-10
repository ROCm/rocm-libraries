// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_fwd_cr_args.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_cr.hpp"

namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdKernel;

// CR owns its grid mapping and epilogue. Keep the compact BSHD argument ABI
// alongside tensor views for GQA, independent strides and variable lengths.
template <typename Pipeline_,
          typename Epilogue_,
          typename Tuning_ = FmhaTdmCrTuning<>,
          bool Strided_    = !Pipeline_::kCompact>
struct FmhaFwdTdmCrKernel
{
    using FmhaPipeline                   = Pipeline_;
    using Tuning                         = Tuning_;
    using Kargs                          = std::conditional_t<Strided_,
                                                              detail::fmha_cr::StridedArguments,
                                                              typename FmhaPipeline::Arguments>;
    static constexpr index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    template <typename OtherTuning>
    using WithTuning = FmhaFwdTdmCrKernel<Pipeline_, Epilogue_, OtherTuning, Strided_>;

    using StridedKernel = FmhaFwdTdmCrKernel<Pipeline_, Epilogue_, Tuning_, true>;

    // Validate the complete input before compact conversion discards its layout.
    // Input exposes the named fields of CK's forward tensor/sequence view; it
    // need not be the example API's fmha_fwd_args type.
    template <typename Input>
    CK_TILE_HOST static bool IsSupportedInput(const Input& a)
    {
        using Problem = typename FmhaPipeline::Problem;
        return detail::fmha_fwd_cr_supported_args(a) &&
               Problem::kIsGroupMode == (a.seqstart_q_ptr != nullptr) &&
               FmhaPipeline::kCausal == (a.mask_type != 0) &&
               (!FmhaPipeline::kStoreLSE || a.lse_ptr) && a.hdim_q == FmhaPipeline::kHeadDim &&
               a.hdim_v == FmhaPipeline::kValueDim &&
               (Problem::kIsGroupMode || Problem::kPadSeqLenQ ||
                a.seqlen_q % FmhaPipeline::kM0 == 0) &&
               (Problem::kIsGroupMode || Problem::kPadSeqLenK ||
                a.seqlen_k % FmhaPipeline::kN0 == 0);
    }

    // As with FmhaFwdKernel::MakeKargs, callers validate support before launch.
    // Compact construction additionally requires fmha_fwd_cr_compact_args(a).
    template <typename Input>
    CK_TILE_HOST static Kargs MakeKargs(const Input& a)
    {
        if constexpr(Strided_)
        {
            Kargs x{};
            x.q            = a.q_ptr;
            x.k            = a.k_ptr;
            x.v            = a.v_ptr;
            x.o            = a.o_ptr;
            x.lse          = static_cast<float*>(a.lse_ptr);
            x.batch        = a.batch;
            x.heads        = a.nhead_q;
            x.kv_heads     = a.nhead_k;
            x.sq           = FmhaPipeline::Problem::kIsGroupMode ? a.max_seqlen_q : a.seqlen_q;
            x.sk           = a.seqlen_k;
            x.scale        = a.scale_s;
            x.stride_q     = a.stride_q;
            x.stride_k     = a.stride_k;
            x.stride_v     = a.stride_v;
            x.stride_o     = a.stride_o;
            x.head_q       = a.nhead_stride_q;
            x.head_k       = a.nhead_stride_k;
            x.head_v       = a.nhead_stride_v;
            x.head_o       = a.nhead_stride_o;
            x.head_lse     = a.nhead_stride_lse;
            x.batch_q      = a.batch_stride_q;
            x.batch_k      = a.batch_stride_k;
            x.batch_v      = a.batch_stride_v;
            x.batch_o      = a.batch_stride_o;
            x.batch_lse    = a.batch_stride_lse;
            x.start_q      = static_cast<const int32_t*>(a.seqstart_q_ptr);
            x.start_k      = static_cast<const int32_t*>(a.seqstart_k_ptr);
            x.length_q     = static_cast<const int32_t*>(a.seqlen_q_ptr);
            x.length_k     = static_cast<const int32_t*>(a.seqlen_k_ptr);
            x.cumulative_q = static_cast<const int32_t*>(a.cu_seqlen_q_ptr);
            x.cumulative_k = static_cast<const int32_t*>(a.cu_seqlen_k_ptr);
            x.group        = FmhaPipeline::Problem::kIsGroupMode;
            x.mask_type    = a.mask_type;
            x.left         = a.window_size_left;
            x.right        = a.window_size_right;
            return x;
        }
        else
            return {a.q_ptr,
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

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& a)
    {
        detail::FmhaFwdCrInput view{};
        view.hdim_q   = FmhaPipeline::kHeadDim;
        view.hdim_v   = FmhaPipeline::kValueDim;
        view.seqlen_q = view.max_seqlen_q = a.sq;
        if constexpr(Strided_)
        {
            view.q_ptr             = a.q;
            view.k_ptr             = a.k;
            view.v_ptr             = a.v;
            view.o_ptr             = a.o;
            view.lse_ptr           = a.lse;
            view.batch             = a.batch;
            view.nhead_q           = a.heads;
            view.nhead_k           = a.kv_heads;
            view.seqlen_k          = a.sk;
            view.scale_s           = a.scale;
            view.stride_q          = a.stride_q;
            view.stride_k          = a.stride_k;
            view.stride_v          = a.stride_v;
            view.stride_o          = a.stride_o;
            view.nhead_stride_q    = a.head_q;
            view.nhead_stride_k    = a.head_k;
            view.nhead_stride_v    = a.head_v;
            view.nhead_stride_o    = a.head_o;
            view.nhead_stride_lse  = a.head_lse;
            view.batch_stride_q    = a.batch_q;
            view.batch_stride_k    = a.batch_k;
            view.batch_stride_v    = a.batch_v;
            view.batch_stride_o    = a.batch_o;
            view.batch_stride_lse  = a.batch_lse;
            view.seqstart_q_ptr    = a.start_q;
            view.seqstart_k_ptr    = a.start_k;
            view.seqlen_q_ptr      = a.length_q;
            view.seqlen_k_ptr      = a.length_k;
            view.cu_seqlen_q_ptr   = a.cumulative_q;
            view.cu_seqlen_k_ptr   = a.cumulative_k;
            view.mask_type         = a.mask_type;
            view.window_size_left  = a.left;
            view.window_size_right = a.right;
            if(a.group != FmhaPipeline::Problem::kIsGroupMode)
                return false;
        }
        else
        {
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
            view.mask_type                            = FmhaPipeline::kCausal ? 1 : 0;
            view.window_size_left                     = -1;
            view.window_size_right                    = FmhaPipeline::kCausal ? 0 : -1;
            if(!FmhaPipeline::kCompact || !detail::fmha_fwd_cr_compact_args(view))
                return false;
            if constexpr(Tuning::kSequence != 0)
                if(a.batch != 32 || a.heads != 8 || a.sq != Tuning::kSequence ||
                   a.sk != Tuning::kSequence)
                    return false;
            if constexpr(Tuning::kQueryTiles != 0)
            {
                if(a.heads != 8 || (int64_t{a.sq} + FmhaPipeline::kM0 - 1) / FmhaPipeline::kM0 !=
                                       Tuning::kQueryTiles)
                    return false;
                if constexpr(Tuning::kKeyTiles != 0)
                    if(a.sq != Tuning::kQueryTiles * FmhaPipeline::kM0 ||
                       a.sk != Tuning::kKeyTiles * FmhaPipeline::kN0)
                        return false;
            }
            if constexpr(Tuning::kTailFragments)
                if(a.sq % 16 != 15 || a.sk <= 512 || a.sk >= 640 || a.sk % 128 <= 112)
                    return false;
        }
        return IsSupportedInput(view);
    }

    CK_TILE_HOST static dim3 BlockSize() { return dim3(128); }
    CK_TILE_HOST static dim3 GridSize(const Kargs& a)
    {
        return dim3((a.sq + FmhaPipeline::kM0 - 1) / FmhaPipeline::kM0, a.batch * a.heads);
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return FmhaPipeline::GetSmemSize();
    }
    CK_TILE_DEVICE void operator()(Kargs a) const { FmhaPipeline::template Run<Tuning>(a); }
};

// CR is a self-contained execution specialization of the common forward entry.
// Keep its native launch types (and therefore device ABI) for tuning variants.
template <typename Problem, typename Epilogue>
struct FmhaFwdKernel<BlockFmhaPipelineQRKSVSTdmCr<Problem>, Epilogue>
    : FmhaFwdTdmCrKernel<BlockFmhaPipelineQRKSVSTdmCr<Problem>, Epilogue>
{
};

} // namespace ck_tile
