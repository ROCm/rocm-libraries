// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// SDPA (scaled-dot-product / flash-attention forward) op adapter -- the third
// proof op, and the deferred prize: attention is ~76% of LTX-Video device time.
// It decodes a single-node SdpaAttributes graph into a ProblemShape keyed by
// dtype/D/H/H_kv/S_q/S_kv/causal and resolves the 15-arg launch ABI the gfx1151
// WMMA flash-attention .co expects (Q,K,V,O ptrs; scale_log2 f32; seqlen_q/k
// i32; then 8 within-batch i32 strides {q,k,v,o} x {token,head}).
//
// The shipped kernel is rocKE's `build_wmma_fmha_fwd` (a thin adapter over the
// unified `mfma_attention_fwd_inner_body`), built mask_mode="none", D=64, H=32,
// MHA (H_kv==H), for f16 and NATIVE bf16. Because head_size/head_count are
// compile-time but seqlen is a runtime arg, one D64/H32 kernel per dtype serves
// both LTX self-attn (Sq=Sk=4096) AND cross-attn (Sq=4096, Sk=128).
//
// SCALE GOTCHA: the kernel takes `scale_log2 = attn_scale * log2(e)` (softmax is
// computed base-2 via exp2), NOT the raw scale -- this adapter does that multiply.
//
// The adapter fails closed (declines) on everything LTX does not need, mirroring
// the ASM engine's SdpaFwdPlanBuilder::isApplicable: masks (causal/alibi/padding/
// attn_mask), dropout, paged-KV, stats, GQA (H_kv != H), a runtime scale tensor,
// varlen/group mode (seq_len_* tensors), a non-D64 head dim, sequence lengths not
// a multiple of 16, and any batch layout the single token/head stride pair cannot
// fold (B>1 unless batch_stride == seqlen*stride_token). Declining lets another
// engine serve the graph rather than risk a wrong result.

#pragma once

#include "ops/IOpAdapter.hpp"

namespace aot_catalog_engine::ops
{

class SdpaAdapter : public IOpAdapter
{
public:
    const char* opKind() const override
    {
        return "sdpa";
    }

    std::optional<catalog::ProblemShape> decode(const IGraph& graph) const override;

    catalog::LaunchBindings buildBindings(const IGraph& graph,
                                          const catalog::ProblemShape& problem,
                                          const catalog::KernelEntry& kernel) const override;

    launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                    const catalog::KernelEntry& kernel) const override;
};

} // namespace aot_catalog_engine::ops
