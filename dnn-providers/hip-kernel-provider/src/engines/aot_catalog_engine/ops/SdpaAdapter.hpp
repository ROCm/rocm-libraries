// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// SDPA (scaled-dot-product / flash-attention FORWARD) op adapter -- universal
// across archs: one adapter serves gfx1151 today and gfx942/gfx950 (and beyond)
// as pure data. It decodes a single-node SdpaAttributes graph into a ProblemShape
// and lets per-kernel family.json constraints decide applicability, rather than
// hard-coding any one kernel's feature set in C++.
//
// decode() publishes the full CAPABILITY VOCABULARY as ProblemShape keys: the
// numeric shape (dtype/B/H/H_kv/S_q/S_kv/D/gqa_ratio) plus a fact per feature
// (causal, causal_bottom_right, has_alibi, has_padding_mask, has_attn_mask,
// has_block_mask, has_sink, has_dropout, paged, varlen, gen_stats, fp8,
// runtime_scale) and per structural property (d_contiguous, batch_foldable). A
// kernel opts in/out of each via a constraint; a graph no kernel accepts yields
// no candidate, so the engine declines and another serves it (aggregate fail-
// closed). Only universal, memory-safety invariants stay as C++ declines
// (single SdpaAttributes node; rank-4 BHSD Q/K/V/O; K/V agree on H_kv/S_kv/D; O
// mirrors Q; consistent supported dtype; integer gqa_ratio; rank-4 strides).
//
// buildBindings() emits a SUPERSET of named arguments. Always: Q,K,V,O; scale_log2
// and scale_raw; seqlen_q/k; per-tensor token/head/batch strides in element AND
// byte units; H/H_kv/D/B/gqa_ratio. Bound only when the graph carries them (each a
// forward feature decode() also flags as a fact): attn_mask, block_mask, sink,
// scale_tensor (runtime scale), seqlen_q_ptr/seqlen_kv_ptr (varlen), page_table_k/v
// (paged), dropout_mask/dropout_scale/dropout_seed/dropout_offset/rng_dump,
// descale_q/k/v/s + scale_s/scale_o + amax_s/amax_o (fp8), and stats/lse/max/sum_exp
// (softmax stats outputs). Each family's args_signature selects and orders the
// subset its kernel takes; launch::bindArgs resolves by name and fails closed on an
// unemitted name. A new quantity a future kernel needs is one added emission here --
// the single, explicit extension point.
//
// SCALE GOTCHA: a base-2 (exp2) softmax kernel takes `scale_log2 = attn_scale *
// log2(e)`, NOT the raw scale; the adapter emits both so a family names whichever.
//
// See the engine README for the arg-vocabulary and capability-key reference
// tables and the "authoring a forward SDPA family on a new arch" checklist.

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
