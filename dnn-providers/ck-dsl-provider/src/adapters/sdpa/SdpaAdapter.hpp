// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <unordered_map>

#include "SdpaSpec.hpp"

namespace ck_dsl_provider {

/// Walks a single hipDNN SDPA node plus the surrounding tensor map and
/// produces a fully-populated ``SdpaSpec``.
///
/// Q/K/V/O tensors follow the rank-4 [B, H, S, D] convention:
///   * Q: [B, Hq,  Sq,  D ]
///   * K: [B, Hkv, Skv, D ]
///   * V: [B, Hkv, Skv, Dv]  (Dv == D enforced)
///   * O: [B, Hq,  Sq,  Dv]  (== [B, Hq, Sq, D])
///
/// The shape fields (B, Hq, Hkv, Sq, Skv, D) and dtype/mask_mode are the
/// codegen inputs folded into the cache signature; the eight stride_*
/// scalars and scale_log2 are launch-time kernel arguments carried on
/// the spec for the plan builder.
///
/// This is the HONEST capability gate for the unified paged/varlen SDPA
/// kernel. Every validation failure throws ``HipdnnPluginException``;
/// ``SdpaFwdPlanBuilder::isApplicable`` wraps the call in ``tryBuildSpec``
/// and converts the throw into ``isApplicable=false`` + the reason string
/// (the clean-decline path). There is NO hybrid dense routing -- a variant
/// the single kernel cannot do is declined, never silently downgraded.
///
/// ACCEPTED (the broad-but-safe matrix), extracted into the spec:
///   * dtype FP16 AND BF16 (Q drives codegen; K/V/O must match Q);
///     spec.dtype is "f16" or "bf16"
///   * head_size D in {64, 128, 256} (a deliberate POC narrowing from the
///     prior {32, 64, 128, 192, 256}: the paged kernel only does these)
///   * GQA: Hq % Hkv == 0 (head counts from tensor dims)
///   * top-left causal masking (the kernel applies causal
///     unconditionally); mask_mode is always "causal"
///   * sliding window: top-left + left_bound > 0 -> spec.sliding_window
///   * sinks: a present Sink_token tensor -> spec.use_sinks
///   * varlen: present seq_len_q AND seq_len_kv -> spec.is_varlen
///   * real paged KV: present page_table_k AND page_table_v with identical
///     block-table layouts -> spec.is_paged + spec.block_size derived
///     (in {16, 32, 64}) from the page-table dims + max_seq_len_kv
///   * scalar attn_scale (attn_scale_value) -> scale_log2
///
/// Structural validation (also throws on failure): Q/K/V/O dims non-null
/// and 4-D; batch matches across Q/K/V/O; head_size matches (K.D == Q.D,
/// V.Dv == D, O last dim == D); K.Skv == V.Skv; O.Hq == Q.Hq;
/// O.Sq == Q.Sq; V.Hkv == K.Hkv; all dims positive; Sq % 16 == 0 and
/// Skv % 16 == 0.
///
/// DECLINED cleanly (isApplicable=false + reason; NO hybrid dense routing):
///   * non-causal / bidirectional (the prior "no mask" case) -- the kernel
///     applies causal unconditionally
///   * bottom-right causal / diagonal_alignment == BOTTOM_RIGHT
///   * right_bound != 0 (only a left causal window is modelled)
///   * non-BSHD layout (head-dim not unit-stride, or batch stride !=
///     seqlen * sequence-stride for batch>1)
///   * mismatched page_table_k/page_table_v (single-table kernel)
///   * LSE/stats output (generate_stats) -- a REGRESSION vs the dense path,
///     which can emit LSE; the paged kernel cannot (Vidya follow-up)
///   * additive attn_mask (bias), per-element scale tensor, dropout
///     (tensors and dropout_probability), block mask, padding mask, ALiBi
///   * FP8 descale/scale tensors (deferred), amax_s/amax_o, rng_dump,
///     max/sum_exp outputs
///   * head_size 32 and 192 (outside the {64, 128, 256} set)
///
/// All extracted scalars are narrowed from int64_t to int32_t via
/// ``narrowToI32``, which first checks the value fits (the DSL's
/// signature is i32 for shape + stride scalars).
class SdpaAdapter {
   public:
    using SdpaAttributes = hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes;
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorMap = std::unordered_map<std::int64_t, const TensorAttributes*>;

    /// Build the spec from a single SDPA node's attributes + the tensor
    /// map that resolves its Q/K/V/O UIDs.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` on any
    /// validation failure listed in the class docstring.
    static SdpaSpec buildSpec(const SdpaAttributes& sdpaAttr, const TensorMap& tensorMap);

   private:
    SdpaAdapter() = delete;
};

}  // namespace ck_dsl_provider
