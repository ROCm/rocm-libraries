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
/// Validation (throws HipdnnPluginException on any failure -- callers
/// can catch + return ``false`` from ``isApplicable``):
///   * Q/K/V/O dims non-null and 4-D
///   * Q/K/V/O data_type == HALF (FP16 only)
///   * batch matches across Q/K/V/O
///   * head_size matches: K.D == Q.D, V.Dv == D, O last dim == D
///     (single head_size kernel)
///   * K.Skv == V.Skv; O.Hq == Q.Hq; O.Sq == Q.Sq; V.Hkv == K.Hkv
///   * GQA: Hq % Hkv == 0
///   * head_size D in {32, 64, 128, 192, 256}
///   * all dims positive (B, Hq, Hkv, Sq, Skv, D > 0); checked before
///     the seqlen multiple-of-16 rule so a zero seqlen cannot slip
///     through ``0 % 16 == 0``
///   * Sq % 16 == 0 and Skv % 16 == 0
///   * layout (BSHD-compatible, since the kernel has no batch stride):
///     the head-dim (last axis) is unit-stride for every tensor; and
///     for batch>1 the batch stride equals seqlen * sequence-stride
///     (Q/O use seqlen_q, K/V use seqlen_k). A contiguous BHSD tensor
///     with B>1 and H>1 is rejected.
///   * mask: alibi, padding, bottom-right causal, and sliding-window
///     (left_bound/right_bound) are all rejected; mask_mode is
///     "causal" when causal_mask is set, otherwise "none"
///   * every advanced feature is rejected when present: additive
///     attn_mask, per-element scale, stats/LSE outputs, generate_stats,
///     variable-length sequences (seq_len_q / seq_len_kv), dropout
///     (mask / scale / seed / offset tensors AND dropout_probability),
///     paged KV (page_table_k/v), block mask, sink tokens, the max /
///     sum_exp outputs, FP8 descale/scale tensors (descale_q/k/v/s,
///     scale_s/o), the amax_s / amax_o outputs, and rng_dump
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
