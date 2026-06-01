// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <unordered_map>

#include "SdpaBwdSpec.hpp"

namespace ck_dsl_provider {

/// Walks a single hipDNN SDPA-backward node plus the surrounding tensor
/// map and produces a fully-populated ``SdpaBwdSpec``.
///
/// Q/K/V/dO and the gradients follow the rank-4 [B, H, S, D] convention:
///   * Q:  [B, Hq,  Sq,  D] (HALF)
///   * K:  [B, Hkv, Skv, D] (HALF)
///   * V:  [B, Hkv, Skv, D] (HALF)
///   * O:  [B, Hq,  Sq,  D] (HALF; resolved for validation, unused by
///         the kernel)
///   * dO: [B, Hq,  Sq,  D] (HALF)
///   * dQ: [B, Hq,  Sq,  D] (FLOAT; f32 accumulator)
///   * dK: [B, Hkv, Skv, D] (FLOAT; f32 accumulator)
///   * dV: [B, Hkv, Skv, D] (FLOAT; f32 accumulator)
///   * stats: [B, Hq, Sq] (FLOAT; natural-log LSE, head-major contiguous)
///
/// The shape fields (B, Hq, Hkv, Sq, Skv, D) and dtype/mask_mode are the
/// codegen inputs folded into the cache signatures; the eleven stride_*
/// scalars and the two scale_* values are launch-time kernel arguments
/// carried on the spec for the plan builder.
///
/// Validation (throws HipdnnPluginException on any failure -- callers
/// can catch + return ``false`` from ``isApplicable``):
///   * Q/K/V/O/dO data_type == HALF (FP16); dQ/dK/dV/stats data_type
///     == FLOAT (f32 accumulators + natural-log LSE)
///   * rank-4 for Q/K/V/O/dO/dQ/dK/dV; stats rank-3 [B, Hq, Sq] or
///     rank-4 [B, Hq, Sq, 1] (either accepted; B/Hq/Sq validated)
///   * batch matches across all tensors
///   * head_size matches: K.D == V.D == dO.D == dQ.D == dK.D == dV.D
///     == Q.D
///   * Skv shared by K/V/dK/dV; Sq shared by Q/O/dO/dQ
///   * Hq for Q/O/dO/dQ; Hkv for K/V/dK/dV
///   * GQA: Hq % Hkv == 0
///   * head_size in {64, 128, 192, 256} and % 64 == 0 (the bwd kernel
///     needs head_size >= WARP_SIZE; 32 is rejected even though the
///     forward path accepts it)
///   * Sq % 16 == 0 and Skv % 16 == 0
///   * layout (BSHD-compatible) for Q/K/V/dO and dQ/dK/dV: head-dim
///     unit-stride; for batch>1 batch stride == seqlen * sequence-stride
///   * gradient head stride must equal the matching input head stride
///     (dQ↔Q, dK↔K, dV↔V): the kernel reuses the input head stride for
///     the gradient writes
///   * stats contiguous head-major: strides == {Hq*Sq, Sq, 1} so the
///     LSE-prep kernel can read [B, Hq, Sq] as a flat contiguous buffer
///   * mask: alibi, padding, bottom-right causal, and sliding-window
///     (left_bound/right_bound) are all rejected; mask_mode is
///     "causal" when causal_mask is set, otherwise "none"
///   * dropout (probability AND the dropout tensors), additive
///     attn_mask, per-element scale, variable-length sequences
///     (seq_len_q / seq_len_kv), seed/offset, and dbias are all
///     rejected when present
///
/// All extracted scalars are narrowed from int64_t to int32_t via
/// ``narrowToI32``, which first checks the value fits (the DSL's
/// signature is i32 for shape + stride scalars).
class SdpaBwdAdapter {
   public:
    using SdpaBackwardAttributes = hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes;
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorMap = std::unordered_map<std::int64_t, const TensorAttributes*>;

    /// Build the spec from a single SDPA-backward node's attributes + the
    /// tensor map that resolves its Q/K/V/dO/stats/dQ/dK/dV UIDs.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` on any
    /// validation failure listed in the class docstring.
    static SdpaBwdSpec buildSpec(const SdpaBackwardAttributes& sdpaAttr,
                                 const TensorMap& tensorMap);

   private:
    SdpaBwdAdapter() = delete;
};

}  // namespace ck_dsl_provider
