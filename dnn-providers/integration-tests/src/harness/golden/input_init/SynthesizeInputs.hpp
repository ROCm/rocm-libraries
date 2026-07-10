// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/golden/input_init/SynthesisTracker.hpp"

namespace hipdnn_integration_tests::golden
{

// ── Per-op fill functions ─────────────────────────────────────────────────────
// Each function declares inputs for one node in the graph. A single
// SynthesisTracker is shared across all nodes in the graph — the caller
// (synthesizeInputs in the harness .cpp) creates it with the whole-graph leaf
// input UIDs, passes it through each fill function, then calls finish() once
// after all nodes have been processed. This graph-level tracking is essential
// for fused/multi-node graphs: each node only accounts for its own UIDs, and
// the final finish() verifies that every leaf input was covered by some node.
//
// Every function follows the same pattern:
//   1. Cast the node to its concrete attribute type.
//   2. Declare each input as FREE (fill with random values), STRUCTURED (can't
//      synthesize — needs specific format), or DERIVED (must come from another
//      op's output). See SynthesisTracker.hpp for role definitions.
//   3. Return ok() if the attribute cast succeeded, or unsupported() if not.
//
// Fills must be deterministic given `rng` so re-running the same graph produces
// identical inputs for reproducible comparisons.
//
// To add a new op: copy fillConvFwdInputs (simplest example), adapt for your
// op's attributes, and add one case to the switch in synthesizeNodeInputs().
// Function names follow the pattern fill<AttributeName>Inputs.

// ── Convolution ───────────────────────────────────────────────────────────────

inline SynthesisResult fillConvFwdInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         SynthesisTracker& tracker,
                                         std::mt19937& rng)
{
    const auto* a = node.attributes_as_ConvolutionFwdAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not ConvolutionFwdAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->w_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

inline SynthesisResult fillConvBwdDataInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                             SynthesisTracker& tracker,
                                             std::mt19937& rng)
{
    const auto* a = node.attributes_as_ConvolutionBwdAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not ConvolutionBwdAttributes");
    }
    tracker.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->w_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

inline SynthesisResult
    fillConvBwdWeightsInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                             SynthesisTracker& tracker,
                             std::mt19937& rng)
{
    const auto* a = node.attributes_as_ConvolutionWrwAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not ConvolutionWrwAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── Batchnorm ─────────────────────────────────────────────────────────────────

inline SynthesisResult
    fillBatchnormInferenceInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 SynthesisTracker& tracker,
                                 std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormInferenceAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->mean_tensor_uid(), -0.1f, 0.1f, rng);
    tracker.fillFree(a->inv_variance_tensor_uid(), 0.5f, 1.5f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

inline SynthesisResult
    fillBatchnormInferenceVarianceInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         SynthesisTracker& tracker,
                                         std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormInferenceAttributesVarianceExt");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->mean_tensor_uid(), -0.1f, 0.1f, rng);
    tracker.fillFree(a->variance_tensor_uid(), 0.5f, 1.5f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// peer_stats holds references to other GPUs' memory for multi-GPU batchnorm —
// randomly generated values would point to invalid cross-device memory.
inline SynthesisResult
    fillBatchnormTrainingInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                SynthesisTracker& tracker,
                                std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    tracker.fillFree(a->prev_running_mean_tensor_uid(), -0.1f, 0.1f, rng);
    tracker.fillFree(a->prev_running_variance_tensor_uid(), 0.5f, 1.5f, rng);
    tracker.fillFree(a->momentum_tensor_uid(), 0.0f, 1.0f, rng);

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            tracker.markStructured(uid, "peer_stats");
        }
    }

    return SynthesisResult::ok();
}

// mean/inv_variance are optional (may come from forward). peer_stats: see above.
inline SynthesisResult
    fillBatchnormBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                SynthesisTracker& tracker,
                                std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormBackwardAttributes");
    }
    tracker.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->mean_tensor_uid(), -0.1f, 0.1f, rng);
    tracker.fillFree(a->inv_variance_tensor_uid(), 0.5f, 1.5f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            tracker.markStructured(uid, "peer_stats");
        }
    }

    return SynthesisResult::ok();
}

// ── Matmul ────────────────────────────────────────────────────────────────────

inline SynthesisResult fillMatmulInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                        SynthesisTracker& tracker,
                                        std::mt19937& rng)
{
    const auto* a = node.attributes_as_MatmulAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not MatmulAttributes");
    }
    tracker.fillFree(a->a_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->b_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── Pointwise ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillPointwiseInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                           SynthesisTracker& tracker,
                                           std::mt19937& rng)
{
    const auto* a = node.attributes_as_PointwiseAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not PointwiseAttributes");
    }
    tracker.fillFree(a->in_0_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->in_1_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->in_2_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->axis_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── Reduction ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillReductionInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                           SynthesisTracker& tracker,
                                           std::mt19937& rng)
{
    const auto* a = node.attributes_as_ReductionAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not ReductionAttributes");
    }
    tracker.fillFree(a->in_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillLayernormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                           SynthesisTracker& tracker,
                                           std::mt19937& rng)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not LayernormAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// mean and inv_variance are computed by the forward pass — a standalone backward
// can't produce correct gradients without them.
inline SynthesisResult
    fillLayernormBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                SynthesisTracker& tracker,
                                std::mt19937& rng)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not LayernormBackwardAttributes");
    }
    tracker.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.markDerived(a->mean_tensor_uid(), "mean (forward output)");
    tracker.markDerived(a->inv_variance_tensor_uid(), "inv_variance (forward output)");
    tracker.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

inline SynthesisResult fillRmsnormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         SynthesisTracker& tracker,
                                         std::mt19937& rng)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not RMSNormAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    tracker.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// inv_rms is computed by the forward pass.
inline SynthesisResult
    fillRmsnormBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                              SynthesisTracker& tracker,
                              std::mt19937& rng)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not RMSNormBackwardAttributes");
    }
    tracker.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.markDerived(a->inv_rms_tensor_uid(), "inv_rms (forward output)");
    return SynthesisResult::ok();
}

// ── Resample ──────────────────────────────────────────────────────────────────

inline SynthesisResult fillResampleFwdInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                             SynthesisTracker& tracker,
                                             std::mt19937& rng)
{
    const auto* a = node.attributes_as_ResampleFwdAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not ResampleFwdAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── Block-scale quantization ──────────────────────────────────────────────────

// Scale tensor holds per-block quantization factors that must match the
// quantized data — random scales would produce garbage dequantized values.
inline SynthesisResult
    fillBlockScaleDequantizeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                   SynthesisTracker& tracker,
                                   std::mt19937& rng)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BlockScaleDequantizeAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.markStructured(a->scale_tensor_uid(), "scale (block quantization scales)");
    return SynthesisResult::ok();
}

inline SynthesisResult
    fillBlockScaleQuantizeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 SynthesisTracker& tracker,
                                 std::mt19937& rng)
{
    const auto* a = node.attributes_as_BlockScaleQuantizeAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BlockScaleQuantizeAttributes");
    }
    tracker.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    return SynthesisResult::ok();
}

// ── SDPA ──────────────────────────────────────────────────────────────────────

// Q/K/V/mask accept random values, as does scale (the softmax multiplier, e.g.
// 1/sqrt(head_dim) — any positive value is mathematically valid). The FP8/MX
// descale/scale factors are STRUCTURED, NOT free: each must equal the actual
// quantization factor used to produce its tensor's data. A random descale does
// not break the engine-vs-reference comparison (both read the same shared value)
// but it lets values drift out of FP8 range and saturate identically on both
// sides — a vacuous pass that verifies nothing. We therefore refuse to fabricate
// them, mirroring fillBlockScaleDequantizeInputs. Real FP8 coverage comes from
// authored bundles that ship the matching scales as data. The remaining inputs
// are STRUCTURED for their own reasons: seq lengths encode actual sequence
// boundaries, page tables map to allocated GPU memory chunks, block masks define
// sparse attention patterns, and dropout seed/offset must match between fwd and
// bwd. Most of these are optional — absent ones (uid 0) are silently ignored.
inline SynthesisResult fillSdpaForwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                             SynthesisTracker& tracker,
                                             std::mt19937& rng)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not SdpaAttributes");
    }

    tracker.fillFree(a->q_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->k_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->v_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->attn_mask_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), 0.1f, 1.0f, rng);

    // FP8/MX quantization scale factors must match the data's true scale — see
    // the header comment. Refuse rather than fabricate a meaningless value.
    tracker.markStructured(a->descale_q_tensor_uid(), "descale_q");
    tracker.markStructured(a->descale_k_tensor_uid(), "descale_k");
    tracker.markStructured(a->descale_v_tensor_uid(), "descale_v");
    tracker.markStructured(a->descale_s_tensor_uid(), "descale_s");
    tracker.markStructured(a->scale_s_tensor_uid(), "scale_s");
    tracker.markStructured(a->scale_o_tensor_uid(), "scale_o");

    tracker.markStructured(a->seq_len_q_tensor_uid(), "seq_len_q");
    tracker.markStructured(a->seq_len_kv_tensor_uid(), "seq_len_kv");
    tracker.markStructured(a->page_table_k_tensor_uid(), "page_table_k");
    tracker.markStructured(a->page_table_v_tensor_uid(), "page_table_v");
    tracker.markStructured(a->block_mask_tensor_uid(), "block_mask");
    tracker.markStructured(a->seed_tensor_uid(), "dropout_seed");
    tracker.markStructured(a->offset_tensor_uid(), "dropout_offset");

    return SynthesisResult::ok();
}

// Q/K/V/dO accept random values. O (the forward output) and stats (softmax
// statistics) are DERIVED — they must come from a forward pass to produce
// correct gradients. In a fused forward+backward graph these are virtual
// inter-node tensors (not owned, so silently skipped). A standalone backward
// without a forward is refused.
inline SynthesisResult
    fillSdpaBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                           SynthesisTracker& tracker,
                           std::mt19937& rng)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not SdpaBackwardAttributes");
    }

    tracker.fillFree(a->q_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->k_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->v_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->do_tensor_uid(), -1.0f, 1.0f, rng);
    tracker.fillFree(a->scale_tensor_uid(), 0.1f, 1.0f, rng);
    tracker.fillFree(a->dropout_scale_tensor_uid(), 0.1f, 1.0f, rng);
    tracker.fillFree(a->dropout_scale_inv_tensor_uid(), 0.1f, 1.0f, rng);
    tracker.fillFree(a->attn_mask_tensor_uid(), -1.0f, 1.0f, rng);

    tracker.markDerived(a->o_tensor_uid(), "o (forward output)");
    tracker.markDerived(a->stats_tensor_uid(), "stats (forward softmax stats)");

    tracker.markStructured(a->seq_len_q_tensor_uid(), "seq_len_q");
    tracker.markStructured(a->seq_len_kv_tensor_uid(), "seq_len_kv");
    tracker.markStructured(a->seed_tensor_uid(), "dropout_seed");
    tracker.markStructured(a->offset_tensor_uid(), "dropout_offset");

    return SynthesisResult::ok();
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
// Routes a node to its fill function based on the flatbuffer attribute type.
// The harness calls this once per node in the graph — for a fused graph like
// conv+bias+relu, each node is dispatched separately with only its own inputs.
// Returns ok() when all of the node's inputs were filled, or unsupported() with
// a diagnostic when the op is unrecognized or an input can't be synthesized.

inline SynthesisResult synthesizeNodeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                            SynthesisTracker& tracker,
                                            std::mt19937& rng)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::ConvolutionFwdAttributes:
        return fillConvFwdInputs(node, tracker, rng);
    case NA::ConvolutionBwdAttributes:
        return fillConvBwdDataInputs(node, tracker, rng);
    case NA::ConvolutionWrwAttributes:
        return fillConvBwdWeightsInputs(node, tracker, rng);
    case NA::BatchnormInferenceAttributes:
        return fillBatchnormInferenceInputs(node, tracker, rng);
    case NA::BatchnormInferenceAttributesVarianceExt:
        return fillBatchnormInferenceVarianceInputs(node, tracker, rng);
    case NA::BatchnormAttributes:
        return fillBatchnormTrainingInputs(node, tracker, rng);
    case NA::BatchnormBackwardAttributes:
        return fillBatchnormBackwardInputs(node, tracker, rng);
    case NA::MatmulAttributes:
        return fillMatmulInputs(node, tracker, rng);
    case NA::PointwiseAttributes:
        return fillPointwiseInputs(node, tracker, rng);
    case NA::ReductionAttributes:
        return fillReductionInputs(node, tracker, rng);
    case NA::LayernormAttributes:
        return fillLayernormInputs(node, tracker, rng);
    case NA::LayernormBackwardAttributes:
        return fillLayernormBackwardInputs(node, tracker, rng);
    case NA::RMSNormAttributes:
        return fillRmsnormInputs(node, tracker, rng);
    case NA::RMSNormBackwardAttributes:
        return fillRmsnormBackwardInputs(node, tracker, rng);
    case NA::ResampleFwdAttributes:
        return fillResampleFwdInputs(node, tracker, rng);
    case NA::BlockScaleDequantizeAttributes:
        return fillBlockScaleDequantizeInputs(node, tracker, rng);
    case NA::BlockScaleQuantizeAttributes:
        return fillBlockScaleQuantizeInputs(node, tracker, rng);
    case NA::SdpaAttributes:
        return fillSdpaForwardInputs(node, tracker, rng);
    case NA::SdpaBackwardAttributes:
        return fillSdpaBackwardInputs(node, tracker, rng);
    default:
        return SynthesisResult::unsupported("no input synthesis registered for this op");
    }
}

} // namespace hipdnn_integration_tests::golden
