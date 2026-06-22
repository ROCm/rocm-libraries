// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/golden/input_init/SynthesisTracker.hpp"

namespace hipdnn_integration_tests::golden
{

// ── Per-op fill functions ─────────────────────────────────────────────────────
// Each function synthesizes inputs for one node in the graph. A node "owns" the
// leaf input tensors declared in its flatbuffer attributes — virtual tensors
// (inter-node edges in a fused graph) and output tensors are excluded.
//
// Every function follows the same pattern:
//   1. Cast the node to its concrete attribute type.
//   2. Create a SynthesisTracker with the node's owned uids.
//   3. Declare each input as FREE (fill with random values), STRUCTURED (can't
//      synthesize — needs specific format), or DERIVED (must come from another
//      op's output). See SynthesisTracker.hpp for role definitions.
//   4. Call finish() — returns ok() if all owned inputs were filled, or
//      unsupported() with a diagnostic listing what couldn't be synthesized.
//
// Fills must be deterministic given `rng` so re-running the same graph produces
// identical inputs for reproducible comparisons.
//
// To add a new op: copy fillConvFwdInputs (simplest example), adapt for your
// op's attributes, and add one case to the switch in synthesizeNodeInputs().
// Function names follow the pattern fill<AttributeName>Inputs.

// ── Convolution ───────────────────────────────────────────────────────────────

inline SynthesisResult fillConvFwdInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                     const std::vector<int64_t>& ownedLeafInputUids,
                                     InputTensorMap& inputs,
                                     std::mt19937& rng)
{
    const auto* a = node.attributes_as_ConvolutionFwdAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not ConvolutionFwdAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->w_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("ConvolutionFwd");
}

inline SynthesisResult fillConvBwdDataInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         const std::vector<int64_t>& ownedLeafInputUids,
                                         InputTensorMap& inputs,
                                         std::mt19937& rng)
{
    const auto* a = node.attributes_as_ConvolutionBwdAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not ConvolutionBwdAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->w_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("ConvolutionBwdData");
}

inline SynthesisResult fillConvBwdWeightsInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                            const std::vector<int64_t>& ownedLeafInputUids,
                                            InputTensorMap& inputs,
                                            std::mt19937& rng)
{
    const auto* a = node.attributes_as_ConvolutionWrwAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not ConvolutionWrwAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("ConvolutionBwdWeights");
}

// ── Batchnorm ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillBatchnormInferenceInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not BatchnormInferenceAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->mean_tensor_uid(), -0.1f, 0.1f, rng);
    acct.fillFree(a->inv_variance_tensor_uid(), 0.5f, 1.5f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("BatchnormInference");
}

inline SynthesisResult fillBatchnormInferenceVarianceInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(!a)
    {
        return SynthesisResult::unsupported("not BatchnormInferenceAttributesVarianceExt");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->mean_tensor_uid(), -0.1f, 0.1f, rng);
    acct.fillFree(a->variance_tensor_uid(), 0.5f, 1.5f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    return acct.finish("BatchnormInferenceVarianceExt");
}

// peer_stats holds references to other GPUs' memory for multi-GPU batchnorm —
// randomly generated values would point to invalid cross-device memory.
inline SynthesisResult fillBatchnormTrainingInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not BatchnormAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    acct.fillFree(a->prev_running_mean_tensor_uid().value_or(0), -0.1f, 0.1f, rng);
    acct.fillFree(a->prev_running_variance_tensor_uid().value_or(0), 0.5f, 1.5f, rng);
    acct.fillFree(a->momentum_tensor_uid().value_or(0), 0.0f, 1.0f, rng);

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            acct.markStructured(uid, "peer_stats");
        }
    }

    return acct.finish("BatchnormTraining");
}

// mean/inv_variance are optional (may come from forward). peer_stats: see above.
inline SynthesisResult fillBatchnormBackwardInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not BatchnormBackwardAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->mean_tensor_uid().value_or(0), -0.1f, 0.1f, rng);
    acct.fillFree(a->inv_variance_tensor_uid().value_or(0), 0.5f, 1.5f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            acct.markStructured(uid, "peer_stats");
        }
    }

    return acct.finish("BatchnormBackward");
}

// ── Matmul ────────────────────────────────────────────────────────────────────

inline SynthesisResult fillMatmulInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    const std::vector<int64_t>& ownedLeafInputUids,
                                    InputTensorMap& inputs,
                                    std::mt19937& rng)
{
    const auto* a = node.attributes_as_MatmulAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not MatmulAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->a_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->b_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("Matmul");
}

// ── Pointwise ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillPointwiseInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       const std::vector<int64_t>& ownedLeafInputUids,
                                       InputTensorMap& inputs,
                                       std::mt19937& rng)
{
    const auto* a = node.attributes_as_PointwiseAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not PointwiseAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->in_0_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->in_1_tensor_uid().value_or(0), -1.0f, 1.0f, rng);
    acct.fillFree(a->in_2_tensor_uid().value_or(0), -1.0f, 1.0f, rng);
    acct.fillFree(a->axis_tensor_uid().value_or(0), -1.0f, 1.0f, rng);
    return acct.finish("Pointwise");
}

// ── Reduction ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillReductionInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       const std::vector<int64_t>& ownedLeafInputUids,
                                       InputTensorMap& inputs,
                                       std::mt19937& rng)
{
    const auto* a = node.attributes_as_ReductionAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not ReductionAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->in_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("Reduction");
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

inline SynthesisResult fillLayernormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       const std::vector<int64_t>& ownedLeafInputUids,
                                       InputTensorMap& inputs,
                                       std::mt19937& rng)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not LayernormAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->bias_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    return acct.finish("Layernorm");
}

// mean and inv_variance are computed by the forward pass — a standalone backward
// can't produce correct gradients without them.
inline SynthesisResult fillLayernormBackwardInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not LayernormBackwardAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.markDerived(a->mean_tensor_uid().value_or(0), "mean (forward output)");
    acct.markDerived(a->inv_variance_tensor_uid().value_or(0), "inv_variance (forward output)");
    acct.fillFree(a->epsilon_tensor_uid().value_or(0), 0.0f, 1.0f, rng);
    return acct.finish("LayernormBackward");
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

inline SynthesisResult fillRmsnormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                     const std::vector<int64_t>& ownedLeafInputUids,
                                     InputTensorMap& inputs,
                                     std::mt19937& rng)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not RMSNormAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->epsilon_tensor_uid(), 0.0f, 1.0f, rng);
    acct.fillFree(a->bias_tensor_uid().value_or(0), -1.0f, 1.0f, rng);
    return acct.finish("RMSNorm");
}

// inv_rms is computed by the forward pass.
inline SynthesisResult fillRmsnormBackwardInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not RMSNormBackwardAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->dy_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.markDerived(a->inv_rms_tensor_uid(), "inv_rms (forward output)");
    return acct.finish("RMSNormBackward");
}

// ── Resample ──────────────────────────────────────────────────────────────────

inline SynthesisResult fillResampleFwdInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         const std::vector<int64_t>& ownedLeafInputUids,
                                         InputTensorMap& inputs,
                                         std::mt19937& rng)
{
    const auto* a = node.attributes_as_ResampleFwdAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not ResampleFwdAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("ResampleFwd");
}

// ── Block-scale quantization ──────────────────────────────────────────────────

// Scale tensor holds per-block quantization factors that must match the
// quantized data — random scales would produce garbage dequantized values.
inline SynthesisResult fillBlockScaleDequantizeInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not BlockScaleDequantizeAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.markStructured(a->scale_tensor_uid(), "scale (block quantization scales)");
    return acct.finish("BlockScaleDequantize");
}

inline SynthesisResult fillBlockScaleQuantizeInputs(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::vector<int64_t>& ownedLeafInputUids,
    InputTensorMap& inputs,
    std::mt19937& rng)
{
    const auto* a = node.attributes_as_BlockScaleQuantizeAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not BlockScaleQuantizeAttributes");
    }
    SynthesisTracker acct(ownedLeafInputUids, inputs);
    acct.fillFree(a->x_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("BlockScaleQuantize");
}

// ── SDPA ──────────────────────────────────────────────────────────────────────

// Q/K/V/mask/scale accept random values. The remaining inputs are STRUCTURED:
// seq lengths encode actual sequence boundaries, page tables map to allocated
// GPU memory chunks, block masks define sparse attention patterns, and dropout
// seed/offset must match between forward and backward passes.
// Most of these are optional — absent ones (uid 0) are silently ignored.
inline SynthesisResult fillSdpaForwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         const std::vector<int64_t>& ownedLeafInputUids,
                                         InputTensorMap& inputs,
                                         std::mt19937& rng)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not SdpaAttributes");
    }

    SynthesisTracker acct(ownedLeafInputUids, inputs);

    acct.fillFree(a->q_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->k_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->v_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->attn_mask_tensor_uid().value_or(0), -1.0f, 1.0f, rng);
    acct.fillFree(a->scale_tensor_uid().value_or(0), 0.1f, 1.0f, rng);

    acct.markStructured(a->seq_len_q_tensor_uid().value_or(0), "seq_len_q");
    acct.markStructured(a->seq_len_kv_tensor_uid().value_or(0), "seq_len_kv");
    acct.markStructured(a->page_table_k_tensor_uid().value_or(0), "page_table_k");
    acct.markStructured(a->page_table_v_tensor_uid().value_or(0), "page_table_v");
    acct.markStructured(a->block_mask_tensor_uid().value_or(0), "block_mask");
    acct.markStructured(a->seed_tensor_uid().value_or(0), "dropout_seed");
    acct.markStructured(a->offset_tensor_uid().value_or(0), "dropout_offset");

    return acct.finish("Sdpa");
}

// Q/K/V/dO accept random values. O (the forward output) and stats (softmax
// statistics) are DERIVED — they must come from a forward pass to produce
// correct gradients. In a fused forward+backward graph these are virtual
// inter-node tensors (not owned, so silently skipped). A standalone backward
// without a forward is refused.
inline SynthesisResult fillSdpaBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                          const std::vector<int64_t>& ownedLeafInputUids,
                                          InputTensorMap& inputs,
                                          std::mt19937& rng)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(!a)
    {
        return SynthesisResult::unsupported("not SdpaBackwardAttributes");
    }

    SynthesisTracker acct(ownedLeafInputUids, inputs);

    acct.fillFree(a->q_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->k_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->v_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(a->do_tensor_uid(), -1.0f, 1.0f, rng);

    acct.markDerived(a->o_tensor_uid(), "o (forward output)");
    acct.markDerived(a->stats_tensor_uid(), "stats (forward softmax stats)");

    return acct.finish("SdpaBackward");
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
// Routes a node to its fill function based on the flatbuffer attribute type.
// The harness calls this once per node in the graph — for a fused graph like
// conv+bias+relu, each node is dispatched separately with only its own inputs.
// Returns ok() when all of the node's inputs were filled, or unsupported() with
// a diagnostic when the op is unrecognized or an input can't be synthesized.

inline SynthesisResult synthesizeNodeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                        const std::vector<int64_t>& ownedLeafInputUids,
                                        InputTensorMap& inputs,
                                        std::mt19937& rng)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::ConvolutionFwdAttributes:
        return fillConvFwdInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::ConvolutionBwdAttributes:
        return fillConvBwdDataInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::ConvolutionWrwAttributes:
        return fillConvBwdWeightsInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::BatchnormInferenceAttributes:
        return fillBatchnormInferenceInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::BatchnormInferenceAttributesVarianceExt:
        return fillBatchnormInferenceVarianceInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::BatchnormAttributes:
        return fillBatchnormTrainingInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::BatchnormBackwardAttributes:
        return fillBatchnormBackwardInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::MatmulAttributes:
        return fillMatmulInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::PointwiseAttributes:
        return fillPointwiseInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::ReductionAttributes:
        return fillReductionInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::LayernormAttributes:
        return fillLayernormInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::LayernormBackwardAttributes:
        return fillLayernormBackwardInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::RMSNormAttributes:
        return fillRmsnormInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::RMSNormBackwardAttributes:
        return fillRmsnormBackwardInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::ResampleFwdAttributes:
        return fillResampleFwdInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::BlockScaleDequantizeAttributes:
        return fillBlockScaleDequantizeInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::BlockScaleQuantizeAttributes:
        return fillBlockScaleQuantizeInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::SdpaAttributes:
        return fillSdpaForwardInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::SdpaBackwardAttributes:
        return fillSdpaBackwardInputs(node, ownedLeafInputUids, inputs, rng);
    default:
        return SynthesisResult::unsupported("no input synthesis registered for this op");
    }
}

} // namespace hipdnn_integration_tests::golden
