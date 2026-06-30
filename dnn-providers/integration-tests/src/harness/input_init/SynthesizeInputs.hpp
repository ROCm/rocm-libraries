// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/input_init/InputSynthesizer.hpp"

namespace hipdnn_integration_tests
{

// ── Per-op declaration functions ─────────────────────────────────────────────
// Each function declares inputs for one node in the graph. A single
// InputSynthesizer is shared across all nodes — synth.synthesize(graph)
// walks the graph, calls each declaration function, then resolves and
// fills all owned tensors.
//
// Every function follows the same pattern:
//   1. Cast the node to its concrete attribute type.
//   2. Call range(uid, lo, hi) for op-specific ranges, markStructured/markDerived
//      for non-synthesizable inputs. Uids that need default [-1,1] need no call.
//   3. Return ok() if the attribute cast succeeded, or unsupported() if not.
//
// To add a new op: write a declare*Inputs function that calls range() /
// markStructured() / markDerived() for exceptions, add a case to the switch
// in declareNodeInputs(). Ops where all inputs are default FREE need no
// function — just add the case to the fallthrough block in the switch.

// ── Batchnorm ─────────────────────────────────────────────────────────────────

inline SynthesisResult
    declareBatchnormInferenceInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormInferenceAttributes");
    }
    synth.range(a->mean_tensor_uid(), -0.1f, 0.1f);
    synth.range(a->inv_variance_tensor_uid(), 0.5f, 1.5f);
    return SynthesisResult::ok();
}

inline SynthesisResult
    declareBatchnormInferenceVarianceInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                            InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormInferenceAttributesVarianceExt");
    }
    synth.range(a->mean_tensor_uid(), -0.1f, 0.1f);
    synth.range(a->variance_tensor_uid(), 0.5f, 1.5f);
    synth.range(a->epsilon_tensor_uid(), 0.0f, 1.0f);
    return SynthesisResult::ok();
}

// peer_stats holds references to other GPUs' memory for multi-GPU batchnorm —
// randomly generated values would point to invalid cross-device memory.
inline SynthesisResult
    declareBatchnormTrainingInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                   InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormAttributes");
    }
    synth.range(a->epsilon_tensor_uid(), 0.0f, 1.0f);
    synth.range(a->prev_running_mean_tensor_uid(), -0.1f, 0.1f);
    synth.range(a->prev_running_variance_tensor_uid(), 0.5f, 1.5f);
    synth.range(a->momentum_tensor_uid(), 0.0f, 1.0f);

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            synth.markStructured(uid, "peer_stats");
        }
    }

    return SynthesisResult::ok();
}

// mean/inv_variance are optional (may come from forward). peer_stats: see above.
inline SynthesisResult
    declareBatchnormBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                   InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BatchnormBackwardAttributes");
    }
    synth.range(a->mean_tensor_uid(), -0.1f, 0.1f);
    synth.range(a->inv_variance_tensor_uid(), 0.5f, 1.5f);

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            synth.markStructured(uid, "peer_stats");
        }
    }

    return SynthesisResult::ok();
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

inline SynthesisResult
    declareLayernormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                           InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not LayernormAttributes");
    }
    synth.range(a->epsilon_tensor_uid(), 0.0f, 1.0f);
    return SynthesisResult::ok();
}

// mean and inv_variance are computed by the forward pass — a standalone backward
// can't produce correct gradients without them.
inline SynthesisResult
    declareLayernormBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                   InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not LayernormBackwardAttributes");
    }
    synth.markDerived(a->mean_tensor_uid(), "mean (forward output)");
    synth.markDerived(a->inv_variance_tensor_uid(), "inv_variance (forward output)");
    synth.range(a->epsilon_tensor_uid(), 0.0f, 1.0f);
    return SynthesisResult::ok();
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

inline SynthesisResult declareRmsnormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                            InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not RMSNormAttributes");
    }
    synth.range(a->epsilon_tensor_uid(), 0.0f, 1.0f);
    return SynthesisResult::ok();
}

// inv_rms is computed by the forward pass.
inline SynthesisResult
    declareRmsnormBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not RMSNormBackwardAttributes");
    }
    synth.markDerived(a->inv_rms_tensor_uid(), "inv_rms (forward output)");
    return SynthesisResult::ok();
}

// ── Block-scale quantization ──────────────────────────────────────────────────

// Scale tensor holds per-block quantization factors that must match the
// quantized data — random scales would produce garbage dequantized values.
inline SynthesisResult
    declareBlockScaleDequantizeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not BlockScaleDequantizeAttributes");
    }
    synth.markStructured(a->scale_tensor_uid(), "scale (block quantization scales)");
    return SynthesisResult::ok();
}

// ── SDPA ──────────────────────────────────────────────────────────────────────

inline SynthesisResult
    declareSdpaForwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                             InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not SdpaAttributes");
    }

    synth.range(a->scale_tensor_uid(), 0.1f, 1.0f);

    synth.markStructured(a->descale_q_tensor_uid(), "descale_q");
    synth.markStructured(a->descale_k_tensor_uid(), "descale_k");
    synth.markStructured(a->descale_v_tensor_uid(), "descale_v");
    synth.markStructured(a->descale_s_tensor_uid(), "descale_s");
    synth.markStructured(a->scale_s_tensor_uid(), "scale_s");
    synth.markStructured(a->scale_o_tensor_uid(), "scale_o");

    synth.markStructured(a->seq_len_q_tensor_uid(), "seq_len_q");
    synth.markStructured(a->seq_len_kv_tensor_uid(), "seq_len_kv");
    synth.markStructured(a->page_table_k_tensor_uid(), "page_table_k");
    synth.markStructured(a->page_table_v_tensor_uid(), "page_table_v");
    synth.markStructured(a->block_mask_tensor_uid(), "block_mask");
    synth.markStructured(a->seed_tensor_uid(), "dropout_seed");
    synth.markStructured(a->offset_tensor_uid(), "dropout_offset");

    return SynthesisResult::ok();
}

// Q/K/V/dO accept random values. O (the forward output) and stats (softmax
// statistics) are DERIVED — they must come from a forward pass to produce
// correct gradients.
inline SynthesisResult
    declareSdpaBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                              InputSynthesizer& synth)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return SynthesisResult::unsupported("not SdpaBackwardAttributes");
    }

    synth.range(a->scale_tensor_uid(), 0.1f, 1.0f);
    synth.range(a->dropout_scale_tensor_uid(), 0.1f, 1.0f);
    synth.range(a->dropout_scale_inv_tensor_uid(), 0.1f, 1.0f);

    synth.markDerived(a->o_tensor_uid(), "o (forward output)");
    synth.markDerived(a->stats_tensor_uid(), "stats (forward softmax stats)");

    synth.markStructured(a->seq_len_q_tensor_uid(), "seq_len_q");
    synth.markStructured(a->seq_len_kv_tensor_uid(), "seq_len_kv");
    synth.markStructured(a->seed_tensor_uid(), "dropout_seed");
    synth.markStructured(a->offset_tensor_uid(), "dropout_offset");

    return SynthesisResult::ok();
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

inline SynthesisResult declareNodeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         InputSynthesizer& synth)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    // All inputs default FREE — no exceptions to declare.
    case NA::ConvolutionFwdAttributes:
    case NA::ConvolutionBwdAttributes:
    case NA::ConvolutionWrwAttributes:
    case NA::MatmulAttributes:
    case NA::PointwiseAttributes:
    case NA::ReductionAttributes:
    case NA::ResampleFwdAttributes:
    case NA::BlockScaleQuantizeAttributes:
        return SynthesisResult::ok();

    // Ops with exceptions — custom ranges, structured, or derived inputs.
    case NA::BatchnormInferenceAttributes:
        return declareBatchnormInferenceInputs(node, synth);
    case NA::BatchnormInferenceAttributesVarianceExt:
        return declareBatchnormInferenceVarianceInputs(node, synth);
    case NA::BatchnormAttributes:
        return declareBatchnormTrainingInputs(node, synth);
    case NA::BatchnormBackwardAttributes:
        return declareBatchnormBackwardInputs(node, synth);
    case NA::LayernormAttributes:
        return declareLayernormInputs(node, synth);
    case NA::LayernormBackwardAttributes:
        return declareLayernormBackwardInputs(node, synth);
    case NA::RMSNormAttributes:
        return declareRmsnormInputs(node, synth);
    case NA::RMSNormBackwardAttributes:
        return declareRmsnormBackwardInputs(node, synth);
    case NA::BlockScaleDequantizeAttributes:
        return declareBlockScaleDequantizeInputs(node, synth);
    case NA::SdpaAttributes:
        return declareSdpaForwardInputs(node, synth);
    case NA::SdpaBackwardAttributes:
        return declareSdpaBackwardInputs(node, synth);
    default:
        return SynthesisResult::unsupported("no input synthesis registered for this op");
    }
}

// ── InputSynthesizer::synthesize(Graph) — out-of-class definition ────────
// Declared in InputSynthesizer.hpp, defined here to break the include cycle.

inline SynthesisResult
    InputSynthesizer::synthesize(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        auto result = declareNodeInputs(*graph.nodes()->Get(i), *this);
        if(!result.filled)
        {
            return result;
        }
    }
    return synthesize("graph");
}

} // namespace hipdnn_integration_tests
