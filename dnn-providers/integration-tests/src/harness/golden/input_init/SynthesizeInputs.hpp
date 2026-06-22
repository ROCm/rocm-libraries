// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/golden/input_init/RoleAccounting.hpp"

namespace hipdnn_integration_tests::golden
{

// ── Per-op fill functions ─────────────────────────────────────────────────────
// To add an op: copy fillBatchnormInputs, adapt for your op's attributes, and
// add one case to the switch in synthesizeNodeInputs() below. Each function
// fills EVERY leaf input its node owns. The fill must be deterministic given
// `rng` (seeded from BundleMetadata::seed) so a graph-only bundle reproduces
// the same inputs across runs.

// Batchnorm-inference: every input is FREE (fillable from a numeric range).
// Ranges keep the op numerically well-behaved — inv_variance in [0.5, 1.5]
// avoids the blow-up in y = (x-mean)*inv_var*scale+bias.
inline FillOutcome fillBatchnormInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       const std::vector<int64_t>& ownedLeafInputUids,
                                       InputTensorMap& inputs,
                                       std::mt19937& rng)
{
    const auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    if(attrs == nullptr)
    {
        return FillOutcome::unsupported(
            "node is not BatchnormInferenceAttributes (initializer mis-registered)");
    }

    RoleAccounting acct(ownedLeafInputUids, inputs);
    acct.fillFree(attrs->x_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->mean_tensor_uid(), -0.1f, 0.1f, rng);
    acct.fillFree(attrs->inv_variance_tensor_uid(), 0.5f, 1.5f, rng);
    acct.fillFree(attrs->scale_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->bias_tensor_uid(), -1.0f, 1.0f, rng);
    return acct.finish("BatchnormInference");
}

// SDPA-forward: Q/K/V/mask/scale are FREE; sequence lengths, page tables, block
// masks, and dropout state are STRUCTURED (refused if present as leaf inputs).
// A plain Q/K/V graph fills fine; the moment a STRUCTURED input is actually
// present the bundle is refused (SKIP).
inline FillOutcome fillSdpaForwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         const std::vector<int64_t>& ownedLeafInputUids,
                                         InputTensorMap& inputs,
                                         std::mt19937& rng)
{
    const auto* attrs = node.attributes_as_SdpaAttributes();
    if(attrs == nullptr)
    {
        return FillOutcome::unsupported("node is not SdpaAttributes (initializer mis-registered)");
    }

    RoleAccounting acct(ownedLeafInputUids, inputs);

    acct.fillFree(attrs->q_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->k_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->v_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->attn_mask_tensor_uid().value_or(0), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->scale_tensor_uid().value_or(0), 0.1f, 1.0f, rng);

    acct.markStructured(attrs->seq_len_q_tensor_uid().value_or(0), "seq_len_q");
    acct.markStructured(attrs->seq_len_kv_tensor_uid().value_or(0), "seq_len_kv");
    acct.markStructured(attrs->page_table_k_tensor_uid().value_or(0), "page_table_k");
    acct.markStructured(attrs->page_table_v_tensor_uid().value_or(0), "page_table_v");
    acct.markStructured(attrs->block_mask_tensor_uid().value_or(0), "block_mask");
    acct.markStructured(attrs->seed_tensor_uid().value_or(0), "dropout_seed");
    acct.markStructured(attrs->offset_tensor_uid().value_or(0), "dropout_offset");

    return acct.finish("Sdpa");
}

// SDPA-backward: Q/K/V/dO are FREE; O and stats are DERIVED (must match a
// forward pass). A standalone backward graph-only bundle is refused; when
// forward+backward are fused in one graph, O/stats are virtual inter-node edges
// and never reach this function.
inline FillOutcome fillSdpaBackwardInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                          const std::vector<int64_t>& ownedLeafInputUids,
                                          InputTensorMap& inputs,
                                          std::mt19937& rng)
{
    const auto* attrs = node.attributes_as_SdpaBackwardAttributes();
    if(attrs == nullptr)
    {
        return FillOutcome::unsupported(
            "node is not SdpaBackwardAttributes (initializer mis-registered)");
    }

    RoleAccounting acct(ownedLeafInputUids, inputs);

    acct.fillFree(attrs->q_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->k_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->v_tensor_uid(), -1.0f, 1.0f, rng);
    acct.fillFree(attrs->do_tensor_uid(), -1.0f, 1.0f, rng);

    acct.markDerived(attrs->o_tensor_uid(), "o (forward output)");
    acct.markDerived(attrs->stats_tensor_uid(), "stats (forward softmax stats)");

    return acct.finish("SdpaBackward");
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
// Maps a node's attribute type to its fill function. Unknown ops return
// unsupported — the harness SKIPs and records it in the unverifiable report.

inline FillOutcome synthesizeNodeInputs(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                        const std::vector<int64_t>& ownedLeafInputUids,
                                        InputTensorMap& inputs,
                                        std::mt19937& rng)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::BatchnormInferenceAttributes:
        return fillBatchnormInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::SdpaAttributes:
        return fillSdpaForwardInputs(node, ownedLeafInputUids, inputs, rng);
    case NA::SdpaBackwardAttributes:
        return fillSdpaBackwardInputs(node, ownedLeafInputUids, inputs, rng);
    default:
        return FillOutcome::unsupported("no input synthesis registered for this op");
    }
}

} // namespace hipdnn_integration_tests::golden
