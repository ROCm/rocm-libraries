// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/IReferenceExecutors.hpp"
#include "harness/input-init/FillInputs.hpp"

namespace hipdnn_integration_tests
{

/// Overwrites the filled O and stats (LSE) inputs of an SDPA backward node with the
/// forward of its filled Q, K and V. The backward recomputes P = exp(Q K^T * scale - lse)
/// from them, and only the forward's LSE makes the rows of P sum to 1, as they do in
/// training.
///
/// The forward is an SDPA forward graph built from the node: its Q, K and V, its forward
/// attributes (scale or scale tensor, diagonal band and causal flags, additive mask,
/// ALiBi, padding with seq_len_q and seq_len_kv, the explicit dropout mask and scale) and
/// any ragged offsets, with O and stats as outputs; rank-3 stats are declared
/// [B, H, Sq, 1] over the same memory. It runs through `referenceExecutors`, so it models
/// what the forward references model. With `onDevice` and a visible HIP device the GPU
/// reference executor runs it where the device fill wrote Q, K and V; a graph it declines,
/// or rejects before it launches (ReferenceCapabilityError or std::invalid_argument), goes
/// to the CPU reference executor, as does every graph otherwise. The GPU executor is only
/// asked for when it is to run.
///
/// Returns nullopt when O and stats were derived, and when neither is in `ownedUids`, as
/// in a fused forward-backward graph, whose forward node computes them. Otherwise returns
/// why they cannot be derived, which leaves the node with no meaningful O and stats to
/// verify against: Philox dropout, a forward operand or ragged offsets the graph computes,
/// or a forward graph no reference executor models or that the CPU reference rejects.
/// Never returns nullopt without deriving. Throws DeviceInputError when the device forward
/// fails after the GPU executor accepted the graph: that is a failure, not a fallback.
std::optional<std::string> deriveSdpaBackwardForwardOutputs(
    const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attributes,
    InputTensorMap& inputs,
    const std::vector<int64_t>& ownedUids,
    IReferenceExecutors& referenceExecutors,
    bool onDevice);

} // namespace hipdnn_integration_tests
