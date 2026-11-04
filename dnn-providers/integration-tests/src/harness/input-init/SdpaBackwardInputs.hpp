// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/FillInputs.hpp"

namespace hipdnn_integration_tests
{

/// Overwrites the filled O and stats (LSE) inputs of an SDPA backward node with the
/// forward of its filled Q, K and V, under the node's scale, diagonal band and additive
/// mask. The backward recomputes P = exp(Q K^T * scale - lse) from them, and only the
/// forward's LSE makes the rows of P sum to 1, as they do in training.
///
/// Returns why the forward cannot be derived, which leaves the node with no meaningful
/// O and stats to verify against: ALiBi, padding, variable sequence lengths, dropout,
/// ragged tensors, mixed or unsupported data types, a scale tensor that is not a
/// pass-by-value scalar, a mask that does not broadcast to the scores, or a Q, K, V or
/// mask that the graph computes.
///
/// Returns nullopt when O and stats were derived; when neither is in `ownedUids`, as in
/// a fused forward-backward graph, whose forward node computes them; and when the CPU
/// forward rejects the shapes or band, such as stats not shaped [B, H, Sq, 1]. In that
/// last case O and stats keep their fill, and the CPU backward reference, which runs the
/// same checks, fails the graph instead of skipping it.
std::optional<std::string> deriveSdpaBackwardForwardOutputs(
    const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attributes,
    InputTensorMap& inputs,
    const std::vector<int64_t>& ownedUids);

} // namespace hipdnn_integration_tests
