// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/FillInputs.hpp"

namespace hipdnn_integration_tests
{

/// Overwrites the filled O and stats (LSE) inputs of an SDPA backward node with the
/// forward of its filled Q, K and V, under the node's scale and diagonal band. The
/// backward recomputes P = exp(Q K^T * scale - lse) from them, and only the forward's
/// LSE makes the rows of P sum to 1, as they do in training.
///
/// Leaves O and stats as filled when that forward cannot be reproduced: an additive
/// attention mask, ALiBi, padding, variable sequence lengths, dropout, ragged tensors,
/// a data type or shape the CPU reference does not take, or a scale tensor that is not
/// a pass-by-value scalar. Also leaves them when Q, K, V, O or stats is not one of
/// `ownedUids`, which is the case in a fused forward-backward graph, whose forward node
/// produces O and stats.
void deriveSdpaBackwardForwardOutputs(
    const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& attributes,
    InputTensorMap& inputs,
    const std::vector<int64_t>& ownedUids);

} // namespace hipdnn_integration_tests
