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

/// Overwrites the filled index input of a max-pooling resample backward node with the
/// argmax indices a forward produces. A random fill gives indices that no forward
/// emits: negative, or past the end of the dx window they select. The forward runs on
/// a scratch x, shaped like dx and filled uniformly in [-1, 1] from `seed`, under the
/// node's pre-padding, stride, window and padding mode, as the CPU reference forward
/// computes them. A window whose maximum is a zero-padding element gets index -1, as
/// in the forward, and routes its gradient to no dx element.
///
/// `seed` must be the same for every bundle of one graph, so the device and reference
/// bundles get the same indices.
///
/// Returns nullopt when the indices were derived; when the node is not a max pooling;
/// when it has no index tensor; when the index is not in `ownedUids`, as in a fused
/// forward-backward graph, whose forward node computes it; and when the CPU forward
/// rejects the shapes or parameters. In that last case the index keeps its fill, and
/// the CPU backward reference, which runs the same checks, fails the graph instead of
/// skipping it.
///
/// Returns why the indices cannot be derived otherwise: the index or dx is not in the
/// graph, or the index is not INT32, the only index type the references take.
std::optional<std::string> deriveResampleBackwardIndex(
    const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& attributes,
    InputTensorMap& inputs,
    const std::vector<int64_t>& ownedUids,
    unsigned int seed);

} // namespace hipdnn_integration_tests
