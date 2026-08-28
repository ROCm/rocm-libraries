// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

/// The ops each reference executor is *required* to handle.
///
/// This is a commitment, not a description. A bundle whose every node type appears
/// in a reference's set is registered for validation against that reference and
/// must pass — the reference harness has no skip path, because "the reference
/// could not run this" is a gap in the reference, not a property of the bundle.
///
/// That inverts the previous arrangement, where a reference that could not handle a
/// graph produced a silent skip and the bundle went unverified. Here the set is the
/// contract: adding an op obliges someone to implement it for that reference;
/// leaving it out means bundles using it are simply not validated by that
/// reference, visibly, by their absence from the registered suite.
///
/// Keyed on the flatbuffer node type rather than the bundle's optional `operation`
/// metadata string, because that is what both executors actually dispatch on and it
/// cannot drift from the graph.
const std::set<NodeAttributes>& referenceSupportedOps(ReferenceExecutorType type);

/// Node types this graph uses. Empty if the buffer cannot be walked.
std::set<NodeAttributes> graphNodeTypes(const void* graphBuffer, size_t size);

/// True iff every node in the graph is inside `type`'s required-op set.
bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// Human-readable node types the given reference does not cover, for diagnostics.
std::vector<std::string>
    uncoveredNodeTypes(ReferenceExecutorType type, const void* graphBuffer, size_t size);

} // namespace hipdnn_integration_tests::bundle
