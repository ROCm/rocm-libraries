// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests::bundle
{

using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

/// Stands in for the node list of a graph whose buffer failed verification, in the
/// one place a node list is only ever printed.
inline constexpr std::string_view K_UNREADABLE_GRAPH = "<unreadable graph>";

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

/// Node types this graph uses, or nullopt when the buffer cannot be walked.
///
/// The two cases are distinct and callers must keep them apart: a graph with no
/// nodes is covered by every reference, an unreadable one is covered by none.
/// Collapsing both onto an empty set makes referenceCoversGraph() and
/// uncoveredNodeTypes() disagree -- "not covered, but nothing is uncovered".
std::optional<std::set<NodeAttributes>> graphNodeTypes(const void* graphBuffer, size_t size);

/// True iff the graph is readable and every node in it is inside `type`'s
/// required-op set.
bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// Human-readable node types the given reference does not cover, for diagnostics.
///
/// An unreadable graph yields a single sentinel entry rather than nothing, so a
/// caller printing this never reports an exclusion with no reason attached.
std::vector<std::string>
    uncoveredNodeTypes(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// The parenthesised op list appended to the registration summary, or "" when the
/// set is empty.
///
/// Split out from the summary line itself so it is testable without calling
/// registerReferenceValidationTests(), which is inline and reaches
/// sharedReferenceExecutors() -- the unit target deliberately does not link that.
std::string formatUncoveredOps(const std::set<std::string>& uncoveredOps);

} // namespace hipdnn_integration_tests::bundle
