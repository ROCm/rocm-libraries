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

/// Exclusion reasons that are not an op name. All start with '<' so they sort ahead
/// of the op names they appear beside.
inline constexpr std::string_view K_UNREADABLE_GRAPH = "<unreadable graph>";
inline constexpr std::string_view K_RAGGED_TENSORS = "<ragged tensors>";
inline constexpr std::string_view K_FP8_TENSORS = "<fp8 tensors>";
inline constexpr std::string_view K_NO_NODES = "<no nodes>";

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
///
/// The limit of that choice is that a node type says nothing about the tensors the
/// node reads, so this set alone cannot express a feature the reference does not
/// handle. The commitment is therefore "op set *and* feature set": see
/// exclusionReasons(), which folds this set together with the graph-feature queries
/// below.
const std::set<NodeAttributes>& referenceSupportedOps(ReferenceExecutorType type);

/// Node types this graph uses, or nullopt when the buffer cannot be walked.
///
/// The two cases are distinct and callers must keep them apart: a graph with no
/// nodes is readable but has nothing to validate, an unreadable one is covered by
/// none. Collapsing both onto an empty set loses that distinction.
std::optional<std::set<NodeAttributes>> graphNodeTypes(const void* graphBuffer, size_t size);

/// Whether the graph declares a ragged tensor, or nullopt when it cannot be walked.
///
/// Ragged lives on TensorAttributes (`ragged_offset_tensor_uid`), not on the node
/// type, so referenceSupportedOps() structurally cannot see it: a ragged SDPA graph
/// and a dense one are both SdpaAttributes. That is why the gate needs this second
/// axis at all.
///
/// Deliberately not parameterized on ReferenceExecutorType: no reference plan
/// builder, CPU or GPU, reads ragged offsets, so a per-reference answer would be two
/// copies of the same `false`. When one op on one reference gains ragged support it
/// gains that parameter, and the reason list already carries the result.
std::optional<bool> graphUsesRaggedTensors(const void* graphBuffer, size_t size);

/// Whether any tensor in the graph is an FP8 type, or nullopt when it cannot be
/// walked. Same tensor-not-node argument as graphUsesRaggedTensors(); unlike ragged,
/// only the GPU reference is excluded on it (see exclusionReasons()).
std::optional<bool> graphUsesFp8Tensors(const void* graphBuffer, size_t size);

/// Every reason this reference is not required to run this graph, for diagnostics;
/// empty means it is. Reasons compose: an unsupported op and a ragged tensor both
/// appear rather than the first one short-circuiting.
///
/// An unreadable graph yields a single sentinel entry rather than nothing, so a
/// caller printing this never reports an exclusion with no reason attached.
std::vector<std::string>
    exclusionReasons(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// True iff exclusionReasons() is empty -- defined in terms of it rather than
/// recomputed, so verdict and reason cannot disagree.
///
/// They previously could: the two were independent implementations of one rule, and
/// had already diverged on a zero-node graph, which referenceCoversGraph() called
/// "not covered" while uncoveredNodeTypes() returned {}. That is how the
/// registration summary could report an exclusion with no reason attached.
bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// The parenthesised reason list appended to the registration summary, or "" when
/// the set is empty.
///
/// Split out from the summary line itself so it is testable without calling
/// registerReferenceValidationTests(), which is inline and reaches
/// sharedReferenceExecutors() -- the unit target deliberately does not link that.
std::string formatExclusionReasons(const std::set<std::string>& reasons);

} // namespace hipdnn_integration_tests::bundle
