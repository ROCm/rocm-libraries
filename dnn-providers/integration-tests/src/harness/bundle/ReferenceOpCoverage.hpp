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
inline constexpr std::string_view K_SEQ_LEN_TENSORS = "<seq-len tensors>";
inline constexpr std::string_view K_NO_NODES = "<no nodes>";

/// The node types each reference executor is required to handle. This is a contract:
/// covered bundles are registered with no skip path, so adding an op obliges implementing
/// it. Keyed on the flatbuffer node type, which is what the executors dispatch on.
/// Graph features (ragged, FP8, seq-len) are gated separately by exclusionReasons().
const std::set<NodeAttributes>& referenceSupportedOps(ReferenceExecutorType type);

/// Node types this graph uses, or nullopt when the buffer cannot be walked.
///
/// The two cases are distinct and callers must keep them apart: a graph with no
/// nodes is readable but has nothing to validate, an unreadable one is covered by
/// none. Collapsing both onto an empty set loses that distinction.
std::optional<std::set<NodeAttributes>> graphNodeTypes(const void* graphBuffer, size_t size);

/// Whether any tensor in the graph is ragged, or nullopt when it cannot be walked.
/// No reference executor, CPU or GPU, supports ragged tensors.
std::optional<bool> graphUsesRaggedTensors(const void* graphBuffer, size_t size);

/// Whether any tensor in the graph is an FP8 type, or nullopt when it cannot be walked.
std::optional<bool> graphUsesFp8Tensors(const void* graphBuffer, size_t size);

/// Whether any SDPA node sets a seq_len_q/kv tensor, or nullopt when it cannot be walked.
/// No reference executor, CPU or GPU, supports per-batch sequence lengths.
std::optional<bool> graphUsesSeqLenTensors(const void* graphBuffer, size_t size);

/// Every reason this reference is not required to run this graph; empty means it is.
/// An unreadable graph yields a single sentinel entry rather than nothing.
std::vector<std::string>
    exclusionReasons(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// True iff exclusionReasons() is empty.
bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size);

/// The parenthesised reason list appended to the registration summary, or "" when
/// the set is empty.
///
/// Split out from the summary line itself so it is testable without calling
/// registerReferenceValidationTests(), which is inline and reaches
/// sharedReferenceExecutors() -- the unit target deliberately does not link that.
std::string formatExclusionReasons(const std::set<std::string>& reasons);

} // namespace hipdnn_integration_tests::bundle
