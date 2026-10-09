// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <optional>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor
{

/// The registry template under its ingestor spelling. It is one template, shared with the
/// UHD scorer registry, so `ingestor::NativeRegistry<T>` and
/// `hipdnn_plugin_sdk::NativeRegistry<T>` name the same type.
template <typename T>
using NativeRegistry = hipdnn_plugin_sdk::NativeRegistry<T>;

/// @brief Descriptor-native hooks retain their typed graph and kernel operands.
using GraphMatchFn = std::optional<BoundTokens> (*)(const MatchContext&);
using GraphCriterionFn = bool (*)(const MatchContext&, const BoundTokens&);
using KernelMatcherFn = bool (*)(const MatchContext&, const BoundTokens&, const KernelDefinition&);
using ScoreFn = double (*)(const MatchContext&, const BoundTokens&, const KernelDefinition&);

/// A kernel-scoped matcher's set-level refinement, for an applicability rule that is a
/// property of the candidate set rather than of one kernel ("only the widest variant
/// the graph admits", say). A pack registers one under the symbol of a kernel-scoped
/// UMD it already declares; no descriptor names it. Once every pack has been matched,
/// it sees every kernel that UMD admitted for the graph, across all packs listing it,
/// in catalog order, and returns one keep flag per kernel. It can only narrow the
/// catalog: what it keeps stays in order, and ranking orders it as before. It resolves
/// with its matcher, once per engine's shared state (KernelIngestorSharedState), so it
/// must be registered before the engine loads. Must be thread-safe, like the hooks above.
using CandidateFilterFn = std::vector<bool> (*)(const MatchContext&,
                                                const BoundTokens&,
                                                const std::vector<const KernelDefinition*>&);

using GraphMatchRegistry = NativeRegistry<GraphMatchFn>;
using GraphCriterionRegistry = NativeRegistry<GraphCriterionFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;
using CandidateFilterRegistry = NativeRegistry<CandidateFilterFn>;
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
