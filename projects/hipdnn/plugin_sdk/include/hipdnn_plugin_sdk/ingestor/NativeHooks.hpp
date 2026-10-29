// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <optional>

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
/// Resolves the symbol a UED's `graph_match.native` arm names (RFC 0020 §4.5). Register a
/// GraphMatchFn here only for that arm. A `graph_match.nodes` pattern (RFC 0020 §4.3) is
/// compiled from the descriptor against the op-schema registry and needs no registration.
using GraphMatchRegistry = NativeRegistry<GraphMatchFn>;
using GraphCriterionRegistry = NativeRegistry<GraphCriterionFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
