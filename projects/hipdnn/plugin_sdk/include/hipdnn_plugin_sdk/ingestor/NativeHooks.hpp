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

/// @brief Descriptor-native hooks retain their typed graph and kernel operands.
using GraphMatchFn = std::optional<BoundTokens> (*)(const MatchContext&);
using GraphCriterionFn = bool (*)(const MatchContext&, const BoundTokens&);
using KernelMatcherFn = bool (*)(const MatchContext&, const BoundTokens&, const KernelDefinition&);
using ScoreFn = double (*)(const MatchContext&, const BoundTokens&, const KernelDefinition&);
using GraphMatchRegistry = NativeRegistry<GraphMatchFn>;
using GraphCriterionRegistry = NativeRegistry<GraphCriterionFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
