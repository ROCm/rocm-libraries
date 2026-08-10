// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string_view>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @file NativeSymbolNames.hpp
 * @brief The symbol names this pack's descriptors resolve through.
 *
 * A descriptor names a native implementation by symbol, and the provider registers that
 * implementation under the same name. These constants are that contract, written once so
 * the descriptor side and the implementation side cannot drift apart.
 *
 * Each of these is a place a declarative descriptor field goes once the follow-up RFCs
 * land: the matcher symbols become criteria expressions, the score symbol becomes a
 * model artifact and its feature signature, and the dispatch symbol becomes grid, block,
 * workspace, and argument-signature formulas.
 */

/// Graph-scoped applicability: a single-node pointwise ADD over 1-element tensors.
inline constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.pointwise_add.graph_match";

/// Kernel-scoped applicability: the kernel's dtype must be the graph's dtype.
inline constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.pointwise_add.kernel_match";

/// Kernel-selection score.
inline constexpr std::string_view SCORE_SYMBOL = "hipkernel.pointwise_add.score";

/// Workspace sizing and launch.
inline constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.pointwise_add.dispatch";

/**
 * @brief The engine name, hashed into hipDNN's engine-id space.
 *
 * Scoped with a prefix, as RFC 0017 requires of a descriptor-backed engine: names must
 * be globally unique because they hash into a shared id space, and a scope is what makes
 * that tractable without a central allocation authority.
 *
 * Deliberately absent from EngineNames.hpp's registry. A descriptor-backed engine is
 * defined by data, so it registers its name when it is constructed rather than at build
 * time — which is the behavior this pack exists to demonstrate.
 */
inline constexpr std::string_view ENGINE_NAME = "hipkernel:PointwiseAdd";

/// The KMD fields this engine's kernels vary along.
inline constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
inline constexpr std::string_view DTYPE_FIELD = "dtype";

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
