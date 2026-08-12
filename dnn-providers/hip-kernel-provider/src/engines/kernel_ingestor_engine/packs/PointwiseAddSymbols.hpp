// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string_view>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @file PointwiseAddSymbols.hpp
 * @brief The symbol names this pack's descriptors resolve through.
 *
 * Each constant is a placeholder for a declarative descriptor field once the
 * follow-up RFCs land.
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
 * Prefixed per RFC 0017's global-uniqueness rule. Absent from EngineNames.hpp's
 * registry: registered at construction, not at build time.
 */
inline constexpr std::string_view ENGINE_NAME = "hipkernel:PointwiseAdd";

/// The KMD fields this engine's kernels vary along.
inline constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
inline constexpr std::string_view DTYPE_FIELD = "dtype";

/**
 * @brief Tokens the graph matcher binds for the dispatch handler to read.
 *
 * Native stand-in for the `$`-prefixed names a descriptor's dispatch formulas
 * would reference (RFC 0017 §5).
 */
inline constexpr std::string_view INPUT_A_TOKEN = "pointwise_add.input_a.uid";
inline constexpr std::string_view INPUT_B_TOKEN = "pointwise_add.input_b.uid";
inline constexpr std::string_view OUTPUT_TOKEN = "pointwise_add.output.uid";

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
