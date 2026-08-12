// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @brief Graph-scoped applicability: is this a single-node pointwise ADD over
 *        1-element tensors?
 *
 * Reads only graph facts, so it is evaluated once per (graph, device); a failure
 * disqualifies every kernel in the pack without per-kernel work.
 */
bool pointwiseAddGraphMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                              hipdnn_plugin_sdk::ingestor::BoundTokens& bound);

/**
 * @brief Kernel-scoped applicability: does this kernel's dtype match the graph's?
 *
 * Reads kernel metadata, so it is evaluated once per candidate and disqualifies that
 * candidate alone. Without this check, the graph-level gate above would accept any
 * dtype the pack serves and an f32 graph could reach an f16 binary, which returns
 * wrong numbers rather than failing.
 */
bool pointwiseAddKernelMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                               const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// @brief Ranks kernels for a pointwise add.
double pointwiseAddScore(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                         const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// @brief The tensor uids a matched pointwise-add graph binds, in argument order.
struct PointwiseAddBinding
{
    int64_t inputA = 0;
    int64_t inputB = 0;
    int64_t output = 0;
};

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * The matcher proves the graph is a pointwise add and identifies which tensor is
 * which; dispatch reads those same uids here rather than re-deriving them.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
PointwiseAddBinding pointwiseAddBinding(const hipdnn_plugin_sdk::ingestor::BoundTokens& bound);

/// @brief Registers this pack's matchers and scorer under their symbol names.
///
/// Called once, from the pack's translation unit, rather than at static-init time.
/// Atomic: if one symbol is already taken, whichever of the others this call already
/// installed is unregistered before it rethrows.
void registerPointwiseAddMatchers();

/// @brief Undoes registerPointwiseAddMatchers(), for a caller that must roll this pack
/// back after a later registration step of its own fails.
void unregisterPointwiseAddMatchers();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
