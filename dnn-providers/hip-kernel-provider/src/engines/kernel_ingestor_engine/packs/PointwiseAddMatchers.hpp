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
 * Reads only graph facts, so the ingestor evaluates it once per (graph, device) and one
 * failure disqualifies every kernel in the pack without any per-kernel work.
 */
bool pointwiseAddGraphMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                              hipdnn_plugin_sdk::ingestor::BoundTokens& bound);

/**
 * @brief Kernel-scoped applicability: does this kernel's dtype match the graph's?
 *
 * Reads kernel metadata, so it is evaluated once per candidate and disqualifies that
 * candidate alone.
 *
 * This check is what a prebuilt-kernel system cannot omit: the graph-level gate above
 * accepts any dtype the pack serves, so without pinning the kernel's baked dtype against
 * the graph's, an f32 graph could be handed to an f16 binary — which does not fail
 * outright, it silently returns wrong numbers.
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
 * The matcher proves the graph is a pointwise add and, in doing so, identifies which
 * tensor is which. Dispatch needs those same uids to bind its arguments, so it asks for
 * them here rather than re-deriving them from the graph with a second, possibly
 * divergent, notion of what the graph shape is.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
PointwiseAddBinding pointwiseAddBinding(const hipdnn_plugin_sdk::ingestor::BoundTokens& bound);

/// @brief Registers this pack's matchers and scorer under their symbol names.
///
/// Called once, from the pack's translation unit, rather than at static-init time: an
/// engine that is never constructed should not have mutated process-wide registries.
void registerPointwiseAddMatchers();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
