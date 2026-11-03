// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Internal header: the pieces of gemm::compute_total_latency() that the leveled
// estimation cascade calls on their own. Not installed and not exported.

#pragma once

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

/**
 * @brief Context-free rejection rules applied by compute_total_latency.
 *
 * Returns true exactly when compute_total_latency() rejects @p config (returns the maximum
 * latency) without building a context or running a latency model: invalid configs,
 * heuristic rejections, the ANALYTICAL_GEMM_PICK filter and, for the estimation model,
 * its short-circuit rules.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return true if the config is rejected.
 */
bool fast_reject(const problem_t& problem, const hardware_t& hardware, const config_t& config);

/**
 * @brief Compute the estimation-model total latency from a prebuilt context.
 *
 * This is the part of compute_total_latency() that runs after the context is built. It does
 * not apply fast_reject(); callers are expected to filter first.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context built from @p problem, @p hardware and @p config.
 * @return double Latency in cycles.
 */
double estimation_latency_from_context(const problem_t& problem,
                                       const hardware_t& hardware,
                                       const config_t& config,
                                       const context_t& context);

}  // namespace gemm
}  // namespace origami
