// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "origami/core/config.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/prediction_result.hpp"
#include "origami/core/problem.hpp"
#include "origami/cost_models/cost_model.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"

namespace origami {
namespace selection {

/**
 * @brief Write the debug log line that reports a rejected configuration.
 *
 * @param config Rejected configuration
 * @param rejection Why the cost model does not admit it
 */
ORIGAMI_EXPORT void log_rejection(const config_t& config, rejection_t rejection);

/**
 * @brief Latency a configuration is ranked by.
 *
 * @tparam Model Model tag (@see origami::cost_model)
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Candidate configuration
 * @return double The cost model's predicted latency, or REJECTED_LATENCY (after logging the reason)
 * when the cost model does not admit @p config.
 */
template <class Model>
double score(const problem_t& problem, const hardware_t& hardware, const config_t& config) {
  const evaluation_t evaluation = cost_model_traits<Model>::evaluate(problem, hardware, config);
  if (evaluation.rejection != rejection_t::none) log_rejection(config, evaluation.rejection);
  return evaluation.latency;
}

/**
 * @brief Order configurations by their latencies, best first, and break ties among the leaders.
 *
 * Configurations whose latency is REJECTED_LATENCY are left out unless every configuration has
 * it.  The leading configurations within the tolerance of the best latency (runtime_options'
 * heuristics_variance, or an absolute 1e-9 when that is zero) are then reordered by a sequence
 * of stable sorts, so each later sort takes precedence over the earlier ones:
 *  1. buffer load/store before non-buffer (only when the best configuration carries TensileLite
 *     parameters),
 *  2. higher arithmetic intensity of the macro tile,
 *  3. among the leaders that share the top arithmetic intensity, larger tile along the larger
 *     problem dimension (skipped for M == N),
 *  4. among the same leaders, larger macro tile (M, then N, then K), then occupancy closest to
 *     what the tile grid can use.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs Candidate configurations
 * @param latencies Latency of each configuration, in the order of @p configs
 * @return std::vector<prediction_result_t> Configurations with latencies, best first
 * @throws std::runtime_error if @p configs is empty.
 * @throws std::invalid_argument if @p latencies and @p configs differ in size.
 */
ORIGAMI_EXPORT std::vector<prediction_result_t> rank_by_latency(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    const std::vector<double>& latencies);

}  // namespace selection

/**
 * @brief Rank configurations of one cost model based on predicted performance.
 *
 * @tparam Model Model tag (@see origami::cost_model); its cost_model_traits specialization
 * must be visible.
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @return std::vector<prediction_result_t> Configurations with latencies ranked by performance
 * (best first). If every candidate is rejected, returns a ranking of the rejected configs
 * (max latency) instead of throwing.
 * @throws std::runtime_error if @p configs is empty.
 */
template <class Model>
std::vector<prediction_result_t> rank(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const std::vector<config_t>& configs) {
  std::vector<double> latencies;
  latencies.reserve(configs.size());
  for (const auto& config : configs) {
    latencies.push_back(selection::score<Model>(problem, hardware, config));
  }
  return selection::rank_by_latency(problem, hardware, configs, latencies);
}

/**
 * @brief Rank configurations based on predicted performance.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @param model Model type to use for ranking (gemm or attention)
 * @return std::vector<prediction_result_t> Configurations with latencies ranked by performance
 * (best first). If every candidate is rejected, returns a ranking of the rejected configs
 * (max latency) instead of throwing.
 *
 * Non-temporal availability is read from each config
 * (config_t::non_temporal_a_available / non_temporal_b_available). Call
 * memoize_non_temporal_availability() once when the candidate list is built.
 * @throws std::runtime_error if @p configs is empty.
 */
ORIGAMI_EXPORT std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                                             const hardware_t& hardware,
                                                             const std::vector<config_t>& configs,
                                                             model_t model = model_t::gemm);

/**
 * @brief Record per-macro-tile non-temporal availability on each config.
 *
 * A non-temporal read (cache hint 4) counts only for configs with the same
 * mt.m, mt.n, and mt.k. Call once when @p configs is assembled. rank_configs()
 * then loads the two bools and does not rebuild this map.
 */
ORIGAMI_EXPORT void memoize_non_temporal_availability(std::vector<config_t>& configs);

}  // namespace origami
