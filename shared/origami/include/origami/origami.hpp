// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <set>
#include <tuple>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Based on the provided problem and configs; selects the best config.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs Vector of all possible valid configurations.
 * @return prediction_result_t Configurations with best latency.
 */
prediction_result_t select_config(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const std::vector<config_t>& configs);

/**
 * @brief Select best workgroup-mapping for the given tile size.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro-tile of the kernel.
 * @param mi Matrix-instruction of the kernel.
 * @param wgms List of possible workgroup-mappings.
 * @return std::tuple<size_t, size_t>
 */
std::tuple<size_t, size_t> select_workgroup_mapping(const problem_t& problem,
                                                    const hardware_t& hardware,
                                                    const config_t& config,
                                                    size_t skGrid);

/**
 * @brief Rank configurations based on predicted performance.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @return std::vector<prediction_result_t> Configurations with latencies ranked by performance
 * (best first)
 */
std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs);

/**
 * @brief Select best configuration based only on M, N, K dimensions with default settings.
 *
 * @param M Problem dimension M
 * @param N Problem dimension N
 * @param K Problem dimension K
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations
 * @return prediction_result_t Configurations with best latency.
 */
prediction_result_t select_config_mnk(std::size_t M,
                                      std::size_t N,
                                      std::size_t K,
                                      const hardware_t& hardware,
                                      const std::vector<config_t>& configs);

/**
 * @brief Select top K configurations based on performance ranking.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @param topk Number of top configurations to return
 * @return std::vector<prediction_result_t> Top K configurations ranked by performance (best first)
 */
std::vector<prediction_result_t> select_topk_configs(const problem_t& problem,
                                                     const hardware_t& hardware,
                                                     const std::vector<config_t>& configs,
                                                     int topk);

/**
 * @brief Given a latency, compute the achieved throughput in gflops.
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param problem Problem description (M, N, K, etc.)
 * @param latency Kernel latency.
 * @return double Throughput in gflops/s.
 */
double compute_perf_gflops(const hardware_t& hardware,
                           const problem_t& problem,
                           const double latency);

}  // namespace origami