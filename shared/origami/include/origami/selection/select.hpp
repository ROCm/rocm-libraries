// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

#include "origami/core/config.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/prediction_result.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Based on the provided problem and configs; selects the best config.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs Vector of all possible valid configurations.
 * @param model Model type to use for ranking (gemm or attention)
 * @return prediction_result_t Configurations with best latency.
 */
ORIGAMI_EXPORT prediction_result_t select_config(const problem_t& problem,
                                                 const hardware_t& hardware,
                                                 const std::vector<config_t>& configs,
                                                 model_t model = model_t::gemm);

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
ORIGAMI_EXPORT prediction_result_t select_config_mnk(std::size_t M,
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
 * @param model Model type to use for ranking (gemm or attention)
 * @return std::vector<prediction_result_t> Top K configurations ranked by performance (best first)
 */
ORIGAMI_EXPORT std::vector<prediction_result_t> select_topk_configs(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    std::size_t topk,
    model_t model = model_t::gemm);

}  // namespace origami
