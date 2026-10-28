// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "origami/hardware.hpp"
#include "origami/nn/boost/loaded_model.hpp"
#include "origami/nn/boost/types.hpp"
#include "origami/types.hpp"

namespace origami::nn::boost {

/**
 * @brief Re-rank Origami's analytical results in place (inference_mode_t::analytical_rerank).
 *
 * Checks the problem (booster_t::check_problem), takes the first top_n results @p source accepts
 * as candidates (all results, with no kernel fields, when @p source is null), ranks them
 * (booster_t::rank) and on success moves them to the front of @p results in the model's order.
 * Reports the outcome to @p source.
 *
 * @return true if @p results were re-ordered; false leaves them unchanged.
 */
bool rerank_configs(const booster_t&                  booster,
                    const problem_t&                  problem,
                    const hardware_t&                 hardware,
                    std::vector<prediction_result_t>& results,
                    rerank_source_t*                  source);

}  // namespace origami::nn::boost
