// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/core/config.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/problem.hpp"
#include "origami/cost_models/attention/model.hpp"
#include "origami/cost_models/cost_model.hpp"
#include "origami/hardware.hpp"
#include "origami/method/method.hpp"

namespace origami {

/**
 * @brief Analytical method of flash attention.
 *
 * Attention has no other method, so every prediction mode evaluates this one.
 */
template <>
struct method::evaluator<cost_model::attention, method::analytical> {
  /**
   * @brief Latency, in cycles, of @p config for @p problem on @p hardware.
   */
  static double latency(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) {
    return attention::compute_total_latency(problem, hardware, config);
  }
};

/**
 * @brief The flash-attention cost model.
 */
template <>
struct cost_model_traits<cost_model::attention> {
  static constexpr const char* name = "attention";
  static constexpr model_t model    = model_t::attention;

  /**
   * @brief Check that the macro tile of @p config fits the hardware.
   *
   * @return rejection_t::rf_capacity when the tiles exceed the register file, otherwise
   * rejection_t::lds_capacity when they exceed LDS, otherwise rejection_t::none.
   */
  static rejection_t admit(const problem_t& problem,
                           const hardware_t& hardware,
                           const config_t& config) {
    if (!attention::check_rf_capacity(hardware, config.mt, problem.a_dtype)) {
      return rejection_t::rf_capacity;
    }
    if (!attention::check_lds_capacity(hardware, config.mt, problem.a_dtype)) {
      return rejection_t::lds_capacity;
    }
    return rejection_t::none;
  }

  /**
   * @brief Latency, in cycles, of @p config under the method its prediction_mode names.
   */
  static double latency(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) {
    return method::latency<cost_model::attention>(problem, hardware, config);
  }

  /**
   * @brief Admission and latency of @p config in one call.
   */
  static evaluation_t evaluate(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config) {
    const rejection_t rejection = admit(problem, hardware, config);
    if (rejection != rejection_t::none) return {rejection, REJECTED_LATENCY};
    return {rejection_t::none, latency(problem, hardware, config)};
  }
};

}  // namespace origami
