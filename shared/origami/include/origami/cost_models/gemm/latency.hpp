// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/core/config.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/problem.hpp"
#include "origami/cost_models/cost_model.hpp"
#include "origami/cost_models/gemm/compute.hpp"
#include "origami/cost_models/gemm/tile.hpp"
#include "origami/hardware.hpp"
#include "origami/method/method.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

/**
 * @brief Decides whether the model scores @p config for @p problem.
 *
 * The rules run in the order of the GEMM enumerators of rejection_t.  The shape and non-temporal
 * rules belong to the tile model; a config evaluated by the simulator
 * (prediction_modes_t::simulation) or by the W4A16 decode model only has to pass the first three.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return rejection_t none, or the first rule the config fails.
 */
ORIGAMI_EXPORT rejection_t admit(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config);

// ---------------------------------------------------------------------------
// Parallel reduction kernel
// ---------------------------------------------------------------------------

/**
 * @brief Compute the latency of the parallel reduction kernel (separate kernel launch).
 *
 * For parallel reduction, the GEMM kernel writes partials to workspace and a second
 * kernel is launched to read all partials, accumulate, and write the final output.
 * This function estimates the latency of that second kernel including launch overhead.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Latency in cycles (0 if no parallel reduction).
 */
ORIGAMI_EXPORT double compute_parallel_reduction_latency(const problem_t& problem,
                                                         const hardware_t& hardware,
                                                         const config_t& config,
                                                         const context_t& context);

// ---------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------

/**
 * @brief Latency of an admitted config under the analytical method.
 *
 * Every scheduling round runs one tile per CU; the kernel launch comes on top, and a second
 * kernel when split-K reduces in parallel.  A W4A16 decode kernel has its own model instead of
 * the tile model.  Does not run the admission rules (@see admit).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_analytical_latency(const problem_t& problem,
                                                 const hardware_t& hardware,
                                                 const config_t& config);

/**
 * @brief Latency of an admitted config under the simulation method (the Formocast simulator).
 *
 * Does not run the admission rules (@see admit).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in microseconds, as Formocast reports it (not cycles).
 */
ORIGAMI_EXPORT double compute_simulated_latency(const problem_t& problem,
                                                const hardware_t& hardware,
                                                const config_t& config);

/**
 * @brief Runs the admission rules and, for an admitted config, the method its prediction_mode
 * names.
 *
 * This is compute_total_latency with the rejection reason; it looks the config's heuristics up
 * once for both steps.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return evaluation_t The first rule the config fails and REJECTED_LATENCY, or none and the
 * latency.
 */
ORIGAMI_EXPORT evaluation_t evaluate(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config);

/**
 * @brief Latency of a config under the method its prediction_mode names.
 *
 * A config the admission rules reject (@see admit) gets the maximum latency.  The analytical
 * model is the latency of one timestep multiplied by the number of timesteps
 * (@see compute_tile_latency).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration. Non-temporal availability is read from
 * config.non_temporal_a_available and config.non_temporal_b_available.
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_total_latency(const problem_t& problem,
                                            const hardware_t& hardware,
                                            const config_t& config);

}  // namespace gemm
}  // namespace origami

namespace origami {

// ---------------------------------------------------------------------------
// GEMM as a cost model
// ---------------------------------------------------------------------------

/**
 * @brief Analytical method of GEMM.
 */
template <>
struct method::evaluator<cost_model::gemm, method::analytical> {
  /**
   * @brief Latency, in cycles, of @p config for @p problem on @p hardware.
   */
  static double latency(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) {
    return gemm::compute_analytical_latency(problem, hardware, config);
  }
};

/**
 * @brief Simulation method of GEMM (the Formocast simulator).
 */
template <>
struct method::evaluator<cost_model::gemm, method::simulation> {
  /**
   * @brief Latency, in microseconds, of @p config for @p problem on @p hardware.
   */
  static double latency(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) {
    return gemm::compute_simulated_latency(problem, hardware, config);
  }
};

/**
 * @brief The GEMM cost model.
 */
template <>
struct cost_model_traits<cost_model::gemm> {
  static constexpr const char* name = "gemm";
  static constexpr model_t model    = model_t::gemm;

  /**
   * @brief Check that the macro tile of @p config fits the hardware, then the model's own rules.
   *
   * @return rejection_t::lds_capacity when the A and B tiles exceed LDS, otherwise what
   * gemm::admit reports.
   */
  static rejection_t admit(const problem_t& problem,
                           const hardware_t& hardware,
                           const config_t& config) {
    if (!gemm::check_lds_capacity(
            hardware, config.mt, gemm::lds_dtype_a(problem, hardware), problem.b_dtype)) {
      return rejection_t::lds_capacity;
    }
    return gemm::admit(problem, hardware, config);
  }

  /**
   * @brief Latency of an admitted @p config, in the unit of the method its
   * prediction_mode names.
   */
  static double latency(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) {
    return method::latency<cost_model::gemm>(problem, hardware, config);
  }

  /**
   * @brief Admission and latency of @p config in one call.
   */
  static evaluation_t evaluate(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config) {
    if (!gemm::check_lds_capacity(
            hardware, config.mt, gemm::lds_dtype_a(problem, hardware), problem.b_dtype)) {
      return {rejection_t::lds_capacity, REJECTED_LATENCY};
    }
    return gemm::evaluate(problem, hardware, config);
  }
};

}  // namespace origami
