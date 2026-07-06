// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Uniform cost-model contract used by the ranking driver.
 *
 * A cost model answers two questions about a candidate kernel config for a given
 * problem and hardware: is it feasible, and what is its predicted latency. Every
 * prediction strategy -- analytical estimation, Formocast simulation, and future
 * learned/ML models -- implements this same interface, so the ranking driver and
 * the multi-phase pipeline never depend on which strategy is in use. Concrete
 * models are resolved through @ref get_model and are stateless and thread-safe.
 *
 * Any internal coarse-to-fine *refinement* (a model evaluating its own cheap
 * proxies before its full estimate, reusing per-config intermediate data) is a
 * private implementation detail behind @ref score_candidates. The cross-model
 * pipeline only stacks whole models (e.g. analytical estimation then Formocast
 * simulation); it does not see or drive a model's internal levels, because those
 * levels share data and the pipeline boundary does not.
 */
struct ORIGAMI_EXPORT CostModel {
  virtual ~CostModel() = default;

  /**
   * @brief Short human-readable model name (for logging / debugging).
   *
   * @return const char* Stable, null-terminated model name.
   */
  virtual const char* name() const = 0;

  /**
   * @brief Score a survivor subset of @p configs for ranking.
   *
   * Returns (cost, original config index) pairs sorted ascending by cost, with
   * infeasible / model-disqualified configs dropped. A model may refine across
   * its own internal detail levels here -- evaluating cheap proxies first and
   * reusing per-config intermediate data as it sharpens -- but the returned costs
   * are the model's full-detail estimate for the survivors it keeps. The default
   * implementation scores each survivor once via @ref feasible + @ref latency.
   *
   * @param problem Problem descriptor (M, N, K, dtypes, etc.).
   * @param hardware Hardware characteristics (@see origami::hardware_t).
   * @param configs Full candidate array (indexed by @p survivors).
   * @param survivors Indices into @p configs to score.
   * @return Sorted (cost, index) pairs; lower cost is better.
   */
  virtual scored_configs_t score_candidates(
      const problem_t& problem,
      const hardware_t& hardware,
      const std::vector<config_t>& configs,
      const std::vector<std::size_t>& survivors) const;

  /**
   * @brief Whether the config can run at all (capacity / disqualification gates).
   *
   * @param problem Problem descriptor (M, N, K, dtypes, etc.).
   * @param hardware Hardware characteristics (@see origami::hardware_t).
   * @param config Candidate kernel configuration.
   * @return bool True if the config is feasible and should be scored.
   */
  virtual bool feasible(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) const = 0;

  /**
   * @brief Predicted latency in cycles for a feasible config.
   *
   * May return std::numeric_limits<double>::max() to signal that the config is
   * disqualified by the model itself; the ranking driver drops such configs.
   *
   * @param problem Problem descriptor (M, N, K, dtypes, etc.).
   * @param hardware Hardware characteristics (@see origami::hardware_t).
   * @param config Candidate kernel configuration.
   * @return double Predicted latency in cycles (lower is better).
   */
  virtual double latency(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config) const = 0;
};

/**
 * @brief Resolve the cost model for a (model, target, fidelity) triple.
 *
 * The registry owns one stateless instance per supported combination. Estimation
 * is currently target-agnostic (the analytical model); simulation is provided by
 * the tensilelite Formocast model. Requesting an unregistered combination throws.
 *
 * @param model Operation model (gemm or attention).
 * @param target Target backend whose kernels are being modeled.
 * @param fidelity Prediction fidelity (estimation or simulation).
 * @return const CostModel& Reference to the resolved, owned model instance.
 */
ORIGAMI_EXPORT const CostModel& get_model(model_t model,
                                          target_t target,
                                          prediction_modes_t fidelity);

}  // namespace origami
