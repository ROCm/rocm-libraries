// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "origami/attention.hpp"
#include "origami/estimation_leveled.hpp"
#include "origami/simulator/tensilelite/formocast_simulator.hpp"
#include "origami/gemm.hpp"
#include "origami/model.hpp"

namespace origami {

namespace {

/**
 * @brief GEMM cost model adapter.
 *
 * Thin bridge between the @ref CostModel interface and the @c gemm:: functions.
 * The fidelity is fixed at construction so a single class serves every GEMM
 * phase. The estimation scoring cascade (the coarse-to-fine leveled walk) lives
 * in @ref gemm::score_estimation_leveled; this adapter just dispatches to it.
 * Simulation is single-level (Formocast).
 */
class GemmModel : public CostModel {
 public:
  explicit GemmModel(prediction_modes_t fidelity) : fidelity_(fidelity) {}

  bool feasible(const problem_t& problem,
                const hardware_t& hardware,
                const config_t& config) const override {
    // tensilelite candidates are LDS-validated by the library before they reach
    // origami, so the per-config capacity check is redundant for that target.
    // Other targets may feed unvalidated configs, so they keep the check.
    if (config.target == target_t::tensilelite) return true;
    return gemm::check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype);
  }

  double latency(const problem_t& problem,
                 const hardware_t& hardware,
                 const config_t& config) const override {
    // Estimation and simulation are distinct models with their own latency entry.
    if (fidelity_ == prediction_modes_t::simulation) {
      return gemm::compute_formocast_latency(problem, hardware, config);
    }
    return gemm::compute_total_latency(problem, hardware, config);
  }

  const char* name() const override {
    return fidelity_ == prediction_modes_t::simulation ? "gemm/simulation" : "gemm/estimation";
  }

  scored_configs_t score_candidates(
      const problem_t& problem,
      const hardware_t& hardware,
      const std::vector<config_t>& configs,
      const std::vector<std::size_t>& survivors) const override {
    // Simulation is single-level: score each survivor once (Formocast).
    if (fidelity_ == prediction_modes_t::simulation) {
      return CostModel::score_candidates(problem, hardware, configs, survivors);
    }
    // Estimation: the leveled coarse-to-fine cascade lives with the GEMM model.
    return gemm::score_estimation_leveled(problem, hardware, configs, survivors);
  }

 private:
  prediction_modes_t fidelity_;
};

/**
 * @brief Attention cost model adapter (analytical estimation).
 */
class AttentionModel : public CostModel {
 public:
  bool feasible(const problem_t& problem,
                const hardware_t& hardware,
                const config_t& config) const override {
    return attention::check_rf_capacity(hardware, config.mt, problem.a_dtype) &&
           attention::check_lds_capacity(hardware, config.mt, problem.a_dtype);
  }

  double latency(const problem_t& problem,
                 const hardware_t& hardware,
                 const config_t& config) const override {
    return attention::compute_total_latency(problem, hardware, config);
  }

  const char* name() const override { return "attention"; }
};

/// Registry key: (model, target, fidelity).
using model_key_t = std::tuple<model_t, target_t, prediction_modes_t>;

/// Resolve a (model, target, fidelity) triple to its owned model, or nullptr if
/// unregistered. Shared by the throwing get_model and the non-throwing has_model so
/// the registry has a single definition.
const CostModel* find_model(model_t model, target_t target, prediction_modes_t fidelity) {
  // Stateless, thread-safe singletons. One instance per distinct behavior is
  // shared across every (target) it applies to.
  static const GemmModel      gemm_estimation{prediction_modes_t::estimation};
  static const GemmModel      gemm_simulation{prediction_modes_t::simulation};
  static const AttentionModel attention_estimation{};

  // All backend targets the registry currently spans. Estimation is
  // target-agnostic today, so every target maps to the same analytical model.
  static constexpr target_t all_targets[] = {target_t::generic,
                                             target_t::tensilelite,
                                             target_t::rocroller,
                                             target_t::triton,
                                             target_t::composable_kernel};

  static const std::map<model_key_t, const CostModel*> registry = [] {
    std::map<model_key_t, const CostModel*> table;
    for (target_t t : all_targets) {
      // Analytical estimation is currently identical across targets.
      table[{model_t::gemm, t, prediction_modes_t::estimation}]      = &gemm_estimation;
      // Attention currently exposes a single fidelity; map both so an attention
      // request never depends on the prediction mode.
      table[{model_t::attention, t, prediction_modes_t::estimation}] = &attention_estimation;
      table[{model_t::attention, t, prediction_modes_t::simulation}] = &attention_estimation;
    }
    // Simulation is provided by the tensilelite Formocast model.
    table[{model_t::gemm, target_t::tensilelite, prediction_modes_t::simulation}] =
        &gemm_simulation;
    return table;
  }();

  auto it = registry.find({model, target, fidelity});
  return it == registry.end() ? nullptr : it->second;
}

}  // namespace

scored_configs_t CostModel::score_candidates(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    const std::vector<std::size_t>& survivors) const {
  // Default: score each survivor once at full detail, dropping infeasible /
  // disqualified configs. Single-level models (and the simulation fidelity) use
  // this directly; leveled models override to refine internally.
  scored_configs_t scored;
  scored.reserve(survivors.size());
  for (std::size_t idx : survivors) {
    const config_t& config = configs[idx];
    if (!feasible(problem, hardware, config)) continue;
    const double cost = latency(problem, hardware, config);
    if (cost != std::numeric_limits<double>::max()) scored.emplace_back(cost, idx);
  }
  std::stable_sort(scored.begin(), scored.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  return scored;
}

const CostModel& get_model(model_t model, target_t target, prediction_modes_t fidelity) {
  if (const CostModel* m = find_model(model, target, fidelity)) { return *m; }
  throw std::runtime_error(
      "origami::get_model: no cost model registered for (model=" + model_to_string(model) +
      ", target=" + target_to_string(target) +
      ", fidelity=" + prediction_modes_to_string(fidelity) + ")");
}

bool has_model(model_t model, target_t target, prediction_modes_t fidelity) {
  return find_model(model, target, fidelity) != nullptr;
}

}  // namespace origami
