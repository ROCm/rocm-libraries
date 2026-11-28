// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "origami/core/config.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/logger.hpp"
#include "origami/core/math.hpp"
#include "origami/core/prediction_result.hpp"
#include "origami/core/problem.hpp"
#include "origami/core/runtime_options.hpp"
#include "origami/cost_models/attention/traits.hpp"
#include "origami/cost_models/cost_model.hpp"
#include "origami/cost_models/gemm/latency.hpp"
#include "origami/hardware.hpp"
#include "origami/selection/ranking.hpp"

namespace origami {
namespace {

/// A latency and the candidate it belongs to, without copying the candidate.
struct scored_config_t {
  double latency;
  std::reference_wrapper<const config_t> config;
};

// Pair every config with its latency and keep the ones that are not rejected, in input order.
// When every candidate is rejected, keep them all so the caller still gets a usable kernel
// rather than failing.
std::vector<scored_config_t> admitted_configs(const std::vector<config_t>& configs,
                                              const std::vector<double>& latencies) {
  std::vector<scored_config_t> valid_configs;
  std::vector<scored_config_t> invalid_configs;
  valid_configs.reserve(configs.size());
  invalid_configs.reserve(configs.size());

  for (size_t i = 0; i < configs.size(); ++i) {
    const double latency = latencies[i];

    if (latency != REJECTED_LATENCY) {
      valid_configs.push_back({latency, std::cref(configs[i])});
    } else {
      invalid_configs.push_back({latency, std::cref(configs[i])});
    }
  }

  if (valid_configs.empty()) { valid_configs = std::move(invalid_configs); }
  return valid_configs;
}

// Arithmetic intensity of a macro tile.
// Flops = 2 * MT_M * MT_N * MT_K, Memory traffic = MT_M*MT_K + MT_K*MT_N + MT_M*MT_N
double arithmetic_intensity(const config_t& config) {
  const auto MT_M = config.mt.m;
  const auto MT_N = config.mt.n;
  const auto MT_K = config.mt.k;

  const double flops          = static_cast<double>(2ull * MT_M * MT_N * MT_K);
  const double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

  if (memory_traffic == 0.0) return 0.0;
  return flops / memory_traffic;
}

// Occupancy the workload can use: grid waves/CU, clamped.  Small grids favor
// low occupancy (fatter waves); saturated grids favor high occupancy (latency
// hiding).
double occupancy_need(const problem_t& problem, const hardware_t& hardware, const config_t& c) {
  constexpr double CAP  = 4.0;
  const size_t occ_n_cu = std::max<size_t>(hardware.N_CU, 1);
  const size_t tiles    = math::safe_ceil_div(problem.size.m, std::max<size_t>(c.mt.m, 1)) *
                       math::safe_ceil_div(problem.size.n, std::max<size_t>(c.mt.n, 1)) *
                       std::max<size_t>(problem.batch, 1);
  const double waves_per_cu = std::ceil(static_cast<double>(tiles) / static_cast<double>(occ_n_cu));
  return std::clamp(waves_per_cu, 1.0, CAP);
}

// Number of leading results whose latency is within tolerance of the best one.
size_t count_latency_ties(const std::vector<prediction_result_t>& results) {
  double best_latency = results.front().latency;
  size_t num_the_same = 0;

  constexpr double epsilon = 1e-9;
  // variance is set through environment variable ORIGAMI_HEURISTICS_VARIANCE
  const double top_N_heuristic = origami::runtime_options::get().heuristics_variance;

  for (const auto& res : results) {
    bool within_top;
    const double diff = std::abs(res.latency - best_latency);

    if (top_N_heuristic <= epsilon) {
      // Absolute tolerance path
      within_top = diff < epsilon;
    } else {
      // Relative tolerance path (guard denom)
      const double denom = std::max(std::abs(best_latency), epsilon);
      // If it's within top_N_heuristic%, include it.
      within_top = (diff / denom) < top_N_heuristic;
    }

    if (within_top)
      ++num_the_same;
    else
      break;
  }
  return num_the_same;
}

// Number of leading results, out of the first num_the_same, that have the arithmetic intensity
// of the first one.
size_t count_intensity_ties(const std::vector<prediction_result_t>& results, size_t num_the_same) {
  double first_ai    = arithmetic_intensity(results.front().config);
  size_t num_same_ai = 1;
  for (size_t i = 1; i < num_the_same; ++i) {
    double current_ai = arithmetic_intensity(results[i].config);
    if (std::abs(current_ai - first_ai) < 1e-6) {
      num_same_ai++;
    } else {
      break;
    }
  }
  return num_same_ai;
}

// Buffer load and store first; a non-buffer load weighs more than a non-buffer store.
bool prefers_buffer_access(const prediction_result_t& a, const prediction_result_t& b) {
  auto rank = [](bool bufferLoad, bool bufferStore) {
    return (static_cast<int>(!bufferLoad) << 1) | static_cast<int>(!bufferStore);
  };
  return rank(a.config.tensile().buffer_load, a.config.tensile().buffer_store) <
         rank(b.config.tensile().buffer_load, b.config.tensile().buffer_store);
}

// Highest arithmetic intensity first.
bool has_higher_intensity(const prediction_result_t& a, const prediction_result_t& b) {
  return arithmetic_intensity(a.config) > arithmetic_intensity(b.config);
}

// Larger MT_M first, then larger MT_N, then larger MT_K.  For a bit-identical macro tile, the
// occupancy variant that best matches the workload (this never changes which tile wins).
bool has_larger_tile_then_closer_occupancy(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const prediction_result_t& a,
                                           const prediction_result_t& b) {
  if (a.config.mt.m != b.config.mt.m) return a.config.mt.m > b.config.mt.m;
  if (a.config.mt.n != b.config.mt.n) return a.config.mt.n > b.config.mt.n;
  if (a.config.mt.k != b.config.mt.k) return a.config.mt.k > b.config.mt.k;
  // The need of b is the same: both have the same macro tile.
  const double need = occupancy_need(problem, hardware, a.config);
  return std::abs(static_cast<double>(a.config.occupancy) - need) <
         std::abs(static_cast<double>(b.config.occupancy) - need);
}

// Reorder the leading results whose latencies tie.  The criteria are applied as successive
// stable sorts, so a later sort takes precedence and an earlier one decides among the results
// the later ones leave tied.
void break_ties(const problem_t& problem,
                const hardware_t& hardware,
                std::vector<prediction_result_t>& results) {
  const size_t num_the_same = count_latency_ties(results);
  if (num_the_same <= 1) return;
  const auto latency_ties_end = results.begin() + num_the_same;

  // Buffer load/store preference is the lowest-priority tie-break, so it is sorted first and
  // applies within every subgroup the later criteria leave tied.
  if (results.front().config.has_tensile_params()) {
    std::stable_sort(results.begin(), latency_ties_end, prefers_buffer_access);
  }

  std::stable_sort(results.begin(), latency_ties_end, has_higher_intensity);

  // The remaining criteria apply to the results that still tie on arithmetic intensity.
  const size_t num_same_ai = count_intensity_ties(results, num_the_same);
  if (num_same_ai <= 1) return;
  const auto intensity_ties_end = results.begin() + num_same_ai;

  // Final tie-breaker: consistently prefer larger tiles, which makes the selection
  // deterministic regardless of input order.
  std::stable_sort(results.begin(),
                   intensity_ties_end,
                   [&](const prediction_result_t& a, const prediction_result_t& b) {
                     return has_larger_tile_then_closer_occupancy(problem, hardware, a, b);
                   });
}

}  // namespace

namespace selection {

void log_rejection(const config_t& config, rejection_t rejection) {
  OLOG_DEBUG("  Config MT=(" << config.mt.m << "," << config.mt.n << "," << config.mt.k << ") MI=("
                             << config.mi.m << "," << config.mi.n << "," << config.mi.k
                             << ") REJECTED: " << to_string(rejection));
}

std::vector<prediction_result_t> rank_by_latency(const problem_t& problem,
                                                 const hardware_t& hardware,
                                                 const std::vector<config_t>& configs,
                                                 const std::vector<double>& latencies) {
  if (configs.empty()) { throw std::runtime_error("No configurations provided."); }
  if (latencies.size() != configs.size()) {
    throw std::invalid_argument("One latency per configuration is required.");
  }

  std::vector<scored_config_t> valid_configs = admitted_configs(configs, latencies);

  std::stable_sort(valid_configs.begin(), valid_configs.end(), [](const auto& a, const auto& b) {
    return a.latency < b.latency;
  });

  std::vector<prediction_result_t> results;
  results.reserve(valid_configs.size());
  std::transform(valid_configs.begin(),
                 valid_configs.end(),
                 std::back_inserter(results),
                 [&](const auto& r) -> prediction_result_t { return {r.latency, r.config.get()}; });

  break_ties(problem, hardware, results);

  OLOG_DEBUG("rank_configs selected MT=(" << results[0].config.mt.m << "," << results[0].config.mt.n
                                          << "," << results[0].config.mt.k << ")"
                                          << " latency=" << results[0].latency);

  return results;
}

}  // namespace selection

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs,
                                              model_t model) {
  return cost_model::dispatch(
      model, [&](auto tag) { return rank<decltype(tag)>(problem, hardware, configs); });
}

void memoize_non_temporal_availability(std::vector<config_t>& configs) {
  // Same macro tile only. An NT-B kernel at 128x256 must not mark a 256x256
  // temporal kernel as having a non-temporal sibling.
  std::map<std::tuple<size_t, size_t, size_t>, std::pair<bool, bool>> nt_by_tile;
  for (const auto& config : configs) {
    auto& flags = nt_by_tile[{config.mt.m, config.mt.n, config.mt.k}];
    flags.first |= config.cache_hints_a == 4;
    flags.second |= config.cache_hints_b == 4;
  }

  for (auto& config : configs) {
    const auto& flags               = nt_by_tile.at({config.mt.m, config.mt.n, config.mt.k});
    config.non_temporal_a_available = flags.first;
    config.non_temporal_b_available = flags.second;
  }
}

}  // namespace origami
