// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>

#include "origami/targets/triton/heuristics.hpp"

namespace origami::triton {

heuristics_database_t::heuristics_database_t() {
  initialize_defaults();
}

heuristics_database_t& heuristics_database_t::get_instance() {
  static heuristics_database_t instance;
  return instance;
}

heuristic_params_t heuristics_database_t::lookup(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config) const {

  heuristic_params_t result = default_params_;

  std::vector<std::pair<size_t, const heuristic_params_t*>> matches;
  for (const auto& [key, params] : entries_) {
    if (key.matches(problem, hardware, config))
      matches.push_back({key.specificity(), &params});
  }

  std::sort(matches.begin(), matches.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  for (const auto& [spec, params] : matches)
    result.merge_with(*params);

  return result;
}

void heuristics_database_t::add_entry(const heuristic_key_t& key,
                                      const heuristic_params_t& params) {
  entries_.push_back({key, params});
}

void heuristics_database_t::initialize_defaults() {
  // Heuristic 1: prefer 256x256x64 on all architectures.
  // Empirical 262K-shape benchmarks show 256x256x64 has the best wavefront
  // utilization and occupancy balance. A 5% latency discount makes it win
  // close races without overriding tiles that are clearly better.
  {
    auto key = make_tile_key(256, 256, 64);
    heuristic_params_t params;
    params.weight_tile_total = 0.95;
    add_entry(key, params);
  }

  // Heuristic 2: penalize 256x256x128 on gfx950.
  // The analytical model's arithmetic-intensity tie-breaker incorrectly
  // favors K_block=128 on gfx950. Register pressure and MFMA pipeline
  // differences make K_block=64 the better choice for the majority of
  // shapes. Apply a 10% latency penalty to steer selection away.
  {
    auto key = make_tile_key(256, 256, 128);
    key.arch = hardware_t::architecture_t::gfx950;
    heuristic_params_t params;
    params.weight_tile_total = 1.10;
    add_entry(key, params);
  }
}

}  // namespace origami::triton
