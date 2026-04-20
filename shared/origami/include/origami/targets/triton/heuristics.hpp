// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/heuristics.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Triton-specific heuristics database.
 *
 * Separate from the main heuristics_database_t which contains Tensile/CMS
 * entries. This database holds tile preferences and latency adjustments
 * that are specific to Triton-generated kernels.
 *
 * Reuses the same heuristic_key_t / heuristic_params_t infrastructure so
 * entries are expressed identically — just stored in a Triton-only container.
 *
 * Queried by select_config when ranking Triton candidates. Entries here
 * do not affect Tensile kernel selection.
 */
class triton_heuristics_database_t {
 public:
  /**
   * @brief Lookup Triton-specific heuristic params for a config.
   *
   * Returns default (neutral) params if no entry matches.
   */
  heuristic_params_t lookup(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) const;

  /**
   * @brief Add a heuristic entry.
   */
  void add_entry(const heuristic_key_t& key, const heuristic_params_t& params);

  /**
   * @brief Get the global Triton heuristics database instance.
   */
  static triton_heuristics_database_t& get_instance();

 private:
  triton_heuristics_database_t();

  std::vector<std::pair<heuristic_key_t, heuristic_params_t>> entries_;
  heuristic_params_t default_params_;

  void initialize_defaults();
};

/**
 * @brief Convenience function to get Triton-specific heuristic params.
 */
inline heuristic_params_t get_triton_heuristic_params(const problem_t& problem,
                                                      const hardware_t& hardware,
                                                      const config_t& config) {
  return triton_heuristics_database_t::get_instance().lookup(problem, hardware, config);
}

}  // namespace origami
