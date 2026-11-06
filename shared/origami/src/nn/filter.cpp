// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/filter.hpp"

#include "origami/gemm.hpp"

namespace origami::nn::filter {

bool is_kernel_feasible(const problem_t& problem, const config_t& config) {
  return gemm::is_config_feasible(problem, config);
}

filter_result_t filter_configs(const problem_t& problem,
                               const hardware_t& hardware,
                               const std::vector<config_t>& configs) {
  filter_result_t result;
  result.feasible_indices.reserve(configs.size());
  result.rejected_indices.reserve(configs.size());

  for (std::size_t i = 0; i < configs.size(); ++i) {
    const auto& config = configs[i];
    if (!gemm::check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype) ||
        !is_kernel_feasible(problem, config)) {
      result.rejected_indices.push_back(i);
    } else {
      result.feasible_indices.push_back(i);
    }
  }
  return result;
}

}  // namespace origami::nn::filter
