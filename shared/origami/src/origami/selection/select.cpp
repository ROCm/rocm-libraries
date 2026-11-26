// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <vector>

#include "origami/core/data_type.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/prediction_result.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/selection/ranking.hpp"
#include "origami/selection/select.hpp"

namespace origami {

prediction_result_t select_config(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const std::vector<config_t>& configs,
                                  model_t model) {
  auto ranked_configs = rank_configs(problem, hardware, configs, model);

  // Return the top configuration
  return ranked_configs[0];
}

prediction_result_t select_config_mnk(size_t M,
                                      size_t N,
                                      size_t K,
                                      const hardware_t& hardware,
                                      const std::vector<config_t>& configs) {
  // Create a default problem_t with the provided M, N, K and reasonable defaults
  problem_t problem;
  problem.size.m          = M;
  problem.size.n          = N;
  problem.size.k          = K;
  problem.batch           = 1;
  problem.a_transpose     = transpose_t::T;     // Default to T
  problem.b_transpose     = transpose_t::N;     // Default to N
  problem.a_dtype         = data_type_t::Half;  // Default to fp16
  problem.b_dtype         = data_type_t::Half;  // Default to fp16
  problem.c_dtype         = data_type_t::Half;  // Default to fp16
  problem.d_dtype         = data_type_t::Half;  // Default to fp16
  problem.mi_dtype        = data_type_t::Half;  // Default to fp16
  problem.a_mx_block_size = 0;                  // Default MX block size
  problem.b_mx_block_size = 0;                  // Default MX block size

  // Use the existing select_config function with the constructed problem
  return select_config(problem, hardware, configs);
}

std::vector<prediction_result_t> select_topk_configs(const problem_t& problem,
                                                     const hardware_t& hardware,
                                                     const std::vector<config_t>& configs,
                                                     std::size_t topk,
                                                     model_t model) {
  // Use rank_configs to get configurations with latencies ranked by performance
  auto ranked_configs = rank_configs(problem, hardware, configs, model);

  // Return only the top K configurations
  std::vector<prediction_result_t> topk_configs;
  size_t count = std::min(topk, ranked_configs.size());
  topk_configs.reserve(count);
  for (size_t i = 0; i < count; ++i) { topk_configs.push_back(ranked_configs[i]); }
  return topk_configs;
}

}  // namespace origami
