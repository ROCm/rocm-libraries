// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/filter.hpp"

#include "origami/gemm.hpp"

namespace origami::nn::filter {
namespace {

int dtype_bits_feasible(data_type_t dt) {
  switch (dt) {
    case data_type_t::Float:
    case data_type_t::XFloat32:
      return 32;
    case data_type_t::Half:
    case data_type_t::BFloat16:
      return 16;
    case data_type_t::Float8:
    case data_type_t::Float8_fnuz:
    case data_type_t::BFloat8:
    case data_type_t::BFloat8_fnuz:
      return 8;
    default:
      return 16;
  }
}

}  // namespace

bool is_kernel_feasible(const problem_t& problem, const config_t& config) {
  const long long M    = static_cast<long long>(problem.size.m);
  const long long N    = static_cast<long long>(problem.size.n);
  const long long K    = static_cast<long long>(problem.size.k);
  const long long B    = static_cast<long long>(problem.batch);
  const long long MT_M = static_cast<long long>(config.mt.m);
  const long long MT_N = static_cast<long long>(config.mt.n);
  const long long MT_K = static_cast<long long>(config.mt.k);
  const long long MI_M = static_cast<long long>(config.mi.m);
  const long long MI_N = static_cast<long long>(config.mi.n);
  const long long MI_K = static_cast<long long>(config.mi.k);
  const int       cha  = config.cache_hints_a;
  const int       chb  = config.cache_hints_b;
  const bool      a_trans =
      (problem.a_transpose == transpose_t::T);
  const bool b_trans =
      (problem.b_transpose == transpose_t::T);
  const long long a_bits = dtype_bits_feasible(problem.a_dtype);
  const long long b_bits = dtype_bits_feasible(problem.b_dtype);

  if (M <= 256 && N <= 256 && K < 1024 && B != 1 && (MT_M < M || MT_N < N)) return false;
  if (MI_M == 1 && MI_N == 1 && MI_K == 64 && M > 2) return false;

  const long long K_mod_128b    = (K * a_bits) % 1024;
  const long long MT_K_mod_128b = (MT_K * a_bits) % 1024;
  if (K_mod_128b == 0 && MT_K_mod_128b == 0) {
    if (M <= MT_M * 2 && (!b_trans) &&
        ((N * b_bits) / std::max<long long>(M * a_bits, 1) > 5)) {
      if (chb != 4) return false;
    } else if (N <= MT_N * 2 && a_trans &&
               ((M * a_bits) / std::max<long long>(N * b_bits, 1) > 5)) {
      if (cha != 4) return false;
    } else {
      if (cha || chb) return false;
    }
  } else if (cha || chb) {
    return false;
  }
  return true;
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
