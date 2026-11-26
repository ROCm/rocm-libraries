// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "origami/core/runtime_options.hpp"

namespace origami {

const char* env_var_t::read() const {
  const char* value = std::getenv(name);
  return (value == nullptr && legacy != nullptr) ? std::getenv(legacy) : value;
}

runtime_options::runtime_options(bool debug, bool heuristics, double variance)
    : debug_enabled(debug), heuristics_enabled(heuristics), heuristics_variance(variance) {}

bool runtime_options::read_debug_from_env() {
  const char* env = env::DEBUG.read();
  return env && std::string(env) == "1";
}

bool runtime_options::read_heuristics_from_env() {
  const char* env = env::HEURISTICS.read();
  return !(env && std::string(env) == "0");
}

double runtime_options::read_heuristics_variance_from_env() {
  constexpr double default_variance = 0.0;  // Disabled by default

  if (const char* env = env::HEURISTICS_VARIANCE.read()) {
    try {
      double val = std::stod(env);
      if (std::isfinite(val) && val >= 0.0) { return val; }
    } catch (...) {
      // fall through to default
    }
  }
  return default_variance;
}

dim3_t runtime_options::read_gemm_pick_from_env() {
  if (const char* env = env::GEMM_PICK.read()) {
    size_t m = 0, n = 0, k = 0;
    if (std::sscanf(env, "%zux%zux%zu", &m, &n, &k) == 3) { return {m, n, k}; }
  }
  return {0, 0, 0};
}

void runtime_options::update_from_env() {
  debug_enabled       = read_debug_from_env();
  heuristics_enabled  = read_heuristics_from_env();
  heuristics_variance = read_heuristics_variance_from_env();
  gemm_pick           = read_gemm_pick_from_env();
}

runtime_options::runtime_options() { update_from_env(); }

}  // namespace origami
