// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Catch2 test suite for the standalone tilewright kernel-recommender API.
// Self-contained: depends only on tilewright's public headers (no HIP / GEMM
// framework headers).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#ifndef TILEWRIGHT_TEST_WEIGHTS
#define TILEWRIGHT_TEST_WEIGHTS ""
#endif

namespace {

// Resolve the weights path: TILEWRIGHT_WEIGHTS env override wins, else the in-tree
// default injected by CMake (TILEWRIGHT_TEST_WEIGHTS).
std::string weights_path() {
  if (const char* env = std::getenv("TILEWRIGHT_WEIGHTS")) {
    if (env[0] != '\0') return std::string(env);
  }
  return std::string(TILEWRIGHT_TEST_WEIGHTS);
}

tilewright::Problem make_problem() {
  tilewright::Problem p{};
  p.size        = tilewright::Dim3{8192, 8192, 8192};
  p.batch       = 1;
  p.a_transpose = tilewright::Transpose::T;
  p.b_transpose = tilewright::Transpose::N;
  p.a_dtype     = tilewright::DataType::BFloat16;
  p.b_dtype     = tilewright::DataType::BFloat16;
  p.c_dtype     = tilewright::DataType::BFloat16;
  p.d_dtype     = tilewright::DataType::BFloat16;
  p.mi_dtype    = tilewright::DataType::BFloat16;
  return p;
}

std::vector<tilewright::Config> make_configs() {
  std::vector<tilewright::Config> configs;
  for (int i = 0; i < 5; ++i) {
    tilewright::Config c{};
    c.mt            = tilewright::Dim3{static_cast<std::size_t>(128 + i * 32),
                            static_cast<std::size_t>(128 + i * 16),
                            static_cast<std::size_t>(64)};
    c.mi            = tilewright::Dim3{16, 16, 32};
    c.occupancy     = 1 + i;
    c.cache_hints_a = 0;
    c.cache_hints_b = 0;
    c.grvw_a        = 8;
    c.grvw_b        = 8;
    c.gwvw_d        = 4;
    c.index         = static_cast<std::size_t>(1000 + i);
    configs.push_back(c);
  }
  return configs;
}

tilewright::Hardware make_hardware() {
  tilewright::Hardware hw{};
  hw.N_CU                       = 256;  // gfx950-ish
  hw.lds_capacity               = 65536;
  hw.L2_capacity                = 4194304;
  hw.parallel_mi_cu             = 1;
  hw.mem_bw_per_wg_coefficients = std::make_tuple(0.0, 0.008, 0.0);
  return hw;
}

}  // namespace

TEST_CASE("tilewright: rank_configs ranking contract", "[tilewright]") {
  const std::string bin = weights_path();
  if (!tilewright::load_weights(bin)) {
    // Weights are an external artifact and may be absent in source checkouts.
    SUCCEED("tilewright weights not loadable from '" + bin +
            "'; skipping rank_configs ranking checks.");
    return;
  }
  REQUIRE(tilewright::weights_loaded());

  const auto problem = make_problem();
  const auto configs = make_configs();
  const auto hw      = make_hardware();

  auto results = tilewright::rank_configs(problem, hw, configs);

  // 1. Every input config is covered exactly once.
  REQUIRE(results.size() == configs.size());
  std::vector<bool> seen(configs.size(), false);
  for (const auto& r : results) {
    REQUIRE(r.config_index < configs.size());
    REQUIRE_FALSE(seen[r.config_index]);
    seen[r.config_index] = true;
  }

  // 2. Survivors (scored == true) come first, before any scored == false.
  bool seen_unscored = false;
  for (const auto& r : results) {
    if (!r.scored) {
      seen_unscored = true;
    } else {
      // A scored entry must never appear after an unscored one.
      REQUIRE_FALSE(seen_unscored);
    }
  }

  // 3. Survivor scores are finite and in non-increasing order.
  double prev    = 0.0;
  bool have_prev = false;
  for (const auto& r : results) {
    if (!r.scored) break;
    REQUIRE(std::isfinite(r.score));
    if (have_prev) { REQUIRE(r.score <= prev); }
    prev      = r.score;
    have_prev = true;
  }
}

TEST_CASE("tilewright: route returns a valid cell index", "[tilewright]") {
  const std::string bin = weights_path();
  if (!tilewright::load_weights(bin)) {
    SUCCEED("tilewright weights not loadable from '" + bin + "'; skipping route() check.");
    return;
  }

  const auto problem = make_problem();
  // -1 means "no trained ancestor cell"; any real cell index is >= 0.
  REQUIRE(tilewright::route(problem) >= -1);
}
