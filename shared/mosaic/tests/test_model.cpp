// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Catch2 test suite for the standalone mosaic kernel-recommender API.
// Self-contained: depends only on mosaic's public headers (no HIP / GEMM
// framework headers).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "mosaic/model.hpp"
#include "mosaic/types.hpp"

#ifndef MOSAIC_TEST_WEIGHTS
#define MOSAIC_TEST_WEIGHTS ""
#endif

namespace {

// Resolve the weights path: MOSAIC_WEIGHTS env override wins, else the in-tree
// default injected by CMake (MOSAIC_TEST_WEIGHTS).
std::string weights_path() {
  if (const char* env = std::getenv("MOSAIC_WEIGHTS")) {
    if (env[0] != '\0') return std::string(env);
  }
  return std::string(MOSAIC_TEST_WEIGHTS);
}

mosaic::Problem make_problem() {
  mosaic::Problem p{};
  p.size        = mosaic::Dim3{8192, 8192, 8192};
  p.batch       = 1;
  p.a_transpose = mosaic::Transpose::T;
  p.b_transpose = mosaic::Transpose::N;
  p.a_dtype     = mosaic::DataType::BFloat16;
  p.b_dtype     = mosaic::DataType::BFloat16;
  p.c_dtype     = mosaic::DataType::BFloat16;
  p.d_dtype     = mosaic::DataType::BFloat16;
  p.mi_dtype    = mosaic::DataType::BFloat16;
  return p;
}

std::vector<mosaic::Config> make_configs() {
  std::vector<mosaic::Config> configs;
  for (int i = 0; i < 5; ++i) {
    mosaic::Config c{};
    c.mt = mosaic::Dim3{static_cast<std::size_t>(128 + i * 32),
                        static_cast<std::size_t>(128 + i * 16),
                        static_cast<std::size_t>(64)};
    c.mi = mosaic::Dim3{16, 16, 32};
    c.occupancy       = 1 + i;
    c.cache_hints_a   = 0;
    c.cache_hints_b   = 0;
    c.grvw_a          = 8;
    c.grvw_b          = 8;
    c.gwvw_d          = 4;
    c.index           = static_cast<std::size_t>(1000 + i);
    configs.push_back(c);
  }
  return configs;
}

mosaic::Hardware make_hardware() {
  mosaic::Hardware hw{};
  hw.N_CU                       = 256;       // gfx950-ish
  hw.lds_capacity               = 65536;
  hw.L2_capacity                = 4194304;
  hw.parallel_mi_cu             = 1;
  hw.mem_bw_per_wg_coefficients = std::make_tuple(0.0, 0.008, 0.0);
  return hw;
}

}  // namespace

TEST_CASE("mosaic: rank_configs ranking contract", "[mosaic]") {
  const std::string bin = weights_path();
  if (!mosaic::load_weights(bin)) {
    // Weights are an external artifact and may be absent in source checkouts.
    SUCCEED("mosaic weights not loadable from '" + bin +
            "'; skipping rank_configs ranking checks.");
    return;
  }
  REQUIRE(mosaic::weights_loaded());

  const auto problem = make_problem();
  const auto configs = make_configs();
  const auto hw      = make_hardware();

  auto results = mosaic::rank_configs(problem, hw, configs, nullptr);

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
  double prev = 0.0;
  bool have_prev = false;
  for (const auto& r : results) {
    if (!r.scored) break;
    REQUIRE(std::isfinite(r.score));
    if (have_prev) { REQUIRE(r.score <= prev); }
    prev      = r.score;
    have_prev = true;
  }
}

TEST_CASE("mosaic: route returns a valid cell index", "[mosaic]") {
  const std::string bin = weights_path();
  if (!mosaic::load_weights(bin)) {
    SUCCEED("mosaic weights not loadable from '" + bin +
            "'; skipping route() check.");
    return;
  }

  const auto problem = make_problem();
  // -1 means "no trained ancestor cell"; any real cell index is >= 0.
  REQUIRE(mosaic::route(problem) >= -1);
}
