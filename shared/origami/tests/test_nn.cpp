// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <vector>

#include "common.hpp"
#include "origami/nn/config.hpp"
#include "origami/nn/features/gemm_tilewright.hpp"
#include "origami/nn/filter.hpp"
#include "origami/origami.hpp"

#if ORIGAMI_ENABLE_NN

TEST_CASE("NN: rank_options_t defaults", "[nn]") {
  origami::rank_options_t options;
  REQUIRE(options.analytical_model == origami::model_t::gemm);
  REQUIRE(options.inference == origami::inference_mode_t::analytical);
  REQUIRE(options.nn_backend == origami::nn_backend_t::auto_select);
  REQUIRE(options.nn_model == origami::nn::invalid_handle);
  REQUIRE(options.library_models == nullptr);
}

TEST_CASE("NN: rank_configs rank_options_t matches model_t overload", "[nn]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(4096, 4096, 4096);
  std::vector<origami::config_t> configs = {
      make_config(256, 256, 64),
      make_config(128, 128, 32),
      make_config(64, 64, 32),
  };

  const auto via_model = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm);
  origami::rank_options_t options;
  const auto via_options = origami::rank_configs(problem, hardware, configs, options);

  REQUIRE(via_model.size() == via_options.size());
  for (std::size_t i = 0; i < via_model.size(); ++i) {
    REQUIRE(via_model[i].config.mt.m == via_options[i].config.mt.m);
    REQUIRE(via_model[i].config.mt.n == via_options[i].config.mt.n);
    REQUIRE(via_model[i].config.mt.k == via_options[i].config.mt.k);
    REQUIRE(via_model[i].latency == Catch::Approx(via_options[i].latency));
  }
}

TEST_CASE("NN: rank_configs inference=nn throws without model", "[nn]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(1024, 1024, 1024);
  std::vector<origami::config_t> configs = {make_config(128, 128, 32)};

  origami::rank_options_t options;
  options.inference = origami::inference_mode_t::nn;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

TEST_CASE("NN: rank_configs inference=nn_fallback uses analytical", "[nn]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(2048, 2048, 2048);
  std::vector<origami::config_t> configs = {
      make_config(256, 256, 64),
      make_config(128, 128, 32),
  };

  origami::rank_options_t options;
  options.inference = origami::inference_mode_t::nn_fallback;
  const auto results = origami::rank_configs(problem, hardware, configs, options);
  REQUIRE_FALSE(results.empty());
  REQUIRE(results.front().latency > 0.0);
}

TEST_CASE("NN: gemm_tilewright_v1 feature dimensions", "[nn]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(1024, 1024, 1024);
  const auto config   = make_config(128, 128, 32);

  std::array<float, origami::nn::features::gemm_tilewright_v1::query_dim> query{};
  std::array<float, origami::nn::features::gemm_tilewright_v1::item_dim> item{};
  std::array<float, origami::nn::features::gemm_tilewright_v1::interaction_dim> inter{};

  origami::nn::features::gemm_tilewright_v1::build_query(problem, hardware, query.data());
  origami::nn::features::gemm_tilewright_v1::build_item(config, item.data());
  origami::nn::features::gemm_tilewright_v1::build_interaction(
      problem, config, hardware, inter.data());

  REQUIRE(query.size() == 55);
  REQUIRE(item.size() == 12);
  REQUIRE(inter.size() == 37);

  for (float v : query) {
    REQUIRE(std::isfinite(v));
  }
  for (float v : item) {
    REQUIRE(std::isfinite(v));
  }
  for (float v : inter) {
    REQUIRE(std::isfinite(v));
  }
}

TEST_CASE("NN: filter_configs rejects LDS-infeasible kernels", "[nn]") {
  const auto hardware = make_hardware(950);
  auto problem        = make_problem(4096, 4096, 4096);

  origami::config_t huge = make_config(512, 512, 512);
  origami::config_t ok    = make_config(128, 128, 32);

  const auto result =
      origami::nn::filter::filter_configs(problem, hardware, {huge, ok});

  REQUIRE(result.feasible_indices.size() + result.rejected_indices.size() == 2);
  REQUIRE(origami::nn::filter::check_lds_capacity(problem, huge, hardware) == false);
  REQUIRE(origami::nn::filter::check_lds_capacity(problem, ok, hardware) == true);
}

TEST_CASE("NN: is_kernel_feasible cache-hint gate", "[nn]") {
  auto problem = make_problem(4096, 256, 1024, origami::transpose_t::N, origami::transpose_t::N);

  origami::config_t bad_hints = make_config(256, 256, 64);
  bad_hints.cache_hints_a     = 1;
  bad_hints.cache_hints_b     = 1;

  origami::config_t good_hints = make_config(256, 256, 64);
  good_hints.cache_hints_a     = 0;
  good_hints.cache_hints_b     = 0;

  REQUIRE(origami::nn::filter::is_kernel_feasible(problem, bad_hints) == false);
  REQUIRE(origami::nn::filter::is_kernel_feasible(problem, good_hints) == true);
}

#else

TEST_CASE("NN: rank_configs inference=nn throws when compiled out", "[nn]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(1024, 1024, 1024);
  std::vector<origami::config_t> configs = {make_config(128, 128, 32)};

  origami::rank_options_t options;
  options.inference = origami::inference_mode_t::nn;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

#endif
