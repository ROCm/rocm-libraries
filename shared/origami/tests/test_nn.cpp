// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "origami/nn/nn.hpp"
#include "origami/origami.hpp"

TEST_CASE("NN: rank_options_t defaults", "[nn]") {
  origami::rank_options_t options;
  REQUIRE(options.analytical_model == origami::model_t::gemm);
  REQUIRE(options.inference == origami::inference_mode_t::analytical);
  REQUIRE(options.nn_backend == origami::nn_backend_t::auto_select);
  REQUIRE(options.nn_model == origami::nn::invalid_handle);
  REQUIRE(options.library_models == nullptr);
}

TEST_CASE("NN: rank_configs rank_options_t matches model_t overload", "[nn]") {
  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(4096, 4096, 4096);
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
  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(1024, 1024, 1024);
  std::vector<origami::config_t> configs = {make_config(128, 128, 32)};

  origami::rank_options_t options;
  options.inference = origami::inference_mode_t::nn;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

TEST_CASE("NN: rank_configs inference=nn_fallback uses analytical", "[nn]") {
  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(2048, 2048, 2048);
  std::vector<origami::config_t> configs = {
      make_config(256, 256, 64),
      make_config(128, 128, 32),
  };

  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn_fallback;
  const auto results = origami::rank_configs(problem, hardware, configs, options);
  REQUIRE_FALSE(results.empty());
  REQUIRE(results.front().latency > 0.0);
}

TEST_CASE("NN: registry without models", "[nn]") {
  REQUIRE(origami::nn::model_info(origami::nn::invalid_handle) == nullptr);
  REQUIRE(origami::nn::default_model(origami::nn::backend_id_t::tilewright_v1) ==
          origami::nn::invalid_handle);
  REQUIRE(origami::nn::load_model_by_index("TensileLibrary_no_such_stem",
                                           origami::nn::backend_id_t::tilewright_v1,
                                           "/nonexistent") == origami::nn::invalid_handle);
}
