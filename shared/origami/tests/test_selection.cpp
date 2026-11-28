// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common.hpp"
#include "origami/attention.hpp"
#include "origami/origami.hpp"
#include "origami/types.hpp"

namespace {

// Macro tiles of mixed shapes, with duplicates and occupancy variants so that latencies tie.
std::vector<origami::config_t> sample_configs() {
  std::vector<origami::config_t> configs;
  for (int occupancy : {1, 2, 4}) {
    configs.push_back(make_config(64, 64, 64, 32, 32, 8, false, 1, occupancy));
    configs.push_back(make_config(128, 128, 32, 32, 32, 8, false, 1, occupancy));
    configs.push_back(make_config(128, 64, 64, 32, 32, 8, false, 1, occupancy));
    configs.push_back(make_config(64, 128, 64, 32, 32, 8, false, 1, occupancy));
    configs.push_back(make_config(256, 256, 32, 32, 32, 8, false, 1, occupancy));
  }
  return configs;
}

void require_same_ranking(const std::vector<origami::prediction_result_t>& a,
                          const std::vector<origami::prediction_result_t>& b) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].latency == b[i].latency);
    REQUIRE(a[i].config == b[i].config);
    REQUIRE(a[i].config.occupancy == b[i].config.occupancy);
  }
}

}  // namespace

TEST_CASE("Selection: cost model dispatch maps model_t to its tag", "[selection][cost-model]") {
  using origami::model_t;
  namespace cost_model = origami::cost_model;

  auto name = [](auto tag) -> std::string {
    return origami::cost_model_traits<decltype(tag)>::name;
  };
  REQUIRE(cost_model::dispatch(model_t::gemm, name) == "gemm");
  REQUIRE(cost_model::dispatch(model_t::attention, name) == "attention");
  // A model that names no cost model takes the gemm tag.
  REQUIRE(cost_model::dispatch(model_t::count, name) == "gemm");
  REQUIRE(cost_model::dispatch(model_t::none, name) == "gemm");

  auto model = [](auto tag) { return origami::cost_model_traits<decltype(tag)>::model; };
  REQUIRE(cost_model::dispatch(model_t::gemm, model) == model_t::gemm);
  REQUIRE(cost_model::dispatch(model_t::attention, model) == model_t::attention);
}

TEST_CASE("Selection: cost dispatch maps prediction_modes_t to its tag", "[selection][cost]") {
  using origami::prediction_modes_t;
  namespace method = origami::method;

  auto is_simulation = [](auto tag) { return std::is_same_v<decltype(tag), method::simulation>; };
  REQUIRE_FALSE(method::dispatch(prediction_modes_t::estimation, is_simulation));
  REQUIRE(method::dispatch(prediction_modes_t::simulation, is_simulation));
  // A mode that names no cost model takes the analytical tag.
  REQUIRE_FALSE(method::dispatch(prediction_modes_t::count, is_simulation));
  REQUIRE_FALSE(method::dispatch(prediction_modes_t::none, is_simulation));
}

TEST_CASE("Selection: backend dispatch maps target_t to its tag", "[selection][backend]") {
  using origami::target_t;
  namespace backend = origami::backend;

  auto target = [](auto tag) { return origami::backend_traits<decltype(tag)>::target; };
  for (target_t t : {target_t::generic,
                     target_t::tensilelite,
                     target_t::rocroller,
                     target_t::triton,
                     target_t::composable_kernel}) {
    REQUIRE(backend::dispatch(t, target) == t);
  }
  // A target that names no backend takes the generic tag.
  REQUIRE(backend::dispatch(target_t::none, target) == target_t::generic);

  STATIC_REQUIRE(std::is_same_v<origami::backend_traits<backend::tensilelite>::params_type,
                                origami::tensile_params_t>);
  STATIC_REQUIRE(
      std::is_same_v<origami::backend_traits<backend::triton>::params_type, std::monostate>);
}

TEST_CASE("Selection: method evaluators agree with the cost models", "[selection][method]") {
  namespace method     = origami::method;
  namespace cost_model = origami::cost_model;

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 32, 32, 32, 8, false, 1, 2);

      const double gemm_latency = origami::gemm::compute_total_latency(problem, hardware, config);
      REQUIRE(method::evaluator<cost_model::gemm, method::analytical>::latency(
                  problem, hardware, config) == gemm_latency);
      REQUIRE(method::latency<cost_model::gemm>(problem, hardware, config) == gemm_latency);
      REQUIRE(origami::cost_model_traits<cost_model::gemm>::latency(problem, hardware, config) ==
              gemm_latency);

      const double attention_latency =
          origami::attention::compute_total_latency(problem, hardware, config);
      REQUIRE(method::latency<cost_model::attention>(problem, hardware, config) ==
              attention_latency);

      // Attention provides no simulation model, so simulation evaluates the analytical one.
      config.prediction_mode = origami::prediction_modes_t::simulation;
      REQUIRE(method::latency<cost_model::attention>(problem, hardware, config) ==
              attention_latency);
      REQUIRE(method::evaluator<cost_model::attention, method::simulation>::latency(
                  problem, hardware, config) == attention_latency);
    }
  }
}

TEST_CASE("Selection: rejection reasons", "[selection][cost-model]") {
  using origami::rejection_t;
  namespace cost_model = origami::cost_model;

  REQUIRE(std::string(origami::to_string(rejection_t::none)) == "none");
  REQUIRE(std::string(origami::to_string(rejection_t::lds_capacity)) == "LDS capacity exceeded");
  REQUIRE(std::string(origami::to_string(rejection_t::rf_capacity)) ==
          "Register File (RF) capacity exceeded");

  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto fits     = make_config(64, 64, 64, 32, 32, 8, false, 1, 6);
  auto too_big  = make_config(512, 512, 256, 32, 32, 8, false, 1, 6);

  SECTION("gemm admits by LDS capacity") {
    using traits = origami::cost_model_traits<cost_model::gemm>;
    REQUIRE(traits::admit(problem, hardware, fits) == rejection_t::none);
    REQUIRE(traits::admit(problem, hardware, too_big) == rejection_t::lds_capacity);
  }

  SECTION("attention checks the register file before LDS") {
    using traits = origami::cost_model_traits<cost_model::attention>;
    for (const auto& config : {fits, too_big}) {
      const bool rf  = origami::attention::check_rf_capacity(hardware, config.mt, problem.a_dtype);
      const bool lds = origami::attention::check_lds_capacity(hardware, config.mt, problem.a_dtype);
      const rejection_t expected = !rf    ? rejection_t::rf_capacity
                                   : !lds ? rejection_t::lds_capacity
                                          : rejection_t::none;
      REQUIRE(traits::admit(problem, hardware, config) == expected);
    }
    REQUIRE(traits::admit(problem, hardware, too_big) == rejection_t::rf_capacity);
  }

  SECTION("a rejected config scores the rejected latency") {
    REQUIRE(origami::selection::score<cost_model::gemm>(problem, hardware, too_big) ==
            origami::REJECTED_LATENCY);
    REQUIRE(origami::REJECTED_LATENCY == std::numeric_limits<double>::max());
    REQUIRE(origami::selection::score<cost_model::gemm>(problem, hardware, fits) ==
            origami::gemm::compute_total_latency(problem, hardware, fits));
  }
}

TEST_CASE("Selection: rank<Model> equals rank_configs", "[selection][ranking]") {
  namespace cost_model = origami::cost_model;
  using origami::model_t;

  const auto configs = sample_configs();
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      for (const auto& problem : {make_problem(4096, 4096, 1024),
                                  make_problem(8192, 512, 256),
                                  make_problem(96, 4096, 64)}) {
        const auto gemm_ranking = origami::rank<cost_model::gemm>(problem, hardware, configs);
        REQUIRE(gemm_ranking.size() == configs.size());
        require_same_ranking(gemm_ranking,
                             origami::rank_configs(problem, hardware, configs, model_t::gemm));
        require_same_ranking(gemm_ranking, origami::rank_configs(problem, hardware, configs));

        require_same_ranking(origami::rank<cost_model::attention>(problem, hardware, configs),
                             origami::rank_configs(problem, hardware, configs, model_t::attention));
      }
    }
  }
}

TEST_CASE("Selection: rank<Model> on empty and all-rejected lists", "[selection][ranking]") {
  namespace cost_model = origami::cost_model;

  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);

  REQUIRE_THROWS_AS(origami::rank<cost_model::gemm>(problem, hardware, {}), std::runtime_error);
  REQUIRE_THROWS_AS(origami::rank<cost_model::attention>(problem, hardware, {}),
                    std::runtime_error);

  // When every candidate is rejected the ranking keeps them all, in input order.
  std::vector<origami::config_t> rejected = {make_config(512, 512, 256, 32, 32, 8, false, 1, 6),
                                             make_config(128, 128, 512, 32, 32, 8, false, 1, 6),
                                             make_config(256, 256, 512, 32, 32, 8, false, 1, 6)};
  const auto ranking = origami::rank<cost_model::gemm>(problem, hardware, rejected);
  REQUIRE(ranking.size() == rejected.size());
  for (const auto& result : ranking) { REQUIRE(result.latency == origami::REJECTED_LATENCY); }

  REQUIRE_THROWS_AS(origami::selection::rank_by_latency(problem, hardware, rejected, {1.0}),
                    std::invalid_argument);
}

TEST_CASE("Selection: mapping selectors dispatch on the architecture family",
          "[selection][mapping]") {
  auto problem = make_problem(4096, 4096, 1024);
  auto config  = make_config(128, 128, 32, 32, 32, 8, false, 1, 2);

  // gfx1250 (cdna5) maps WGM only and runs no StaggerU remap.
  auto cdna5         = make_hardware(1250);
  const auto mapping = origami::select_workgroup_mapping(problem, cdna5, config, 1024);
  REQUIRE(mapping.wgmxccsplitk == 0);
  REQUIRE(mapping.wgmxccchunk == 0);
  REQUIRE(mapping.wgmxcc == 1);
  REQUIRE(mapping.wgm >= 1);
  const auto stagger = origami::select_staggerU(problem, cdna5, config, 1024, 1);
  REQUIRE(stagger.staggerU == 0);

  // Invalid problem sizes return the disabled mapping on every family.
  auto empty_problem = make_problem(0, 4096, 1024);
  for (int gpu_arch : test_architectures) {
    auto hardware = make_hardware(gpu_arch);
    const auto m  = origami::select_workgroup_mapping(empty_problem, hardware, config, 1024);
    REQUIRE(m.wgmxccsplitk == 0);
    REQUIRE(m.wgmxccchunk == 0);
    REQUIRE(m.wgmxcc == 0);
    REQUIRE(m.wgm == 1);
  }
}
