// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "common.hpp"
#include "origami/nn/config.hpp"
#include "origami/nn/nn.hpp"
#include "origami/origami.hpp"

#if ORIGAMI_ENABLE_NN && ORIGAMI_ENABLE_NN_TILEWRIGHT

#  include "tilewright/model.hpp"
#  include "tilewright/types.hpp"

#  ifndef ORIGAMI_TEST_TW_WEIGHTS_YAML
#    define ORIGAMI_TEST_TW_WEIGHTS_YAML ""
#  endif
#  ifndef ORIGAMI_TEST_TW_WEIGHTS_DIR
#    define ORIGAMI_TEST_TW_WEIGHTS_DIR ""
#  endif
#  ifndef ORIGAMI_TEST_TW_LOGIC_STEM
#    define ORIGAMI_TEST_TW_LOGIC_STEM ""
#  endif

namespace {

std::string weights_yaml_path() {
  if (const char* env = std::getenv("ORIGAMI_NN_WEIGHTS")) {
    if (env[0] != '\0') return std::string(env);
  }
  return std::string(ORIGAMI_TEST_TW_WEIGHTS_YAML);
}

std::string weights_dir() { return std::string(ORIGAMI_TEST_TW_WEIGHTS_DIR); }

std::string logic_stem() { return std::string(ORIGAMI_TEST_TW_LOGIC_STEM); }

std::vector<origami::config_t> make_rank_configs() {
  std::vector<origami::config_t> configs;
  for (int i = 0; i < 5; ++i) {
    auto c = make_config(static_cast<std::size_t>(128 + i * 32),
                         static_cast<std::size_t>(128 + i * 16),
                         64,
                         16,
                         16,
                         32);
    c.grvw_a = 8;
    c.grvw_b = 8;
    c.gwvw_d = 4;
    c.index  = static_cast<std::size_t>(1000 + i);
    configs.push_back(c);
  }
  return configs;
}

const origami::prediction_result_t* first_finite(
    const std::vector<origami::prediction_result_t>& results) {
  for (const auto& r : results) {
    if (std::isfinite(r.latency)) return &r;
  }
  return nullptr;
}

tilewright::Config to_tilewright_config(const origami::config_t& c) {
  tilewright::Config tc;
  tc.mt            = {c.mt.m, c.mt.n, c.mt.k};
  tc.mi            = {c.mi.m, c.mi.n, c.mi.k};
  tc.occupancy     = c.occupancy;
  tc.cache_hints_a = c.cache_hints_a;
  tc.cache_hints_b = c.cache_hints_b;
  tc.grvw_a        = c.grvw_a;
  tc.grvw_b        = c.grvw_b;
  tc.gwvw_d        = c.gwvw_d;
  tc.index         = c.index;
  return tc;
}

}  // namespace

TEST_CASE("NN tilewright: load_model registers handle", "[nn][tilewright]") {
  const std::string yaml = weights_yaml_path();
  if (yaml.empty()) {
    SUCCEED("tilewright weights path not configured; skipping.");
    return;
  }

  const auto handle = origami::nn::load_model(yaml);
  REQUIRE(handle >= 0);

  const auto* info = origami::nn::model_info(handle);
  REQUIRE(info != nullptr);
  REQUIRE(info->backend == origami::nn::backend_id_t::tilewright_v1);
  REQUIRE(info->features.query_dim == 55);
  REQUIRE(info->features.item_dim == 12);
  REQUIRE(info->features.interaction_dim == 37);
}

TEST_CASE("NN tilewright: load_models_for_logic via tilewright_index", "[nn][tilewright]") {
  const std::string dir  = weights_dir();
  const std::string stem = logic_stem();
  if (dir.empty() || stem.empty()) {
    SUCCEED("tilewright index paths not configured; skipping.");
    return;
  }

  const auto models = origami::nn::load_models_for_logic(stem, dir);
  REQUIRE(models.tilewright >= 0);
  REQUIRE(models.embedding_similarity == origami::nn::invalid_handle);
}

TEST_CASE("NN tilewright: rank_configs matches tilewright engine", "[nn][tilewright]") {
  const std::string yaml = weights_yaml_path();
  if (yaml.empty()) {
    SUCCEED("tilewright weights path not configured; skipping.");
    return;
  }

  const auto handle = origami::nn::load_model(yaml);
  REQUIRE(handle >= 0);

  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(8192, 8192, 8192);
  const auto configs  = make_rank_configs();

  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn;
  options.nn_backend = origami::nn_backend_t::tilewright;
  options.nn_model   = handle;

  const auto via_origami = origami::rank_configs(problem, hardware, configs, options);
  const auto* origami_top = first_finite(via_origami);
  REQUIRE(origami_top != nullptr);

  tilewright::Problem tp;
  tp.size        = {problem.size.m, problem.size.n, problem.size.k};
  tp.batch       = problem.batch;
  tp.a_transpose = static_cast<tilewright::Transpose>(static_cast<int>(problem.a_transpose));
  tp.b_transpose = static_cast<tilewright::Transpose>(static_cast<int>(problem.b_transpose));
  tp.a_dtype     = static_cast<tilewright::DataType>(static_cast<int>(problem.a_dtype));
  tp.b_dtype     = static_cast<tilewright::DataType>(static_cast<int>(problem.b_dtype));
  tp.c_dtype     = static_cast<tilewright::DataType>(static_cast<int>(problem.c_dtype));
  tp.d_dtype     = static_cast<tilewright::DataType>(static_cast<int>(problem.d_dtype));
  tp.mi_dtype    = static_cast<tilewright::DataType>(static_cast<int>(problem.mi_dtype));

  tilewright::Hardware th;
  th.N_CU                       = hardware.N_CU;
  th.lds_capacity               = hardware.lds_capacity;
  th.L2_capacity                = hardware.L2_capacity;
  th.parallel_mi_cu             = hardware.parallel_mi_cu;
  th.mem_bw_per_wg_coefficients = hardware.mem_bw_per_wg_coefficients;

  std::vector<tilewright::Config> tw_configs;
  tw_configs.reserve(configs.size());
  for (const auto& c : configs) {
    tw_configs.push_back(to_tilewright_config(c));
  }

  const auto tw_results = tilewright::rank_configs(handle, tp, th, tw_configs);
  const tilewright::Result* tw_top = nullptr;
  for (const auto& r : tw_results) {
    if (r.scored) {
      tw_top = &r;
      break;
    }
  }
  REQUIRE(tw_top != nullptr);

  REQUIRE(origami_top->config.mt.m == configs[tw_top->config_index].mt.m);
  REQUIRE(origami_top->config.mt.n == configs[tw_top->config_index].mt.n);
  REQUIRE(origami_top->config.mt.k == configs[tw_top->config_index].mt.k);
  REQUIRE(origami_top->latency == Catch::Approx(-tw_top->score));
}

TEST_CASE("NN tilewright: inference=nn throws without model", "[nn][tilewright]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(1024, 1024, 1024);
  const auto configs  = make_rank_configs();

  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn;
  options.nn_backend = origami::nn_backend_t::tilewright;

  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

TEST_CASE("NN tilewright: nn_fallback uses analytical when model missing", "[nn][tilewright]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(2048, 2048, 2048);
  const auto configs  = make_rank_configs();

  const auto analytical =
      origami::rank_configs(problem, hardware, configs, origami::model_t::gemm);

  origami::rank_options_t options;
  options.inference = origami::inference_mode_t::nn_fallback;
  const auto fallback = origami::rank_configs(problem, hardware, configs, options);

  REQUIRE(fallback.size() == analytical.size());
  REQUIRE(fallback.front().latency == Catch::Approx(analytical.front().latency));
}

#endif
