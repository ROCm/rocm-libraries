// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "common.hpp"
#include "origami/origami.hpp"

#if ORIGAMI_ENABLE_NN
#include "origami/nn/features/gemm_tilewright.hpp"
#include "origami/nn/filter.hpp"
#include "origami/nn/nn.hpp"

namespace {

#ifndef ORIGAMI_TEST_TW_WEIGHTS_DIR
#define ORIGAMI_TEST_TW_WEIGHTS_DIR ""
#endif

bool tilewright_weights_available() {
  const std::string weights_dir = ORIGAMI_TEST_TW_WEIGHTS_DIR;
  if (weights_dir.empty()) return false;
  const std::filesystem::path manifest =
      std::filesystem::path(weights_dir) /
      "gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.tilewright.yaml";
  return std::filesystem::exists(manifest);
}

}  // namespace

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

TEST_CASE("NN: gemm_tilewright feature dimensions", "[nn]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(1024, 1024, 1024);
  const auto config   = make_config(128, 128, 32);

  std::array<float, origami::nn::features::gemm_tilewright::query_dim> query{};
  std::array<float, origami::nn::features::gemm_tilewright::item_dim> item{};
  std::array<float, origami::nn::features::gemm_tilewright::interaction_dim> inter{};

  origami::nn::features::gemm_tilewright::build_query(problem, hardware, query.data());
  origami::nn::features::gemm_tilewright::build_item(config, item.data());
  origami::nn::features::gemm_tilewright::build_interaction(
      problem, config, hardware, inter.data());

  REQUIRE(query.size() == 55);
  REQUIRE(item.size() == 12);
  REQUIRE(inter.size() == 37);

  for (float v : query) { REQUIRE(std::isfinite(v)); }
  for (float v : item) { REQUIRE(std::isfinite(v)); }
  for (float v : inter) { REQUIRE(std::isfinite(v)); }
}

TEST_CASE("NN: load_models_for_logic loads gfx950 tilewright weights", "[nn][tilewright]") {
#ifndef ORIGAMI_TEST_TW_LOGIC_STEM
#define ORIGAMI_TEST_TW_LOGIC_STEM ""
#endif

  const std::string weights_dir = ORIGAMI_TEST_TW_WEIGHTS_DIR;
  const std::string logic_stem  = ORIGAMI_TEST_TW_LOGIC_STEM;
  if (!tilewright_weights_available() || logic_stem.empty()) {
    SUCCEED("tilewright weights not present; skipping.");
    return;
  }

  const auto models = origami::nn::load_models_for_logic(logic_stem, weights_dir);
  REQUIRE(models.tilewright >= 0);

  const origami::nn::model_info_t* info = origami::nn::model_info(models.tilewright);
  REQUIRE(info != nullptr);
  REQUIRE(info->backend == origami::nn::backend_id_t::tilewright_v1);
  REQUIRE(info->arch == "gfx950");
  REQUIRE(info->features.query_dim == 55);
  REQUIRE(info->features.item_dim == 12);
  REQUIRE(info->features.interaction_dim == 37);
  REQUIRE(info->n_cells > 0);
  REQUIRE(info->n_splits > 0);
}

TEST_CASE("NN: load_models_for_logic uses bundled gfx950 weights when hint_dir empty",
          "[nn][tilewright]") {
  if (!tilewright_weights_available()) {
    SUCCEED("tilewright weights not present; skipping.");
    return;
  }

  const char* logic_stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                           "Bljk_Cijk_Dijk_ID75a0_gfx950";
  const auto models      = origami::nn::load_models_for_logic(logic_stem, "");
  REQUIRE(models.tilewright >= 0);
}

TEST_CASE("NN: rank_configs inference=nn ranks with loaded tilewright model", "[nn][tilewright]") {
  if (!tilewright_weights_available()) {
    SUCCEED("tilewright weights not present; skipping.");
    return;
  }

  const char* logic_stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                           "Bljk_Cijk_Dijk_ID75a0_gfx950";
  const auto models      = origami::nn::load_models_for_logic(logic_stem, "");
  REQUIRE(models.tilewright >= 0);

  const auto hardware = make_hardware(950);
  auto problem = make_problem(1024, 3072, 2048, origami::transpose_t::T, origami::transpose_t::N);
  problem.a_dtype  = origami::data_type_t::BFloat16;
  problem.b_dtype  = origami::data_type_t::BFloat16;
  problem.c_dtype  = origami::data_type_t::BFloat16;
  problem.d_dtype  = origami::data_type_t::BFloat16;
  problem.mi_dtype = origami::data_type_t::BFloat16;

  std::vector<origami::config_t> configs = {
      make_config(128, 128, 128, 32, 32, 32, false, 1, 6, 0, 0),
      make_config(352, 224, 64, 32, 32, 32, false, 1, 6, 0, 0),
  };

  origami::rank_options_t options;
  options.inference      = origami::inference_mode_t::nn;
  options.nn_backend     = origami::nn_backend_t::tilewright;
  options.library_models = &models;

  const auto results = origami::rank_configs(problem, hardware, configs, options);
  REQUIRE_FALSE(results.empty());
  REQUIRE(std::isfinite(results.front().latency));

  bool any_scored = false;
  for (const auto& r : results) {
    if (std::isfinite(r.latency)) any_scored = true;
  }
  REQUIRE(any_scored);
}

TEST_CASE("NN: filter_configs rejects LDS-infeasible kernels", "[nn]") {
  const auto hardware = make_hardware(950);
  auto problem        = make_problem(4096, 4096, 4096);

  origami::config_t huge = make_config(512, 512, 512);
  origami::config_t ok   = make_config(128, 128, 32);

  const auto result = origami::nn::filter::filter_configs(problem, hardware, {huge, ok});

  REQUIRE(result.feasible_indices.size() + result.rejected_indices.size() == 2);
  REQUIRE(origami::gemm::check_lds_capacity(hardware, huge.mt, problem.a_dtype, problem.b_dtype) ==
          false);
  REQUIRE(origami::gemm::check_lds_capacity(hardware, ok.mt, problem.a_dtype, problem.b_dtype) ==
          true);
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

TEST_CASE("NN: unload_model releases model info and payload", "[nn]") {
  if (!tilewright_weights_available()) {
    SUCCEED("tilewright weights not present; skipping.");
    return;
  }

  const char* logic_stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                           "Bljk_Cijk_Dijk_ID75a0_gfx950";
  const origami::nn::model_handle_t handle =
      origami::nn::load_models_for_logic(logic_stem, "").tilewright;
  REQUIRE(handle >= 0);
  REQUIRE(origami::nn::model_info(handle) != nullptr);

  origami::nn::unload_model(handle);
  REQUIRE(origami::nn::model_info(handle) == nullptr);

  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(1024, 3072, 2048);
  std::vector<origami::config_t> configs = {make_config(128, 128, 128)};

  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn;
  options.nn_backend = origami::nn_backend_t::tilewright;
  options.nn_model   = handle;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

TEST_CASE("NN: rank_options force_cell overrides routing", "[nn][tilewright]") {
  if (!tilewright_weights_available()) {
    SUCCEED("tilewright weights not present; skipping.");
    return;
  }

  const char* logic_stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                           "Bljk_Cijk_Dijk_ID75a0_gfx950";
  const origami::nn::model_handle_t handle =
      origami::nn::load_models_for_logic(logic_stem, "").tilewright;
  REQUIRE(handle >= 0);

  const auto hardware = make_hardware(950);
  auto problem = make_problem(1024, 3072, 2048, origami::transpose_t::T, origami::transpose_t::N);
  problem.a_dtype  = origami::data_type_t::BFloat16;
  problem.b_dtype  = origami::data_type_t::BFloat16;
  problem.c_dtype  = origami::data_type_t::BFloat16;
  problem.d_dtype  = origami::data_type_t::BFloat16;
  problem.mi_dtype = origami::data_type_t::BFloat16;

  std::vector<origami::config_t> configs = {
      make_config(128, 128, 128, 16, 16, 32, false, 1, 6, 0, 0),
      make_config(352, 224, 64, 32, 32, 32, false, 1, 6, 0, 0),
  };

  origami::rank_options_t routed;
  routed.inference  = origami::inference_mode_t::nn;
  routed.nn_backend = origami::nn_backend_t::tilewright;
  routed.nn_model   = handle;

  origami::rank_options_t forced = routed;
  forced.nn.force_cell           = 0;

  const auto routed_results = origami::rank_configs(problem, hardware, configs, routed);
  const auto forced_results = origami::rank_configs(problem, hardware, configs, forced);

  REQUIRE_FALSE(routed_results.empty());
  REQUIRE_FALSE(forced_results.empty());
  REQUIRE(std::isfinite(forced_results.front().latency));
}

#else

TEST_CASE("NN: rank_configs inference=nn throws when compiled out", "[nn]") {
  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(1024, 1024, 1024);
  std::vector<origami::config_t> configs = {make_config(128, 128, 32)};

  origami::rank_options_t options;
  options.inference = origami::inference_mode_t::nn;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

#endif
