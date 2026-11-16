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

TEST_CASE("NN: analytical gemm ranking ignores the param-only tensile fields", "[nn][param]") {
  // hipBLASLt fills global_split_u and the stagger fields for the param backend; the analytical
  // model's ranking of a library must not change because of them.
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(3000, 1000, 5000);
  std::vector<origami::config_t> plain = {make_config(256, 256, 64), make_config(128, 128, 32),
                                          make_config(64, 64, 32)};
  auto filled = plain;
  for (auto& c : filled) {
    origami::tensile_params_t t;
    t.global_split_u       = 0;
    t.stagger_u            = 32;
    t.stagger_u_mapping    = 1;
    t.stagger_stride_shift = 2;
    c.backend              = t;
  }
  for (auto& c : plain) c.backend = origami::tensile_params_t{};
  const auto a = origami::rank_configs(problem, hardware, plain, origami::model_t::gemm);
  const auto b = origami::rank_configs(problem, hardware, filled, origami::model_t::gemm);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].config.mt.m == b[i].config.mt.m);
    REQUIRE(a[i].latency == b[i].latency);
  }
}

#if ORIGAMI_TEST_HAS_MSGPACK
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "origami/nn/types.hpp"

namespace {

const std::string param_data_dir = ORIGAMI_TEST_DATA_DIR "/param";

std::vector<std::vector<long long>> read_table(const std::string& name) {
  std::ifstream in(param_data_dir + "/" + name);
  REQUIRE(in.good());
  std::vector<std::vector<long long>> table;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::vector<long long> row;
    long long v;
    while (fields >> v) row.push_back(v);
    table.push_back(row);
  }
  return table;
}

// The fixture's rows as candidates: MT0 MT1 DU GSU WGM SU SUM SUS PGR TLDS LDSB DTVB.
std::vector<origami::config_t> fixture_configs() {
  std::vector<origami::config_t> configs;
  for (const auto& r : read_table("param_fixture_rows.txt")) {
    REQUIRE(r.size() == 12);
    origami::config_t c;
    c.mt                = {std::size_t(r[0]), std::size_t(r[1]), std::size_t(r[2])};
    c.workgroup_mapping = int(r[4]);
    c.index             = configs.size();
    origami::tensile_params_t t;
    t.global_split_u       = std::int16_t(r[3]);
    t.stagger_u            = int(r[5]);
    t.stagger_u_mapping    = int(r[6]);
    t.stagger_stride_shift = int(r[7]);
    t.prefetch_global_read = int(r[8]);
    t.unroll_major_lds_b   = int(r[9]);
    t.one_lds_buffer       = r[10] != 0;
    t.direct_to_vgpr_b     = r[11] != 0;
    c.backend              = t;
    configs.push_back(c);
  }
  return configs;
}

origami::rank_options_t param_options(origami::nn::model_handle_t handle) {
  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn;
  options.nn_backend = origami::nn_backend_t::param;
  options.nn_model   = handle;
  return options;
}

}  // namespace

TEST_CASE("NN param: load, self-test and problem key", "[nn][param]") {
  const auto handle = origami::nn::load_model_by_index(
      "TensileLibrary_param_fixture", origami::nn::backend_id_t::param_v1, param_data_dir);
  REQUIRE(handle != origami::nn::invalid_handle);
  const auto* info = origami::nn::model_info(handle);
  REQUIRE(info != nullptr);
  REQUIRE(info->backend == origami::nn::backend_id_t::param_v1);
  REQUIRE(origami::nn::param_problem_key(handle) == "gfx1151_Cijk_Alik_Bljk_TEST");
  // Loading the same file again returns the cached handle.
  REQUIRE(origami::nn::load_model_by_index("TensileLibrary_param_fixture",
                                           origami::nn::backend_id_t::param_v1,
                                           param_data_dir) == handle);
  REQUIRE(origami::nn::param_problem_key(origami::nn::invalid_handle).empty());
}

TEST_CASE("NN param: ranking matches the Python reference order", "[nn][param]") {
  const auto handle = origami::nn::load_model_by_index(
      "TensileLibrary_param_fixture", origami::nn::backend_id_t::param_v1, param_data_dir);
  REQUIRE(handle != origami::nn::invalid_handle);
  const auto hardware = make_hardware(950);
  const auto configs  = fixture_configs();

  for (const auto& e : read_table("param_fixture_expected.txt")) {
    REQUIRE(e.size() == 4 + configs.size());
    auto problem  = make_problem(std::size_t(e[0]), std::size_t(e[1]), std::size_t(e[2]));
    problem.batch = std::size_t(e[3]);
    const auto results = origami::rank_configs(problem, hardware, configs, param_options(handle));
    REQUIRE(results.size() == configs.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
      CHECK(results[i].config.index == std::size_t(e[4 + i]));
      REQUIRE(std::isfinite(results[i].latency));
      if (i > 0) REQUIRE(results[i].latency >= results[i - 1].latency);  // latency = -score
    }
  }
}

TEST_CASE("NN param: a value outside the head's vocabulary scores unk_logp", "[nn][param]") {
  const auto handle = origami::nn::load_model_by_index(
      "TensileLibrary_param_fixture", origami::nn::backend_id_t::param_v1, param_data_dir);
  REQUIRE(handle != origami::nn::invalid_handle);
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(4096, 4096, 4096);

  // Row 0 has WGM 1 (class 0 of {1, 8}); with WGM 5 its score drops to unk_logp (-20) on that head.
  auto configs = fixture_configs();
  configs.resize(1);
  auto unknown              = configs[0];
  unknown.workgroup_mapping = 5;
  unknown.index             = 1;
  auto other                = configs[0];
  other.workgroup_mapping   = 8;
  other.index               = 2;
  configs.push_back(unknown);
  configs.push_back(other);
  const auto results = origami::rank_configs(problem, hardware, configs, param_options(handle));
  REQUIRE(results.size() == 3);
  double score[3];
  for (const auto& r : results) score[r.config.index] = -r.latency;
  // log p(WGM=1) and log p(WGM=8) are the two classes of one softmax: exp sums to 1.
  const double head_known = std::log(std::exp(score[0] - score[1] - 20.0) + std::exp(score[2] - score[1] - 20.0));
  REQUIRE(head_known == Catch::Approx(0.0).margin(1e-9));
  REQUIRE(results.back().config.index == 1);
}

TEST_CASE("NN param: unreadable candidates rank last with NaN latency", "[nn][param]") {
  const auto handle = origami::nn::load_model_by_index(
      "TensileLibrary_param_fixture", origami::nn::backend_id_t::param_v1, param_data_dir);
  REQUIRE(handle != origami::nn::invalid_handle);
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(1024, 1024, 1024);
  auto configs        = fixture_configs();
  configs[3].backend  = std::monostate{};  // no tensile params: GSU, SU, ... unreadable
  const auto results  = origami::rank_configs(problem, hardware, configs, param_options(handle));
  REQUIRE(results.size() == configs.size());
  REQUIRE(results.back().config.index == 3);
  REQUIRE(std::isnan(results.back().latency));
  for (std::size_t i = 0; i + 1 < results.size(); ++i) REQUIRE(std::isfinite(results[i].latency));
}

TEST_CASE("NN param: bad or missing model files are refused", "[nn][param]") {
  REQUIRE(origami::nn::load_model_by_index("TensileLibrary_param_not_msgpack",
                                           origami::nn::backend_id_t::param_v1,
                                           param_data_dir) == origami::nn::invalid_handle);
  REQUIRE(origami::nn::load_model_by_index("TensileLibrary_param_missing",
                                           origami::nn::backend_id_t::param_v1,
                                           param_data_dir) == origami::nn::invalid_handle);
  // The index binds the fixture to the param backend only.
  REQUIRE(origami::nn::load_model_by_index("TensileLibrary_param_fixture",
                                           origami::nn::backend_id_t::tilewright_v1,
                                           param_data_dir) == origami::nn::invalid_handle);
}

TEST_CASE("NN param: nn_fallback without a param model uses analytical", "[nn][param]") {
  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(2048, 2048, 2048);
  std::vector<origami::config_t> configs = {make_config(256, 256, 64), make_config(128, 128, 32)};
  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn_fallback;
  options.nn_backend = origami::nn_backend_t::param;
  const auto results = origami::rank_configs(problem, hardware, configs, options);
  REQUIRE(results.size() == 2);
  REQUIRE(results.front().latency > 0.0);
}
#endif  // ORIGAMI_TEST_HAS_MSGPACK
