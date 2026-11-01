// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"
#include "origami/nn/boost/load_validate.hpp"
#include "origami/nn/boost/rank.hpp"
#include "origami/nn/nn.hpp"
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

// Boost (Origami Boost, boost_v1): the re-rank backend, built with or without ORIGAMI_ENABLE_NN.
// The fixtures are the small models of data/boost; each test loads its own copy, since the
// registry caches models by path.

namespace {

#ifndef ORIGAMI_BOOST_TEST_DATA
#define ORIGAMI_BOOST_TEST_DATA ""
#endif

namespace nnb = origami::nn::boost;

std::filesystem::path boost_fixture(const char* name) {
  return std::filesystem::path(ORIGAMI_BOOST_TEST_DATA) / name;
}

bool boost_fixtures_available() {
  return std::filesystem::exists(boost_fixture("boost_model_bf16.orr"));
}

std::string boost_fixture_copy(const char* name) {
  static std::mt19937_64 rng{std::random_device{}()};
  const auto dst = std::filesystem::temp_directory_path() /
                   ("origami_nn_boost_" + std::to_string(rng()) + "_" + name);
  std::filesystem::copy_file(boost_fixture(name), dst,
                             std::filesystem::copy_options::overwrite_existing);
  return dst.string();
}

// A fixture with the stamps it was exported with: its own build identity and training library.
struct boost_model_t {
  std::string path;
  std::shared_ptr<const nnb::ensemble_t> model;
  nnb::build_identity_t identity;
  nnb::deployment_t target;
};

boost_model_t boost_model(const char* name) {
  boost_model_t b;
  b.path = boost_fixture_copy(name);
  nnb::status_t status;
  std::string detail;
  b.model = nnb::ensemble_t::load_file(b.path, status, detail);
  REQUIRE(b.model);
  b.identity = {b.model->head_string("origami_commit"), b.model->head_string("origami_source_hash"),
                b.model->head_string("feature_schema_id"), b.model->head_string("feature_schema_hash")};
  b.target   = {b.model->head_string("arch"), b.model->head_string("dtype"),
                &b.model->library_manifest()};
  return b;
}

origami::hardware_t boost_hardware() {
  return origami::hardware_t::get_hardware_for_arch(origami::hardware_t::architecture_t::gfx950,
                                                    256, 163840, 512 * 1024, 4 * 1024 * 1024,
                                                    2'400'000, std::nullopt);
}

std::vector<origami::config_t> boost_configs() {
  struct spec_t {
    size_t mt_m, mt_n, mt_k, mi_m, mi_n, mi_k;
    int wg_m, wg_n, occ, sk;
  };
  const spec_t specs[] = {{256, 256, 64, 16, 16, 32, 2, 2, 1, 3}, {256, 128, 64, 16, 16, 32, 2, 2, 1, 3},
                          {128, 256, 64, 16, 16, 32, 2, 2, 1, 3}, {128, 128, 64, 16, 16, 32, 2, 2, 2, 3},
                          {256, 224, 64, 16, 16, 32, 2, 2, 1, 0}, {192, 256, 64, 16, 16, 32, 2, 2, 1, 0},
                          {128, 128, 128, 16, 16, 32, 2, 2, 1, 0}, {64, 128, 128, 16, 16, 32, 1, 2, 2, 3},
                          {256, 64, 128, 32, 32, 16, 2, 1, 2, 3}, {128, 64, 64, 32, 32, 16, 2, 2, 2, 0},
                          {64, 64, 256, 16, 16, 32, 2, 2, 2, 0},  {32, 64, 256, 16, 16, 32, 1, 2, 2, 3}};
  std::vector<origami::config_t> configs;
  for (std::size_t i = 0; i < std::size(specs); ++i) {
    const auto& s = specs[i];
    auto c = make_config(s.mt_m, s.mt_n, s.mt_k, s.mi_m, s.mi_n, s.mi_k, false, 8, s.occ, 0, 0, s.sk);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
    c.index                     = i;
    auto& tp                    = c.tensile();
    tp.wave_group_m             = s.wg_m;
    tp.wave_group_n             = s.wg_n;
    tp.prefetch_global_read     = 2;
    configs.push_back(c);
  }
  origami::memoize_non_temporal_availability(configs);
  return configs;
}

// One workgroup per resident slot, at most one per tile.
nnb::kernel_fields_t boost_fields(const origami::problem_t& p, const origami::config_t& c) {
  nnb::kernel_fields_t f;
  f.num_loads_coalesced_a = 1;
  f.num_loads_coalesced_b = 1;
  auto& sk                    = f.streamk;
  sk.valid                    = true;
  sk.is_streamk               = c.stream_k != 0;
  sk.mt_x                     = c.mt.m;
  sk.mt_y                     = c.mt.n;
  sk.workspace_size_per_elem_c = 4;
  sk.cu_occupancy             = c.occupancy;
  sk.cu_count                 = 256;
  if (!sk.is_streamk) return f;
  auto ceil_div     = [](std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; };
  sk.tiles          = ceil_div(p.size.m, c.mt.m) * ceil_div(p.size.n, c.mt.n) * p.batch;
  sk.iters_per_tile = std::max<std::uint64_t>(ceil_div(p.size.k, c.mt.k), 1);
  sk.final_grid     = std::min<std::uint64_t>(sk.tiles, 256 * static_cast<std::uint64_t>(c.occupancy));
  sk.sk_tiles       = sk.tiles;
  sk.sk_split       = 1;
  sk.reduction      = origami::reduction_t::tree;
  return f;
}

class boost_source_t final : public nnb::rerank_source_t {
 public:
  explicit boost_source_t(const origami::problem_t& problem) : problem_(problem) {}

  void prepare(std::vector<origami::prediction_result_t>& results) override {
    ++prepared;
    if (adjust) adjust(results);
  }

  bool candidate(const origami::config_t& config, nnb::kernel_fields_t& fields) override {
    REQUIRE(prepared == 1);
    offered.push_back(config.index);
    if (std::find(reject.begin(), reject.end(), config.index) != reject.end()) return false;
    if (with_fields) fields = boost_fields(problem_, config);
    return true;
  }

  void outcome(nnb::status_t s, const std::string&, const std::vector<size_t>& o) override {
    ++outcomes;
    status = s;
    order  = o;
  }

  std::function<void(std::vector<origami::prediction_result_t>&)> adjust;
  std::vector<std::size_t> reject;
  bool with_fields = true;
  std::vector<std::size_t> offered;
  int prepared         = 0;
  int outcomes         = 0;
  nnb::status_t status = nnb::status_t::ok;
  std::vector<size_t> order;

 private:
  const origami::problem_t& problem_;
};

std::vector<std::size_t> indices(const std::vector<origami::prediction_result_t>& results) {
  std::vector<std::size_t> out;
  for (const auto& r : results) out.push_back(r.config.index);
  return out;
}

}  // namespace

TEST_CASE("NN: boost defaults", "[nn][boost]") {
  origami::rank_options_t options;
  REQUIRE(options.nn.rerank_source == nullptr);
  const origami::nn::library_models_t models;
  REQUIRE(models.boost == origami::nn::invalid_handle);
}

TEST_CASE("NN: load_boost_model loads an ORIBOOST model for its deployment", "[nn][boost]") {
  if (!boost_fixtures_available()) {
    SUCCEED("boost fixtures not present; skipping.");
    return;
  }
  const auto b = boost_model("boost_model_bf16.orr");

  nnb::status_t status = nnb::status_t::bad_file;
  std::string detail;
  const auto handle = origami::nn::load_boost_model(b.path, b.target, b.identity, &status, &detail);
  REQUIRE(handle >= 0);
  REQUIRE(status == nnb::status_t::ok);
  REQUIRE(detail.rfind("loaded 300 trees, 161 features", 0) == 0);

  const origami::nn::model_info_t* info = origami::nn::model_info(handle);
  REQUIRE(info != nullptr);
  REQUIRE(info->backend == origami::nn::backend_id_t::boost_v1);
  REQUIRE(info->arch == "gfx950");
  REQUIRE(info->features.catalog_id == "gemm_boost");
  REQUIRE(info->features.interaction_dim == 161);

  // Cached by path.
  REQUIRE(origami::nn::load_boost_model(b.path, b.target, b.identity) == handle);

  // Not a ranking backend.
  const auto hardware = boost_hardware();
  const auto problem  = make_problem(4096, 4096, 4096, origami::transpose_t::N, origami::transpose_t::N);
  const auto configs  = boost_configs();
  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn_fallback;
  options.nn_backend = origami::nn_backend_t::boost;
  options.nn_model   = handle;
  REQUIRE(indices(origami::rank_configs(problem, hardware, configs, options)) ==
          indices(origami::rank_configs(problem, hardware, configs)));

  origami::nn::unload_model(handle);
  REQUIRE(origami::nn::model_info(handle) == nullptr);
}

TEST_CASE("NN: load_boost_model refuses a model of other sources or another library", "[nn][boost]") {
  if (!boost_fixtures_available()) {
    SUCCEED("boost fixtures not present; skipping.");
    return;
  }
  const auto b = boost_model("boost_model_bf16.orr");
  nnb::status_t status = nnb::status_t::ok;
  std::string detail;

  auto other = b.identity;
  other.origami_source_hash = "sha256:other";
  REQUIRE(origami::nn::load_boost_model(b.path, b.target, other, &status, &detail) ==
          origami::nn::invalid_handle);
  REQUIRE(status == nnb::status_t::incompatible);
  REQUIRE(detail.find("origami source") != std::string::npos);

  auto no_library            = b.target;
  no_library.library_kernels = nullptr;
  REQUIRE(origami::nn::load_boost_model(b.path, no_library, b.identity, &status, &detail) ==
          origami::nn::invalid_handle);
  REQUIRE(detail.find("kernel library") != std::string::npos);

  auto tf32  = b.target;
  tf32.dtype = "tf32";
  REQUIRE(origami::nn::load_boost_model(b.path, tf32, b.identity, &status, &detail) ==
          origami::nn::invalid_handle);
  REQUIRE(detail.find("dtype") != std::string::npos);
}

TEST_CASE("NN: index_weights_path maps a logic stem to its boost model", "[nn][boost]") {
  static std::mt19937_64 rng{std::random_device{}()};
  const auto dir = std::filesystem::temp_directory_path() / ("origami_nn_index_" + std::to_string(rng()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "origami_nn_index")
      << "# origami_nn_index: <logic_stem>  <backend>  <weights_manifest>\n"
      << "TensileLibrary_BB_BB_stem_gfx950\ttilewright\tstem.tilewright.yaml\n"
      << "TensileLibrary_BB_BB_stem_gfx950\tboost\torigami_boost_gfx950_bf16.orr\n"
      << "TensileLibrary_SS_SS_stem_gfx950\tboost\torigami_boost_gfx950_tf32.orr  # tf32\n";
  using origami::nn::backend_id_t;
  REQUIRE(origami::nn::index_weights_path("TensileLibrary_BB_BB_stem_gfx950", backend_id_t::boost_v1,
                                          dir.string()) ==
          (dir / "origami_boost_gfx950_bf16.orr").string());
  REQUIRE(origami::nn::index_weights_path("TensileLibrary_SS_SS_stem_gfx950", backend_id_t::boost_v1,
                                          dir.string()) ==
          (dir / "origami_boost_gfx950_tf32.orr").string());
  REQUIRE(origami::nn::index_weights_path("TensileLibrary_HH_HH_stem_gfx950", backend_id_t::boost_v1,
                                          dir.string())
              .empty());
  REQUIRE(origami::nn::index_weights_path("TensileLibrary_SS_SS_stem_gfx950",
                                          backend_id_t::tilewright_v1, dir.string())
              .empty());
  // A boost entry needs a deployment: load_model_by_index does not load it.
  REQUIRE(origami::nn::load_model_by_index("TensileLibrary_BB_BB_stem_gfx950", backend_id_t::boost_v1,
                                           dir.string()) == origami::nn::invalid_handle);
  std::filesystem::remove_all(dir);
}

TEST_CASE("NN: rank_configs analytical_rerank re-orders the top candidates with boost", "[nn][boost]") {
  if (!boost_fixtures_available()) {
    SUCCEED("boost fixtures not present; skipping.");
    return;
  }
  const auto b      = boost_model("boost_model_bf16.orr");
  const auto handle = origami::nn::load_boost_model(b.path, b.target, b.identity);
  REQUIRE(handle >= 0);

  const auto hardware = boost_hardware();
  const auto configs  = boost_configs();
  nnb::status_t status;
  std::string detail;
  const auto booster = nnb::booster_t::create(b.model, b.target, b.identity, status, detail);
  REQUIRE(booster);

  // A problem the model re-orders: booster_t::rank on the analytical top ten.
  auto rank_top = [&](const origami::problem_t& p,
                      const std::vector<origami::prediction_result_t>& analytical) {
    std::vector<nnb::candidate_t> candidates(std::min<std::size_t>(10, analytical.size()));
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      candidates[i].config          = &analytical[i].config;
      candidates[i].origami_latency = analytical[i].latency;
      candidates[i].fields          = boost_fields(p, analytical[i].config);
    }
    return booster->rank(p, hardware, candidates.data(), candidates.size());
  };
  origami::problem_t problem;
  std::vector<origami::prediction_result_t> analytical;
  nnb::boost_result_t res;
  for (const auto& mnk : std::vector<std::array<size_t, 3>>{{4096, 4096, 4096}, {1024, 1024, 8192},
                                                           {8192, 1024, 2048}, {512, 4096, 4096},
                                                           {2048, 2048, 512}, {16384, 512, 1024}}) {
    problem = make_problem(mnk[0], mnk[1], mnk[2], origami::transpose_t::N, origami::transpose_t::N);
    analytical = origami::rank_configs(problem, hardware, configs);
    res        = rank_top(problem, analytical);
    REQUIRE(res.status == nnb::status_t::ok);
    std::vector<size_t> identity(res.order.size());
    std::iota(identity.begin(), identity.end(), size_t{0});
    if (res.order != identity) break;
  }
  REQUIRE(analytical.size() > 10);
  REQUIRE_FALSE(std::is_sorted(res.order.begin(), res.order.end()));

  origami::nn::library_models_t models;
  models.boost = handle;
  boost_source_t source(problem);
  origami::rank_options_t options;
  options.inference        = origami::inference_mode_t::analytical_rerank;
  options.nn_backend       = origami::nn_backend_t::boost;
  options.library_models   = &models;
  options.nn.rerank_source = &source;

  SECTION("the first top_n move to the front in booster_t::rank's order") {
    const auto reranked = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.outcomes == 1);
    REQUIRE(source.status == nnb::status_t::ok);
    REQUIRE(source.offered.size() == 10);
    REQUIRE(source.order == res.order);
    REQUIRE(reranked.size() == analytical.size());
    for (std::size_t i = 0; i < reranked.size(); ++i) {
      const auto& want = analytical[i < 10 ? res.order[i] : i];
      REQUIRE(reranked[i].config.index == want.config.index);
      REQUIRE(reranked[i].latency == want.latency);
    }
  }
  SECTION("the candidates come from the order prepare() leaves") {
    source.adjust = [](std::vector<origami::prediction_result_t>& results) {
      std::reverse(results.begin(), results.end());
    };
    const auto reranked = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.prepared == 1);
    REQUIRE(source.offered.size() == 10);
    for (std::size_t i = 0; i < 10; ++i)
      REQUIRE(source.offered[i] == analytical[analytical.size() - 1 - i].config.index);
    for (std::size_t i = 10; i < reranked.size(); ++i)
      REQUIRE(reranked[i].config.index == analytical[analytical.size() - 1 - i].config.index);
  }
  SECTION("a CU budget below the physical count is re-ranked") {
    problem.num_cus = 224;
    origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.outcomes == 1);
    REQUIRE(source.status == nnb::status_t::ok);
    REQUIRE(source.offered.size() == 10);
  }
  SECTION("auto_select resolves the boost handle") {
    options.nn_backend  = origami::nn_backend_t::auto_select;
    const auto reranked = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.status == nnb::status_t::ok);
    REQUIRE(source.order == res.order);
    REQUIRE(indices(reranked) != indices(analytical));
  }
  SECTION("configs the source declines are not candidates and keep their place in the rest") {
    source.reject       = {analytical[1].config.index};
    const auto reranked = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.status == nnb::status_t::ok);
    REQUIRE(source.offered.size() == 11);
    const auto order = indices(reranked);
    REQUIRE(std::find(order.begin(), order.begin() + 10, analytical[1].config.index) ==
            order.begin() + 10);
    REQUIRE(reranked[10].config.index == analytical[1].config.index);
    for (std::size_t i = 11; i < reranked.size(); ++i)
      REQUIRE(reranked[i].config.index == analytical[i].config.index);
  }
  SECTION("analytical inference does not consult the boost model") {
    options.inference   = origami::inference_mode_t::analytical;
    const auto reranked = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.prepared == 0);
    REQUIRE(source.outcomes == 0);
    REQUIRE(source.offered.empty());
    REQUIRE(indices(reranked) == indices(analytical));
  }
  SECTION("another backend has no re-rank") {
    options.nn_backend  = origami::nn_backend_t::tilewright;
    const auto reranked = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.outcomes == 0);
    REQUIRE(indices(reranked) == indices(analytical));
  }
  origami::nn::unload_model(handle);
}

TEST_CASE("NN: analytical_rerank keeps the analytical order when boost declines", "[nn][boost]") {
  if (!boost_fixtures_available()) {
    SUCCEED("boost fixtures not present; skipping.");
    return;
  }
  const auto b      = boost_model("boost_model_bf16.orr");
  const auto handle = origami::nn::load_boost_model(b.path, b.target, b.identity);
  REQUIRE(handle >= 0);

  const auto hardware = boost_hardware();
  const auto configs  = boost_configs();
  auto problem        = make_problem(4096, 4096, 4096, origami::transpose_t::N, origami::transpose_t::N);

  origami::nn::library_models_t models;
  models.boost = handle;
  boost_source_t source(problem);
  origami::rank_options_t options;
  options.inference        = origami::inference_mode_t::analytical_rerank;
  options.nn_backend       = origami::nn_backend_t::boost;
  options.library_models   = &models;
  options.nn.rerank_source = &source;

  auto expect_stock = [&](nnb::status_t want) {
    const auto analytical = origami::rank_configs(problem, hardware, configs);
    const auto reranked   = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(source.prepared == 1);
    REQUIRE(source.outcomes == 1);
    REQUIRE(source.status == want);
    REQUIRE(source.order.empty());
    REQUIRE(indices(reranked) == indices(analytical));
  };
  SECTION("problem out of the training ranges") {
    problem.size.m = 100000;
    expect_stock(nnb::status_t::out_of_range);
    REQUIRE(source.offered.empty());
  }
  SECTION("a decline keeps the order prepare() leaves") {
    source.adjust = [](std::vector<origami::prediction_result_t>& results) {
      std::reverse(results.begin(), results.end());
    };
    problem.size.m        = 100000;
    auto       analytical = origami::rank_configs(problem, hardware, configs);
    const auto reranked   = origami::rank_configs(problem, hardware, configs, options);
    std::reverse(analytical.begin(), analytical.end());
    REQUIRE(source.prepared == 1);
    REQUIRE(source.status == nnb::status_t::out_of_range);
    REQUIRE(indices(reranked) == indices(analytical));
  }
  SECTION("fewer than two candidates") {
    for (const auto& c : configs) source.reject.push_back(c.index);
    source.reject.pop_back();
    expect_stock(nnb::status_t::too_few_candidates);
  }
  SECTION("kernel fields not provided") {
    source.with_fields = false;
    expect_stock(nnb::status_t::missing_feature);
  }
  origami::nn::unload_model(handle);
}
