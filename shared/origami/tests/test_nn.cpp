// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "origami/origami.hpp"

#if ORIGAMI_ENABLE_NN
#include "origami/nn/features/gemm_tilewright.hpp"
#include "origami/nn/filter.hpp"
#include "origami/nn/nn.hpp"
#include "origami/nn/twrec/b64_decode.hpp"

namespace {

#ifndef ORIGAMI_TEST_TW_WEIGHTS_DIR
#define ORIGAMI_TEST_TW_WEIGHTS_DIR ""
#endif

#ifndef ORIGAMI_TEST_TW_FIXTURE_DIR
#define ORIGAMI_TEST_TW_FIXTURE_DIR ""
#endif

// The checked-in tiny model in tests/fixtures/tilewright, written by fixtures/gen_tiny_twrec.py.
// Its score has a closed form, see expected_score() and the generator's header.
constexpr const char* kFixtureStem = "tiny_gfx950_stem";
constexpr const char* kFixtureName = "tiny_gfx950";

// Cells of the fixture, in manifest order.
constexpr int kCellA = 0;  // M <= 2048:               prefers mt.m, temperature 1, offset 0
constexpr int kCellB = 1;  // M > 2048, N <= 2048:     prefers mt.n, temperature 2, offset 100
constexpr int kCellC = 2;  // M > 2048, N > 2048:      prefers mt.k, temperature 1, offset 200

// score = log2(mt[axis]) / max(|temperature|, 0.1) + offset, exact for power-of-two tiles.
double expected_score(int cell, const origami::config_t& c) {
  switch (cell) {
    case kCellA: return std::log2(static_cast<double>(c.mt.m));
    case kCellB: return std::log2(static_cast<double>(c.mt.n)) / 2.0 + 100.0;
    case kCellC: return std::log2(static_cast<double>(c.mt.k)) + 200.0;
    default: return std::numeric_limits<double>::quiet_NaN();
  }
}

// Loads the fixture through the index, as hipBLASLt does, and unloads it on destruction.
class FixtureModel {
 public:
  FixtureModel()
      : handle_(origami::nn::load_models_for_logic(kFixtureStem, ORIGAMI_TEST_TW_FIXTURE_DIR)
                    .tilewright) {}
  ~FixtureModel() { origami::nn::unload_model(handle_); }
  FixtureModel(const FixtureModel&)            = delete;
  FixtureModel& operator=(const FixtureModel&) = delete;
  origami::nn::model_handle_t handle() const { return handle_; }

  origami::rank_options_t options(origami::inference_mode_t mode = origami::inference_mode_t::nn) const {
    origami::rank_options_t o;
    o.inference  = mode;
    o.nn_backend = origami::nn_backend_t::tilewright;
    o.nn_model   = handle_;
    return o;
  }

 private:
  origami::nn::model_handle_t handle_;
};

// A feasible bf16 config on gfx950. The mi shape and cache hints are the same for every tile, so
// they match the smart-K signatures in the fixture.
origami::config_t fixture_config(size_t mt_m, size_t mt_n, size_t mt_k) {
  return make_config(mt_m, mt_n, mt_k, 16, 16, 32, false, 1, 6, 0, 0);
}


// Copy of the fixture manifest in a scratch directory with its feature_names_hash replaced.
// The weights sidecar is linked, not copied. The directory is removed on destruction.
class ScratchManifest {
 public:
  explicit ScratchManifest(const std::string& new_hash) {
    namespace fs = std::filesystem;
    const std::string stem = kFixtureName;
    const fs::path src     = fs::path(ORIGAMI_TEST_TW_FIXTURE_DIR);
    dir_                   = fs::temp_directory_path() /
             ("origami_nn_hash_" + std::to_string(std::chrono::steady_clock::now()
                                                      .time_since_epoch()
                                                      .count()));
    fs::create_directories(dir_);
    fs::create_symlink(fs::absolute(src / (stem + ".tilewright.wts.yaml")),
                       dir_ / (stem + ".tilewright.wts.yaml"));

    std::ifstream in(src / (stem + ".tilewright.yaml"));
    std::ostringstream buf;
    buf << in.rdbuf();
    std::string text            = buf.str();
    const std::string needle    = "feature_names_hash: ";
    const std::size_t at        = text.find(needle);
    REQUIRE(at != std::string::npos);
    const std::size_t value_at  = at + needle.size();
    const std::size_t value_end = text.find('\n', value_at);
    text.replace(value_at, value_end - value_at, new_hash);

    path_ = (dir_ / (stem + ".tilewright.yaml")).string();
    std::ofstream out(path_);
    out << text;
  }
  ~ScratchManifest() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  const std::string& path() const { return path_; }

 private:
  std::filesystem::path dir_;
  std::string path_;
};

bool tilewright_weights_available() {
  const std::string weights_dir = ORIGAMI_TEST_TW_WEIGHTS_DIR;
  if (weights_dir.empty()) return false;
  const std::filesystem::path manifest =
      std::filesystem::path(weights_dir) /
      "gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.tilewright.yaml";
  return std::filesystem::exists(manifest);
}

}  // namespace

// Weight YAML is downloaded separately (see data/nn/README.md). Report the case as skipped,
// not passed, when it is absent.
#define SKIP_UNLESS_TW_WEIGHTS()                                                          \
  do {                                                                                    \
    if (!tilewright_weights_available()) {                                                \
      SKIP("tilewright weights not present in " ORIGAMI_TEST_TW_WEIGHTS_DIR);             \
    }                                                                                     \
  } while (0)

TEST_CASE("NN: nn_inference_requested follows the resolved mode", "[nn]") {
  ScopedEnvVar mode("ORIGAMI_INFERENCE_MODE", "analytical");
  REQUIRE_FALSE(origami::nn::nn_inference_requested());
  REQUIRE_FALSE(origami::nn::nn_inference_requested(origami::inference_mode_t::analytical));
  REQUIRE(origami::nn::nn_inference_requested(origami::inference_mode_t::nn));
  REQUIRE(origami::nn::nn_inference_requested(origami::inference_mode_t::nn_fallback));

  // The environment override applies to a call that asked for the default.
  mode.set("nn_fallback");
  REQUIRE(origami::nn::nn_inference_requested());
  mode.set("nn");
  REQUIRE(origami::nn::nn_inference_requested());
}

TEST_CASE("NN: a missing index lookup is cached", "[nn][tilewright]") {
  // A loadable manifest in a scratch directory that has no index yet.
  ScratchManifest good(origami::nn::features::gemm_tilewright::feature_names_hash);
  const std::filesystem::path manifest(good.path());
  const std::string dir  = manifest.parent_path().string();
  const std::string name = manifest.filename().string();

  REQUIRE(origami::nn::load_models_for_logic("stem_a", dir).tilewright ==
          origami::nn::invalid_handle);

  // Now the index names the manifest for stem_a. The earlier miss is remembered for the
  // process, so many libraries asking for the same stem do not each read the index. Weights
  // are expected to exist before the first lookup.
  {
    std::ofstream out(std::filesystem::path(dir) / "origami_nn_index");
    out << "stem_a tilewright " << name << "\n" << "stem_b tilewright " << name << "\n";
  }
  REQUIRE(origami::nn::load_models_for_logic("stem_a", dir).tilewright ==
          origami::nn::invalid_handle);

  // A stem that was never looked up reads the index and loads the same manifest, so the miss
  // above came from the cache and not from an unloadable file.
  const origami::nn::model_handle_t handle = origami::nn::load_models_for_logic("stem_b", dir).tilewright;
  REQUIRE(handle >= 0);
  origami::nn::unload_model(handle);
}

TEST_CASE("NN: a rejected manifest shared by two stems gives no model for either",
          "[nn][tilewright]") {
  // The failed load is remembered, so the second stem does not parse the manifest again; the
  // observable result is that both stems report no model.
  ScratchManifest bad("0000000000000000");
  const std::string manifest_name = std::filesystem::path(bad.path()).filename().string();
  // The manifest and its sidecar live in the ScratchManifest directory; put the index there.
  const std::string dir = std::filesystem::path(bad.path()).parent_path().string();
  {
    std::ofstream out(std::filesystem::path(dir) / "origami_nn_index");
    out << "stem_x tilewright " << manifest_name << "\n"
        << "stem_y tilewright " << manifest_name << "\n";
  }
  REQUIRE(origami::nn::load_models_for_logic("stem_x", dir).tilewright ==
          origami::nn::invalid_handle);
  REQUIRE(origami::nn::load_models_for_logic("stem_y", dir).tilewright ==
          origami::nn::invalid_handle);
}

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
  // The analytical model reports a latency and leaves score NaN.
  for (const auto& r : results) REQUIRE(std::isnan(r.score));
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
  SKIP_UNLESS_TW_WEIGHTS();
  if (logic_stem.empty()) {
    SKIP("ORIGAMI_TEST_TW_LOGIC_STEM is not set.");
  }

  const auto models = origami::nn::load_models_for_logic(logic_stem, weights_dir);
  REQUIRE(models.tilewright >= 0);

  const origami::nn::model_info_t* info = origami::nn::model_info(models.tilewright);
  REQUIRE(info != nullptr);
  REQUIRE(info->backend == origami::nn::backend_id_t::tilewright_v1);
  REQUIRE(info->arch == "gfx950");
  REQUIRE(info->features.feature_names_hash ==
          origami::nn::features::gemm_tilewright::feature_names_hash);
  REQUIRE(info->features.query_dim == 55);
  REQUIRE(info->features.item_dim == 12);
  REQUIRE(info->features.interaction_dim == 37);
  REQUIRE(info->n_cells > 0);
  REQUIRE(info->n_splits > 0);
}

TEST_CASE("NN: load_models_for_logic falls back to ORIGAMI_NN_WEIGHTS_DIR when hint_dir empty",
          "[nn][tilewright]") {
  ScopedEnvVar weights_dir_env("ORIGAMI_NN_WEIGHTS_DIR", ORIGAMI_TEST_TW_FIXTURE_DIR);
  const auto models = origami::nn::load_models_for_logic(kFixtureStem, "");
  REQUIRE(models.tilewright >= 0);
  origami::nn::unload_model(models.tilewright);
}

TEST_CASE("NN: load_model rejects a manifest with a mismatched feature hash",
          "[nn][tilewright]") {
  // Control: an unmodified hash loads, so the rejection below is due to the hash alone.
  ScratchManifest matching(origami::nn::features::gemm_tilewright::feature_names_hash);
  const origami::nn::model_handle_t ok = origami::nn::load_model(matching.path());
  REQUIRE(ok >= 0);
  origami::nn::unload_model(ok);

  ScratchManifest mismatched("0000000000000000");
  REQUIRE(origami::nn::load_model(mismatched.path()) == origami::nn::invalid_handle);
}

TEST_CASE("NN: a model for another architecture is not used", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);  // a gfx950 model

  auto problem = make_problem(1024, 3072, 2048, origami::transpose_t::T, origami::transpose_t::N);
  std::vector<origami::config_t> configs = {
      fixture_config(128, 128, 64),
      fixture_config(256, 256, 32),
  };

  // Control: the same model scores the same problem on gfx950, so the fallback below is due to
  // the architecture alone.
  const auto scored =
      origami::rank_configs(problem, make_hardware(950), configs, fixture.options());
  REQUIRE(std::isfinite(scored.front().score));

  const auto hardware = make_hardware(942);

  const auto analytical = origami::rank_configs(problem, hardware, configs, origami::rank_options_t{});
  REQUIRE_FALSE(analytical.empty());

  // nn_fallback declines the gfx950 model on gfx942 and returns the analytical ranking.
  auto options        = fixture.options(origami::inference_mode_t::nn_fallback);
  const auto fallback = origami::rank_configs(problem, hardware, configs, options);
  REQUIRE(fallback.size() == analytical.size());
  for (std::size_t i = 0; i < fallback.size(); ++i) {
    REQUIRE(fallback[i].config.mt.m == analytical[i].config.mt.m);
    REQUIRE(fallback[i].config.mt.n == analytical[i].config.mt.n);
    REQUIRE(fallback[i].latency == analytical[i].latency);
    REQUIRE(std::isnan(fallback[i].score));
  }

  // nn has no fallback, so the mismatch is an error.
  options.inference = origami::inference_mode_t::nn;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

TEST_CASE("NN: rank_configs inference=nn ranks with loaded tilewright model", "[nn][tilewright]") {
  SKIP_UNLESS_TW_WEIGHTS();

  const char* logic_stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_"
                           "Bljk_Cijk_Dijk_ID75a0_gfx950";
  const auto models      = origami::nn::load_models_for_logic(logic_stem, ORIGAMI_TEST_TW_WEIGHTS_DIR);
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
  REQUIRE(std::isfinite(results.front().score));

  // The model produces a score, not a time: latency is NaN on every NN entry.
  for (const auto& r : results) REQUIRE(std::isnan(r.latency));

  // Scored entries come first, best score first; an unscored entry has no score either.
  bool seen_unscored = false;
  double previous    = std::numeric_limits<double>::infinity();
  for (const auto& r : results) {
    if (std::isnan(r.score)) {
      seen_unscored = true;
      continue;
    }
    REQUIRE_FALSE(seen_unscored);
    REQUIRE(r.score <= previous);
    previous = r.score;
  }
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

  const auto hardware = make_hardware(950);
  REQUIRE(origami::nn::filter::is_kernel_feasible(problem, hardware, bad_hints) == false);
  REQUIRE(origami::nn::filter::is_kernel_feasible(problem, hardware, good_hints) == true);
}

TEST_CASE("NN: unload_model releases model info and payload", "[nn][tilewright]") {
  const origami::nn::model_handle_t handle =
      origami::nn::load_models_for_logic(kFixtureStem, ORIGAMI_TEST_TW_FIXTURE_DIR).tilewright;
  REQUIRE(handle >= 0);
  REQUIRE(origami::nn::model_info(handle) != nullptr);

  origami::nn::unload_model(handle);
  REQUIRE(origami::nn::model_info(handle) == nullptr);

  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(1024, 3072, 2048);
  std::vector<origami::config_t> configs = {fixture_config(128, 128, 64)};

  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn;
  options.nn_backend = origami::nn_backend_t::tilewright;
  options.nn_model   = handle;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, options), std::runtime_error);
}

TEST_CASE("NN: unload_model while ranking is in flight", "[nn][tilewright]") {
  const origami::nn::model_handle_t handle =
      origami::nn::load_models_for_logic(kFixtureStem, ORIGAMI_TEST_TW_FIXTURE_DIR).tilewright;
  REQUIRE(handle >= 0);

  const auto hardware                    = make_hardware(950);
  const auto problem                     = make_problem(1024, 3072, 2048);
  std::vector<origami::config_t> configs = {fixture_config(128, 128, 64),
                                            fixture_config(256, 256, 32)};

  origami::rank_options_t options;
  options.inference  = origami::inference_mode_t::nn_fallback;
  options.nn_backend = origami::nn_backend_t::tilewright;
  options.nn_model   = handle;

  // Ranking holds a reference to the model, so a concurrent unload must not free it.
  // Either result is valid: NN scores before the unload lands, analytical fallback after.
  // Run under ASan/TSan to catch a regression.
  std::atomic<bool> stop{false};
  std::thread ranker([&] {
    while (!stop.load()) {
      const auto results = origami::rank_configs(problem, hardware, configs, options);
      if (results.size() != configs.size()) std::abort();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  origami::nn::unload_model(handle);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  stop.store(true);
  ranker.join();

  REQUIRE(origami::nn::model_info(handle) == nullptr);
}

// The tests below use the checked-in fixture, so they run on any machine. The fixture's score is
// log2(mt[axis]) / max(|temperature|, 0.1) + offset for the cell that routing picks; see
// expected_score() and fixtures/gen_tiny_twrec.py.

TEST_CASE("NN fixture: loads through the index and reports its structure", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);

  const origami::nn::model_info_t* info = origami::nn::model_info(fixture.handle());
  REQUIRE(info != nullptr);
  REQUIRE(info->backend == origami::nn::backend_id_t::tilewright_v1);
  REQUIRE(info->arch == "gfx950");
  REQUIRE(info->features.feature_names_hash ==
          origami::nn::features::gemm_tilewright::feature_names_hash);
  REQUIRE(info->features.query_dim == 55);
  REQUIRE(info->features.item_dim == 12);
  REQUIRE(info->features.interaction_dim == 37);
  REQUIRE(info->n_cells == 3);
  REQUIRE(info->n_splits == 2);
}

TEST_CASE("NN fixture: scores follow the closed form and are sorted best first",
          "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);
  const auto hardware = make_hardware(950);

  // Cell A. Both tiles are on its smart-K whitelist, with distinct scores log2(mt.m).
  const auto problem = make_problem(1024, 4096, 4096);
  const std::vector<origami::config_t> configs = {
      fixture_config(128, 128, 64),  // 7
      fixture_config(256, 256, 32),  // 8
  };

  const auto results = origami::rank_configs(problem, hardware, configs, fixture.options());
  REQUIRE(results.size() == configs.size());
  REQUIRE(results[0].config.mt.m == 256);
  REQUIRE(results[1].config.mt.m == 128);
  for (const auto& r : results) {
    REQUIRE(std::isnan(r.latency));
    REQUIRE(r.score == Catch::Approx(expected_score(kCellA, r.config)));
  }
  REQUIRE(results[0].score > results[1].score);
}

TEST_CASE("NN fixture: routing picks the cell from the splits", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);
  const auto hardware = make_hardware(950);

  // One tile that every cell can score, so the offset in the score identifies the cell.
  const std::vector<origami::config_t> configs = {fixture_config(256, 256, 32)};

  struct Case {
    size_t m, n, k;
    int cell;
    const char* why;
  };
  const Case cases[] = {
      {1024, 4096, 4096, kCellA, "M below the first split"},
      {2048, 4096, 4096, kCellA, "the M threshold is inclusive"},
      {2049, 1024, 4096, kCellB, "M just above the first split, N below the second"},
      {4096, 1024, 4096, kCellB, "the missing N <= 2048 cell resolves to its parent"},
      {4096, 2048, 4096, kCellB, "the N threshold is inclusive"},
      {4096, 2049, 4096, kCellC, "N just above the second split"},
      {2049, 4096, 4096, kCellC, "M just above the first split, N above the second"},
      {8192, 8192, 4096, kCellC, "M and N far above both splits"},
  };
  for (const Case& c : cases) {
    INFO(c.why << ": M=" << c.m << " N=" << c.n << " K=" << c.k);
    const auto results = origami::rank_configs(
        make_problem(c.m, c.n, c.k), hardware, configs, fixture.options());
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].score == Catch::Approx(expected_score(c.cell, configs[0])));
  }
}

TEST_CASE("NN fixture: each cell scores along its own axis and temperature", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);
  const auto hardware = make_hardware(950);

  // A tile that differs from the next one on every axis. In each cell the winner is the tile
  // that is larger on that cell's preferred axis (A: mt.m, B: mt.n, C: mt.k).
  const std::vector<origami::config_t> configs = {
      fixture_config(256, 64, 32),   // large m
      fixture_config(64, 256, 32),   // large n
      fixture_config(64, 64, 128),   // large k
  };
  struct Case {
    size_t m, n;
    int cell;
    size_t winner;
  };
  const Case cases[] = {
      {1024, 4096, kCellA, 0},
      {4096, 1024, kCellB, 1},
      {4096, 4096, kCellC, 2},
  };
  for (const Case& c : cases) {
    INFO("cell " << c.cell);
    // Rank with the whitelist off so every feasible tile is scored in every cell.
    auto options                          = fixture.options();
    options.nn.use_smart_k_whitelist      = false;
    const auto results =
        origami::rank_configs(make_problem(c.m, c.n, 4096), hardware, configs, options);
    REQUIRE(results.size() == configs.size());
    REQUIRE(results.front().config.mt.m == configs[c.winner].mt.m);
    REQUIRE(results.front().config.mt.n == configs[c.winner].mt.n);
    REQUIRE(results.front().config.mt.k == configs[c.winner].mt.k);
    for (const auto& r : results) {
      REQUIRE(r.score == Catch::Approx(expected_score(c.cell, r.config)));
    }
  }
}

TEST_CASE("NN fixture: smart-K whitelist tiers the candidates", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);
  const auto hardware = make_hardware(950);

  // Cell A. The first tile is not on the whitelist, and it would score highest.
  const auto problem                           = make_problem(1024, 4096, 4096);
  const origami::config_t off_list             = fixture_config(512, 64, 32);  // score 9
  const origami::config_t on_list_low          = fixture_config(128, 128, 64);  // score 7
  const origami::config_t on_list_high         = fixture_config(256, 256, 32);  // score 8
  const std::vector<origami::config_t> configs = {off_list, on_list_low, on_list_high};

  SECTION("only the whitelisted tiles are scored; the rest follow unscored") {
    const auto results = origami::rank_configs(problem, hardware, configs, fixture.options());
    REQUIRE(results.size() == 3);
    REQUIRE(results[0].config.mt.m == 256);
    REQUIRE(results[1].config.mt.m == 128);
    REQUIRE(results[0].score == Catch::Approx(8.0));
    REQUIRE(results[1].score == Catch::Approx(7.0));
    // The tile off the whitelist keeps its place after the scored ones and has no score.
    REQUIRE(results[2].config.mt.m == 512);
    REQUIRE(std::isnan(results[2].score));
    REQUIRE(std::isnan(results[2].latency));
  }

  SECTION("min_scored scores the tiles off the whitelist after the whitelisted ones") {
    auto options        = fixture.options();
    options.nn.min_scored = 3;
    const auto results  = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(results.size() == 3);
    // Tier one is sorted, then tier two is appended; the list is not sorted as a whole, so the
    // off-list tile, which scores highest, comes last.
    REQUIRE(results[0].score == Catch::Approx(8.0));
    REQUIRE(results[1].score == Catch::Approx(7.0));
    REQUIRE(results[2].config.mt.m == 512);
    REQUIRE(results[2].score == Catch::Approx(9.0));
  }

  SECTION("switching the whitelist off scores every feasible tile and sorts them all") {
    auto options                     = fixture.options();
    options.nn.use_smart_k_whitelist = false;
    const auto results               = origami::rank_configs(problem, hardware, configs, options);
    REQUIRE(results.size() == 3);
    REQUIRE(results[0].config.mt.m == 512);
    REQUIRE(results[0].score == Catch::Approx(9.0));
    REQUIRE(results[1].score == Catch::Approx(8.0));
    REQUIRE(results[2].score == Catch::Approx(7.0));
  }

  SECTION("a cell with no whitelist scores every feasible tile") {
    // Cell B has an empty smart-K list, so nothing is filtered out.
    const auto results = origami::rank_configs(
        make_problem(4096, 1024, 4096), hardware, configs, fixture.options());
    REQUIRE(results.size() == 3);
    for (const auto& r : results) {
      REQUIRE(r.score == Catch::Approx(expected_score(kCellB, r.config)));
    }
  }

  SECTION("a whitelist that matches no feasible tile falls back to scoring all of them") {
    // Cell C lists one tile that none of these configs uses.
    const auto results = origami::rank_configs(
        make_problem(4096, 4096, 4096), hardware, configs, fixture.options());
    REQUIRE(results.size() == 3);
    for (const auto& r : results) {
      REQUIRE(r.score == Catch::Approx(expected_score(kCellC, r.config)));
    }
  }
}

TEST_CASE("NN fixture: rank_options force_cell overrides routing", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);
  const auto hardware = make_hardware(950);

  const std::vector<origami::config_t> configs = {fixture_config(256, 256, 32)};
  const auto problem                           = make_problem(1024, 4096, 4096);  // routes to A

  auto routed = fixture.options();
  REQUIRE(origami::rank_configs(problem, hardware, configs, routed)[0].score ==
          Catch::Approx(expected_score(kCellA, configs[0])));

  for (const int cell : {kCellA, kCellB, kCellC}) {
    INFO("force_cell " << cell);
    auto forced           = fixture.options();
    forced.nn.force_cell  = cell;
    const auto results    = origami::rank_configs(problem, hardware, configs, forced);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].score == Catch::Approx(expected_score(cell, configs[0])));
  }

  // An index that is not a cell scores nothing, which nn reports as an error.
  auto out_of_range          = fixture.options();
  out_of_range.nn.force_cell = 3;
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, out_of_range),
                    std::runtime_error);
}

TEST_CASE("NN fixture: a problem with no cell is unscored", "[nn][tilewright]") {
  const FixtureModel fixture;
  REQUIRE(fixture.handle() >= 0);
  const auto hardware = make_hardware(950);

  // M <= 128 is the "Small" tier, which the fixture has no cell or split for.
  const auto problem                           = make_problem(64, 4096, 4096);
  const std::vector<origami::config_t> configs = {fixture_config(128, 128, 64),
                                                  fixture_config(256, 256, 32)};

  // Nothing is scored, so nn has no result to return.
  REQUIRE_THROWS_AS(origami::rank_configs(problem, hardware, configs, fixture.options()),
                    std::runtime_error);

  // nn_fallback falls back to the analytical ranking, with a latency and no score.
  const auto results = origami::rank_configs(
      problem, hardware, configs, fixture.options(origami::inference_mode_t::nn_fallback));
  REQUIRE(results.size() == configs.size());
  for (const auto& r : results) {
    REQUIRE(std::isfinite(r.latency));
    REQUIRE(std::isnan(r.score));
  }
}

namespace {

std::vector<std::uint8_t> bytes_of(const std::string& s) {
  return std::vector<std::uint8_t>(s.begin(), s.end());
}

// Decodes `in`; returns whether it was accepted. `out` is checked on success only.
bool decode(const std::string& in, std::vector<std::uint8_t>* out) {
  return origami::nn::twrec::detail::b64_decode(in, out);
}

}  // namespace

TEST_CASE("NN: b64_decode accepts well-formed input", "[nn][b64]") {
  std::vector<std::uint8_t> out;
  REQUIRE(decode("", &out));
  REQUIRE(out.empty());

  REQUIRE(decode("TQ==", &out));
  REQUIRE(out == bytes_of("M"));
  REQUIRE(decode("TWE=", &out));
  REQUIRE(out == bytes_of("Ma"));
  REQUIRE(decode("TWFu", &out));
  REQUIRE(out == bytes_of("Man"));
  REQUIRE(decode("TWFuTWFu", &out));
  REQUIRE(out == bytes_of("ManMan"));

  // Whitespace is skipped and does not count toward the length.
  REQUIRE(decode("TWFu\nTWE=\r\n", &out));
  REQUIRE(out == bytes_of("ManMa"));
  REQUIRE(decode(" T W F u ", &out));
  REQUIRE(out == bytes_of("Man"));
}

TEST_CASE("NN: b64_decode rejects malformed padding", "[nn][b64]") {
  std::vector<std::uint8_t> out;

  // Padding that is not at the end.
  REQUIRE_FALSE(decode("AB=C", &out));
  REQUIRE_FALSE(decode("A=BC", &out));
  REQUIRE_FALSE(decode("TQ==TWFu", &out));
  REQUIRE_FALSE(decode("TWE=TWFu", &out));

  // More padding than a group can hold.
  REQUIRE_FALSE(decode("A===", &out));
  REQUIRE_FALSE(decode("====", &out));
  REQUIRE_FALSE(decode("TQ===", &out));

  // Padding with the wrong number of data characters in front of it.
  REQUIRE_FALSE(decode("T=", &out));
  REQUIRE_FALSE(decode("TWE==", &out));
  REQUIRE_FALSE(decode("TWFu=", &out));
  REQUIRE_FALSE(decode("=", &out));

  // A rejected input leaves no partial output behind.
  REQUIRE(out.empty());
}

TEST_CASE("NN: b64_decode rejects lengths that are not a multiple of 4", "[nn][b64]") {
  std::vector<std::uint8_t> out;
  REQUIRE_FALSE(decode("T", &out));
  REQUIRE_FALSE(decode("TQ", &out));
  REQUIRE_FALSE(decode("TWE", &out));
  REQUIRE_FALSE(decode("TWFuT", &out));
  REQUIRE_FALSE(decode("TWFuTQ", &out));
  REQUIRE_FALSE(decode("TWFuTWE", &out));
  REQUIRE(out.empty());
}

TEST_CASE("NN: b64_decode rejects characters outside the alphabet and stray bits", "[nn][b64]") {
  std::vector<std::uint8_t> out;
  REQUIRE_FALSE(decode("TW-u", &out));
  REQUIRE_FALSE(decode("TW_u", &out));
  REQUIRE_FALSE(decode(std::string("TW\0u", 4), &out));  // an embedded NUL, not a terminator

  // "TR==" would decode to the same byte as "TQ==" but sets bits the encoder never writes.
  REQUIRE_FALSE(decode("TR==", &out));
  REQUIRE_FALSE(decode("TWF=", &out));
  REQUIRE(out.empty());
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
