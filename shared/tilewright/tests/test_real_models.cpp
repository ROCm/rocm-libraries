// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Loads every shipped model under weights/hipblaslt; pointer files are skipped.

#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TILEWRIGHT_TEST_WEIGHTS_DIR
#define TILEWRIGHT_TEST_WEIGHTS_DIR ""
#endif
#ifndef TILEWRIGHT_TEST_KERNELS
#define TILEWRIGHT_TEST_KERNELS ""
#endif

namespace tw = tilewright;
namespace fs = std::filesystem;
using namespace tilewright_test;

namespace {

bool is_lfs_pointer(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  char head[40] = {};
  f.read(head, sizeof head);
  return std::strncmp(head, "version https://git-lfs.github.com/spec/", 40) == 0;
}

std::vector<fs::path> shipped_models(std::size_t* pointers) {
  std::vector<fs::path> out;
  *pointers = 0;
  const fs::path root(TILEWRIGHT_TEST_WEIGHTS_DIR);
  std::error_code ec;
  if (root.empty() || !fs::is_directory(root, ec)) return out;
  for (const auto& e : fs::recursive_directory_iterator(root)) {
    const std::string name = e.path().filename().string();
    if (!e.is_regular_file() || name.size() < 15 ||
        name.compare(name.size() - 15, 15, ".tilewright.bin") != 0)
      continue;
    if (is_lfs_pointer(e.path()))
      ++*pointers;
    else
      out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Kernel pool of real Tensile kernel parameters (tests/data/sample_kernels.tsv).
std::vector<tw::Config> sample_pool() {
  std::vector<tw::Config> pool;
  std::ifstream in(TILEWRIGHT_TEST_KERNELS);
  std::string line;
  bool in_configs = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream s(line);
    if (line.rfind("NCONFIGS", 0) == 0) {
      in_configs = true;
      continue;
    }
    if (!in_configs) continue;
    long v[12];
    for (long& x : v) s >> x;
    if (!s) continue;
    tw::Config c = make_config(static_cast<std::size_t>(v[0]),
                               static_cast<std::size_t>(v[1]),
                               static_cast<std::size_t>(v[2]),
                               static_cast<std::size_t>(v[3]),
                               static_cast<std::size_t>(v[4]),
                               static_cast<std::size_t>(v[5]),
                               static_cast<int>(v[7]),
                               static_cast<int>(v[8]),
                               static_cast<int>(std::max(1L, v[6])));
    c.grvw_a     = static_cast<std::size_t>(std::max(1L, v[9]));
    c.grvw_b     = static_cast<std::size_t>(std::max(1L, v[10]));
    c.gwvw_d     = static_cast<std::size_t>(std::max(1L, v[11]));
    c.index      = pool.size();
    pool.push_back(c);
  }
  return pool;
}

std::vector<tw::Config> synthetic_pool() {
  std::vector<tw::Config> pool;
  for (std::size_t mt_m : {32, 64, 128, 256})
    for (std::size_t mt_n : {32, 64, 128, 256})
      for (std::size_t mt_k : {32, 64})
        for (const tw::Dim3& mi :
             {tw::Dim3{16, 16, 32}, tw::Dim3{16, 16, 128}, tw::Dim3{32, 32, 64}})
          for (int ch : {0, 4}) {
            tw::Config c = make_config(mt_m, mt_n, mt_k, mi.m, mi.n, mi.k, ch, 0);
            c.index      = pool.size();
            pool.push_back(c);
          }
  return pool;
}

}  // namespace

TEST_CASE("shipped models load, route and rank", "[real]") {
  std::size_t pointers               = 0;
  const std::vector<fs::path> models = shipped_models(&pointers);
  if (pointers > 0) WARN(pointers << " shipped model(s) are Git LFS pointers; skipped");
  if (models.empty()) SKIP("no materialized shipped models under " TILEWRIGHT_TEST_WEIGHTS_DIR);

  const std::vector<tw::Config> real_pool  = sample_pool();
  const std::vector<tw::Config> synth_pool = synthetic_pool();
  REQUIRE_FALSE(real_pool.empty());
  const tw::Hardware hw{128, 65536, 4u << 20};
  for (const fs::path& path : models) {
    INFO(path.string());
    std::string err;
    const tw::ModelPtr model = tw::load_model(path.string(), &err);
    INFO(err);
    REQUIRE(model);
    const tw::ModelInfo info = tw::describe(*model);
    CHECK(info.arch.rfind(path.parent_path().filename().string().substr(0, 6), 0) == 0);
    CHECK(info.feature_catalog_hash == tw::feature_catalog_hash());
    CHECK(info.n_cells > 0);

    const tw::CandidateSet real(model, real_pool);
    const tw::CandidateSet synth(model, synth_pool);
    std::size_t routed = 0, scored = 0;
    for (std::size_t m : {1, 64, 777, 4096, 12345})
      for (std::size_t n : {16, 640, 9000})
        for (std::size_t k : {16, 1000, 16384})
          for (std::size_t b : {1, 4}) {
            const tw::Problem p = make_problem(m, n, k, b, tw::DataType::BFloat16);
            const int cell      = tw::route(*model, p);
            REQUIRE(cell >= -1);
            REQUIRE(cell < static_cast<int>(info.n_cells));
            if (cell < 0) continue;
            ++routed;
            REQUIRE_FALSE(tw::cell_label(*model, cell).empty());
            for (const tw::CandidateSet* set : {&real, &synth}) {
              const std::vector<tw::Result> r = set->rank(p, hw, b == 4 ? 1000 : 0);
              REQUIRE(r.size() == set->configs().size());
              REQUIRE(same_results(
                  r, tw::rank_configs(*model, p, hw, set->configs(), b == 4 ? 1000 : 0)));
              bool tail = false;
              for (const tw::Result& x : r) {
                if (!x.scored) {
                  tail = true;
                  continue;
                }
                REQUIRE_FALSE(tail);
                REQUIRE(std::isfinite(x.score));
                ++scored;
              }
            }
          }
    CHECK(routed > 0);
    CHECK(scored > 0);
  }
}

TEST_CASE("shipped models rank repeated kernels like kernels scored one by one", "[real]") {
  std::size_t pointers               = 0;
  const std::vector<fs::path> models = shipped_models(&pointers);
  if (models.empty()) SKIP("no materialized shipped models under " TILEWRIGHT_TEST_WEIGHTS_DIR);

  std::vector<tw::Config> pool;
  const std::vector<tw::Config> real_pool = sample_pool();
  for (std::int64_t copy = 0; copy < 3; ++copy)
    for (std::size_t i = 0; i < real_pool.size(); ++i) {
      tw::Config c = real_pool[(i * 5 + static_cast<std::size_t>(copy)) % real_pool.size()];
      c.index      = pool.size();
      c.attributes = {{"workgroup_mapping", copy}, {"wave_num", 4}};
      pool.push_back(c);
    }
  const tw::Hardware hw{128, 65536, 4u << 20};
  const tw::ExecutionContext budget{hw.N_CU / 2, tw::Schedule::Default};
  for (const fs::path& path : models) {
    INFO(path.string());
    const tw::ModelPtr model = tw::load_model(path.string());
    REQUIRE(model);
    const tw::CandidateSet set(model, pool);
    for (std::size_t m : {1, 300, 4096})
      for (std::size_t n : {16, 2048})
        for (std::size_t k : {64, 8192})
          for (std::size_t b : {1, 2}) {
            const tw::Problem p = make_problem(m, n, k, b, tw::DataType::BFloat16);
            for (std::size_t min_scored : {std::size_t{0}, std::size_t{1000}}) {
              const std::vector<tw::Result> expect = tw::detail::rank_keys(*model,
                                                                           p,
                                                                           hw,
                                                                           tw::ExecutionContext{},
                                                                           pool,
                                                                           singleton_keys(pool),
                                                                           min_scored,
                                                                           tw::TieBreak::PoolOrder,
                                                                           nullptr);
              REQUIRE(same_results(set.rank(p, hw, min_scored), expect));
              REQUIRE(same_results(tw::rank_configs(*model, p, hw, pool, min_scored), expect));
              const std::vector<tw::Result> none = set.rank(p, hw, budget, min_scored);
              for (std::size_t j = 0; j < none.size(); ++j) {
                REQUIRE_FALSE(none[j].scored);
                REQUIRE(none[j].config_index == j);
              }
            }
          }
  }
}

TEST_CASE("shipped cells whose occupancy never varied score unseen occupancies alike", "[real]") {
  std::size_t pointers               = 0;
  const std::vector<fs::path> models = shipped_models(&pointers);
  if (models.empty()) SKIP("no materialized shipped models under " TILEWRIGHT_TEST_WEIGHTS_DIR);

  constexpr std::size_t kOccupancy        = 8;
  const std::vector<tw::Config> real_pool = sample_pool();
  std::vector<tw::Config> pool;
  for (std::size_t i = 0; i < real_pool.size(); i += 4)
    for (int occ = 1; occ <= 6; ++occ) {
      tw::Config c = real_pool[i];
      c.occupancy  = occ;
      pool.push_back(c);
    }
  const tw::Hardware hw{128, 65536, 4u << 20};
  std::size_t constant_cells = 0;
  for (const fs::path& path : models) {
    INFO(path.string());
    const tw::ModelPtr model = tw::load_model(path.string());
    REQUIRE(model);
    for (std::size_t m : {64, 777, 4096, 12345})
      for (std::size_t n : {16, 640, 9000})
        for (std::size_t k : {16, 1000, 16384}) {
          const tw::Problem p = make_problem(m, n, k, 1, tw::DataType::BFloat16);
          const int cell      = tw::route(*model, p);
          if (cell < 0 || !model->weights(static_cast<std::size_t>(cell)).i_constant[kOccupancy])
            continue;
          ++constant_cells;
          std::vector<double> score(pool.size(), std::nan(""));
          for (const tw::Result& r : tw::rank_configs(*model, p, hw, pool, 1000))
            if (r.scored) score[r.config_index] = r.score;
          for (std::size_t base = 0; base < pool.size(); base += 6)
            for (std::size_t occ = 2; occ < 6; ++occ) {
              const double a = score[base + 1], b = score[base + occ];
              if (!std::isnan(a) && !std::isnan(b)) CHECK(std::memcmp(&a, &b, sizeof a) == 0);
            }
        }
  }
  if (constant_cells == 0) WARN("no shipped cell has a constant occupancy feature");
}

TEST_CASE("every tilewright_index entry resolves to its model", "[real]") {
  const fs::path root(TILEWRIGHT_TEST_WEIGHTS_DIR);
  std::error_code ec;
  if (root.empty() || !fs::is_directory(root, ec)) SKIP("no weights directory");
  std::size_t entries = 0;
  for (const auto& e : fs::recursive_directory_iterator(root)) {
    if (e.path().filename() != "tilewright_index") continue;
    const fs::path dir = e.path().parent_path();
    std::ifstream in(e.path());
    std::string line;
    while (std::getline(in, line)) {
      const std::size_t hash = line.find('#');
      if (hash != std::string::npos) line.resize(hash);
      std::istringstream s(line);
      std::string stem, file;
      if (!(s >> stem >> file)) continue;
      INFO(e.path().string() << ": " << stem);
      REQUIRE(fs::is_regular_file(dir / file));
      ++entries;
      if (is_lfs_pointer(dir / file)) continue;
      std::string err;
      const tw::ModelPtr model = tw::load_model_by_index(stem, dir.string(), &err);
      INFO(err);
      REQUIRE(model);
      CHECK(model == tw::load_model((dir / file).string()));
    }
  }
  CHECK(entries > 0);
}
