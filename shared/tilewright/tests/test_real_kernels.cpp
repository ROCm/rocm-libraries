// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Ranks REAL GEMM kernel parameters (harvested from a Tensile UserArgs.yaml,
// the same source the benchmark harness uses) through the tilewright engine.
// Unlike test_model.cpp's synthetic configs, this builds the candidate list from
// actual kernel macro-tile / MI / cache-hint / vector-width values so the
// feasibility filter, smart-K signature match, and two-tower scorer exercise the
// real config space. Self-contained: parses the bundled tests/data fixture; no
// GEMM-framework headers.

#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TILEWRIGHT_TEST_KERNELS
#define TILEWRIGHT_TEST_KERNELS ""
#endif
#ifndef TILEWRIGHT_TEST_WEIGHTS
#define TILEWRIGHT_TEST_WEIGHTS ""
#endif

namespace {

struct Fixture {
  tilewright::DataType a_dt, b_dt, c_dt, d_dt, mi_dt;
  tilewright::Transpose ta, tb;
  std::vector<tilewright::Dim3> problems;  // (m,n,k); batch carried separately
  std::vector<std::size_t> batches;
  std::vector<tilewright::Config> configs;  // built from REAL kernel params
};

bool next_data_line(std::ifstream& in, std::string& out) {
  while (std::getline(in, out)) {
    const auto a = out.find_first_not_of(" \t\r\n");
    if (a == std::string::npos || out[a] == '#') continue;
    return true;
  }
  return false;
}

// Parse the F1-microbench TSV format (PROBLEM_HEADER / NPROBLEMS / NCONFIGS).
bool load_fixture(const std::string& path, Fixture& fx) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line, tag;

  if (!next_data_line(in, line)) return false;
  {
    std::istringstream s(line);
    int adt, bdt, cdt, ddt, midt, tra, trb;
    s >> tag >> adt >> bdt >> cdt >> ddt >> midt >> tra >> trb;
    if (tag != "PROBLEM_HEADER") return false;
    fx.a_dt  = static_cast<tilewright::DataType>(adt);
    fx.b_dt  = static_cast<tilewright::DataType>(bdt);
    fx.c_dt  = static_cast<tilewright::DataType>(cdt);
    fx.d_dt  = static_cast<tilewright::DataType>(ddt);
    fx.mi_dt = static_cast<tilewright::DataType>(midt);
    fx.ta    = static_cast<tilewright::Transpose>(tra);
    fx.tb    = static_cast<tilewright::Transpose>(trb);
  }

  std::size_t np = 0;
  if (!next_data_line(in, line)) return false;
  {
    std::istringstream s(line);
    s >> tag >> np;
  }
  for (std::size_t i = 0; i < np; ++i) {
    if (!next_data_line(in, line)) return false;
    std::istringstream s(line);
    std::size_t m, n, k, b;
    s >> m >> n >> k >> b;
    if (!s) return false;
    fx.problems.push_back(tilewright::Dim3{m, n, k});
    fx.batches.push_back(b);
  }

  std::size_t nc = 0;
  if (!next_data_line(in, line)) return false;
  {
    std::istringstream s(line);
    s >> tag >> nc;
  }
  for (std::size_t i = 0; i < nc; ++i) {
    if (!next_data_line(in, line)) return false;
    std::istringstream s(line);
    int mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, cha, chb, grvwa, grvwb, gwvwd;
    // remaining columns (chc..lsu) are extended/ML fields the engine ignores.
    s >> mt_m >> mt_n >> mt_k >> mi_m >> mi_n >> mi_k >> occ >> cha >> chb >> grvwa >> grvwb >>
        gwvwd;
    if (!s) return false;
    tilewright::Config c;
    c.mt            = tilewright::Dim3{(std::size_t)mt_m, (std::size_t)mt_n, (std::size_t)mt_k};
    c.mi            = tilewright::Dim3{(std::size_t)mi_m, (std::size_t)mi_n, (std::size_t)mi_k};
    c.occupancy     = occ < 1 ? 1 : occ;
    c.cache_hints_a = cha;
    c.cache_hints_b = chb;
    c.grvw_a        = (std::size_t)(grvwa < 1 ? 1 : grvwa);
    c.grvw_b        = (std::size_t)(grvwb < 1 ? 1 : grvwb);
    c.gwvw_d        = (std::size_t)(gwvwd < 1 ? 1 : gwvwd);
    c.index         = i;
    fx.configs.push_back(c);
  }
  return true;
}

tilewright::Hardware gfx950_hardware() {
  tilewright::Hardware hw;
  hw.N_CU                       = 256;
  hw.lds_capacity               = 65536;
  hw.L2_capacity                = 4194304;
  hw.parallel_mi_cu             = 4;
  hw.mem_bw_per_wg_coefficients = std::make_tuple(0.0, 0.0, 1.0);
  return hw;
}

bool ensure_weights() {
  if (tilewright::weights_loaded()) return true;
  if (const char* env = std::getenv("TILEWRIGHT_WEIGHTS"))
    if (env[0] && tilewright::load_weights(env)) return true;
  const std::string def = TILEWRIGHT_TEST_WEIGHTS;
  if (!def.empty() && tilewright::load_weights(def)) return true;
  return tilewright::weights_loaded();
}

}  // namespace

TEST_CASE("tilewright: rank a real-kernel config list", "[tilewright][real]") {
  Fixture fx;
  const std::string kernels = TILEWRIGHT_TEST_KERNELS;
  if (kernels.empty() || !load_fixture(kernels, fx)) {
    SUCCEED("real-kernel fixture not available; skipping");
    return;
  }
  REQUIRE(fx.configs.size() >= 1);
  REQUIRE(fx.problems.size() >= 1);

  if (!ensure_weights()) {
    SUCCEED("tilewright weights not available; skipping (fixture parsed: " +
            std::to_string(fx.configs.size()) + " real kernels)");
    return;
  }

  const tilewright::Hardware hw = gfx950_hardware();

  for (std::size_t pi = 0; pi < fx.problems.size(); ++pi) {
    tilewright::Problem p;
    p.size        = fx.problems[pi];
    p.batch       = fx.batches[pi];
    p.a_transpose = fx.ta;
    p.b_transpose = fx.tb;
    p.a_dtype     = fx.a_dt;
    p.b_dtype     = fx.b_dt;
    p.c_dtype     = fx.c_dt;
    p.d_dtype     = fx.d_dt;
    p.mi_dtype    = fx.mi_dt;

    const auto res = tilewright::rank_configs(p, hw, fx.configs);

    // Contract: every input config covered exactly once.
    REQUIRE(res.size() == fx.configs.size());
    std::vector<char> seen(fx.configs.size(), 0);
    bool seen_unscored = false;
    double prev        = std::numeric_limits<double>::infinity();
    for (const auto& r : res) {
      REQUIRE(r.config_index < fx.configs.size());
      REQUIRE(seen[r.config_index] == 0);
      seen[r.config_index] = 1;
      if (r.scored) {
        // survivors come first, in non-increasing score order, all finite.
        REQUIRE_FALSE(seen_unscored);
        REQUIRE(std::isfinite(r.score));
        REQUIRE(r.score <= prev);
        prev = r.score;
      } else {
        seen_unscored = true;
      }
    }
    for (char c : seen) REQUIRE(c == 1);
  }
}

// Depth-on-demand (min_scored): requesting more solutions than the per-cell
// smart_K whitelist provides must (a) score at least as many configs, (b) leave
// the whitelist tier -- and hence the top-1 -- byte-for-byte unchanged, and
// (c) keep all the ranking-contract invariants.
TEST_CASE("tilewright: rank_configs honors min_scored depth (tier-2)", "[tilewright][real]") {
  Fixture fx;
  const std::string kernels = TILEWRIGHT_TEST_KERNELS;
  if (kernels.empty() || !load_fixture(kernels, fx)) {
    SUCCEED("real-kernel fixture not available; skipping");
    return;
  }
  if (!ensure_weights()) {
    SUCCEED("tilewright weights not available; skipping");
    return;
  }
  const tilewright::Hardware hw = gfx950_hardware();

  auto count_scored = [](const std::vector<tilewright::Result>& r) {
    std::size_t n = 0;
    for (const auto& e : r)
      if (e.scored) ++n;
    return n;
  };

  bool exercised_tier2 = false;
  for (std::size_t pi = 0; pi < fx.problems.size(); ++pi) {
    tilewright::Problem p;
    p.size        = fx.problems[pi];
    p.batch       = fx.batches[pi];
    p.a_transpose = fx.ta;
    p.b_transpose = fx.tb;
    p.a_dtype     = fx.a_dt;
    p.b_dtype     = fx.b_dt;
    p.c_dtype     = fx.c_dt;
    p.d_dtype     = fx.d_dt;
    p.mi_dtype    = fx.mi_dt;

    const auto base = tilewright::rank_configs(p, hw, fx.configs);     // default == min_scored 0
    const auto z0   = tilewright::rank_configs(p, hw, fx.configs, 0);  // explicit 0
    // "all available" (rsn=-1 maps to SIZE_MAX) and a request far larger than the
    // feasible set (e.g. 999 when only N are feasible): both must rank exactly the
    // feasible set without overrunning.
    const auto deep =
        tilewright::rank_configs(p, hw, fx.configs, std::numeric_limits<std::size_t>::max());
    const auto big = tilewright::rank_configs(p, hw, fx.configs, 999);

    // 1. default and explicit-0 are identical (back-compat).
    REQUIRE(base.size() == z0.size());
    for (std::size_t i = 0; i < base.size(); ++i) {
      REQUIRE(base[i].config_index == z0[i].config_index);
      REQUIRE(base[i].scored == z0[i].scored);
    }

    const std::size_t s0 = count_scored(base);
    const std::size_t sN = count_scored(deep);

    // 2. Depth is monotonic: requesting more never scores fewer.
    REQUIRE(sN >= s0);
    REQUIRE(deep.size() == fx.configs.size());

    // 2b. "all" clamps to the feasible set: an over-large finite request scores the
    //     same number as SIZE_MAX (no overrun, no break), and never exceeds the
    //     total config count.
    REQUIRE(count_scored(big) == sN);
    REQUIRE(sN <= fx.configs.size());
    for (std::size_t i = 0; i < deep.size(); ++i)
      REQUIRE(big[i].config_index == deep[i].config_index);

    // 3. Tier-1 is preserved on top: the first s0 ranked entries are byte-for-byte
    //    identical (index AND score) -> top-1..top-s0 picks never regress.
    for (std::size_t i = 0; i < s0; ++i) {
      REQUIRE(deep[i].config_index == base[i].config_index);
      REQUIRE(deep[i].score == base[i].score);
    }

    // 4. Contract still holds on the deep result: scored-first, finite scores,
    //    and the tier-1 block (first s0) is non-increasing.
    bool seen_unscored = false;
    double prev        = std::numeric_limits<double>::infinity();
    std::vector<char> seen(fx.configs.size(), 0);
    for (std::size_t i = 0; i < deep.size(); ++i) {
      const auto& r = deep[i];
      REQUIRE(r.config_index < fx.configs.size());
      REQUIRE(seen[r.config_index] == 0);
      seen[r.config_index] = 1;
      if (r.scored) {
        REQUIRE_FALSE(seen_unscored);
        REQUIRE(std::isfinite(r.score));
        if (i < s0) {
          REQUIRE(r.score <= prev);
          prev = r.score;
        }
      } else {
        seen_unscored = true;
      }
    }
    for (char c : seen) REQUIRE(c == 1);

    if (sN > s0) exercised_tier2 = true;
  }
  // Informational: at least one problem should have a whitelist smaller than the
  // feasible set for this fixture (so tier-2 is actually exercised).
  if (!exercised_tier2)
    WARN("min_scored never triggered tier-2 (whitelist covered all feasible "
         "configs for every fixture problem)");
}
