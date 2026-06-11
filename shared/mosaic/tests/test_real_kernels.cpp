// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Ranks REAL GEMM kernel parameters (harvested from a Tensile UserArgs.yaml,
// the same source the gemmaiperf F1 microbench uses) through the mosaic engine.
// Unlike test_model.cpp's synthetic configs, this builds the candidate list from
// actual kernel macro-tile / MI / cache-hint / vector-width values so the
// feasibility filter, smart-K signature match, and two-tower scorer exercise the
// real config space. Self-contained: parses the bundled tests/data fixture; no
// GEMM-framework headers.

#include <catch2/catch_test_macros.hpp>

#include "mosaic/model.hpp"
#include "mosaic/types.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef MOSAIC_TEST_KERNELS
#define MOSAIC_TEST_KERNELS ""
#endif
#ifndef MOSAIC_TEST_WEIGHTS
#define MOSAIC_TEST_WEIGHTS ""
#endif

namespace {

struct Fixture {
  mosaic::DataType a_dt, b_dt, c_dt, d_dt, mi_dt;
  mosaic::Transpose ta, tb;
  std::vector<mosaic::Dim3> problems;       // (m,n,k); batch carried separately
  std::vector<std::size_t> batches;
  std::vector<mosaic::Config> configs;      // built from REAL kernel params
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
    fx.a_dt  = static_cast<mosaic::DataType>(adt);
    fx.b_dt  = static_cast<mosaic::DataType>(bdt);
    fx.c_dt  = static_cast<mosaic::DataType>(cdt);
    fx.d_dt  = static_cast<mosaic::DataType>(ddt);
    fx.mi_dt = static_cast<mosaic::DataType>(midt);
    fx.ta    = static_cast<mosaic::Transpose>(tra);
    fx.tb    = static_cast<mosaic::Transpose>(trb);
  }

  std::size_t np = 0;
  if (!next_data_line(in, line)) return false;
  { std::istringstream s(line); s >> tag >> np; }
  for (std::size_t i = 0; i < np; ++i) {
    if (!next_data_line(in, line)) return false;
    std::istringstream s(line);
    std::size_t m, n, k, b;
    s >> m >> n >> k >> b;
    if (!s) return false;
    fx.problems.push_back(mosaic::Dim3{m, n, k});
    fx.batches.push_back(b);
  }

  std::size_t nc = 0;
  if (!next_data_line(in, line)) return false;
  { std::istringstream s(line); s >> tag >> nc; }
  for (std::size_t i = 0; i < nc; ++i) {
    if (!next_data_line(in, line)) return false;
    std::istringstream s(line);
    int mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, cha, chb, grvwa, grvwb, gwvwd;
    // remaining columns (chc..lsu) are extended/ML fields the engine ignores.
    s >> mt_m >> mt_n >> mt_k >> mi_m >> mi_n >> mi_k >> occ >> cha >> chb
      >> grvwa >> grvwb >> gwvwd;
    if (!s) return false;
    mosaic::Config c;
    c.mt = mosaic::Dim3{(std::size_t)mt_m, (std::size_t)mt_n, (std::size_t)mt_k};
    c.mi = mosaic::Dim3{(std::size_t)mi_m, (std::size_t)mi_n, (std::size_t)mi_k};
    c.occupancy      = occ < 1 ? 1 : occ;
    c.cache_hints_a  = cha;
    c.cache_hints_b  = chb;
    c.grvw_a         = (std::size_t)(grvwa < 1 ? 1 : grvwa);
    c.grvw_b         = (std::size_t)(grvwb < 1 ? 1 : grvwb);
    c.gwvw_d         = (std::size_t)(gwvwd < 1 ? 1 : gwvwd);
    c.index          = i;
    fx.configs.push_back(c);
  }
  return true;
}

mosaic::Hardware gfx950_hardware() {
  mosaic::Hardware hw;
  hw.N_CU                       = 256;
  hw.lds_capacity               = 65536;
  hw.L2_capacity                = 4194304;
  hw.parallel_mi_cu             = 4;
  hw.mem_bw_per_wg_coefficients = std::make_tuple(0.0, 0.0, 1.0);
  return hw;
}

bool ensure_weights() {
  if (mosaic::weights_loaded()) return true;
  if (const char* env = std::getenv("MOSAIC_WEIGHTS"))
    if (env[0] && mosaic::load_weights(env)) return true;
  const std::string def = MOSAIC_TEST_WEIGHTS;
  if (!def.empty() && mosaic::load_weights(def)) return true;
  return mosaic::weights_loaded();
}

}  // namespace

TEST_CASE("mosaic: rank a real-kernel config list", "[mosaic][real]") {
  Fixture fx;
  const std::string kernels = MOSAIC_TEST_KERNELS;
  if (kernels.empty() || !load_fixture(kernels, fx)) {
    SUCCEED("real-kernel fixture not available; skipping");
    return;
  }
  REQUIRE(fx.configs.size() >= 1);
  REQUIRE(fx.problems.size() >= 1);

  if (!ensure_weights()) {
    SUCCEED("mosaic weights not available; skipping (fixture parsed: "
            + std::to_string(fx.configs.size()) + " real kernels)");
    return;
  }

  const mosaic::Hardware hw = gfx950_hardware();

  for (std::size_t pi = 0; pi < fx.problems.size(); ++pi) {
    mosaic::Problem p;
    p.size        = fx.problems[pi];
    p.batch       = fx.batches[pi];
    p.a_transpose = fx.ta;
    p.b_transpose = fx.tb;
    p.a_dtype     = fx.a_dt;
    p.b_dtype     = fx.b_dt;
    p.c_dtype     = fx.c_dt;
    p.d_dtype     = fx.d_dt;
    p.mi_dtype    = fx.mi_dt;

    const auto res = mosaic::rank_configs(p, hw, fx.configs, /*configs_ml=*/nullptr);

    // Contract: every input config covered exactly once.
    REQUIRE(res.size() == fx.configs.size());
    std::vector<char> seen(fx.configs.size(), 0);
    bool seen_unscored = false;
    double prev = std::numeric_limits<double>::infinity();
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
