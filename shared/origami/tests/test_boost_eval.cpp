// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The quantized tree evaluator against an independent walk of the model file's TREE section,
// bit for bit, and the timestep column against compute_timestep_latency.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"
#include "origami/gemm.hpp"
#include "origami/nn/boost/load_validate.hpp"
#include "origami/nn/boost/rank.hpp"
#include "origami/nn/boost/sha256.hpp"
#include "origami/nn/features/gemm_boost.hpp"

using namespace origami;
using namespace origami::nn::boost;
namespace features = origami::nn::features::gemm_boost;
namespace fs = std::filesystem;

namespace {

const fs::path kData = ORIGAMI_BOOST_TEST_DATA;

std::vector<std::uint8_t> read_bytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// The TREE section as src/nn/boost/orr_loader.cpp reads it, walked node by node with LightGBM's
// NumericalDecision.
struct ref_tree_t {
  std::vector<std::int32_t> feature, left, right;
  std::vector<std::uint8_t> decision;
  std::vector<double>       threshold, leaf;
};

std::vector<ref_tree_t> read_trees(const std::vector<std::uint8_t>& b) {
  auto get = [&](std::size_t& o, auto v) {
    std::memcpy(&v, b.data() + o, sizeof(v));
    o += sizeof(v);
    return v;
  };
  std::vector<ref_tree_t> trees;
  std::size_t             o = 24;
  const std::size_t       end = b.size() - 32;
  while (o + 12 <= end) {
    const std::string tag(reinterpret_cast<const char*>(b.data() + o), 4);
    o += 4;
    const auto        len  = get(o, std::uint64_t{});
    const std::size_t next = o + len;
    if (tag == "TREE") {
      const auto nt = get(o, std::uint32_t{});
      for (std::uint32_t t = 0; t < nt; ++t) {
        ref_tree_t tr;
        const auto L = get(o, std::uint32_t{});
        const auto I = get(o, std::uint32_t{});
        for (std::uint32_t i = 0; i < I; ++i) {
          tr.feature.push_back(get(o, std::int32_t{}));
          tr.decision.push_back(get(o, std::uint8_t{}));
          o += 3;
          tr.threshold.push_back(get(o, double{}));
          tr.left.push_back(get(o, std::int32_t{}));
          tr.right.push_back(get(o, std::int32_t{}));
        }
        for (std::uint32_t j = 0; j < L; ++j) tr.leaf.push_back(get(o, double{}));
        trees.push_back(std::move(tr));
      }
    }
    o = next;
  }
  return trees;
}

// finite_only: the evaluator's contract for prepared rows under nonfinite policy 1/2 without
// zero-as-missing splits, where every decision is x <= threshold (NaN goes right).
double ref_evaluate(const std::vector<ref_tree_t>& trees, const double* x, double init,
                    bool finite_only) {
  constexpr double kZero = static_cast<double>(1e-35f);
  double           sum   = 0.0;
  for (const auto& tr : trees) {
    std::int32_t node = tr.feature.empty() ? -1 : 0;
    while (node >= 0) {
      const auto i  = static_cast<std::size_t>(node);
      double     v  = x[tr.feature[i]];
      bool       go_left;
      if (finite_only) {
        go_left = v <= tr.threshold[i];
      } else {
        const int mt = (tr.decision[i] >> 2) & 3;
        if (std::isnan(v) && mt != 2) v = 0.0;
        if ((mt == 1 && v >= -kZero && v <= kZero) || (mt == 2 && std::isnan(v)))
          go_left = (tr.decision[i] & 2) != 0;
        else
          go_left = v <= tr.threshold[i];
      }
      node = go_left ? tr.left[i] : tr.right[i];
    }
    sum += tr.leaf[static_cast<std::size_t>(~node)];
  }
  return sum + init;
}

hardware_t gfx950() {
  return hardware_t::get_hardware_for_arch(hardware_t::architecture_t::gfx950, 256, 163840,
                                           512 * 1024, 4 * 1024 * 1024, 2'400'000, std::nullopt);
}

struct candidate_set_t {
  std::vector<config_t>    configs;
  std::vector<candidate_t> candidates;
};

// Origami's top ten of a fixed config set (as test_boost.cpp builds them).
candidate_set_t ranked_candidates(const problem_t& p, const hardware_t& hw) {
  struct spec_t {
    size_t mt_m, mt_n, mt_k, mi_m, mi_n, mi_k;
    int    wg_m, wg_n, occ, sk;
  };
  const spec_t specs[] = {{256, 256, 64, 16, 16, 32, 2, 2, 1, 3}, {256, 128, 64, 16, 16, 32, 2, 2, 1, 3},
                          {128, 256, 64, 16, 16, 32, 2, 2, 1, 3}, {128, 128, 64, 16, 16, 32, 2, 2, 2, 3},
                          {256, 224, 64, 16, 16, 32, 2, 2, 1, 0}, {192, 256, 64, 16, 16, 32, 2, 2, 1, 0},
                          {128, 128, 128, 16, 16, 32, 2, 2, 1, 0}, {64, 128, 128, 16, 16, 32, 1, 2, 2, 3},
                          {256, 64, 128, 32, 32, 16, 2, 1, 2, 3}, {128, 64, 64, 32, 32, 16, 2, 2, 2, 0},
                          {64, 64, 256, 16, 16, 32, 2, 2, 2, 0},  {32, 64, 256, 16, 16, 32, 1, 2, 2, 3}};
  std::vector<config_t> all;
  for (std::size_t i = 0; i < std::size(specs); ++i) {
    const auto& s = specs[i];
    config_t    c = make_config(s.mt_m, s.mt_n, s.mt_k, s.mi_m, s.mi_n, s.mi_k, false, 8, s.occ, 0,
                                0, s.sk);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
    c.index                     = i;
    auto& tp                    = c.tensile();
    tp.wave_group_m             = s.wg_m;
    tp.wave_group_n             = s.wg_n;
    tp.prefetch_global_read     = 2;
    all.push_back(c);
  }
  memoize_non_temporal_availability(all);
  const auto      ranked = rank_configs(p, hw, all);
  candidate_set_t out;
  for (std::size_t i = 0; i < ranked.size() && out.configs.size() < 10; ++i)
    out.configs.push_back(ranked[i].config);
  for (std::size_t i = 0; i < out.configs.size(); ++i) {
    candidate_t cand;
    cand.config                       = &out.configs[i];
    cand.origami_latency              = ranked[i].latency;
    cand.fields.num_loads_coalesced_a = 1;
    cand.fields.num_loads_coalesced_b = 1;
    // A static Stream-K launch: one workgroup per resident slot, at most one per tile.
    const config_t& c   = out.configs[i];
    auto&           sk  = cand.fields.streamk;
    auto ceil_div       = [](std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; };
    sk.valid            = true;
    sk.is_streamk       = c.stream_k != 0;
    sk.mt_x             = c.mt.m;
    sk.mt_y             = c.mt.n;
    sk.workspace_size_per_elem_c = 4;
    sk.cu_occupancy              = c.occupancy;
    sk.cu_count                  = 256;
    sk.tiles      = ceil_div(p.size.m, c.mt.m) * ceil_div(p.size.n, c.mt.n) * std::max<size_t>(p.batch, 1);
    sk.iters_per_tile = std::max<std::uint64_t>(ceil_div(p.size.k, c.mt.k), 1);
    sk.final_grid = std::min<std::uint64_t>(sk.tiles, 256 * static_cast<std::uint64_t>(std::max(c.occupancy, 1)));
    sk.sk_tiles   = sk.tiles;
    sk.sk_split   = 1;
    sk.reduction  = reduction_t::tree;
    out.candidates.push_back(cand);
  }
  return out;
}

std::shared_ptr<const booster_t> booster_for(const std::shared_ptr<const ensemble_t>& m) {
  const build_identity_t id{m->head_string("origami_commit"), m->head_string("origami_source_hash"),
                            m->head_string("feature_schema_id"),
                            m->head_string("feature_schema_hash")};
  const deployment_t target{m->head_string("arch"), m->head_string("dtype"), &m->library_manifest()};
  status_t    st;
  std::string detail;
  return booster_t::create(m, target, id, st, detail);
}

// BF16, or TF32 (A/B/C/D Float with XFloat32 math) for a tf32 model.
std::vector<problem_t> problem_sweep(bool tf32 = false) {
  std::vector<problem_t> out;
  const transpose_t      tr[] = {transpose_t::N, transpose_t::T};
  const std::size_t      sizes[][4] = {{4096, 4096, 4096, 1}, {1024, 8192, 512, 1},
                                       {8192, 1024, 16384, 1}, {2048, 2048, 2048, 8},
                                       {1, 4096, 8192, 1},     {128, 128, 128, 1},
                                       {333, 1777, 4099, 3},   {16384, 16384, 64, 1},
                                       {64, 65536, 1024, 2},   {7680, 3072, 1536, 1}};
  for (const auto& s : sizes)
    for (auto a : tr)
      for (auto b : tr) {
        problem_t p = make_problem(s[0], s[1], s[2], a, b, s[3]);
        if (tf32) {
          p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = data_type_t::Float;
          p.mi_dtype = data_type_t::XFloat32;
        }
        out.push_back(p);
      }
  return out;
}

std::vector<fs::path> exact_models() {
  std::vector<fs::path> out;
  for (const char* f : {"boost_model_bf16.orr", "boost_model_tf32.orr", "boost_model_fp8.orr"})
    if (fs::exists(kData / f)) out.push_back(kData / f);
  return out;
}

}  // namespace

TEST_CASE("Origami Boost: quantized evaluator equals a reference walk bit for bit", "[nn][boost]") {
  const auto models = exact_models();
  if (models.empty()) SKIP("no models");
  const auto hw = gfx950();
  for (const auto& path : models) {
    INFO(path.string());
    status_t    st;
    std::string detail;
    const auto  m = ensemble_t::load_file(path.string(), st, detail);
    REQUIRE(m);
    const auto        trees = read_trees(read_bytes(path));
    const std::size_t nf    = m->features().size();
    REQUIRE(trees.size() == m->num_trees());

    std::vector<std::vector<double>> thr(nf);
    for (const auto& tr : trees)
      for (std::size_t i = 0; i < tr.feature.size(); ++i)
        thr[static_cast<std::size_t>(tr.feature[i])].push_back(tr.threshold[i]);

    // Rows on and next to every split threshold, between them and outside them, with
    // non-finite values that prepare() maps to 0.
    std::mt19937_64     rng(20261021);
    std::vector<double> rows;
    const std::size_t   synthetic = 4000;
    for (std::size_t r = 0; r < synthetic; ++r) {
      for (std::size_t j = 0; j < nf; ++j) {
        const auto& t    = thr[j];
        const auto  pick = rng() % 100;
        double      v;
        if (t.empty() || pick < 5) {
          const double lo = t.empty() ? -1.0 : *std::min_element(t.begin(), t.end()) - 1.0;
          const double hi = t.empty() ? 1.0 : *std::max_element(t.begin(), t.end()) + 1.0;
          v = lo + (hi - lo) * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        } else {
          v = t[rng() % t.size()];
          if (pick < 35) v = std::nextafter(v, -std::numeric_limits<double>::infinity());
          else if (pick < 65) v = std::nextafter(v, std::numeric_limits<double>::infinity());
          else if (pick < 80) v = 0.5 * (v + t[rng() % t.size()]);
          else if (pick < 85) v = pick % 2 ? 0.0 : -0.0;
          else if (pick < 88) v = v * 1e6 + (pick % 2 ? 1.0 : -1.0);
        }
        if (rng() % 50 == 0)
          v = rng() % 2 ? std::numeric_limits<double>::quiet_NaN()
                        : std::numeric_limits<double>::infinity();
        rows.push_back(v);
      }
      m->prepare(rows.data() + rows.size() - nf);
    }
    // The runtime's own feature rows for Origami's top ten over a problem sweep.
    std::size_t runtime_rows = 0;
    if (const auto b = booster_for(m)) {
      std::vector<double> row(nf);
      for (const auto& p : problem_sweep(m->head_string("dtype") == "tf32")) {
        const auto cs = ranked_candidates(p, hw);
        for (const auto& c : cs.candidates) {
          if (!b->model_features(p, hw, c, row.data())) continue;
          m->prepare(row.data());
          rows.insert(rows.end(), row.begin(), row.end());
          ++runtime_rows;
        }
      }
    }
    const std::size_t   n = rows.size() / nf;
    std::vector<double> want(n), got(n);
    // LightGBM's NumericalDecision on prepared rows; the walk on x <= threshold must agree.
    for (std::size_t i = 0; i < n; ++i) {
      want[i] = ref_evaluate(trees, rows.data() + i * nf, 0.0, false);
      REQUIRE(ref_evaluate(trees, rows.data() + i * nf, 0.0, true) == want[i]);
    }

    m->evaluate_batch(rows.data(), n, got.data());
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) differ += std::memcmp(&want[i], &got[i], sizeof(double)) != 0;
    // Every batch size the booster can pass, at every offset of a short window.
    for (std::size_t b = 1; b <= 17; ++b)
      for (std::size_t s = 0; s + b <= std::min<std::size_t>(n, 64); s += b) {
        m->evaluate_batch(rows.data() + s * nf, b, got.data());
        for (std::size_t i = 0; i < b; ++i)
          differ += std::memcmp(&want[s + i], &got[i], sizeof(double)) != 0;
      }
    for (std::size_t i = 0; i < std::min<std::size_t>(n, 200); ++i) {
      const double one = m->evaluate(rows.data() + i * nf);
      differ += std::memcmp(&want[i], &one, sizeof(double)) != 0;
    }
    std::printf("[boost eval] %s: %zu trees, %zu rows (%zu from the runtime): %zu differ\n",
                path.filename().string().c_str(), trees.size(), n, runtime_rows, differ);
    REQUIRE(differ == 0);
  }
}

// The extractor takes f_timestep_latency from compute_tile_latency, which is what
// compute_timestep_latency returns at the stamped gemm.cpp.
TEST_CASE("Origami Boost: timestep latency column equals compute_timestep_latency", "[nn][boost]") {
  const auto  hw    = gfx950();
  const auto& names = features::slot_names();
  const auto  slot  = static_cast<std::size_t>(
      std::find(names.begin(), names.end(), "f_timestep_latency") - names.begin());
  REQUIRE(slot < names.size());
  std::vector<double> f(names.size());
  std::size_t         checked = 0;
  for (const bool tf32 : {false, true})
    for (const auto& p : problem_sweep(tf32)) {
      const auto cs = ranked_candidates(p, hw);
      for (const auto& c : cs.candidates) {
        REQUIRE(features::extract(p, hw, c, f.data()));
        const gemm::context_t ctx(p, hw, *c.config);
        const double          want = gemm::compute_timestep_latency(p, hw, *c.config, ctx);
        REQUIRE(std::memcmp(&f[slot], &want, sizeof(double)) == 0);
        ++checked;
      }
    }
  REQUIRE(checked > 0);
}
