// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "kernels.hpp"
#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace tw = tilewright;
using namespace tilewright_test;

namespace {

const std::string kLarge = "Large|Large|LargeK|Bnone";

// Every config index exactly once; scored entries first with finite scores;
// unscored entries in input order.
void check_structure(const std::vector<tw::Result>& r, std::size_t n) {
  REQUIRE(r.size() == n);
  std::vector<int> seen(n, 0);
  bool unscored        = false;
  std::size_t last_idx = 0;
  for (const tw::Result& x : r) {
    REQUIRE(x.config_index < n);
    REQUIRE(seen[x.config_index]++ == 0);
    if (x.scored) {
      REQUIRE_FALSE(unscored);
      REQUIRE(std::isfinite(x.score));
    } else {
      if (unscored) REQUIRE(x.config_index > last_idx);
      unscored = true;
      last_idx = x.config_index;
    }
  }
}

// check_structure plus non-increasing scores within the first `tier1` scored
// entries and within the scored entries after them (tier1 == 0: one tier).
void check_contract(const std::vector<tw::Result>& r, std::size_t n, std::size_t tier1 = 0) {
  check_structure(r, n);
  for (std::size_t i = 1; i < r.size() && r[i].scored; ++i)
    if (i != tier1) REQUIRE(r[i].score <= r[i - 1].score);
}

std::size_t count_scored(const std::vector<tw::Result>& r) {
  return static_cast<std::size_t>(
      std::count_if(r.begin(), r.end(), [](const tw::Result& x) { return x.scored; }));
}

std::vector<std::size_t> sorted(std::vector<std::size_t> v) {
  std::sort(v.begin(), v.end());
  return v;
}

Signature sig(const tw::Config& c) {
  return {static_cast<int>(c.mt.m),
          static_cast<int>(c.mt.n),
          static_cast<int>(c.mt.k),
          static_cast<int>(c.mi.m),
          static_cast<int>(c.mi.n),
          static_cast<int>(c.mi.k),
          c.cache_hints_a,
          c.cache_hints_b};
}

void zero_all(CellTensors& t) {
  for (std::vector<float>* v : {&t.q_w0,
                                &t.q_b0,
                                &t.q_w2,
                                &t.q_b2,
                                &t.q_w4,
                                &t.q_b4,
                                &t.i_w0,
                                &t.i_b0,
                                &t.i_w2,
                                &t.i_b2,
                                &t.x_w0,
                                &t.x_b0,
                                &t.x_w2,
                                &t.x_b2})
    std::fill(v->begin(), v->end(), 0.0f);
}

ModelSpec split_spec(tw::WeightType wt = tw::WeightType::Fp32) {
  ModelSpec spec;
  spec.weight_type       = wt;
  const std::string mid  = "Mid|Mid|MidK|Bnone";
  const std::string bany = "Mid|Mid|MidK|Bany";
  const std::string neg  = "Small|Small|MidK|Bnone";
  spec.splits            = {
      {mid, 'M', 300, mid + "#M<=300", mid + "#M>300"},
      {mid + "#M<=300", 'K', 100, mid + "#M<=300#K<=100", mid + "#M<=300#K>100"},
      {mid + "#M>300", 'N', 400, mid + "#M>300#N<=400", mid + "#M>300#N>400"},
      {bany, 'B', 4, bany + "#B<=4", bany + "#B>4"},
      {neg, 'K', -1, neg + "#K<=-1", neg + "#K>-1"},
  };
  const std::string trained[] = {mid,
                                 mid + "#M<=300",
                                 mid + "#M<=300#K<=100",
                                 mid + "#M>300#N>400",
                                 bany + "#B<=4",
                                 neg + "#K>-1",
                                 kLarge};
  std::uint64_t seed          = 11;
  for (const std::string& label : trained) {
    CellSpec cell;
    cell.label = label;
    cell.seed  = seed++;
    spec.cells.push_back(cell);
  }
  // Odd widths: int4 tensors with an odd element count and kernel tails.
  spec.cells.back().embed_dim    = 5;
  spec.cells.back().hidden_dim   = 7;
  spec.cells.back().inter_hidden = 3;
  return spec;
}

}  // namespace

TEST_CASE("problems route to their base grid cell at every tier boundary", "[route]") {
  const tw::ModelPtr model = load_spec(grid_model());
  const std::size_t big    = std::numeric_limits<std::size_t>::max();
  for (std::size_t m : {std::size_t{0},
                        std::size_t{1},
                        std::size_t{32},
                        std::size_t{33},
                        std::size_t{128},
                        std::size_t{129},
                        std::size_t{512},
                        std::size_t{513},
                        big})
    for (std::size_t n : {std::size_t{1},
                          std::size_t{32},
                          std::size_t{33},
                          std::size_t{128},
                          std::size_t{129},
                          std::size_t{512},
                          std::size_t{513}})
      for (std::size_t k : {std::size_t{0},
                            std::size_t{32},
                            std::size_t{33},
                            std::size_t{512},
                            std::size_t{513},
                            big})
        for (std::size_t b : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
          const int cell = tw::route(*model, make_problem(m, n, k, b));
          INFO(m << "x" << n << "x" << k << " b" << b);
          REQUIRE(cell >= 0);
          REQUIRE(tw::cell_label(*model, cell) == base_label(m, n, k, b));
        }
}

TEST_CASE("split thresholds and nearest-trained-ancestor fallback", "[route]") {
  const tw::ModelPtr model = load_spec(split_spec());
  const auto label         = [&](std::size_t m, std::size_t n, std::size_t k, std::size_t b) {
    return tw::cell_label(*model, tw::route(*model, make_problem(m, n, k, b)));
  };
  const std::string mid = "Mid|Mid|MidK|Bnone";
  CHECK(label(300, 200, 100, 1) == mid + "#M<=300#K<=100");
  CHECK(label(300, 200, 101, 1) == mid + "#M<=300");
  CHECK(label(129, 512, 33, 1) == mid + "#M<=300#K<=100");
  CHECK(label(301, 400, 200, 1) == mid);
  CHECK(label(301, 401, 200, 1) == mid + "#M>300#N>400");
  CHECK(label(512, 512, 512, 1) == mid + "#M>300#N>400");
  CHECK(label(200, 200, 200, 4) == "Mid|Mid|MidK|Bany#B<=4");
  CHECK(label(200, 200, 200, 0) == "Mid|Mid|MidK|Bany#B<=4");
  CHECK(tw::route(*model, make_problem(200, 200, 200, 5)) == -1);
  CHECK(label(64, 64, 100, 1) == "Small|Small|MidK|Bnone#K>-1");
  CHECK(tw::route(*model, make_problem(8, 8, 8, 1)) == -1);
  CHECK(label(4096, 4096, 4096, 1) == kLarge);
}

TEST_CASE("an unroutable problem, an empty pool or invalid hardware score nothing", "[rank]") {
  const tw::ModelPtr model           = load_spec(split_spec());
  const std::vector<tw::Config> pool = {make_config(128, 128, 64), make_config(64, 64, 64)};
  const tw::CandidateSet set(model, pool);
  const tw::Problem routed   = make_problem(4096, 4096, 4096);
  const tw::Problem unrouted = make_problem(8, 8, 8);
  REQUIRE(count_scored(tw::rank_configs(*model, routed, test_hardware(), pool)) == 2);

  for (const std::vector<tw::Result>& r :
       {tw::rank_configs(*model, unrouted, test_hardware(), pool),
        set.rank(unrouted, test_hardware())}) {
    check_contract(r, pool.size());
    CHECK(count_scored(r) == 0);
  }
  CHECK(tw::rank_configs(*model, routed, test_hardware(), {}).empty());
  CHECK(tw::CandidateSet(model, {}).rank(routed, test_hardware()).empty());

  for (const tw::Hardware& hw : {tw::Hardware{0, 65536, 1u << 22},
                                 tw::Hardware{64, 0, 1u << 22},
                                 tw::Hardware{64, 65536, 0}}) {
    for (const std::vector<tw::Result>& r :
         {tw::rank_configs(*model, routed, hw, pool), set.rank(routed, hw, 100)}) {
      check_contract(r, pool.size());
      CHECK(count_scored(r) == 0);
    }
    const tw::Features f = tw::compute_features(*model, routed, pool[0], hw);
    CHECK(f.query.empty());
    CHECK(f.item.empty());
    CHECK(f.interaction.empty());
  }
}

TEST_CASE("CandidateSet requires a model", "[rank]") {
  CHECK_THROWS_AS(tw::CandidateSet(nullptr, {}), std::invalid_argument);
}

TEST_CASE("feasibility: a small batched problem must fit in one tile", "[feasibility]") {
  const tw::ModelPtr model           = load_spec(grid_model());
  const std::vector<tw::Config> pool = {
      make_config(64, 64, 64), make_config(128, 128, 64), make_config(256, 128, 64)};
  const auto scored = [&](const tw::Problem& p) {
    return sorted(scored_indices(tw::rank_configs(*model, p, test_hardware(), pool)));
  };
  CHECK(scored(make_problem(128, 128, 256, 2)) == std::vector<std::size_t>{1, 2});
  CHECK(scored(make_problem(128, 128, 256, 1)) == std::vector<std::size_t>{0, 1, 2});
  CHECK(scored(make_problem(128, 128, 1024, 2)) == std::vector<std::size_t>{0, 1, 2});
  CHECK(scored(make_problem(257, 128, 256, 2)) == std::vector<std::size_t>{0, 1, 2});
  CHECK(scored(make_problem(200, 100, 256, 3)) == std::vector<std::size_t>{2});
}

TEST_CASE("feasibility: Dot2 kernels only serve M < 3", "[feasibility]") {
  const tw::ModelPtr model           = load_spec(grid_model());
  const std::vector<tw::Config> pool = {make_config(16, 16, 64, 1, 1, 64), make_config(16, 16, 64)};
  const auto scored                  = [&](std::size_t m) {
    return sorted(
        scored_indices(tw::rank_configs(*model, make_problem(m, 16, 256), test_hardware(), pool)));
  };
  CHECK(scored(2) == std::vector<std::size_t>{0, 1});
  CHECK(scored(3) == std::vector<std::size_t>{1});
}

TEST_CASE("feasibility: non-temporal hints, per-operand pool availability", "[feasibility]") {
  const tw::ModelPtr model = load_spec(grid_model());
  const auto scored        = [&](const tw::Problem& p, const std::vector<tw::Config>& pool) {
    return sorted(scored_indices(tw::rank_configs(*model, p, test_hardware(), pool)));
  };
  using V = std::vector<std::size_t>;

  // N-skinny (B is the large operand): NT-B is required when the pool has one.
  const tw::Problem skinny_b = make_problem(64, 8192, 4096);
  CHECK(scored(skinny_b,
               {make_config(64, 256, 64, 16, 16, 32, 0, 0),
                make_config(64, 256, 64, 16, 16, 32, 0, 4),
                make_config(64, 256, 64, 16, 16, 32, 4, 0)}) == V{1});
  CHECK(scored(skinny_b,
               {make_config(64, 256, 64, 16, 16, 32, 0, 0),
                make_config(64, 256, 64, 16, 16, 32, 4, 0)}) == V{0, 1});
  CHECK(scored(skinny_b,
               {make_config(64, 256, 64, 16, 16, 32, 0, 0),
                make_config(64, 256, 64, 16, 16, 32, 0, 1)}) == V{0, 1});

  // M-skinny with A transposed: NT-A is required when the pool has one.
  const tw::Problem skinny_a = make_problem(8192, 64, 4096);
  CHECK(scored(skinny_a,
               {make_config(256, 64, 64, 16, 16, 32, 0, 0),
                make_config(256, 64, 64, 16, 16, 32, 4, 0),
                make_config(256, 64, 64, 16, 16, 32, 0, 4)}) == V{1});
  CHECK(scored(skinny_a,
               {make_config(256, 64, 64, 16, 16, 32, 0, 0),
                make_config(256, 64, 64, 16, 16, 32, 0, 4)}) == V{0, 1});
  const tw::Problem skinny_a_nn =
      make_problem(8192, 64, 4096, 1, tw::DataType::BFloat16, tw::Transpose::N, tw::Transpose::N);
  CHECK(scored(skinny_a_nn,
               {make_config(256, 64, 64, 16, 16, 32, 0, 0),
                make_config(256, 64, 64, 16, 16, 32, 4, 0)}) == V{0});

  // Aligned but not skinny, and unaligned K: hinted kernels are rejected.
  const std::vector<tw::Config> mixed = {make_config(256, 256, 64, 16, 16, 32, 0, 0),
                                         make_config(256, 256, 64, 16, 16, 32, 4, 0),
                                         make_config(256, 256, 64, 16, 16, 32, 0, 4),
                                         make_config(256, 256, 64, 16, 16, 32, 1, 0)};
  CHECK(scored(make_problem(4096, 4096, 4096), mixed) == V{0});
  CHECK(scored(make_problem(64, 8192, 4000), mixed) == V{0});
  CHECK(scored(make_problem(64, 8192, 4096),
               {make_config(64, 256, 40, 16, 16, 32, 0, 0),
                make_config(64, 256, 40, 16, 16, 32, 0, 4)}) == V{0});
}

TEST_CASE("feasibility: LDS capacity gate", "[feasibility]") {
  const tw::ModelPtr model = load_spec(grid_model());
  const auto scored        = [&](const tw::Problem& p, const std::vector<tw::Config>& pool) {
    return sorted(scored_indices(tw::rank_configs(*model, p, test_hardware(), pool)));
  };
  using V = std::vector<std::size_t>;
  CHECK(scored(make_problem(4096, 4096, 4096),
               {make_config(256, 256, 64), make_config(256, 256, 128)}) == V{0});
  CHECK(scored(make_problem(4096, 4096, 4096, 1, tw::DataType::Float),
               {make_config(128, 128, 64), make_config(256, 128, 64)}) == V{0});
  CHECK(scored(make_problem(4096, 4096, 4096, 1, tw::DataType::None), {make_config(64, 64, 64)})
            .empty());
}

TEST_CASE("whitelist tier, tier-2 depth and whitelist fallback", "[rank]") {
  const std::vector<tw::Config> pool = {make_config(64, 64, 64),
                                        make_config(256, 128, 64),
                                        make_config(128, 256, 64),
                                        make_config(128, 128, 64),
                                        make_config(512, 512, 64),
                                        make_config(64, 128, 64)};
  ModelSpec spec;
  CellSpec cell;
  cell.label               = kLarge;
  cell.signatures          = {sig(pool[1]), sig(pool[3]), sig(pool[4])};
  spec.cells               = {cell};
  const tw::ModelPtr model = load_spec(spec);
  const tw::Problem p      = make_problem(4096, 4096, 4096);
  const tw::Hardware& hw   = test_hardware();

  const std::vector<tw::Result> r0 = tw::rank_configs(*model, p, hw, pool);
  check_contract(r0, pool.size());
  CHECK(sorted(scored_indices(r0)) == std::vector<std::size_t>{1, 3});
  CHECK(same_results(tw::rank_configs(*model, p, hw, pool, 2), r0));

  const std::vector<tw::Result> r3 = tw::rank_configs(*model, p, hw, pool, 3);
  check_contract(r3, pool.size(), 2);
  REQUIRE(count_scored(r3) == 5);
  CHECK(std::equal(
      r0.begin(), r0.begin() + 2, r3.begin(), [](const tw::Result& a, const tw::Result& b) {
        return a.config_index == b.config_index && a.score == b.score;
      }));
  CHECK(sorted({r3[2].config_index, r3[3].config_index, r3[4].config_index}) ==
        std::vector<std::size_t>{0, 2, 5});
  CHECK(same_results(tw::rank_configs(*model, p, hw, pool, std::numeric_limits<std::size_t>::max()),
                     r3));

  // Only the LDS-infeasible kernel of the pool is whitelisted: every feasible
  // config forms tier 1 and there is no tier 2.
  const std::vector<tw::Config> no_match = {pool[0], pool[2], pool[4], pool[5]};
  const std::vector<tw::Result> fb       = tw::rank_configs(*model, p, hw, no_match);
  check_contract(fb, no_match.size());
  CHECK(sorted(scored_indices(fb)) == std::vector<std::size_t>{0, 1, 3});
  CHECK(same_results(tw::rank_configs(*model, p, hw, no_match, 100), fb));

  const tw::CandidateSet set(model, pool);
  CHECK(same_results(set.rank(p, hw), r0));
  CHECK(same_results(set.rank(p, hw, 3), r3));
}

TEST_CASE("ties keep input order", "[rank]") {
  ModelSpec spec;
  CellSpec cell;
  cell.label               = kLarge;
  cell.edit                = zero_all;
  spec.cells               = {cell};
  const tw::ModelPtr model = load_spec(spec);
  std::vector<tw::Config> pool;
  for (std::size_t mt : {256, 32, 128, 64, 16}) pool.push_back(make_config(mt, mt, 32));
  const std::vector<tw::Result> r =
      tw::rank_configs(*model, make_problem(4096, 4096, 4096), test_hardware(), pool);
  REQUIRE(count_scored(r) == pool.size());
  for (std::size_t i = 0; i < r.size(); ++i) {
    CHECK(r[i].config_index == i);
    CHECK(r[i].score == r[0].score);
  }
}

TEST_CASE("configs whose score is not finite are unscored", "[rank]") {
  ModelSpec spec;
  CellSpec cell;
  cell.label = kLarge;
  cell.edit  = [](CellTensors& t) {
    zero_all(t);
    std::fill(t.i_mean.begin(), t.i_mean.end(), 0.0f);
    std::fill(t.i_std.begin(), t.i_std.end(), 1.0f);
    t.i_mean[8] = 1.0f / 9.0f;  // occupancy / 9 is 0 after whitening for occupancy 1
    t.i_w0[8]   = 1e30f;
    for (std::size_t e = 0; e < 8; ++e) t.i_w2[e * 16] = 1e30f;
    std::fill(t.q_b4.begin(), t.q_b4.end(), 1.0f);
  };
  spec.cells                         = {cell};
  const tw::ModelPtr model           = load_spec(spec);
  const std::vector<tw::Config> pool = {make_config(128, 128, 64, 16, 16, 32, 0, 0, 1),
                                        make_config(128, 128, 64, 16, 16, 32, 0, 0, 9),
                                        make_config(64, 64, 64, 16, 16, 32, 0, 0, 1)};
  const tw::Problem p                = make_problem(4096, 4096, 4096);
  for (const std::vector<tw::Result>& r :
       {tw::rank_configs(*model, p, test_hardware(), pool),
        tw::CandidateSet(model, pool).rank(p, test_hardware())}) {
    check_contract(r, pool.size());
    CHECK(sorted(scored_indices(r)) == std::vector<std::size_t>{0, 2});
  }
  const std::vector<tw::Config> all_bad = {pool[1]};
  CHECK(count_scored(tw::rank_configs(*model, p, test_hardware(), all_bad)) == 0);
}

TEST_CASE("CandidateSet::rank is bitwise identical to rank_configs", "[rank]") {
  std::mt19937_64 rng(1234);
  const auto pick = [&](std::initializer_list<std::size_t> values) {
    return *(values.begin() + rng() % values.size());
  };
  const auto random_pool = [&](std::size_t n) {
    std::vector<tw::Config> pool;
    for (std::size_t i = 0; i < n; ++i) {
      // Sequenced draws: every compiler builds the same pools.
      const std::size_t mt_m = pick({16, 32, 64, 128, 192, 256, 512});
      const std::size_t mt_n = pick({16, 32, 64, 96, 128, 256});
      const std::size_t mt_k = pick({16, 32, 64, 128, 256});
      tw::Config c           = make_config(mt_m, mt_n, mt_k);
      const std::size_t mi   = rng() % 4;
      c.mi                   = mi == 0   ? tw::Dim3{1, 1, 64}
                               : mi == 1 ? tw::Dim3{32, 32, 16}
                               : mi == 2 ? tw::Dim3{16, 16, 128}
                                         : tw::Dim3{16, 16, 32};
      c.cache_hints_a        = static_cast<int>(pick({0, 0, 0, 1, 4}));
      c.cache_hints_b        = static_cast<int>(pick({0, 0, 0, 4}));
      c.occupancy            = static_cast<int>(1 + rng() % 4);
      c.grvw_a               = pick({1, 2, 4, 8, 16});
      c.grvw_b               = pick({1, 2, 4, 8, 16});
      c.gwvw_d               = pick({1, 2, 4, 8});
      c.index                = 1000 + i;
      pool.push_back(c);
    }
    return pool;
  };
  const auto random_problem = [&]() {
    const tw::DataType dts[] = {tw::DataType::BFloat16,
                                tw::DataType::Half,
                                tw::DataType::Float,
                                tw::DataType::Float8,
                                tw::DataType::Float8_fnuz,
                                tw::DataType::XFloat32};
    const std::size_t m      = 1 + rng() % 9000;
    const std::size_t n      = 1 + rng() % 9000;
    const std::size_t k      = 1 + rng() % 9000;
    const std::size_t batch  = pick({1, 1, 1, 2, 8});
    const tw::DataType dt    = dts[rng() % 6];
    const tw::Transpose ta   = rng() % 2 ? tw::Transpose::T : tw::Transpose::N;
    const tw::Transpose tb   = rng() % 2 ? tw::Transpose::T : tw::Transpose::N;
    tw::Problem p            = make_problem(m, n, k, batch, dt, ta, tb);
    if (rng() % 4 == 0) p.size.m = pick({32, 33, 128, 129, 300, 301, 512, 513});
    return p;
  };

  ModelSpec whitelisted = grid_model(tw::WeightType::Int4);
  for (CellSpec& cell : whitelisted.cells) {
    cell.embed_dim    = 24;
    cell.hidden_dim   = 40;
    cell.inter_hidden = 21;
    for (const tw::Config& c : random_pool(6)) cell.signatures.push_back(sig(c));
  }
  for (const ModelSpec& spec : {whitelisted, split_spec(tw::WeightType::Bf16)}) {
    const tw::ModelPtr model = load_spec(spec);
    for (int pool_round = 0; pool_round < 6; ++pool_round) {
      std::vector<tw::Config> pool = random_pool(1 + rng() % 40);
      for (const Signature& s : spec.cells[rng() % spec.cells.size()].signatures) {
        tw::Config c = make_config(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        pool.push_back(c);
      }
      const tw::CandidateSet set(model, pool);
      for (int i = 0; i < 60; ++i) {
        const tw::Problem p          = random_problem();
        const std::size_t min_scored = pick({0, 1, 3, 10, 1000});
        const std::vector<tw::Result> expect =
            tw::rank_configs(*model, p, test_hardware(), pool, min_scored);
        check_structure(expect, pool.size());
        REQUIRE(same_results(set.rank(p, test_hardware(), min_scored), expect));
        REQUIRE(same_results(set.rank(p, test_hardware(), min_scored), expect));
      }
    }
  }
}

TEST_CASE("packed weights score like their fp32 dequantization", "[rank]") {
  std::vector<tw::Config> pool;
  for (std::size_t mt : {32, 64, 128, 256})
    for (std::size_t mi_k : {16, 32}) pool.push_back(make_config(mt, 128, 64, 16, 16, mi_k));
  for (tw::WeightType wt : {tw::WeightType::Bf16, tw::WeightType::Int8, tw::WeightType::Int4}) {
    const ModelSpec packed = split_spec(wt);
    const tw::ModelPtr q   = load_spec(packed);
    const tw::ModelPtr f   = load_spec(dequantized(packed));
    const tw::CandidateSet s(q, pool);
    for (const tw::Problem& p : {make_problem(4096, 4096, 4096),
                                 make_problem(300, 200, 100),
                                 make_problem(301, 401, 200),
                                 make_problem(64, 64, 100)}) {
      const std::vector<tw::Result> expect = tw::rank_configs(*f, p, test_hardware(), pool, 100);
      REQUIRE(count_scored(expect) > 0);
      CHECK(same_results(tw::rank_configs(*q, p, test_hardware(), pool, 100), expect));
      CHECK(same_results(s.rank(p, test_hardware(), 100), expect));
    }
  }
}

// Scores are part of the model contract: they must not change with the host,
// the compiler or its flags. Updating these values means every shipped model
// ranks differently.
TEST_CASE("scores of a fixed synthetic model are stable", "[rank]") {
  ModelSpec spec = grid_model(tw::WeightType::Int4);
  for (CellSpec& c : spec.cells) {
    c.embed_dim    = 24;
    c.hidden_dim   = 40;
    c.inter_hidden = 21;
  }
  const tw::ModelPtr model = load_spec(spec);
  std::vector<tw::Config> pool;
  for (std::size_t mt : {32, 64, 128, 256})
    for (std::size_t mi_k : {16, 32}) pool.push_back(make_config(mt, 128, 64, 16, 16, mi_k));
  struct Expected {
    std::size_t index;
    std::uint64_t bits;
  };
  const std::pair<tw::Problem, std::array<Expected, 3>> cases[] = {
      {make_problem(4096, 4096, 4096),
       {{{7, 0x4096d12220000000ull}, {5, 0x4096946920000000ull}, {1, 0x40962c7840000000ull}}}},
      {make_problem(100, 3000, 700, 2),
       {{{7, 0x407faa7780000000ull}, {6, 0x407d1e4f60000000ull}, {5, 0x4075bc64a0000000ull}}}},
      {make_problem(17, 9000, 1500, 1, tw::DataType::Half),
       {{{6, 0x4032a49b00000000ull}, {4, 0x403007a2a0000000ull}, {7, 0x402d3cec40000000ull}}}}};
  for (const auto& c : cases) {
    const std::vector<tw::Result> r = tw::rank_configs(*model, c.first, test_hardware(), pool);
    for (std::size_t i = 0; i < 3; ++i) {
      std::uint64_t bits;
      std::memcpy(&bits, &r[i].score, sizeof bits);
      CHECK(r[i].config_index == c.second[i].index);
      CHECK(bits == c.second[i].bits);
    }
  }
}

TEST_CASE("every kernel variant returns identical bits", "[kernels]") {
  using tw::detail::Isa;
  const tw::detail::Kernels* scalar = tw::detail::kernels_for(Isa::Scalar);
  REQUIRE(scalar != nullptr);
  CHECK(tw::detail::kernels_for(tw::detail::kernels().isa) == &tw::detail::kernels());
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  int variants = 0;
  for (Isa isa : {Isa::Avx2, Isa::Avx512}) {
    const tw::detail::Kernels* k = tw::detail::kernels_for(isa);
    if (k == nullptr) continue;
    ++variants;
    for (std::size_t m : {1, 2, 3, 4, 5, 7, 8, 9, 13, 16}) {
      for (std::size_t kk = 0; kk <= 200; kk += (kk < 66 ? 1 : 13)) {
        std::vector<float> W(m * kk), b(m), x(kk), o1(m), o2(m);
        for (float& v : W) v = u(rng);
        for (float& v : b) v = u(rng);
        for (float& v : x) v = (rng() % 7 == 0) ? 0.0f : u(rng);
        scalar->linear(W.data(), b.data(), x.data(), m, kk, o1.data());
        k->linear(W.data(), b.data(), x.data(), m, kk, o2.data());
        REQUIRE(std::memcmp(o1.data(), o2.data(), m * sizeof(float)) == 0);
        scalar->linear_relu(W.data(), b.data(), x.data(), m, kk, o1.data());
        k->linear_relu(W.data(), b.data(), x.data(), m, kk, o2.data());
        REQUIRE(std::memcmp(o1.data(), o2.data(), m * sizeof(float)) == 0);
        const float d1 = scalar->dot(W.data(), x.data(), kk);
        const float d2 = k->dot(W.data(), x.data(), kk);
        REQUIRE(std::memcmp(&d1, &d2, sizeof d1) == 0);
        double ref = 0.0;
        for (std::size_t j = 0; j < kk; ++j) ref += static_cast<double>(W[j]) * x[j];
        REQUIRE(std::fabs(d1 - ref) <= 1e-5 * (1.0 + static_cast<double>(kk)));
      }
    }
  }
  if (variants == 0) WARN("only the scalar kernels are available on this host");
}

TEST_CASE("scores follow the two-tower formula", "[rank]") {
  ModelSpec spec;
  CellSpec cell;
  cell.label        = kLarge;
  cell.embed_dim    = 19;
  cell.hidden_dim   = 37;
  cell.inter_hidden = 23;
  cell.seed         = 99;
  // Constant item features: occupancy at its training value, log2(mt_m) and
  // log2(mi_k) away from theirs.
  cell.edit = [](CellTensors& t) {
    t.i_std[8]  = 2e-4f;
    t.i_mean[8] = static_cast<float>(1.0 / 9.0) - 2e-4f;
    t.i_std[0]  = 5e-4f;
    t.i_std[5]  = 0.0f;
  };
  spec.cells               = {cell};
  const tw::ModelPtr model = load_spec(spec);
  const CellTensors t      = tensors_of(cell);
  const tw::Problem p      = make_problem(3000, 5000, 7000);

  const auto layer = [](const std::vector<float>& W,
                        const std::vector<float>& b,
                        const std::vector<double>& x,
                        bool relu) {
    std::vector<double> out(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
      double v = b[i];
      for (std::size_t j = 0; j < x.size(); ++j)
        v += static_cast<double>(W[i * x.size() + j]) * x[j];
      out[i] = relu ? std::max(v, 0.0) : v;
    }
    return out;
  };
  const auto whiten = [](const std::vector<float>& f,
                         const std::vector<float>& mean,
                         const std::vector<float>& sd,
                         bool item) {
    std::vector<double> out(f.size());
    for (std::size_t j = 0; j < f.size(); ++j) {
      const double d = static_cast<double>(f[j]) - mean[j];
      out[j]         = item && sd[j] < 1e-3f && std::fabs(d) > 2.0 * sd[j]
                           ? 0.0
                           : d / (sd[j] < 1e-6f ? 1.0 : sd[j]);
    }
    return out;
  };

  for (const tw::Config& c : {make_config(256, 128, 64),
                              make_config(64, 64, 32, 32, 32, 16),
                              make_config(128, 256, 64, 16, 16, 128)}) {
    const tw::Features f = tw::compute_features(*model, p, c, test_hardware());
    REQUIRE(f.query.size() == 55);
    REQUIRE(f.item.size() == 12);
    REQUIRE(f.interaction.size() == 37);
    const std::vector<double> qe =
        layer(t.q_w4,
              t.q_b4,
              layer(t.q_w2,
                    t.q_b2,
                    layer(t.q_w0, t.q_b0, whiten(f.query, t.q_mean, t.q_std, false), true),
                    true),
              false);
    const std::vector<double> ie =
        layer(t.i_w2,
              t.i_b2,
              layer(t.i_w0, t.i_b0, whiten(f.item, t.i_mean, t.i_std, true), true),
              false);
    const std::vector<double> xh =
        layer(t.x_w0, t.x_b0, whiten(f.interaction, t.x_mean, t.x_std, false), true);
    double expect = 0.0;
    for (std::size_t e = 0; e < qe.size(); ++e) expect += qe[e] * ie[e];
    expect /= std::max(std::fabs(static_cast<double>(t.temperature)), 0.1);
    double inter = t.x_b2[0];
    for (std::size_t j = 0; j < xh.size(); ++j) inter += t.x_w2[j] * xh[j];
    expect += inter;

    const std::vector<tw::Result> r = tw::rank_configs(*model, p, test_hardware(), {c});
    REQUIRE(r.size() == 1);
    REQUIRE(r[0].scored);
    INFO("engine " << r[0].score << " reference " << expect);
    CHECK(std::fabs(r[0].score - expect) <= 1e-4 * std::max(1.0, std::fabs(expect)));
  }
}

TEST_CASE("features use the model's arch constants and the documented formulas", "[features]") {
  ModelSpec spec;
  spec.parallel_mi_cu    = 4.0;
  spec.mi_default_cycles = 40.0;
  spec.bw[0]             = 1e-5;
  spec.bw[1]             = 0.002;
  spec.bw[2]             = 0.05;
  spec.mi_table          = {{16, 16, 32, static_cast<int>(tw::DataType::BFloat16), 16.0},
                            {16, 16, 128, static_cast<int>(tw::DataType::Float8), 8.0},
                            {1, 16, 64, static_cast<int>(tw::DataType::BFloat16), 12.0}};
  CellSpec cell;
  cell.label               = kLarge;
  spec.cells               = {cell};
  const tw::ModelPtr model = load_spec(spec);
  const tw::Hardware hw{64, 65536, 4u << 20};

  tw::Problem p      = make_problem(1024, 2000, 3000);
  const tw::Config c = make_config(256, 128, 64);
  tw::Features f     = tw::compute_features(*model, p, c, hw);
  CHECK(f.query[0] == static_cast<float>(std::log2(1024.0)));
  CHECK(f.query[1] == static_cast<float>(std::log2(2000.0)));
  CHECK(f.query[9] == 1.0f);
  CHECK(f.query[10] == 0.0f);
  CHECK(f.query[12] == 1.0f);
  CHECK(f.query[13] == 0.0f);
  CHECK(f.query[14] == 3.0f);
  CHECK(f.query[21] == 0.0f);
  CHECK(f.query[27] == static_cast<float>(2000 % 256));
  CHECK(f.query[36] == static_cast<float>(std::min(2000 % 64, 64 - 2000 % 64) / 64.0));
  CHECK(f.item[0] == 8.0f);
  CHECK(f.item[1] == 7.0f);
  CHECK(f.item[2] == 6.0f);
  CHECK(f.item[8] == static_cast<float>(1 / 9.0));
  CHECK(f.item[9] == 1.0f);
  CHECK(f.interaction[33] == 4.0f);
  const double tiles  = 4.0 * 16.0;
  const double active = std::min(tiles, 64.0);
  CHECK(f.interaction[35] ==
        static_cast<float>(std::min(1.0, 1e-5 * active * active + 0.002 * active + 0.05)));
  CHECK(f.interaction[36] == static_cast<float>(active / 64.0));

  tw::Config f8 = make_config(256, 128, 128, 16, 16, 128);
  p.mi_dtype    = tw::DataType::Float8_fnuz;
  CHECK(tw::compute_features(*model, p, f8, hw).interaction[33] == 2.0f);
  p.mi_dtype = tw::DataType::Float8;
  CHECK(tw::compute_features(*model, p, f8, hw).interaction[33] == 2.0f);
  p.mi_dtype = tw::DataType::Half;
  CHECK(tw::compute_features(*model, p, f8, hw).interaction[33] == 10.0f);
  p.mi_dtype = tw::DataType::BFloat16;
  CHECK(tw::compute_features(*model, p, make_config(64, 64, 64, 0, 16, 64), hw).interaction[33] ==
        3.0f);

  spec.parallel_mi_cu     = 0.5;
  const tw::ModelPtr slow = load_spec(spec);
  CHECK(tw::compute_features(*slow, make_problem(1024, 2000, 3000), c, hw).interaction[33] ==
        16.0f);
}
