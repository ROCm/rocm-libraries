// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The cache limit: what it counts, what it releases, and that rankings never
// depend on it.

#include "model_impl.hpp"
#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

namespace tw = tilewright;
using namespace tilewright_test;

namespace {

constexpr std::size_t kAll       = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kUnlimited = std::numeric_limits<std::size_t>::max();

// Restores the process-wide limit when a test ends.
struct LimitGuard {
  std::size_t saved = tw::cache_limit();
  LimitGuard()      = default;
  ~LimitGuard() { tw::set_cache_limit(saved); }
};

std::vector<tw::Config> random_pool(std::mt19937_64& rng, std::size_t n) {
  std::vector<tw::Config> pool;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t mt[] = {32, 64, 128, 256};
    tw::Config c           = make_config(mt[rng() % 4], mt[rng() % 4], rng() % 2 ? 32 : 64);
    c.occupancy            = static_cast<int>(1 + rng() % 3);
    c.grvw_a               = rng() % 2 ? 4 : 8;
    c.index                = i;
    if (rng() % 3 == 0 && i > 0) {
      c       = pool[rng() % i];
      c.index = i;
    }
    pool.push_back(c);
  }
  return pool;
}

tw::Problem random_problem(std::mt19937_64& rng) {
  const tw::DataType dts[] = {tw::DataType::BFloat16, tw::DataType::Half, tw::DataType::Float8};
  return make_problem(1 + rng() % 6000,
                      1 + rng() % 6000,
                      1 + rng() % 6000,
                      rng() % 4 == 0 ? 3 : 1,
                      dts[rng() % 3],
                      rng() % 2 ? tw::Transpose::T : tw::Transpose::N,
                      rng() % 2 ? tw::Transpose::T : tw::Transpose::N);
}

ModelSpec listed_grid(tw::WeightType wt, const std::vector<tw::Config>& pool, std::uint64_t seed) {
  ModelSpec spec = grid_model(wt);
  std::mt19937_64 rng(seed);
  for (CellSpec& cell : spec.cells)
    for (const tw::Config& c : pool)
      if (rng() % 5 == 0)
        cell.signatures.push_back({static_cast<int>(c.mt.m),
                                   static_cast<int>(c.mt.n),
                                   static_cast<int>(c.mt.k),
                                   static_cast<int>(c.mi.m),
                                   static_cast<int>(c.mi.n),
                                   static_cast<int>(c.mi.k),
                                   c.cache_hints_a,
                                   c.cache_hints_b});
  return spec;
}

}  // namespace

TEST_CASE("rankings do not depend on the cache limit", "[cache]") {
  const LimitGuard guard;
  std::mt19937_64 rng(77);
  const std::vector<tw::Config> pool = random_pool(rng, 60);
  ModelSpec v2                       = listed_grid(tw::WeightType::Int4, pool, 1);
  ModelSpec v3                       = as_v3(listed_grid(tw::WeightType::Int8, pool, 2));
  v3.v3_cascade_keys                 = 4;
  v3.schedules                       = 0x7;
  v3.min_cu_fraction                 = 0.25;
  for (const ModelSpec& spec : {v2, v3}) {
    std::vector<tw::Problem> problems;
    std::vector<std::size_t> depth;
    for (int i = 0; i < 40; ++i) {
      problems.push_back(random_problem(rng));
      const std::size_t depths[] = {0, 1, 5, 20, kAll};
      depth.push_back(depths[rng() % 5]);
    }
    tw::set_cache_limit(kUnlimited);
    std::vector<std::vector<tw::Result>> expect;
    {
      const tw::ModelPtr model = load_spec(spec);
      const tw::CandidateSet set(model, pool);
      for (std::size_t i = 0; i < problems.size(); ++i)
        expect.push_back(set.rank(problems[i], test_hardware(), depth[i]));
    }
    const std::size_t cell_bytes = tw::detail::cell_weight_bytes(*load_spec(spec), 0);
    for (std::size_t limit : {std::size_t{0}, std::size_t{1}, 3 * cell_bytes, kUnlimited}) {
      tw::set_cache_limit(limit);
      const tw::ModelPtr model = load_spec(spec);
      const tw::CandidateSet set(model, pool);
      const tw::CandidateSet other(model, pool);
      for (int round = 0; round < 2; ++round)
        for (std::size_t i = 0; i < problems.size(); ++i) {
          INFO("v" << spec.version << " limit " << limit << " problem " << i);
          CHECK(same_results(set.rank(problems[i], test_hardware(), depth[i]), expect[i]));
          CHECK(same_results(other.rank_top(problems[i], test_hardware(), {}, depth[i]),
                             scored_prefix(expect[i], depth[i])));
          CHECK(same_results(tw::rank_configs(*model, problems[i], test_hardware(), pool, depth[i]),
                             expect[i]));
        }
    }
  }
}

TEST_CASE("the cache counts expanded cells and candidate-set caches", "[cache]") {
  const LimitGuard guard;
  tw::set_cache_limit(kUnlimited);
  const std::size_t before = tw::cache_usage();
  std::mt19937_64 rng(5);
  const std::vector<tw::Config> pool = random_pool(rng, 30);
  for (std::uint8_t clip : {std::uint8_t{0}, std::uint8_t{1}}) {
    ModelSpec spec = as_v3(listed_grid(tw::WeightType::Int4, pool, 3));
    spec.clip      = clip;
    {
      const tw::ModelPtr model = load_spec(spec);
      const tw::ModelInfo info = tw::describe(*model);
      CHECK(info.packed_bytes == model->payload.size());
      std::size_t expanded = 0;
      for (std::size_t c = 0; c < model->cells.size(); ++c) {
        const std::size_t usage                                = tw::cache_usage();
        const std::shared_ptr<const tw::detail::CellWeights> w = model->weights(c);
        CHECK(tw::detail::weight_bytes(*w) == tw::detail::cell_weight_bytes(*model, c));
        CHECK(tw::cache_usage() == usage + tw::detail::weight_bytes(*w));
        CHECK(model->weights(c) == w);
        CHECK(tw::cache_usage() == usage + tw::detail::weight_bytes(*w));
        expanded += tw::detail::weight_bytes(*w);
      }
      CHECK(info.expanded_bytes == expanded);
      CHECK(tw::cache_usage() == before + expanded);

      const tw::CandidateSet set(model, pool);
      set.rank(make_problem(4096, 4096, 4096), test_hardware(), kAll);
      CHECK(tw::cache_usage() > before + expanded);
    }
    CHECK(tw::cache_usage() == before);
  }
}

TEST_CASE("CandidateSet::prepare builds what rankings would build on first use",
          "[cache][prepare]") {
  const LimitGuard guard;
  tw::set_cache_limit(kUnlimited);
  std::mt19937_64 rng(23);
  const std::vector<tw::Config> pool = random_pool(rng, 40);
  const ModelSpec spec               = listed_grid(tw::WeightType::Int4, pool, 6);
  std::vector<tw::Problem> problems;
  std::vector<std::size_t> depth;
  for (int i = 0; i < 30; ++i) {
    problems.push_back(random_problem(rng));
    depth.push_back(i % 3 == 0 ? kAll : 1);
  }
  std::vector<std::vector<tw::Result>> expect;
  {
    const tw::CandidateSet plain(load_spec(spec), pool);
    for (std::size_t i = 0; i < problems.size(); ++i)
      expect.push_back(plain.rank(problems[i], test_hardware(), depth[i]));
  }
  const auto ranks_alike = [&](const tw::CandidateSet& set, bool builds_nothing) {
    for (std::size_t i = 0; i < problems.size(); ++i) {
      const std::size_t usage = tw::cache_usage();
      INFO("problem " << i);
      CHECK(same_results(set.rank(problems[i], test_hardware(), depth[i]), expect[i]));
      if (builds_nothing) CHECK(tw::cache_usage() == usage);
    }
  };

  {
    const tw::ModelPtr model = load_spec(spec);
    const tw::CandidateSet set(model, pool);
    const std::size_t before = tw::cache_usage();
    set.prepare();
    const std::size_t prepared = tw::cache_usage();
    CHECK(prepared > before + tw::describe(*model).expanded_bytes);
    ranks_alike(set, true);
    set.prepare();
    CHECK(tw::cache_usage() == prepared);
  }
  {
    const tw::ModelPtr model = load_spec(spec);
    const tw::CandidateSet set(model, pool);
    const std::size_t before = tw::cache_usage();
    set.prepare(std::vector<int>{-1, 100000});
    set.prepare(std::vector<int>{});
    CHECK(tw::cache_usage() == before);
    const tw::Problem a = make_problem(4096, 4096, 4096), b = make_problem(100, 3000, 700);
    set.prepare({tw::route(*model, a)});
    const std::size_t one = tw::cache_usage();
    CHECK(one > before);
    set.rank(a, test_hardware(), kAll);
    CHECK(tw::cache_usage() == one);
    set.rank(b, test_hardware(), kAll);
    CHECK(tw::cache_usage() > one);
  }
  {
    const tw::ModelPtr model = load_spec(spec);
    const tw::CandidateSet set(model, pool);
    for (const tw::Problem& p : problems) set.prepare(p, test_hardware());
    ranks_alike(set, true);
    set.prepare(problems[0], tw::Hardware{});
  }
  {
    ModelSpec one_cell;
    one_cell.cells           = {spec.cells[0]};
    const tw::ModelPtr model = load_spec(one_cell);
    const tw::CandidateSet set(model, pool);
    const std::size_t before = tw::cache_usage();
    set.prepare(make_problem(4096, 4096, 4096), test_hardware());
    CHECK(tw::cache_usage() == before);
    set.prepare();
    CHECK(tw::cache_usage() > before);
  }
}

TEST_CASE("the cache stays within its limit and releases the least recently used parts",
          "[cache]") {
  const LimitGuard guard;
  tw::set_cache_limit(kUnlimited);
  std::mt19937_64 rng(11);
  const std::vector<tw::Config> pool = random_pool(rng, 40);
  const tw::ModelPtr model           = load_spec(listed_grid(tw::WeightType::Int4, pool, 4));
  const std::size_t cell_bytes       = tw::detail::cell_weight_bytes(*model, 0);
  const std::size_t before           = tw::cache_usage();
  const tw::Problem a = make_problem(4096, 4096, 4096), b = make_problem(100, 3000, 700);

  // Room for a few cells: the parts of the problem ranked last stay.
  tw::set_cache_limit(before + 6 * cell_bytes);
  const tw::CandidateSet set(model, pool);
  std::size_t largest = 0;
  for (int i = 0; i < 200; ++i) {
    set.rank(random_problem(rng), test_hardware(), i % 3 == 0 ? kAll : 1);
    largest = std::max(largest, tw::cache_usage());
  }
  CHECK(largest <= before + 6 * cell_bytes);
  set.rank(a, test_hardware(), kAll);
  set.rank(b, test_hardware(), kAll);
  const std::shared_ptr<const tw::detail::CellWeights> wa =
      model->weights(static_cast<std::size_t>(tw::route(*model, a)));
  const std::shared_ptr<const tw::detail::CellWeights> wb =
      model->weights(static_cast<std::size_t>(tw::route(*model, b)));
  const std::size_t usage = tw::cache_usage();
  CHECK(model->weights(static_cast<std::size_t>(tw::route(*model, a))) == wa);
  CHECK(tw::cache_usage() == usage);

  // A lower limit releases at once; data still held stays valid.
  tw::set_cache_limit(before);
  CHECK(tw::cache_usage() <= before);
  CHECK(wa->q_b0.size() == model->cells[static_cast<std::size_t>(tw::route(*model, a))].hidden_dim);
  CHECK(same_results(set.rank(a, test_hardware(), kAll),
                     tw::rank_configs(*model, a, test_hardware(), pool, kAll)));
  CHECK(model->weights(static_cast<std::size_t>(tw::route(*model, b))) != wb);
  // With no room, only the part published last stays.
  tw::set_cache_limit(0);
  CHECK(tw::cache_limit() == 0);
  CHECK(tw::cache_usage() == 0);
  set.rank(a, test_hardware(), kAll);
  CHECK(tw::cache_usage() > 0);
  CHECK(tw::cache_usage() <= 2 * cell_bytes);
}
