// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The selection-time options (TW_LEVER_PREFETCH, TW_LEVER_MEMO); built, with
// the rest of the suite, into tilewright-levers-tests (TW_LEVER_MEMO_CHECK=1)
// and tilewright-levers-nocheck-tests (TW_LEVER_MEMO_CHECK=0) only.

#include "model_impl.hpp"
#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !TW_LEVER_PREFETCH || !TW_LEVER_MEMO || !defined(TW_LEVER_MEMO_CHECK)
#error "test_levers.cpp needs TW_LEVER_PREFETCH, TW_LEVER_MEMO and TW_LEVER_MEMO_CHECK (1 or 0)"
#endif

namespace tw = tilewright;
using namespace tilewright_test;

namespace {

constexpr std::size_t kAll = static_cast<std::size_t>(-1);

// FNV-1a 64 over a v3 view key without spec attributes: the signature, each
// field as 8 little-endian bytes.
std::uint64_t key_hash(const tw::Config& c) {
  const std::int64_t v[] = {static_cast<std::int64_t>(c.mt.m),
                            static_cast<std::int64_t>(c.mt.n),
                            static_cast<std::int64_t>(c.mt.k),
                            static_cast<std::int64_t>(c.mi.m),
                            static_cast<std::int64_t>(c.mi.n),
                            static_cast<std::int64_t>(c.mi.k),
                            c.cache_hints_a,
                            c.cache_hints_b};
  std::uint64_t h        = 0xCBF29CE484222325ull;
  for (std::int64_t x : v)
    for (int b = 0; b < 8; ++b)
      h = (h ^ static_cast<unsigned char>(static_cast<std::uint64_t>(x) >> (8 * b))) *
          0x100000001B3ull;
  return h;
}

// Six signatures, then a variant of each that differs in every other Config
// field: indices s and s + 6 share a view key.
std::vector<tw::Config> variant_pool() {
  std::vector<tw::Config> pool;
  for (std::size_t mt : {32, 64, 128, 256}) pool.push_back(make_config(mt, 128, 64));
  pool.push_back(make_config(128, 64, 32, 16, 16, 32, 1, 0));
  pool.push_back(make_config(64, 256, 64, 32, 32, 16));
  for (std::size_t s = 0; s < 6; ++s) {
    tw::Config v = pool[s];
    v.occupancy  = 4;
    v.grvw_b     = 2;
    v.attributes = {{"unread", 3}};
    pool.push_back(v);
  }
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  return pool;
}

// An MLREC_v3 grid model for every schedule and the CU budgets from
// `min_cu_fraction` of the device up whose cells all whitelist the view keys
// of `whitelisted`.
tw::ModelPtr whitelist_model(const std::vector<tw::Config>& whitelisted,
                             bool launch_keys       = false,
                             double min_cu_fraction = 1.0 / 64.0) {
  ModelSpec spec       = as_v3(grid_model());
  spec.schedules       = 0x7;
  spec.min_cu_fraction = min_cu_fraction;
  if (launch_keys) spec.v3_serving_flags |= 0x10;
  for (CellSpec& cell : spec.cells)
    for (const tw::Config& c : whitelisted) cell.view_key_hashes.push_back(key_hash(c));
  return load_spec(spec);
}

tw::ExecutionContext context(std::size_t cu_budget, tw::Schedule schedule) {
  tw::ExecutionContext ctx;
  ctx.cu_budget = cu_budget;
  ctx.schedule  = schedule;
  return ctx;
}

// Plans by config index, plus `shift` workgroups, with the tile queue or the
// partial reduction turned over when asked; launch key: occupancy.
struct ShiftLaunch final : tw::LaunchModel {
  std::size_t shift    = 0;
  bool other_queue     = false;
  bool other_reduction = false;

  tw::LaunchPlan plan(const tw::Config& c,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    const bool tree = (c.index % 3 == 0) != other_reduction;
    return {(c.index % 2 == 0) != other_queue,
            7 + 31 * c.index + shift,
            tree ? tw::PartialReduction::Tree : tw::PartialReduction::None};
  }
  std::uint64_t launch_key(const tw::Config& c) const override {
    return static_cast<std::uint64_t>(c.occupancy);
  }
};

// Plans as ShiftLaunch does, shifted by the CU budget they are asked for.
struct BudgetShiftLaunch final : tw::LaunchModel {
  tw::LaunchPlan plan(const tw::Config& c,
                      const tw::Problem&,
                      const tw::ExecutionContext& context) const override {
    const bool tree = c.index % 3 == 0;
    return {c.index % 2 == 0,
            7 + 31 * c.index + context.cu_budget,
            tree ? tw::PartialReduction::Tree : tw::PartialReduction::None};
  }
};

std::uint64_t replays() { return tw::detail::memo_counts().replays; }
std::uint64_t plan_changes() { return tw::detail::memo_counts().plan_changes; }

}  // namespace

TEST_CASE("the memo answers a ranking that repeats the thread's last one, as a fresh ranking",
          "[levers][memo]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  const std::vector<tw::Config> listed(pool.begin(), pool.begin() + 4);
  const tw::ModelPtr m = whitelist_model(listed);
  auto launch          = std::make_shared<ShiftLaunch>();
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
  const tw::CandidateSet set(m, pool, options);
  // Index i is listed exactly when i % 6 < 4.
  const auto listed_only = [](const std::vector<tw::Result>& r) {
    for (const tw::Result& x : r)
      if (x.scored && x.config_index % 6 >= 4) return false;
    return !scored_indices(r).empty();
  };
  std::vector<tw::Result> top;
  for (const tw::Problem& p : {make_problem(1000, 2000, 512),
                               make_problem(4096, 4096, 256),
                               make_problem(640, 3000, 1024, 3)})
    for (const tw::ExecutionContext& ctx :
         {tw::ExecutionContext{}, context(16, tw::Schedule::Dynamic)}) {
      INFO("m " << p.size.m << " cu_budget " << ctx.cu_budget);
      const std::vector<tw::Result> fresh = tw::rank_configs(*m, p, hw, ctx, pool, options, 1);
      REQUIRE(listed_only(fresh));
      const std::vector<tw::Result> fresh4 = tw::rank_configs(*m, p, hw, ctx, pool, options, 4);
      const std::uint64_t before           = replays();

      // TensileLite's re-rank: rank_top(1), then rank of the same problem.
      set.rank_top(p, hw, ctx, 1, top);
      CHECK(same_results(top, scored_prefix(fresh, 1)));
      CHECK(replays() == before);
      CHECK(same_results(set.rank(p, hw, ctx, 1), fresh));
      CHECK(replays() == before + 1);
      // A replay is spent: the next ranking is fresh, and is remembered.
      CHECK(same_results(set.rank(p, hw, ctx, 1), fresh));
      CHECK(replays() == before + 1);
      set.rank_top(p, hw, ctx, 4, top);
      CHECK(same_results(top, scored_prefix(fresh4, 4)));
      CHECK(replays() == before + 2);
      CHECK(same_results(set.rank(p, hw, ctx, 0), tw::rank_configs(*m, p, hw, ctx, pool, options)));
    }
}

TEST_CASE("the memo leaves every other ranking to a fresh one", "[levers][memo]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  const std::vector<tw::Config> listed(pool.begin(), pool.begin() + 4);
  const tw::ModelPtr m = whitelist_model(listed);
  auto launch          = std::make_shared<ShiftLaunch>();
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
  const tw::CandidateSet set(m, pool, options);
  std::vector<tw::Config> rotated(pool.begin() + 1, pool.end());
  rotated.push_back(pool.front());
  for (std::size_t i = 0; i < rotated.size(); ++i) rotated[i].index = i;
  const tw::CandidateSet other_pool(m, rotated, options);
  const tw::Problem p = make_problem(1000, 2000, 512);
  const tw::Problem q = make_problem(1000, 2000, 513);
  const tw::ExecutionContext all, part = context(16, tw::Schedule::Default);
  std::vector<tw::Result> top;
  // A fresh rank_top(1) of p, remembered: the ranking before it is of q.
  const auto first = [&] {
    set.rank_top(q, hw, all, 1, top);
    const std::uint64_t n = replays();
    set.rank_top(p, hw, all, 1, top);
    CHECK(replays() == n);
    return n;
  };

  std::uint64_t n = first();
  CHECK(same_results(set.rank(q, hw, all, 1), tw::rank_configs(*m, q, hw, all, pool, options, 1)));
  CHECK(replays() == n);

  n = first();
  set.rank_top(p, hw, part, 1, top);
  CHECK(same_results(top, scored_prefix(tw::rank_configs(*m, p, hw, part, pool, options, 1), 1)));
  CHECK(replays() == n);

  n = first();
  CHECK(same_results(other_pool.rank(p, hw, all, 1),
                     tw::rank_configs(*m, p, hw, all, rotated, options, 1)));
  CHECK(replays() == n);

  // More configs than the whitelist holds: a second tier.
  n = first();
  set.rank_top(p, hw, all, 9, top);
  CHECK(same_results(top, scored_prefix(tw::rank_configs(*m, p, hw, all, pool, options, 9), 9)));
  CHECK(replays() == n);

  // A ranking without a pool in between.
  const std::vector<tw::Result> fresh = tw::rank_configs(*m, p, hw, all, pool, options, 1);
  n                                   = first();
  CHECK(same_results(tw::rank_configs(*m, p, hw, all, pool, options, 1), fresh));
  CHECK(same_results(set.rank(p, hw, all, 1), fresh));
  CHECK(replays() == n);

  // A prior orders ties, so the remembered order does not apply.
  const tw::CandidateSet with_prior(m, pool, tw::PoolOptions{tw::TieBreak::Prior, launch});
  const std::vector<double> prior(pool.size(), 1.0);
  with_prior.rank_top(p, hw, all, 1, top);
  n = replays();
  CHECK(same_results(with_prior.rank(p, hw, all, 1, &prior), fresh));
  CHECK(replays() == n);

  // Resetting the scratch's epoch forgets the memo, even at the epoch it
  // remembered (the reset forgets the memo, so rank_top ranks at 101).
  tw::detail::set_rank_epoch(100);
  set.rank_top(p, hw, all, 1, top);
  n = replays();
  tw::detail::set_rank_epoch(101);
  CHECK(same_results(set.rank(p, hw, all, 1), fresh));
  CHECK(replays() == n);

  // A pool built where a destroyed one was (as an allocator tends to place
  // it) is another pool.
  {
    const std::vector<tw::Result> fresh_rotated =
        tw::rank_configs(*m, p, hw, all, rotated, options, 1);
    auto gone = std::make_unique<tw::CandidateSet>(m, pool, options);
    gone->rank_top(q, hw, all, 1, top);
    gone->rank_top(p, hw, all, 1, top);
    gone.reset();
    const tw::CandidateSet next(m, rotated, options);
    n = replays();
    CHECK(same_results(next.rank(p, hw, all, 1), fresh_rotated));
    CHECK(replays() == n);
  }

  // Each thread has its own.
  n = first();
  std::vector<tw::Result> theirs;
  std::uint64_t their_replays = 1;
  std::thread([&] {
    theirs        = set.rank(p, hw, all, 1);
    their_replays = replays();
  }).join();
  CHECK(same_results(theirs, fresh));
  CHECK(their_replays == 0);
  CHECK(same_results(set.rank(p, hw, all, 1), fresh));
  CHECK(replays() == n + 1);
}

TEST_CASE("the memo ranks afresh when the launch model now plans a key differently",
          "[levers][memo]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  const std::vector<tw::Config> listed(pool.begin(), pool.begin() + 4);
  const tw::ModelPtr m = whitelist_model(listed);
  auto launch          = std::make_shared<ShiftLaunch>();
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
  const tw::CandidateSet set(m, pool, options);
  const tw::Problem p = make_problem(1000, 2000, 512);
  const tw::ExecutionContext all;
  // Field 0 plans as before; fields 1 to 3 each change one field of a plan.
  const char* const fields[] = {"none", "grid_wgs", "dynamic_queue", "partial_reduction"};
  const auto plan_with       = [&](int field) {
    launch->shift           = field == 1 ? 1000 : 0;
    launch->other_queue     = field == 2;
    launch->other_reduction = field == 3;
  };
  plan_with(0);
  const std::vector<tw::Result> before = tw::rank_configs(*m, p, hw, all, pool, options, 1);
  plan_with(1);
  REQUIRE_FALSE(same_results(before, tw::rank_configs(*m, p, hw, all, pool, options, 1)));

  std::vector<tw::Result> top;
  for (int field = 1; field <= 3; ++field) {
    INFO("plan field " << fields[field]);
    plan_with(field);
    const std::vector<tw::Result> after = tw::rank_configs(*m, p, hw, all, pool, options, 1);
    plan_with(0);
    set.rank_top(p, hw, all, 1, top);
    plan_with(field);
    const std::uint64_t n = replays(), changed = plan_changes();
    const std::vector<tw::Result> got = set.rank(p, hw, all, 1);
#if TW_LEVER_MEMO_CHECK
    CHECK(same_results(got, after));
    CHECK(replays() == n);
    CHECK(plan_changes() == changed + 1);

    // That fresh ranking is remembered with its own plans, which still hold.
    set.rank_top(p, hw, all, 1, top);
    CHECK(same_results(top, scored_prefix(after, 1)));
    CHECK(replays() == n + 1);
    CHECK(plan_changes() == changed + 1);
#else
    // Without the check the memo relies on the launch model planning alike
    // (README.md) and answers with the remembered ranking.
    CHECK(same_results(got, before));
    CHECK(replays() == n + 1);
    CHECK(plan_changes() == changed);
#endif
  }
}

TEST_CASE("the memo leaves a ranking that differs in one input to a fresh one", "[levers][memo]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  const std::vector<tw::Config> listed(pool.begin(), pool.begin() + 4);
  const tw::ModelPtr m = whitelist_model(listed);
  auto launch          = std::make_shared<ShiftLaunch>();
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
  const tw::CandidateSet set(m, pool, options);
  struct Inputs {
    tw::Problem p;
    tw::Hardware h;
    tw::ExecutionContext c;
  };
  struct Change {
    const char* field;
    Inputs from, to;
  };
  // One change for every field same_ranking_inputs compares, each between two
  // rankings of one cell that the memo answers on their own, so that only
  // that field's compare tells them apart.
  const Inputs base{make_problem(1000, 2000, 512), hw, tw::ExecutionContext{}};
  Inputs batched  = base;
  batched.p.batch = 2;
  std::vector<Change> changes;
  const auto change = [&](const char* field, const Inputs& from, auto&& edit) {
    Change x{field, from, from};
    edit(x.to);
    changes.push_back(x);
  };
  change("size.m", base, [](Inputs& x) { x.p.size.m += 1; });
  change("size.n", base, [](Inputs& x) { x.p.size.n += 1; });
  change("size.k", base, [](Inputs& x) { x.p.size.k -= 12; });
  change("batch", batched, [](Inputs& x) { x.p.batch += 1; });
  change("a_transpose", base, [](Inputs& x) { x.p.a_transpose = tw::Transpose::N; });
  change("b_transpose", base, [](Inputs& x) { x.p.b_transpose = tw::Transpose::T; });
  change("a_dtype", base, [](Inputs& x) { x.p.a_dtype = tw::DataType::Half; });
  change("b_dtype", base, [](Inputs& x) { x.p.b_dtype = tw::DataType::Half; });
  change("c_dtype", base, [](Inputs& x) { x.p.c_dtype = tw::DataType::Half; });
  change("d_dtype", base, [](Inputs& x) { x.p.d_dtype = tw::DataType::Half; });
  change("mi_dtype", base, [](Inputs& x) { x.p.mi_dtype = tw::DataType::Half; });
  change("N_CU", base, [](Inputs& x) { x.h.N_CU = 80; });
  change("lds_capacity", base, [](Inputs& x) { x.h.lds_capacity *= 2; });
  change("L2_capacity", base, [](Inputs& x) { x.h.L2_capacity *= 2; });
  change("cu_budget", base, [](Inputs& x) { x.c.cu_budget = 16; });
  change("schedule", base, [](Inputs& x) { x.c.schedule = tw::Schedule::Dynamic; });

  std::vector<tw::Result> top;
  // rank_top(1), then rank of the same inputs: whether the memo answered it,
  // as a fresh ranking.
  const auto answered = [&](const Inputs& x, const std::vector<tw::Result>& fresh) {
    set.rank_top(x.p, x.h, x.c, 1, top);
    const std::uint64_t n = replays();
    const bool same       = same_results(set.rank(x.p, x.h, x.c, 1), fresh);
    return same && replays() == n + 1;
  };
  for (const Change& x : changes) {
    INFO("changed: " << x.field);
    const Inputs &a = x.from, &b = x.to;
    REQUIRE(tw::detail::route_cell(*m, b.p) == tw::detail::route_cell(*m, a.p));
    const std::vector<tw::Result> fresh_a = tw::rank_configs(*m, a.p, a.h, a.c, pool, options, 1);
    const std::vector<tw::Result> fresh_b = tw::rank_configs(*m, b.p, b.h, b.c, pool, options, 1);
    REQUIRE(answered(a, fresh_a));
    REQUIRE(answered(b, fresh_b));
    // `a` remembered, then `b`.
    set.rank_top(a.p, a.h, a.c, 1, top);
    const std::uint64_t n = replays();
    CHECK(same_results(set.rank(b.p, b.h, b.c, 1), fresh_b));
    CHECK(replays() == n);
  }
}

TEST_CASE("the memo answers a ranking at another CU budget scored at the same context",
          "[levers][memo]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  const std::vector<tw::Config> listed(pool.begin(), pool.begin() + 4);
  // The model's budgets start at a quarter of the 64 CUs: 4 and 8 CUs are
  // scored as 16, with launches planned for 16, and 32 as itself.
  const tw::ModelPtr m = whitelist_model(listed, false, 0.25);
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, std::make_shared<BudgetShiftLaunch>()};
  const tw::CandidateSet set(m, pool, options);
  const tw::Problem p              = make_problem(1000, 2000, 512);
  const tw::ExecutionContext four  = context(4, tw::Schedule::Default);
  const tw::ExecutionContext eight = context(8, tw::Schedule::Default);
  const tw::ExecutionContext half  = context(32, tw::Schedule::Default);
  REQUIRE(tw::scoring_context(*m, four, hw)->cu_budget == 16);
  REQUIRE(tw::scoring_context(*m, eight, hw)->cu_budget == 16);
  const std::vector<tw::Result> at_eight = tw::rank_configs(*m, p, hw, eight, pool, options, 1);
  const std::vector<tw::Result> at_half  = tw::rank_configs(*m, p, hw, half, pool, options, 1);
  REQUIRE(same_results(
      at_eight, tw::rank_configs(*m, p, hw, context(16, tw::Schedule::Default), pool, options, 1)));
  REQUIRE_FALSE(same_results(at_eight, at_half));
  std::vector<tw::Result> top;

  // Remembered at 4 CUs, answered at 8 with the scores of 16.
  set.rank_top(p, hw, four, 1, top);
  std::uint64_t n = replays();
  CHECK(same_results(set.rank(p, hw, eight, 1), at_eight));
  CHECK(replays() == n + 1);

  // A budget in the range is scored at itself: ranked afresh.
  set.rank_top(p, hw, four, 1, top);
  n = replays();
  CHECK(same_results(set.rank(p, hw, half, 1), at_half));
  CHECK(replays() == n);
}

TEST_CASE("prefetching and the memo leave whitelist-tier rankings bit-identical", "[levers]") {
  std::mt19937_64 rng(12345);
  const auto below      = [&](std::size_t n) { return static_cast<std::size_t>(rng() % n); };
  const tw::Hardware hw = test_hardware();
  std::vector<tw::Config> pool;
  for (std::size_t i = 0; i < 60; ++i) {
    const std::size_t sizes[] = {16, 32, 64, 128, 256};
    tw::Config c              = make_config(sizes[below(5)], sizes[below(5)], sizes[1 + below(3)]);
    c.occupancy               = 1 + static_cast<int>(below(3));
    c.cache_hints_a           = below(5) == 0 ? 4 : 0;
    c.index                   = i;
    pool.push_back(c);
  }
  std::vector<tw::Config> listed;
  for (const tw::Config& c : pool)
    if (below(3) == 0) listed.push_back(c);
  std::uint64_t answered = 0;
  for (bool launch_keys : {false, true}) {
    const tw::ModelPtr m = whitelist_model(listed, launch_keys);
    auto launch          = std::make_shared<ShiftLaunch>();
    const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
    const tw::CandidateSet set(m, pool, options);
    std::vector<tw::Result> top;
    for (int i = 0; i < 150; ++i) {
      const tw::Problem p = make_problem(1 + below(6000),
                                         1 + below(6000),
                                         1 + below(4000),
                                         below(4) == 0 ? 1 + below(8) : 1,
                                         below(2) ? tw::DataType::BFloat16 : tw::DataType::Half,
                                         below(2) ? tw::Transpose::T : tw::Transpose::N,
                                         below(2) ? tw::Transpose::T : tw::Transpose::N);
      const tw::ExecutionContext ctx =
          below(2) ? tw::ExecutionContext{} : context(8 + below(56), tw::Schedule::Default);
      const std::size_t k = 1 + below(6);
      INFO("launch keys " << launch_keys << " problem " << i);
      const std::vector<tw::Result> fresh   = tw::rank_configs(*m, p, hw, ctx, pool, options, 1);
      const std::vector<tw::Result> fresh_k = tw::rank_configs(*m, p, hw, ctx, pool, options, k);
      const std::uint64_t n                 = replays();
      set.rank_top(p, hw, ctx, 1, top);
      CHECK(same_results(top, scored_prefix(fresh, 1)));
      CHECK(same_results(set.rank(p, hw, ctx, 1), fresh));
      set.rank_top(p, hw, ctx, k, top);
      CHECK(same_results(top, scored_prefix(fresh_k, k)));
      CHECK(same_results(set.rank(p, hw, ctx, kAll),
                         tw::rank_configs(*m, p, hw, ctx, pool, options, kAll)));
      answered += replays() - n;
    }
  }
  CHECK(answered > 0);
}

TEST_CASE("the memo never answers a ranking of a degraded cell", "[levers][memo][drift]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  const std::vector<tw::Config> listed(pool.begin(), pool.begin() + 4);
  ModelSpec spec       = as_v3(grid_model());
  spec.schedules       = 0x7;
  spec.min_cu_fraction = 1.0 / 64.0;
  for (CellSpec& cell : spec.cells)
    for (const tw::Config& c : listed) cell.view_key_hashes.push_back(key_hash(c));
  const tw::ModelPtr plain = load_spec(spec);
  spec.v3_serving_flags |= 0x400;
  spec.drift_keep_ppm      = 1000000;
  const tw::ModelPtr drift = load_spec(spec);
  // Without the first listed kernel and its variant every cell is degraded; the full pool
  // degrades none, and there the memo answers as before.
  std::vector<tw::Config> lacking;
  for (const tw::Config& c : pool)
    if (key_hash(c) != key_hash(pool[0])) lacking.push_back(c);
  for (std::size_t i = 0; i < lacking.size(); ++i) lacking[i].index = i;
  auto launch = std::make_shared<ShiftLaunch>();
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
  const tw::CandidateSet degraded(drift, lacking, options), whole(drift, pool, options);
  REQUIRE(degraded.degraded_cells().size() == spec.cells.size());
  REQUIRE(whole.degraded_cells().empty());
  std::vector<tw::Result> top;
  for (const tw::Problem& p : {make_problem(1000, 2000, 512), make_problem(640, 3000, 1024, 3)}) {
    const std::vector<tw::Result> fresh =
        tw::rank_configs(*drift, p, hw, tw::ExecutionContext{}, lacking, options, 1);
    std::uint64_t before = replays();
    degraded.rank_top(p, hw, tw::ExecutionContext{}, 1, top);
    CHECK(same_results(top, scored_prefix(fresh, 1)));
    CHECK(same_results(degraded.rank(p, hw, tw::ExecutionContext{}, 1), fresh));
    CHECK(same_results(degraded.rank(p, hw, tw::ExecutionContext{}, 1), fresh));
    CHECK(replays() == before);
    // The same pool ranked by the model without the fallback: its tier is the whitelist alone.
    const std::vector<tw::Result> plain_fresh =
        tw::rank_configs(*plain, p, hw, tw::ExecutionContext{}, pool, options, 1);
    before = replays();
    whole.rank_top(p, hw, tw::ExecutionContext{}, 1, top);
    CHECK(same_results(whole.rank(p, hw, tw::ExecutionContext{}, 1), plain_fresh));
    CHECK(replays() == before + 1);
  }
}
