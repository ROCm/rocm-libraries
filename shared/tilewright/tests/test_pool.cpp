// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pool semantics: configs the feature catalog cannot tell apart, attributes,
// execution contexts, tie-breaks and the whitening of constant item features.

#include "features.hpp"
#include "model_impl.hpp"
#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tw = tilewright;
using namespace tilewright_test;

namespace {

const std::string kLarge   = "Large|Large|LargeK|Bnone";
constexpr std::size_t kAll = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kQ   = 55;
constexpr std::size_t kI   = 12;
constexpr std::size_t kX   = 37;
const double kNaN          = std::numeric_limits<double>::quiet_NaN();
const double kInf          = std::numeric_limits<double>::infinity();

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

void zero_weights(CellTensors& t) {
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

enum class Tower { Query, Item, Inter };

// A one-cell model whose score of every config is exactly the whitened value z
// of feature `j` of `tower`, stored with `mean` and `sd`: two hidden units
// carry relu(z) and relu(-z) and the output is their difference.
ModelSpec probe_model(Tower tower, std::size_t j, float mean, float sd) {
  ModelSpec spec;
  CellSpec cell;
  cell.label               = kLarge;
  const std::size_t hidden = cell.hidden_dim;
  cell.edit                = [=](CellTensors& t) {
    zero_weights(t);
    t.temperature = 1.0f;
    switch (tower) {
      case Tower::Query:
        t.q_w0[j]          = 1.0f;
        t.q_w0[kQ + j]     = -1.0f;
        t.q_w2[0]          = 1.0f;
        t.q_w2[hidden + 1] = 1.0f;
        t.q_w4[0]          = 1.0f;
        t.q_w4[1]          = -1.0f;
        t.i_b2[0]          = 1.0f;
        t.q_mean[j]        = mean;
        t.q_std[j]         = sd;
        break;
      case Tower::Item:
        t.i_w0[j]      = 1.0f;
        t.i_w0[kI + j] = -1.0f;
        t.i_w2[0]      = 1.0f;
        t.i_w2[1]      = -1.0f;
        t.q_b4[0]      = 1.0f;
        t.i_mean[j]    = mean;
        t.i_std[j]     = sd;
        break;
      case Tower::Inter:
        t.x_w0[j]      = 1.0f;
        t.x_w0[kX + j] = -1.0f;
        t.x_w2[0]      = 1.0f;
        t.x_w2[1]      = -1.0f;
        t.x_mean[j]    = mean;
        t.x_std[j]     = sd;
        break;
    }
  };
  spec.cells = {cell};
  return spec;
}

// The engine's whitening of a feature value, in float.
float whitened(float x, float mean, float sd, bool item) {
  const float d = x - mean;
  if (item && sd < 1e-3f && std::fabs(d) > 2.0f * sd) return 0.0f;
  return d / (sd < 1e-6f ? 1.0f : sd);
}

ModelSpec zero_model(std::vector<Signature> whitelist = {}) {
  ModelSpec spec;
  CellSpec cell;
  cell.label      = kLarge;
  cell.edit       = zero_weights;
  cell.signatures = std::move(whitelist);
  spec.cells      = {cell};
  return spec;
}

std::vector<tw::Result> rank_singly(const tw::Model& model,
                                    const tw::Problem& p,
                                    const std::vector<tw::Config>& pool,
                                    std::size_t min_scored,
                                    tw::TieBreak tie_break           = tw::TieBreak::PoolOrder,
                                    const std::vector<double>* prior = nullptr) {
  return tw::detail::rank_keys(model,
                               p,
                               test_hardware(),
                               tw::ExecutionContext{},
                               pool,
                               singleton_keys(pool),
                               min_scored,
                               tie_break,
                               prior);
}

bool prior_before(double a, double b) {
  const bool fa = std::isfinite(a), fb = std::isfinite(b);
  if (fa != fb) return fa;
  return fa && a < b;
}

// The input order stably sorted by descending score, then (when non-null) by
// ascending prior with non-finite values last.
std::vector<std::size_t> expected_order(const std::vector<double>& score,
                                        const std::vector<double>* prior) {
  std::vector<std::size_t> order(score.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (score[a] != score[b]) return score[a] > score[b];
    return prior != nullptr && prior_before((*prior)[a], (*prior)[b]);
  });
  return order;
}

std::vector<std::size_t> indices(const std::vector<tw::Result>& r) {
  std::vector<std::size_t> out;
  for (const tw::Result& x : r) out.push_back(x.config_index);
  return out;
}

bool all_unscored_in_order(const std::vector<tw::Result>& r, std::size_t n) {
  if (r.size() != n) return false;
  for (std::size_t i = 0; i < n; ++i)
    if (r[i].config_index != i || r[i].scored) return false;
  return true;
}

// The ranking contract evaluated without the engine's grouping: tiers from the
// feasibility rules and the cell's whitelist in `spec`, each tier ordered by
// the scores in `r` with ties in input order (or by `prior` first).
void check_contract(const tw::Model& model,
                    const ModelSpec& spec,
                    const tw::Problem& p,
                    const std::vector<tw::Config>& pool,
                    std::size_t min_scored,
                    const std::vector<double>* prior,
                    const std::vector<tw::Result>& r) {
  const std::size_t n = pool.size();
  REQUIRE(r.size() == n);
  std::vector<double> score(n, 0.0);
  for (const tw::Result& x : r) score[x.config_index] = x.score;
  bool nt_a = false, nt_b = false;
  for (const tw::Config& c : pool) {
    nt_a |= c.cache_hints_a == 4;
    nt_b |= c.cache_hints_b == 4;
  }
  std::vector<int> tier(n, 0);
  const int cell = tw::route(model, p);
  if (cell >= 0) {
    const std::vector<Signature>& wl = spec.cells[static_cast<std::size_t>(cell)].signatures;
    std::vector<int> feasible(n, 0);
    std::size_t listed = 0;
    for (std::size_t i = 0; i < n; ++i) {
      feasible[i] =
          tw::detail::check_lds_capacity(test_hardware(), pool[i].mt, p.a_dtype, p.b_dtype) &&
          tw::detail::is_kernel_feasible(p, pool[i], nt_a, nt_b);
      if (feasible[i] && std::find(wl.begin(), wl.end(), sig(pool[i])) != wl.end()) {
        tier[i] = 1;
        ++listed;
      }
    }
    for (std::size_t i = 0; i < n; ++i)
      if (feasible[i] && tier[i] == 0 && (listed == 0 || min_scored > listed))
        tier[i] = listed == 0 ? 1 : 2;
  }
  std::vector<std::size_t> expect;
  for (int t : {1, 2}) {
    std::vector<double> s;
    std::vector<double> pr;
    std::vector<std::size_t> part;
    for (std::size_t i = 0; i < n; ++i)
      if (tier[i] == t) {
        part.push_back(i);
        s.push_back(score[i]);
        if (prior != nullptr) pr.push_back((*prior)[i]);
      }
    for (std::size_t k : expected_order(s, prior != nullptr ? &pr : nullptr))
      expect.push_back(part[k]);
  }
  const std::size_t n_scored = expect.size();
  for (std::size_t i = 0; i < n; ++i)
    if (tier[i] == 0) expect.push_back(i);
  CHECK(indices(r) == expect);
  for (std::size_t k = 0; k < n; ++k) CHECK(r[k].scored == (k < n_scored));
}

struct Random {
  std::mt19937_64 rng;

  explicit Random(std::uint64_t seed) : rng(seed) {}

  std::size_t below(std::size_t n) { return static_cast<std::size_t>(rng() % n); }

  template <typename T>
  T pick(std::initializer_list<T> values) {
    return *(values.begin() + below(values.size()));
  }

  // Occupancy and vector widths include values the features clamp to 1, so
  // some distinct configs score equally. Draws are sequenced so that every
  // compiler builds the same pools.
  tw::Config config() {
    const std::size_t mt_m = pick<std::size_t>({16, 32, 64, 128, 192, 256, 512});
    const std::size_t mt_n = pick<std::size_t>({16, 32, 64, 96, 128, 256});
    const std::size_t mt_k = pick<std::size_t>({16, 32, 64, 128, 256});
    tw::Config c           = make_config(mt_m, mt_n, mt_k);
    const std::size_t mi   = below(4);
    c.mi                   = mi == 0   ? tw::Dim3{1, 1, 64}
                             : mi == 1 ? tw::Dim3{32, 32, 16}
                             : mi == 2 ? tw::Dim3{16, 16, 128}
                                       : tw::Dim3{16, 16, 32};
    c.cache_hints_a        = pick<int>({0, 0, 0, 1, 4});
    c.cache_hints_b        = pick<int>({0, 0, 0, 4});
    c.occupancy            = pick<int>({-1, 0, 1, 1, 2, 4});
    c.grvw_a               = pick<std::size_t>({0, 1, 2, 4, 8, 16});
    c.grvw_b               = pick<std::size_t>({1, 2, 4, 8, 16});
    c.gwvw_d               = pick<std::size_t>({0, 1, 2, 4, 8});
    return c;
  }

  std::vector<tw::Attribute> attributes() {
    static const char* const kNames[] = {"wave_num",
                                         "workgroup_mapping",
                                         "global_split_u",
                                         "stream_k_atomic",
                                         "tile_processing_strategy",
                                         "cluster_dim_x",
                                         "x"};
    std::vector<tw::Attribute> out;
    for (const char* name : kNames)
      if (below(3) == 0)
        out.push_back({name,
                       pick<std::int64_t>({0,
                                           1,
                                           -7,
                                           1 << 20,
                                           std::numeric_limits<std::int64_t>::min(),
                                           std::numeric_limits<std::int64_t>::max()})});
    std::shuffle(out.begin(), out.end(), rng);
    return out;
  }

  // `n` configs drawn from `distinct` random kernels, so most kernels repeat;
  // the copies differ in index and attributes.
  std::vector<tw::Config> pool(std::size_t distinct, std::size_t n) {
    std::vector<tw::Config> kinds;
    for (std::size_t i = 0; i < distinct; ++i) kinds.push_back(config());
    std::vector<tw::Config> out;
    for (std::size_t i = 0; i < n; ++i) {
      tw::Config c = kinds[below(distinct)];
      c.index      = 1000 + i;
      c.attributes = attributes();
      out.push_back(c);
    }
    return out;
  }

  tw::Problem problem() {
    const tw::DataType dts[] = {tw::DataType::BFloat16,
                                tw::DataType::Half,
                                tw::DataType::Float,
                                tw::DataType::Float8,
                                tw::DataType::Float8_fnuz,
                                tw::DataType::XFloat32};
    const std::size_t m      = 1 + below(9000);
    const std::size_t n      = 1 + below(9000);
    const std::size_t k      = 1 + below(9000);
    const std::size_t batch  = pick<std::size_t>({1, 1, 1, 2, 8});
    const tw::DataType dt    = dts[below(6)];
    const tw::Transpose ta   = below(2) ? tw::Transpose::T : tw::Transpose::N;
    const tw::Transpose tb   = below(2) ? tw::Transpose::T : tw::Transpose::N;
    tw::Problem p            = make_problem(m, n, k, batch, dt, ta, tb);
    if (below(4) == 0) p.size.m = pick<std::size_t>({32, 33, 128, 129, 300, 301, 512, 513});
    if (below(8) == 0) p.size.n = pick<std::size_t>({8, 64, 8192});
    return p;
  }

  std::vector<double> prior(std::size_t n) {
    std::vector<double> out;
    for (std::size_t i = 0; i < n; ++i)
      out.push_back(pick<double>({-1.0, 0.0, 0.0, 0.5, 2.0, 2.0, kNaN, kInf, -kInf}));
    return out;
  }
};

// Message of the std::invalid_argument that `f` throws, or "" when it does not.
template <typename F>
std::string invalid_argument_of(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument& e) { return e.what(); }
  return std::string();
}

bool mentions(const std::string& text, const std::string& what) {
  return text.find(what) != std::string::npos;
}

}  // namespace

TEST_CASE("view keys tell every numeric Config field apart and ignore index and attributes",
          "[pool]") {
  const tw::Config base        = make_config(128, 64, 32, 16, 16, 32, 1, 0, 2);
  std::vector<tw::Config> pool = {base};
  const auto vary              = [&](void (*edit)(tw::Config&)) {
    tw::Config c = base;
    edit(c);
    pool.push_back(c);
  };
  vary([](tw::Config& c) { c.mt.m = 256; });
  vary([](tw::Config& c) { c.mt.n = 128; });
  vary([](tw::Config& c) { c.mt.k = 64; });
  vary([](tw::Config& c) { c.mi.m = 32; });
  vary([](tw::Config& c) { c.mi.n = 32; });
  vary([](tw::Config& c) { c.mi.k = 16; });
  vary([](tw::Config& c) { c.occupancy = -2; });
  vary([](tw::Config& c) { c.cache_hints_a = 4; });
  vary([](tw::Config& c) { c.cache_hints_b = -1; });
  vary([](tw::Config& c) { c.grvw_a = 4; });
  vary([](tw::Config& c) { c.grvw_b = 2; });
  vary([](tw::Config& c) { c.gwvw_d = 8; });
  vary([](tw::Config& c) {
    c.index      = 99;
    c.attributes = {{"wave_num", 4}, {"workgroup_mapping", 8}};
  });
  vary([](tw::Config& c) { c.attributes = {{"wave_num", 2}}; });

  const tw::detail::PoolKeys keys = tw::detail::pool_keys(tw::detail::v2_catalog(), pool);
  REQUIRE(keys.first.size() == 13);
  for (std::size_t i = 0; i < 13; ++i) {
    CHECK(keys.key_of[i] == i);
    CHECK(keys.first[i] == i);
  }
  CHECK(keys.key_of[13] == 0);
  CHECK(keys.key_of[14] == 0);
  CHECK(keys.member_begin.size() == 14);
  CHECK(std::vector<std::size_t>(keys.members.begin(), keys.members.begin() + 3) ==
        std::vector<std::size_t>{0, 13, 14});
  CHECK(keys.member_begin[1] == 3);
  CHECK(keys.member_begin[13] == 15);
  CHECK(keys.nt_a);
  CHECK_FALSE(keys.nt_b);
}

TEST_CASE("duplicate configs rank exactly like configs scored one by one", "[pool]") {
  Random rnd(20261005);
  ModelSpec listed = grid_model(tw::WeightType::Int4);
  for (CellSpec& cell : listed.cells) {
    cell.embed_dim    = 24;
    cell.hidden_dim   = 40;
    cell.inter_hidden = 21;
  }
  ModelSpec bf16          = grid_model(tw::WeightType::Bf16);
  std::size_t shared_keys = 0;
  for (int round = 0; round < 8; ++round) {
    const bool whitelisted             = round % 2 == 0;
    ModelSpec spec                     = whitelisted ? listed : bf16;
    const std::size_t kinds            = 2 + rnd.below(12);
    const std::vector<tw::Config> pool = rnd.pool(kinds, 1 + rnd.below(70));
    if (whitelisted)
      for (CellSpec& cell : spec.cells) {
        cell.signatures.clear();
        for (const tw::Config& c : pool)
          if (rnd.below(3) == 0) cell.signatures.push_back(sig(c));
      }
    const tw::ModelPtr model = load_spec(spec);
    const tw::CandidateSet set(model, pool);
    const tw::detail::PoolKeys keys = tw::detail::pool_keys(*model->catalog, pool);
    for (int i = 0; i < 40; ++i) {
      const tw::Problem p          = rnd.problem();
      const std::size_t min_scored = rnd.pick<std::size_t>({0, 1, 3, 10, kAll});
      INFO("round " << round << " problem " << p.size.m << "x" << p.size.n << "x" << p.size.k
                    << " b" << p.batch << " min_scored " << min_scored);
      const std::vector<tw::Result> expect = rank_singly(*model, p, pool, min_scored);
      REQUIRE(same_results(tw::rank_configs(*model, p, test_hardware(), pool, min_scored), expect));
      REQUIRE(same_results(set.rank(p, test_hardware(), min_scored), expect));
      check_contract(*model, spec, p, pool, min_scored, nullptr, expect);

      std::vector<double> bits(keys.first.size(), kNaN);
      for (const tw::Result& x : expect) {
        if (!x.scored) continue;
        const std::size_t key = keys.key_of[x.config_index];
        if (std::isnan(bits[key]))
          bits[key] = x.score;
        else
          ++shared_keys;
        CHECK(std::memcmp(&bits[key], &x.score, sizeof(double)) == 0);
        const std::vector<tw::Result> alone =
            tw::rank_configs(*model, p, test_hardware(), {pool[x.config_index]}, kAll);
        if (alone[0].scored) CHECK(std::memcmp(&alone[0].score, &x.score, sizeof(double)) == 0);
      }
    }
  }
  CHECK(shared_keys > 100);
}

TEST_CASE("equal scores keep input order across view keys and duplicates", "[pool]") {
  const tw::ModelPtr model = load_spec(probe_model(Tower::Item, 0, 0.0f, 1.0f));
  Random rnd(7);
  std::vector<tw::Config> pool;
  for (std::size_t i = 0; i < 40; ++i) {
    const std::size_t mt_m = rnd.pick<std::size_t>({64, 128, 256});
    const std::size_t mt_n = rnd.pick<std::size_t>({32, 64, 128});
    tw::Config c           = make_config(mt_m, mt_n, rnd.pick<std::size_t>({32, 64}));
    c.grvw_a               = rnd.pick<std::size_t>({1, 2});
    c.occupancy            = rnd.pick<int>({1, 2});
    c.index                = i;
    pool.push_back(c);
  }
  std::vector<double> log2_mt_m;
  for (const tw::Config& c : pool) log2_mt_m.push_back(std::log2(static_cast<double>(c.mt.m)));
  const tw::Problem p = make_problem(4096, 4096, 4096);

  const std::vector<tw::Result> r = tw::rank_configs(*model, p, test_hardware(), pool);
  REQUIRE(r.size() == pool.size());
  for (const tw::Result& x : r) CHECK(x.score == log2_mt_m[x.config_index]);
  CHECK(indices(r) == expected_order(log2_mt_m, nullptr));
  CHECK(same_results(tw::CandidateSet(model, pool).rank(p, test_hardware()), r));
  CHECK(same_results(rank_singly(*model, p, pool, 0), r));

  for (int round = 0; round < 20; ++round) {
    const std::vector<double> prior = rnd.prior(pool.size());
    const std::vector<tw::Result> rp =
        tw::rank_configs(*model, p, test_hardware(), {}, pool, 0, tw::TieBreak::Prior, &prior);
    CHECK(indices(rp) == expected_order(log2_mt_m, &prior));
    const tw::CandidateSet set(model, pool, tw::PoolOptions{tw::TieBreak::Prior});
    CHECK(same_results(set.rank(p, test_hardware(), {}, 0, &prior), rp));
    CHECK(same_results(rank_singly(*model, p, pool, 0, tw::TieBreak::Prior, &prior), rp));
  }
}

TEST_CASE("a large pool of a few distinct kernels is scored per kernel", "[pool]") {
  ModelSpec spec = grid_model(tw::WeightType::Int8);
  for (CellSpec& cell : spec.cells) cell.signatures = {{128, 128, 64, 16, 16, 32, 0, 0}};
  const tw::ModelPtr model = load_spec(spec);
  const tw::Config kinds[] = {
      make_config(128, 128, 64), make_config(256, 64, 64), make_config(64, 64, 32, 32, 32, 16)};
  std::vector<tw::Config> pool;
  for (std::size_t i = 0; i < 15000; ++i) {
    pool.push_back(kinds[(i * 7) % 3]);
    pool.back().index = i;
  }
  REQUIRE(tw::detail::pool_keys(*model->catalog, pool).first.size() == 3);
  const tw::CandidateSet set(model, pool);
  for (const tw::Problem& p : {make_problem(4096, 4096, 4096), make_problem(100, 3000, 700, 2)})
    for (std::size_t min_scored : {std::size_t{0}, kAll})
      CHECK(same_results(set.rank(p, test_hardware(), min_scored),
                         rank_singly(*model, p, pool, min_scored)));
}

TEST_CASE("attributes never change scores, feasibility, whitelisting or order", "[attributes]") {
  Random rnd(11);
  ModelSpec spec                      = grid_model(tw::WeightType::Bf16);
  const std::vector<tw::Config> plain = [&] {
    std::vector<tw::Config> pool = rnd.pool(10, 50);
    for (tw::Config& c : pool) c.attributes.clear();
    return pool;
  }();
  for (CellSpec& cell : spec.cells)
    for (std::size_t i = 0; i < plain.size(); i += 4) cell.signatures.push_back(sig(plain[i]));
  const tw::ModelPtr model       = load_spec(spec);
  std::vector<tw::Config> tagged = plain;
  for (tw::Config& c : tagged) c.attributes = rnd.attributes();
  const tw::CandidateSet plain_set(model, plain, tw::PoolOptions{tw::TieBreak::Prior});
  const tw::CandidateSet tagged_set(model, tagged, tw::PoolOptions{tw::TieBreak::Prior});
  for (int i = 0; i < 40; ++i) {
    const tw::Problem p             = rnd.problem();
    const std::size_t min_scored    = rnd.pick<std::size_t>({0, 2, kAll});
    const std::vector<double> prior = rnd.prior(plain.size());
    const std::vector<tw::Result> r0 =
        tw::rank_configs(*model, p, test_hardware(), plain, min_scored);
    CHECK(same_results(tw::rank_configs(*model, p, test_hardware(), tagged, min_scored), r0));
    CHECK(same_results(tagged_set.rank(p, test_hardware(), min_scored), r0));
    const std::vector<tw::Result> rp = plain_set.rank(p, test_hardware(), {}, min_scored, &prior);
    CHECK(same_results(tagged_set.rank(p, test_hardware(), {}, min_scored, &prior), rp));
    CHECK(same_results(
        tw::rank_configs(
            *model, p, test_hardware(), {}, tagged, min_scored, tw::TieBreak::Prior, &prior),
        rp));
  }
}

TEST_CASE("attribute names must be non-empty and unique within a config", "[attributes]") {
  const tw::ModelPtr model = load_spec(zero_model());
  const tw::Problem p      = make_problem(4096, 4096, 4096);
  std::vector<tw::Config> ok(4, make_config(128, 128, 64));
  ok[0].attributes = {{"wave_num", 1}, {"WAVE_NUM", 2}, {"wave_num_", 3}};
  ok[1].attributes = {{"wave_num", 1}};
  ok[2].attributes = {{"wave_num", 1}};
  CHECK_NOTHROW(static_cast<void>(tw::CandidateSet(model, ok)));
  CHECK(tw::rank_configs(*model, p, test_hardware(), ok).size() == 4);

  const auto all_entry_points = [&](const std::vector<tw::Config>& pool) {
    const std::vector<std::string> messages = {
        invalid_argument_of([&] { static_cast<void>(tw::CandidateSet(model, pool)); }),
        invalid_argument_of([&] { tw::rank_configs(*model, p, test_hardware(), pool); }),
        invalid_argument_of([&] {
          tw::rank_configs(*model, p, test_hardware(), {}, pool, 0, tw::TieBreak::Prior, nullptr);
        }),
        invalid_argument_of([&] {
          tw::rank_configs(*model, p, test_hardware(), {1, tw::Schedule::Dynamic}, pool);
        }),
        invalid_argument_of([&] { tw::rank_configs(*model, make_problem(8, 8, 8), {}, pool); })};
    return messages;
  };

  std::vector<tw::Config> empty_name = ok;
  empty_name[2].attributes           = {{"wave_num", 1}, {"", 2}};
  for (const std::string& m : all_entry_points(empty_name)) {
    CHECK(mentions(m, "config 2"));
    CHECK(mentions(m, "empty name"));
  }
  std::vector<tw::Config> duplicate = ok;
  duplicate[3].attributes           = {{"b", 1}, {"a", 2}, {"c", 3}, {"a", 2}};
  for (const std::string& m : all_entry_points(duplicate)) {
    CHECK(mentions(m, "config 3"));
    CHECK(mentions(m, "'a'"));
  }
  std::vector<tw::Config> first = ok;
  first[0].attributes           = {{"x", 1}, {"x", 1}};
  for (const std::string& m : all_entry_points(first)) CHECK(mentions(m, "config 0"));
}

TEST_CASE("v2 models support every CU with the default schedule only", "[context]") {
  const tw::ModelPtr model = load_spec(grid_model());
  const tw::Hardware& hw   = test_hardware();
  const std::size_t n_cu   = hw.N_CU;
  CHECK(tw::supports(*model, {}, hw));
  CHECK(tw::supports(*model, {0, tw::Schedule::Default}, hw));
  CHECK(tw::supports(*model, {n_cu, tw::Schedule::Default}, hw));
  CHECK(tw::supports(*model, {n_cu + 1, tw::Schedule::Default}, hw));
  CHECK(tw::supports(*model, {kAll, tw::Schedule::Default}, hw));
  CHECK_FALSE(tw::supports(*model, {n_cu - 1, tw::Schedule::Default}, hw));
  CHECK_FALSE(tw::supports(*model, {1, tw::Schedule::Default}, hw));
  for (tw::Schedule s : {tw::Schedule::Dynamic, tw::Schedule::Auto, static_cast<tw::Schedule>(3)})
    for (std::size_t budget : {std::size_t{0}, n_cu, n_cu - 1}) {
      INFO("schedule " << static_cast<int>(s) << " budget " << budget);
      CHECK_FALSE(tw::supports(*model, {budget, s}, hw));
    }
}

TEST_CASE("v2 models read no attributes", "[attributes]") {
  CHECK(tw::attribute_names(*load_spec(grid_model())).empty());
}

TEST_CASE("an unsupported context leaves every config unscored", "[context]") {
  Random rnd(3);
  ModelSpec spec                     = grid_model(tw::WeightType::Int4);
  const std::vector<tw::Config> pool = rnd.pool(6, 20);
  for (CellSpec& cell : spec.cells) cell.signatures = {sig(pool[0])};
  const tw::ModelPtr model = load_spec(spec);
  const tw::CandidateSet set(model, pool, tw::PoolOptions{tw::TieBreak::Prior});
  const tw::Hardware& hw = test_hardware();
  for (int i = 0; i < 20; ++i) {
    const tw::Problem p              = rnd.problem();
    const std::vector<double> prior  = rnd.prior(pool.size());
    const std::vector<tw::Result> ex = tw::rank_configs(*model, p, hw, pool, kAll);
    for (const tw::ExecutionContext& ok :
         {tw::ExecutionContext{},
          tw::ExecutionContext{hw.N_CU, tw::Schedule::Default},
          tw::ExecutionContext{3 * hw.N_CU, tw::Schedule::Default}}) {
      CHECK(same_results(set.rank(p, hw, ok, kAll), ex));
      CHECK(same_results(tw::rank_configs(*model, p, hw, ok, pool, kAll), ex));
      CHECK(same_results(
          set.rank(p, hw, ok, kAll, &prior),
          tw::rank_configs(*model, p, hw, ok, pool, kAll, tw::TieBreak::Prior, &prior)));
    }
    for (const tw::ExecutionContext& bad :
         {tw::ExecutionContext{hw.N_CU - 1, tw::Schedule::Default},
          tw::ExecutionContext{0, tw::Schedule::Dynamic},
          tw::ExecutionContext{0, tw::Schedule::Auto}}) {
      CHECK(all_unscored_in_order(set.rank(p, hw, bad, kAll, &prior), pool.size()));
      CHECK(all_unscored_in_order(tw::rank_configs(*model, p, hw, bad, pool, kAll), pool.size()));
    }
    CHECK(same_results(set.rank(p, hw, kAll), ex));
  }
}

TEST_CASE("TieBreak::Prior orders equal scores by ascending prior, non-finite last", "[tiebreak]") {
  const tw::ModelPtr model = load_spec(zero_model());
  const tw::Problem p      = make_problem(4096, 4096, 4096);
  const tw::Hardware& hw   = test_hardware();
  std::vector<tw::Config> pool;
  for (std::size_t mt : {256, 128, 128, 64, 256, 32, 64, 128})
    pool.push_back(make_config(mt, mt, 32));
  const std::vector<double> prior         = {3.0, kNaN, -1.0, 3.0, kInf, -kInf, 0.5, -1.0};
  const std::vector<std::size_t> by_prior = {2, 7, 6, 0, 3, 1, 4, 5};
  const std::vector<std::size_t> in_order = {0, 1, 2, 3, 4, 5, 6, 7};

  const tw::CandidateSet set(model, pool, tw::PoolOptions{tw::TieBreak::Prior});
  const tw::CandidateSet plain(model, pool);
  CHECK(indices(set.rank(p, hw, {}, 0, &prior)) == by_prior);
  CHECK(indices(tw::rank_configs(*model, p, hw, {}, pool, 0, tw::TieBreak::Prior, &prior)) ==
        by_prior);
  CHECK(indices(set.rank(p, hw)) == in_order);
  CHECK(indices(set.rank(p, hw, {}, 0, nullptr)) == in_order);
  CHECK(indices(plain.rank(p, hw, {}, 0, &prior)) == in_order);
  CHECK(indices(tw::rank_configs(*model, p, hw, {}, pool, 0, tw::TieBreak::PoolOrder, &prior)) ==
        in_order);

  const std::vector<double> short_prior(pool.size() - 1, 0.0);
  const std::vector<double> long_prior(pool.size() + 1, 0.0);
  for (const std::vector<double>* bad : {&short_prior, &long_prior}) {
    CHECK_THROWS_AS(set.rank(p, hw, {}, 0, bad), std::invalid_argument);
    CHECK_THROWS_AS(plain.rank(p, hw, {}, 0, bad), std::invalid_argument);
    CHECK_THROWS_AS(set.rank(make_problem(8, 8, 8), hw, {0, tw::Schedule::Auto}, 0, bad),
                    std::invalid_argument);
    CHECK_THROWS_AS(tw::rank_configs(*model, p, hw, {}, pool, 0, tw::TieBreak::Prior, bad),
                    std::invalid_argument);
    CHECK_THROWS_AS(tw::rank_configs(*model, p, hw, {}, pool, 0, tw::TieBreak::PoolOrder, bad),
                    std::invalid_argument);
  }
  const std::vector<double> none;
  CHECK(tw::rank_configs(*model, p, hw, {}, {}, 0, tw::TieBreak::Prior, &none).empty());
  CHECK(tw::CandidateSet(model, {}, tw::PoolOptions{tw::TieBreak::Prior})
            .rank(p, hw, {}, 0, &none)
            .empty());
}

TEST_CASE("the prior breaks ties only, within each tier", "[tiebreak]") {
  const tw::Problem p    = make_problem(4096, 4096, 4096);
  const tw::Hardware& hw = test_hardware();
  const tw::Config a = make_config(64, 64, 64), b = make_config(128, 128, 64),
                   c                 = make_config(256, 64, 64);
  const std::vector<tw::Config> pool = {a, b, a, c, b};
  const std::vector<double> prior    = {5.0, 2.0, 1.0, 0.0, 3.0};
  const tw::ModelPtr model           = load_spec(zero_model({sig(b)}));
  const tw::CandidateSet set(model, pool, tw::PoolOptions{tw::TieBreak::Prior});
  CHECK(indices(set.rank(p, hw, {}, kAll, &prior)) == std::vector<std::size_t>{1, 4, 3, 2, 0});
  const std::vector<tw::Result> top = set.rank(p, hw, {}, 0, &prior);
  CHECK(indices(top) == std::vector<std::size_t>{1, 4, 0, 2, 3});
  CHECK(top[1].scored);
  CHECK_FALSE(top[2].scored);

  // Scores are log2(mt_m): a larger prior never moves a config past a lower score.
  const tw::ModelPtr probe       = load_spec(probe_model(Tower::Item, 0, 0.0f, 1.0f));
  const std::vector<double> flat = {9.0, 1.0, 0.0, -9.0, kNaN};
  CHECK(indices(tw::rank_configs(*probe, p, hw, {}, pool, 0, tw::TieBreak::Prior, &flat)) ==
        std::vector<std::size_t>{3, 1, 4, 2, 0});
}

TEST_CASE("an item feature with a stored std below 1e-3 is constant", "[whitening]") {
  const tw::Problem p    = make_problem(4096, 4096, 4096);
  const tw::Hardware& hw = test_hardware();
  std::vector<tw::Config> pool;
  for (int occ = -1; occ <= 9; ++occ)
    pool.push_back(make_config(128, 128, 64, 16, 16, 32, 0, 0, occ));
  const float one = static_cast<float>(1.0 / 9.0);
  struct Case {
    float mean, sd;
  };
  const Case cases[] = {{one, 0.5f},
                        {one, 1e-3f},
                        {one - 1.87e-4f, 1.87e-4f},
                        {one + 1e-7f, 1e-7f},
                        {one, 0.0f},
                        {one + 0.25f, 0.0f},
                        {one - 2.0f * 5e-4f, 5e-4f},
                        {one + 9.99e-4f, 9.99e-4f},
                        {one + 2.5e-4f, 1e-4f}};
  for (const Case& c : cases) {
    const tw::ModelPtr model = load_spec(probe_model(Tower::Item, 8, c.mean, c.sd));
    const tw::CandidateSet set(model, pool);
    std::vector<double> score(pool.size(), kNaN);
    for (const std::vector<tw::Result>& r :
         {tw::rank_configs(*model, p, hw, pool), set.rank(p, hw)}) {
      REQUIRE(r.size() == pool.size());
      for (const tw::Result& x : r) {
        const float f = tw::compute_features(*model, p, pool[x.config_index], hw).item[8];
        INFO("mean " << c.mean << " std " << c.sd << " occupancy "
                     << pool[x.config_index].occupancy);
        REQUIRE(x.scored);
        CHECK(x.score == static_cast<double>(whitened(f, c.mean, c.sd, true)));
        score[x.config_index] = x.score;
      }
    }
    const bool constant = c.sd < 1e-3f;
    const double occ1   = score[2];
    for (std::size_t i = 3; i < pool.size(); ++i) {
      if (constant) CHECK(score[i] == 0.0);
      if (!constant) CHECK(std::fabs(score[i]) > 0.1);
    }
    CHECK(score[0] == occ1);
    CHECK(score[1] == occ1);
  }

  // A single training value one std from the stored mean whitens as before.
  const tw::ModelPtr single = load_spec(probe_model(Tower::Item, 8, one - 1.87e-4f, 1.87e-4f));
  const std::vector<tw::Result> r = tw::rank_configs(*single, p, hw, pool);
  CHECK(r[0].config_index == 0);
  CHECK(std::fabs(r[0].score - 1.0) < 0.01);
}

TEST_CASE("query and interaction features keep the plain whitening", "[whitening]") {
  const tw::Problem p                = make_problem(4096, 4096, 4096);
  const tw::Hardware& hw             = test_hardware();
  const std::vector<tw::Config> pool = {make_config(128, 128, 64), make_config(256, 64, 64)};
  struct Case {
    Tower tower;
    std::size_t j;
    float offset, sd;
  };
  for (const Case& c : {Case{Tower::Query, 0, 0.5f, 1e-4f},
                        Case{Tower::Query, 1, 0.5f, 0.0f},
                        Case{Tower::Inter, 4, 0.01f, 2e-4f},
                        Case{Tower::Inter, 7, -0.25f, 1e-7f}}) {
    const tw::ModelPtr empty        = load_spec(probe_model(c.tower, c.j, 0.0f, 1.0f));
    const tw::Features f            = tw::compute_features(*empty, p, pool[0], hw);
    const float x                   = c.tower == Tower::Query ? f.query[c.j] : f.interaction[c.j];
    const float mean                = x - c.offset;
    const tw::ModelPtr model        = load_spec(probe_model(c.tower, c.j, mean, c.sd));
    const std::vector<tw::Result> r = tw::rank_configs(*model, p, hw, pool);
    for (const tw::Result& res : r) {
      const tw::Features fi = tw::compute_features(*model, p, pool[res.config_index], hw);
      const float xi        = c.tower == Tower::Query ? fi.query[c.j] : fi.interaction[c.j];
      INFO("feature " << c.j << " std " << c.sd);
      CHECK(res.score == static_cast<double>(whitened(xi, mean, c.sd, false)));
    }
    CHECK(std::fabs(r[0].score) > 0.2);
  }
}

TEST_CASE("the loader selects the feature catalog by the model's hash", "[catalog]") {
  const tw::detail::FeatureCatalog& v2 = tw::detail::v2_catalog();
  CHECK(tw::detail::find_catalog("e7fe4b524851e895") == &v2);
  CHECK(std::string(tw::feature_catalog_hash()) == v2.hash);
  CHECK(v2.query_dim == kQ);
  CHECK(v2.item_dim == kI);
  CHECK(v2.inter_dim == kX);
  CHECK(v2.view_key_size == 12);
  for (const char* other : {"", "e7fe4b524851e89", "e7fe4b524851e8950", "E7FE4B524851E895"})
    CHECK(tw::detail::find_catalog(other) == nullptr);
  const tw::ModelPtr model = load_spec(grid_model());
  CHECK(model->catalog == &v2);
  CHECK(tw::describe(*model).feature_catalog_hash == v2.hash);
}
