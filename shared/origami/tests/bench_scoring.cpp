// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Scoring-latency micro-benchmark for the leveled GEMM cost model.
//
// Builds a large synthetic candidate set (Cartesian product of macro-tiles,
// depths, occupancies, WGMs, and matrix instructions) and times two paths so
// changes to the per-level cost / pruning can be quantified:
//   1. score_candidates  -- the leveled estimation in isolation (per-level
//      proxies, sorts, and prunes), called directly on the resolved cost model.
//   2. rank_configs      -- the full public selection workflow (single
//      estimation phase), i.e. what a caller actually pays end to end.
// This is a developer tool, not a CTest; run the `origami-bench` executable
// directly.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "common.hpp"
#include "origami/math.hpp"
#include "origami/model.hpp"
#include "origami/origami.hpp"

namespace {

// Mirrors GemmModel's internal coarse-prune policy (src/engine/model.cpp). Kept
// in sync manually; the parity check at the end of the breakdown guards drift.
constexpr double      kKeepFractionL1  = 1.0 / 6.0;  // level 1: compute proxy
constexpr double      kKeepFractionL2  = 1.0 / 6.0;  // level 2: roofline
constexpr std::size_t kInternalMinKeep = 32;

struct mi_t {
  size_t m, n, k;
};

// Generate a broad, realistic candidate set. Many entries are intentionally
// infeasible (LDS overflow) or fast-rejected, mirroring a production sweep where
// level 0 culls a large fraction before any context is built.
std::vector<origami::config_t> make_candidate_set() {
  const std::vector<size_t> mt_ms = {16, 32, 64, 96, 128, 160, 192, 224, 256};
  const std::vector<size_t> mt_ns = {16, 32, 64, 96, 128, 160, 192, 224, 256};
  const std::vector<size_t> mt_ks = {16, 32, 64, 128, 256};
  const std::vector<int>    occs  = {1, 2};
  const std::vector<int>    wgms  = {0, 1, 4};
  const std::vector<mi_t>   mis   = {{16, 16, 16}, {32, 32, 8}, {16, 16, 32}};

  std::vector<origami::config_t> configs;
  configs.reserve(mt_ms.size() * mt_ns.size() * mt_ks.size() * occs.size() * wgms.size() *
                  mis.size());
  for (size_t mt_m : mt_ms)
    for (size_t mt_n : mt_ns)
      for (size_t mt_k : mt_ks)
        for (int occ : occs)
          for (int wgm : wgms)
            for (const mi_t& mi : mis)
              configs.push_back(
                  make_config(mt_m, mt_n, mt_k, mi.m, mi.n, mi.k, false, wgm, occ, 0, 0));
  return configs;
}

// Run `f` `iters` times and return the average wall time in ms/run. `f` returns
// a double that is accumulated into a volatile sink so the loop is not optimized
// away.
template <typename F>
double time_ms(int iters, F&& f) {
  volatile double sink = 0.0;
  const auto      t0   = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) sink += f();
  const auto t1 = std::chrono::steady_clock::now();
  (void)sink;
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

using scored_t = std::vector<std::pair<double, std::size_t>>;

// Coarse-level prune: keep the cheapest `max(min, frac*size)` via O(n) selection,
// mirroring GemmModel::score_estimation so the funnel (and thus the per-stage
// survivor counts) matches production.
void prune_coarse(scored_t& scored, double keep_fraction) {
  const auto by_cost = [](const auto& a, const auto& b) { return a.first < b.first; };
  const std::size_t target_keep =
      static_cast<std::size_t>(static_cast<double>(scored.size()) * keep_fraction);
  const std::size_t keep = std::max(kInternalMinKeep, target_keep);
  if (scored.size() > keep) {
    std::nth_element(scored.begin(), scored.begin() + keep, scored.end(), by_cost);
    scored.resize(keep);
  }
}

// Mirrors GemmModel's MI-latency memo so the L1 timing reflects the same caching.
struct mi_latency_cache {
  std::vector<std::pair<std::uint64_t, std::size_t>> entries;
  std::size_t get(const origami::hardware_t& hw, const origami::dim3_t& mi, origami::data_type_t dt) {
    const std::uint64_t key = static_cast<std::uint64_t>(mi.m) |
                              (static_cast<std::uint64_t>(mi.n) << 21) |
                              (static_cast<std::uint64_t>(mi.k) << 42);
    for (const auto& e : entries)
      if (e.first == key) return e.second;
    const std::size_t v = hw.get_mi_latency(mi.m, mi.n, mi.k, dt);
    entries.emplace_back(key, v);
    return v;
  }
};

// Context-free level-1 proxy (replicates GemmModel::score_compute_proxy).
double level1_proxy(const origami::problem_t& problem,
                    const origami::hardware_t& hardware,
                    const origami::config_t& config,
                    mi_latency_cache& lmi) {
  const std::size_t N_MI = origami::gemm::compute_number_matrix_instructions(config.mt, config.mi);
  const std::size_t L_MI = lmi.get(hardware, config.mi, problem.mi_dtype);
  const double      compute = static_cast<double>(N_MI * L_MI);
  const std::size_t grid_m           = origami::math::safe_ceil_div(problem.size.m, config.mt.m);
  const std::size_t grid_n           = origami::math::safe_ceil_div(problem.size.n, config.mt.n);
  const std::size_t num_output_tiles = grid_m * grid_n * problem.batch;
  const double      epilogue = origami::gemm::compute_coarse_epilogue_latency(
      hardware, config.mt, problem.d_dtype, num_output_tiles);
  const std::size_t cheap_timesteps =
      (hardware.N_CU > 0) ? origami::math::safe_ceil_div(num_output_tiles, hardware.N_CU)
                          : num_output_tiles;
  return (compute + epilogue) * static_cast<double>(cheap_timesteps);
}

// Per-level timing breakdown. Replicates the production cascade stage by stage so
// each level (and the L2 context-build vs roofline split) can be timed in
// isolation, WITHOUT adding any timing code to the model. A parity check against
// the real score_candidates guards against the replica drifting out of sync.
void run_level_breakdown(const origami::CostModel& model,
                         const origami::problem_t& problem,
                         const origami::hardware_t& hardware,
                         const std::vector<origami::config_t>& configs,
                         int iters) {
  const auto   by_cost = [](const auto& a, const auto& b) { return a.first < b.first; };
  const double maxv    = std::numeric_limits<double>::max();

  double t_l0 = 0, t_l1 = 0, t_l2a = 0, t_l2b = 0, t_l3 = 0;
  std::size_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  double      best = 0.0;
  std::vector<std::size_t> replica_final;

  std::vector<std::size_t>                     working;
  scored_t                                     scored;
  std::vector<std::optional<origami::gemm::context_t>> ctx;

  for (int it = 0; it < iters; ++it) {
    // L0: feasibility (LDS) + fast rejection, over all candidates.
    working.clear();
    auto t = std::chrono::steady_clock::now();
    for (std::size_t idx = 0; idx < configs.size(); ++idx) {
      const auto& cfg = configs[idx];
      // Mirror GemmModel::feasible: tensilelite skips the LDS check.
      const bool feasible = (cfg.target == origami::target_t::tensilelite)
                                ? true
                                : origami::gemm::check_lds_capacity(
                                      hardware, cfg.mt, problem.a_dtype, problem.b_dtype);
      if (!feasible) continue;
      if (origami::gemm::fast_reject(problem, hardware, cfg)) continue;
      working.push_back(idx);
    }
    t_l0 += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    c0 = working.size();

    // L1: context-free compute proxy + coarse prune.
    mi_latency_cache lmi;
    scored.clear();
    t = std::chrono::steady_clock::now();
    for (std::size_t idx : working) scored.emplace_back(level1_proxy(problem, hardware, configs[idx], lmi), idx);
    t_l1 += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    prune_coarse(scored, kKeepFractionL1);
    working.clear();
    for (const auto& e : scored) working.push_back(e.second);
    c1 = working.size();

    // L2a: build the context (constructor + streamk launch params) for survivors.
    ctx.assign(configs.size(), std::nullopt);
    t = std::chrono::steady_clock::now();
    for (std::size_t idx : working) ctx[idx].emplace(problem, hardware, configs[idx]);
    t_l2a += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();

    // L2b: roofline score (flat-cache memory proxy) + coarse prune.
    scored.clear();
    t = std::chrono::steady_clock::now();
    for (std::size_t idx : working) {
      const auto& c = *ctx[idx];
      if (!c.L_comp_stream)
        c.L_comp_stream =
            static_cast<double>(origami::gemm::compute_mt_compute_latency(problem, hardware, configs[idx]));
      const double compute = *c.L_comp_stream;
      const double mem = origami::gemm::compute_memory_latency<2>(problem, hardware, configs[idx], c);
      const double epi = origami::gemm::compute_coarse_epilogue_latency(hardware, configs[idx], c);
      const double cost = (std::max(compute, mem) + epi) * static_cast<double>(c.num_timesteps);
      if (cost != maxv) scored.emplace_back(cost, idx);
    }
    t_l2b += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    prune_coarse(scored, kKeepFractionL2);
    working.clear();
    for (const auto& e : scored) working.push_back(e.second);
    c2 = working.size();

    // L3: full analytical latency (reuses the carried context).
    scored.clear();
    t = std::chrono::steady_clock::now();
    for (std::size_t idx : working) {
      const double cost =
          origami::gemm::estimation_latency_from_context(problem, hardware, configs[idx], *ctx[idx]);
      if (cost != maxv) scored.emplace_back(cost, idx);
    }
    t_l3 += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    std::stable_sort(scored.begin(), scored.end(), by_cost);
    c3   = scored.size();
    best = scored.empty() ? 0.0 : scored.front().first;

    replica_final.clear();
    for (const auto& e : scored) replica_final.push_back(e.second);
  }

  const double inv = 1.0 / static_cast<double>(iters);
  const double total = (t_l0 + t_l1 + t_l2a + t_l2b + t_l3) * inv;

  std::printf("\n  per-level breakdown (avg over %d iters):\n", iters);
  std::printf("    %-26s %6zu -> %7.3f ms -> %zu\n", "L0 feasible/fast_reject", configs.size(),
              t_l0 * inv, c0);
  std::printf("    %-26s %6zu -> %7.3f ms -> %zu\n", "L1 compute proxy", c0, t_l1 * inv, c1);
  std::printf("    %-26s %6zu -> %7.3f ms\n", "L2a context build", c1, t_l2a * inv);
  std::printf("    %-26s %6zu -> %7.3f ms -> %zu\n", "L2b roofline score", c1, t_l2b * inv, c2);
  std::printf("    %-26s %6zu -> %7.3f ms -> %zu\n", "L3 full estimation", c2, t_l3 * inv, c3);
  std::printf("    %-26s %6s    %7.3f ms\n", "total (sum of levels)", "", total);
  std::printf("    best=%.6g\n", best);

  // Parity check: the replicated cascade must match the real score_candidates.
  std::vector<std::size_t> survivors(configs.size());
  std::iota(survivors.begin(), survivors.end(), 0);
  auto real = model.score_candidates(problem, hardware, configs, survivors);
  std::vector<std::size_t> real_idx;
  for (const auto& e : real) real_idx.push_back(e.second);
  std::vector<std::size_t> a = real_idx, b = replica_final;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  const bool same_set   = (a == b);
  const double real_best = real.empty() ? 0.0 : real.front().first;
  const bool same_best  = (real.size() == replica_final.size()) && (real_best == best);
  if (same_set && same_best) {
    std::printf("    parity: OK (matches score_candidates: %zu survivors, best=%.6g)\n",
                real.size(), real_best);
  } else {
    std::printf("    parity: WARNING -- replica diverged from score_candidates "
                "(replica=%zu/best=%.6g, real=%zu/best=%.6g). Per-level numbers may be stale.\n",
                replica_final.size(), best, real.size(), real_best);
  }
}

}  // namespace

int main() {
  using namespace origami;

  // Pin the heuristic tie-break variance to 0 so survivor counts / best costs are
  // stable run to run (this is timing, not a randomized ranking check).
  portable_setenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE", "0.0", 1);

  const auto hardware = make_hardware(950);
  const auto problem  = make_problem(4096, 4096, 8192);

  // Pre-filter to LDS-valid configs: real tensilelite candidates are validated by
  // the library before reaching origami, so the model skips the LDS check for that
  // target. Mirror that here so the synthetic grid doesn't inject unrunnable
  // configs the model would (correctly) no longer reject.
  std::vector<origami::config_t> configs;
  for (const auto& c : make_candidate_set())
    if (origami::gemm::check_lds_capacity(hardware, c.mt, problem.a_dtype, problem.b_dtype))
      configs.push_back(c);
  std::vector<std::size_t> survivors(configs.size());
  std::iota(survivors.begin(), survivors.end(), 0);

  const CostModel& model =
      get_model(model_t::gemm, target_t::tensilelite, prediction_modes_t::estimation);

  // Single estimation phase: drives the full rank_configs cascade through the
  // same internal leveled estimation that score_candidates exercises directly.
  ranking_phase_t phase;
  phase.model    = model_t::gemm;
  phase.target   = target_t::tensilelite;
  phase.fidelity = prediction_modes_t::estimation;
  ranking_pipeline_t pipeline;
  pipeline.phases.push_back(phase);

  const int iters = 50;

  // --- Path 1: leveled scoring in isolation ---
  std::size_t sc_kept = 0;
  double      sc_best = 0.0;
  time_ms(3, [&] {  // warmup (also primes singletons: heuristics DB, registry)
    auto s = model.score_candidates(problem, hardware, configs, survivors);
    return s.empty() ? 0.0 : s.front().first + static_cast<double>(s.size());
  });
  const double sc_ms = time_ms(iters, [&] {
    auto s  = model.score_candidates(problem, hardware, configs, survivors);
    sc_kept = s.size();
    sc_best = s.empty() ? 0.0 : s.front().first;
    return sc_best + static_cast<double>(sc_kept);
  });

  // --- Path 2: full rank_configs workflow ---
  std::size_t rc_kept = 0;
  double      rc_best = 0.0;
  time_ms(3, [&] {  // warmup
    auto r = rank_configs(problem, hardware, configs, model_t::gemm, pipeline);
    return r.empty() ? 0.0 : r.front().latency + static_cast<double>(r.size());
  });
  const double rc_ms = time_ms(iters, [&] {
    auto r  = rank_configs(problem, hardware, configs, model_t::gemm, pipeline);
    rc_kept = r.size();
    rc_best = r.empty() ? 0.0 : r.front().latency;
    return rc_best + static_cast<double>(rc_kept);
  });

  std::printf("origami scoring benchmark (%s)\n", model.name());
  std::printf("  candidates       : %zu\n", configs.size());
  std::printf("  iterations       : %d\n", iters);
  std::printf("  [score_candidates] survivors=%zu  per-run=%.3f ms  best=%.6g\n",
              sc_kept,
              sc_ms,
              sc_best);
  std::printf("  [rank_configs]     ranked=%zu     per-run=%.3f ms  best=%.6g\n",
              rc_kept,
              rc_ms,
              rc_best);

  run_level_breakdown(model, problem, hardware, configs, iters);
  return 0;
}
