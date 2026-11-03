// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Leveled coarse-to-fine GEMM estimation. A cheap context-free pass scores every
// config, the best ones are kept, and only those pay for the context build and
// the full analytical latency.

#include "estimation_leveled.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "origami/gemm.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"

namespace origami {
namespace gemm {

namespace {

// Cascade tunables. They have not been re-validated against the current latency
// model. Re-measure top-1 agreement with the single-pass ranking and selection time
// before changing them.

// Level-1 keep fraction: after the proxy pass, keep the cheapest 1/6 of the configs.
constexpr double kLeveledKeepFraction = 1.0 / 6.0;

// Keep floor: never prune the working set below this many configs. This is the main
// accuracy dial.
constexpr std::size_t kLeveledMinKeep = 48;

// Split-K-prone test: a reference tiling of kSplitKRefTile x kSplitKRefTile output
// tiles underfills the CU budget, and K has at least kSplitKMinKIters iterations of a
// kSplitKRefDepth-deep k-tile.
constexpr std::size_t kSplitKRefTile   = 256;
constexpr std::size_t kSplitKRefDepth  = 64;
constexpr std::size_t kSplitKMinKIters = 64;

constexpr double kRejectedLatency = std::numeric_limits<double>::max();

// MI-latency memo keyed by packed (mi.m, mi.n, mi.k). One call fixes the arch and
// mi_dtype, and a config list has only a few distinct MIs, so a linear scan is enough.
struct mi_latency_cache {
  std::vector<std::pair<std::uint64_t, std::size_t>> entries;
  std::size_t get(const hardware_t& hardware, const dim3_t& mi, data_type_t mi_dtype) {
    const std::uint64_t key = static_cast<std::uint64_t>(mi.m) |
                              (static_cast<std::uint64_t>(mi.n) << 21) |
                              (static_cast<std::uint64_t>(mi.k) << 42);
    for (const auto& e : entries)
      if (e.first == key) return e.second;
    const std::size_t v = hardware.get_mi_latency(mi.m, mi.n, mi.k, mi_dtype);
    entries.emplace_back(key, v);
    return v;
  }
};

// The winner for split-K-prone shapes depends on cross-CU K-reduction reuse that the
// level-1 proxy cannot see, so the prune would drop it. For these shapes every config
// that passes level 0 reaches the full level.
bool is_split_k_prone(const problem_t& problem, std::size_t n_cu) {
  const std::size_t ref_tiles = math::safe_ceil_div(problem.size.m, kSplitKRefTile) *
                                math::safe_ceil_div(problem.size.n, kSplitKRefTile) * problem.batch;
  const std::size_t ref_k_iters = problem.size.k / kSplitKRefDepth;
  return ref_tiles < n_cu && ref_k_iters >= kSplitKMinKIters;
}

// GEMV shapes (M == 1 or N == 1) are never pruned either. The full model has GEMV and
// M-edge terms the proxy cannot model, and hipBLASLt's predicates reject most configs
// of these lists after the ranking, so a pruned list may keep few runnable configs.
bool is_gemv(const problem_t& problem) { return problem.size.m == 1 || problem.size.n == 1; }

void log_leveled_rejection(const config_t& config, int level, const char* reason) {
  OLOG_DEBUG("  [leveled] Config MT=(" << config.mt.m << "," << config.mt.n << "," << config.mt.k
                                       << ") MI=(" << config.mi.m << "," << config.mi.n << ","
                                       << config.mi.k << ") REJECTED at level " << level << ": "
                                       << reason);
}

// Store latency of one interior MT_M x MT_N output tile. Ignores edge tiles, split-K
// reduction, ACC->VGPR transfer and alignment.
double epilogue_latency_quick(const hardware_t& hardware,
                              const dim3_t& mt,
                              data_type_t d_dtype,
                              std::size_t num_output_tiles,
                              std::size_t n_cu) {
  const double d_bytes = data_type_to_bytes(d_dtype);
  if (d_bytes == 0.0) return 0.0;

  const std::size_t active_cus = std::min(num_output_tiles, n_cu);
  if (active_cus == 0) return 0.0;

  const double mem_bw   = compute_mem_bw_from_occupancy(hardware, active_cus);
  const double store_bw = hardware.mem3_perf_ratio * mem_bw;
  if (store_bw <= 0.0) return 0.0;

  const double per_cu_store_bw = store_bw / static_cast<double>(active_cus);
  const double tile_bytes      = static_cast<double>(mt.m) * mt.n * d_bytes;
  return tile_bytes / per_cu_store_bw;
}

// Tensile params of a config, or the defaults when it has none (as in the full model).
const tensile_params_t& proxy_tparams(const config_t& config) {
  static const tensile_params_t kDefaultTensile{};
  return config.has_tensile_params() ? config.tensile() : kDefaultTensile;
}

// Level-1 context-free proxy, in cycles: a data-parallel schedule of output-tile waves,
// where each tile pays its K-loop, its fixed per-tile costs and the quick epilogue.
//
//   score = ceil(tiles / n_cu) * (iters * max(L_compute, L_mem) + overhead + epilogue)
//
// - Schedule: one output tile per CU per wave, with no K-split. Charging the balanced
//   work per CU instead (as if stream-K spread every K-iteration over all CUs for free)
//   makes few large tiles look nearly free on small and medium grids, although they
//   leave CUs idle or need a split with a costly reduction.
// - Per-tile overhead, as in the full model's main loop: the prologue's first load
//   (L_mem), the PGR fill and the terminal compute iterations (L_compute per prefetched
//   iteration, at most the K-iterations), and the per-K-iteration loop bookkeeping,
//   partly hidden by deep PGR.
// - Memory: A/B bytes of one MT_M x MT_N x MT_K block at L2 bandwidth, with every CU
//   loading. The proxy has no cache hit rates, and charging DRAM bandwidth instead
//   overcharges large tiles that the full model mostly serves from the caches.
//
// The full-model terms the proxy omits: cache hit rates, the split-K/stream-K schedule
// and its reduction, edge tiles and the scalar epilogue, and compute occupancy scaling.
double score_proxy(const problem_t& problem,
                   const hardware_t& hardware,
                   const config_t& config,
                   std::size_t n_cu,
                   mi_latency_cache& lmi) {
  const std::size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  const std::size_t L_MI = lmi.get(hardware, config.mi, problem.mi_dtype);
  const double L_compute = static_cast<double>(N_MI * L_MI);

  const std::size_t grid_m           = math::safe_ceil_div(problem.size.m, config.mt.m);
  const std::size_t grid_n           = math::safe_ceil_div(problem.size.n, config.mt.n);
  const std::size_t num_output_tiles = grid_m * grid_n * problem.batch;

  const double num_iter = static_cast<double>(
      std::max<std::size_t>(1, math::safe_ceil_div(problem.size.k, config.mt.k)));

  const double a_bytes         = data_type_to_bytes(problem.a_dtype);
  const double b_bytes         = data_type_to_bytes(problem.b_dtype);
  const double bytes_per_block = static_cast<double>(config.mt.m * config.mt.k) * a_bytes +
                                 static_cast<double>(config.mt.k * config.mt.n) * b_bytes;
  const double L_mem = (hardware.mem1_perf_ratio > 0.0)
                           ? bytes_per_block *
                                 static_cast<double>(std::max<std::size_t>(hardware.N_CU, 1)) /
                                 hardware.mem1_perf_ratio
                           : 0.0;

  const double pgr = static_cast<double>(std::max(proxy_tparams(config).prefetch_global_read, 0));
  const double loop_overlap = (pgr >= 3.0) ? 1.0 / (pgr - 1.0) : 1.0;
  const double overhead     = L_mem + L_compute * std::min(pgr, num_iter) +
                          heuristic_defaults_t::K_ITER_LOOP_OVERHEAD * num_iter * loop_overlap;

  const double epilogue =
      epilogue_latency_quick(hardware, config.mt, problem.d_dtype, num_output_tiles, n_cu);
  const double waves = static_cast<double>(math::safe_ceil_div(num_output_tiles, n_cu));
  return waves * (std::max(L_compute, L_mem) * num_iter + overhead + epilogue);
}

}  // namespace

leveled_estimation_result_t rank_leveled_estimation(const problem_t& problem,
                                                    const hardware_t& hardware,
                                                    const std::vector<config_t>& configs) {
  const std::size_t n_cu =
      std::max<std::size_t>(resolve_num_cus(problem.num_cus, hardware.N_CU), 1);

  // Levels 0 and 1, fused: reject exactly what the single-pass ranking rejects before
  // building a context, then score the rest with the proxy. (proxy cost, index) pairs.
  std::vector<std::pair<double, std::size_t>> candidates;
  candidates.reserve(configs.size());
  mi_latency_cache lmi;
  for (std::size_t idx = 0; idx < configs.size(); ++idx) {
    const config_t& config = configs[idx];
    if (!check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype)) {
      log_leveled_rejection(config, 0, "LDS capacity exceeded");
      continue;
    }
    if (fast_reject(problem, hardware, config)) {
      log_leveled_rejection(config, 0, "fast_reject");
      continue;
    }
    candidates.emplace_back(score_proxy(problem, hardware, config, n_cu, lmi), idx);
  }

  leveled_estimation_result_t result;

  // Prune to the cheapest max(kLeveledMinKeep, n * kLeveledKeepFraction) by proxy cost.
  // Configs with the same proxy cost as the last kept one are kept as well: the proxy
  // only sees MT, MI, PGR and the data types, so variants that differ in other parameters
  // tie, and the tie-breakers in rank_configs() need the whole group. The kept set
  // therefore depends on proxy values only, not on the order of the config list.
  std::size_t keep =
      (is_split_k_prone(problem, n_cu) || is_gemv(problem))
          ? candidates.size()
          : std::max(kLeveledMinKeep,
                     static_cast<std::size_t>(static_cast<double>(candidates.size()) *
                                              kLeveledKeepFraction));
  if (candidates.size() > keep) {
    std::nth_element(candidates.begin(), candidates.begin() + keep - 1, candidates.end());
    const double cut_cost = candidates[keep - 1].first;
    const auto tie_end    = std::partition(candidates.begin() + keep,
                                        candidates.end(),
                                        [cut_cost](const auto& c) { return c.first == cut_cost; });
    keep                  = static_cast<std::size_t>(tie_end - candidates.begin());

    // The pruned tail, in proxy order with ties broken by index.
    std::sort(candidates.begin() + keep, candidates.end());
    result.pruned.reserve(candidates.size() - keep);
    for (auto it = candidates.begin() + keep; it != candidates.end(); ++it) {
      if (Logger::instance().is_enabled())
        log_leveled_rejection(configs[it->second], 1, "pruned by proxy");
      result.pruned.push_back(it->second);
    }
    candidates.resize(keep);
  }

  // Full level, in the caller's config order so that rank_configs() orders equal
  // latencies the same way as the single-pass ranking.
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.second < b.second;
  });

  result.scored.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const std::size_t idx  = candidate.second;
    const config_t& config = configs[idx];
    context_t context(problem, hardware, config);
    const double latency = estimation_latency_from_context(problem, hardware, config, context);
    if (latency != kRejectedLatency) result.scored.emplace_back(latency, idx);
  }
  return result;
}

}  // namespace gemm
}  // namespace origami
