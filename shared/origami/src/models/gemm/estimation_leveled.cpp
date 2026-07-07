// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Leveled coarse-to-fine GEMM estimation: the analytical model's internal
// scoring cascade. Composed entirely from the gemm:: building blocks declared in
// gemm.hpp (compute / memory / epilogue / full-latency), so it lives in its own
// translation unit, separate from those primitives in gemm_common.cpp.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "origami/estimation_leveled.hpp"
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

namespace {

// The cascade has two stages: levels 0+1 are context-free and fused in one pass
// (feasibility / fast-reject + a compute-vs-memory proxy), pruned to a small working
// set, which then gets the full analytical latency. Split-K-prone problems skip the
// prune (see is_split_k_prone) so every candidate reaches the full-latency level.
// (An intermediate flat-cache roofline level was removed -- it pruned eventual
// winners for no speed benefit.)

// Level-1 keep fraction: after the context-free proxy pass, keep the cheapest 1/6.
constexpr double kL1KeepFraction = 1.0 / 6.0;

// Recall floor: never prune the working set below this many configs -- the dominant
// accuracy dial. 48 keeps the flat model's pick on ~99% of the (non-split-K) problems
// the leveled path handles, while scoring well under the flat per-config cost.
constexpr std::size_t kInternalMinKeep = 48;

// Tiny MI-latency memo keyed by packed (mi.m, mi.n, mi.k). A scoring call fixes
// the arch and mi_dtype, and a sweep has only a handful of distinct MIs, so this
// turns the level-1 proxy's nested-map lookups into a few. Linear scan over a few
// entries beats hashing and stays cache-resident.
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

// Context-free feasibility: tensilelite is LDS-validated upstream by the library.
inline bool coarse_feasible(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) {
  if (config.target == target_t::tensilelite) return true;
  return check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype);
}

// Split-K-prone detection (problem/hardware only, config-independent). The winner
// for these shapes is decided by cross-CU K-reduction reuse that the context-free
// coarse proxy cannot model, so it would prune the eventual StreamK winner. Uses
// StreamK's shape criterion on a large reference tile: even a 256x256 output tiling
// underfills the GPU (tiles < N_CU) AND K is deep enough to split (>= 64 iterations
// of a 64-deep k-tile). For these the cascade skips its coarse prune so every
// candidate reaches the full-latency level -- matching the flat per-config ranking
// exactly where accuracy needs it, while the non-split-K majority still gets pruned.
inline bool is_split_k_prone(const problem_t& problem, const hardware_t& hardware) {
  constexpr std::size_t ref_tile = 256;
  const std::size_t ref_tiles = math::safe_ceil_div(problem.size.m, ref_tile) *
                                math::safe_ceil_div(problem.size.n, ref_tile) * problem.batch;
  const std::size_t ref_k_iters = problem.size.k / 64;  // K-iters for a 64-deep k-tile
  return ref_tiles < static_cast<std::size_t>(hardware.N_CU) && ref_k_iters >= 64;
}

// Debug-only: report the cascade level at which a config drops out. No-op unless
// the origami logger is enabled (OLOG_DEBUG short-circuits), so there is zero cost
// on the normal path.
inline void log_reject(const config_t& config, int level, const char* reason) {
  OLOG_DEBUG("[leveled] config " << config.index << " MT " << config.mt.m << "x" << config.mt.n
                                 << "x" << config.mt.k << " rejected at level " << level << " ("
                                 << reason << ")");
}

// Level-1 context-free proxy: a roofline of the main loop (compute vs A/B read
// bandwidth) scaled by the K-iteration count, plus the coarse tile-store epilogue,
// times a cheap num_output_tiles/N_CU timestep estimate. Avoids the streamk
// launch-param selection (context) entirely.
//
// Critically it scales the per-k-block arms by num_iter ~= K / MT_K. Compute per
// block is N_MI*L_MI ~ MT_K, so compute*num_iter is MT_K-independent (total MACs
// are fixed) -- without this factor the proxy penalized deep-K tiles and pruned the
// eventual StreamK winner. The memory arm (A/B reads per block / bandwidth) is what
// then ranks MT_K choices by their global-read reuse.
inline double score_compute_proxy(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  mi_latency_cache& lmi) {
  const std::size_t N_MI      = compute_number_matrix_instructions(config.mt, config.mi);
  const std::size_t L_MI      = lmi.get(hardware, config.mi, problem.mi_dtype);
  const double      L_compute = static_cast<double>(N_MI * L_MI);

  const std::size_t grid_m           = math::safe_ceil_div(problem.size.m, config.mt.m);
  const std::size_t grid_n           = math::safe_ceil_div(problem.size.n, config.mt.n);
  const std::size_t num_output_tiles = grid_m * grid_n * problem.batch;

  // Number of K main-loop iterations for this tile depth (context-free).
  const std::size_t num_iter =
      (config.mt.k > 0) ? std::max<std::size_t>(1, math::safe_ceil_div(problem.size.k, config.mt.k))
                        : 1;

  // Context-free memory roofline: bytes of A/B read for one MT_M x MT_N x MT_K
  // block, over a bandwidth approximated from the output-tile occupancy.
  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  const double b_bytes = data_type_to_bytes(problem.b_dtype);
  const double bytes_per_block =
      static_cast<double>(config.mt.m * config.mt.k) * a_bytes +
      static_cast<double>(config.mt.k * config.mt.n) * b_bytes;
  const std::size_t active_cus = std::min(num_output_tiles, static_cast<std::size_t>(hardware.N_CU));
  const double      bw         = compute_mem_bw_from_occupancy(hardware, active_cus);
  const double      L_mem      = (bw > 0.0) ? (bytes_per_block / bw) : 0.0;

  // Spread all K-blocks (num_output_tiles x num_iter) across the CUs. This coarse
  // proxy only drives the prune among data-parallel (many-tile) problems; split-K-
  // prone shapes skip the prune entirely (see is_split_k_prone), so the proxy's
  // spread approximation never gates their winner.
  const double      per_block      = std::max(L_compute, L_mem);
  const std::size_t total_k_blocks = num_output_tiles * num_iter;
  const std::size_t work_per_cu =
      (hardware.N_CU > 0) ? math::safe_ceil_div(total_k_blocks, hardware.N_CU) : total_k_blocks;
  const std::size_t epilogue_waves =
      (hardware.N_CU > 0) ? math::safe_ceil_div(num_output_tiles, hardware.N_CU) : num_output_tiles;

  const double epilogue =
      compute_coarse_epilogue_latency(hardware, config.mt, problem.d_dtype, num_output_tiles);
  return per_block * static_cast<double>(work_per_cu) +
         epilogue * static_cast<double>(epilogue_waves);
}

}  // namespace

scored_configs_t score_estimation_leveled(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const std::vector<config_t>& configs,
                                          const std::vector<std::size_t>& survivors) {
  scored_configs_t scored;

  // Resolve the debug-logging state once per call (not per config): when off, the
  // rejection logging below is fully skipped -- no logger lookups, no formatting.
  const bool dbg = Logger::instance().is_enabled();

  const auto by_cost = [](const scored_config_t& a, const scored_config_t& b) {
    return a.first < b.first;
  };

  // Levels 0+1, fused: context-free feasibility / fast-reject + compute-vs-memory
  // proxy, in a single pass (no context built).
  mi_latency_cache lmi;
  scored.reserve(survivors.size());
  for (std::size_t idx : survivors) {
    const config_t& config = configs[idx];
    if (!coarse_feasible(problem, hardware, config)) {  // level 0
      if (dbg) log_reject(config, 0, "infeasible");
      continue;
    }
    if (fast_reject(problem, hardware, config)) {  // level 0
      if (dbg) log_reject(config, 0, "fast_reject");
      continue;
    }
    const double cost = score_compute_proxy(problem, hardware, config, lmi);  // level 1
    if (cost != std::numeric_limits<double>::max()) {
      scored.emplace_back(cost, idx);
    } else if (dbg) {
      log_reject(config, 1, "compute-proxy disqualify");
    }
  }

  // Prune at level 1 to the cheapest max(min_keep, 1/6) survivors (O(n) selection).
  // Split-K-prone problems are the exception: their winner is set by cross-CU K
  // reuse the coarse proxy can't model, so keep everything and let the full-latency
  // level rank them (equivalent to the flat per-config path) rather than risk pruning
  // the winner.
  const std::size_t keep =
      is_split_k_prone(problem, hardware)
          ? scored.size()
          : std::max(kInternalMinKeep,
                     static_cast<std::size_t>(static_cast<double>(scored.size()) * kL1KeepFraction));
  if (scored.size() > keep) {
    std::vector<std::size_t> before_idx;  // debug-only: attribute the L1 keep-fraction drops
    if (dbg)
      for (const auto& e : scored) before_idx.push_back(e.second);
    std::nth_element(scored.begin(), scored.begin() + keep, scored.end(), by_cost);
    scored.resize(keep);
    if (dbg) {
      std::unordered_set<std::size_t> kept;
      for (const auto& e : scored) kept.insert(e.second);
      for (std::size_t i : before_idx)
        if (!kept.count(i)) log_reject(configs[i], 1, "keep-fraction prune");
    }
  }

  // Final level: build the context for each survivor and score at full analytical
  // detail, then sort ascending -- the ranking the caller consumes.
  scored_configs_t final_scored;
  final_scored.reserve(scored.size());
  for (const auto& e : scored) {
    const std::size_t idx = e.second;
    context_t         ctx(problem, hardware, configs[idx]);
    const double cost = estimation_latency_from_context(problem, hardware, configs[idx], ctx);
    if (cost != std::numeric_limits<double>::max()) final_scored.emplace_back(cost, idx);
  }
  std::stable_sort(final_scored.begin(), final_scored.end(), by_cost);
  return final_scored;
}

}  // namespace gemm
}  // namespace origami
