// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/math.hpp"
#include "origami/origami.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Workgroup mapping
// ---------------------------------------------------------------------------

// WGM a gfx1250 launch without clusters walks tiles with, as
// ContractionSolution::calculateAutoWGM picks it: a fixed config.workgroup_mapping if set; a
// StreamK auto mapping (WGM 0, WGMXCC -1) on a nonzero launch grid takes
// select_workgroup_mapping on the solution's tile and hints; any other WGM 0 takes
// ceil(sqrt(N_CU / NUM_XCD)) of the whole device, whatever the problem's CU budget.  A config
// without TensileLite params declares no mapping and takes select_workgroup_mapping.  A
// cluster kernel is dispatched whole and bypasses WGM.  WGMXCC, chunk and split-K stay off.
// The pick builds a context_t, whose constructor predicts a WGM again; the selector gives its own
// context a config with a fixed mapping, which takes the cheap fixed branch below, so the pick
// never recurses.
workgroup_mapping_t cdna5_model::predict_workgroup_mapping(const problem_t& problem,
                                                           const hardware_t& hardware,
                                                           const config_t& config,
                                                           size_t grid_m,
                                                           size_t grid_n,
                                                           size_t splitting_factor) {
  if (std::max<size_t>(config.cluster_dim.m, 1) * std::max<size_t>(config.cluster_dim.n, 1) > 1)
    return {0, 0, 1, 1};

  const tensile_params_t& tp = tparams(config);
  // The launch grid is nonzero exactly when the problem has tiles.
  const bool grid_nonzero = grid_m * grid_n * splitting_factor * problem.batch != 0;
  const bool selected     = !config.has_tensile_params() ||
                        (config.workgroup_mapping == 0 && tp.workgroup_mapping_xcc == -1 &&
                         config.stream_k > 0 && grid_nonzero);
  if (selected) {
    config_t host;
    host.mt                 = config.mt;
    host.cache_hints_a      = config.cache_hints_a;
    host.cache_hints_b      = config.cache_hints_b;
    host.tensile().tdm_inst = tp.tdm_inst;
    return {0, 0, 1, select_workgroup_mapping(problem, hardware, host, 0).wgm};
  }
  if (config.workgroup_mapping != 0) return {0, 0, 1, config.workgroup_mapping};
  const size_t num_xcd = std::max<size_t>(hardware.NUM_XCD, 1);
  return {0,
          0,
          1,
          static_cast<int>(std::ceil(std::sqrt(static_cast<double>(hardware.N_CU / num_xcd))))};
}

// ---------------------------------------------------------------------------
// Cache hit rates
// ---------------------------------------------------------------------------

// Share of a wave's keys (A rows or B columns of one K split and batch) that domain 0 requests,
// when each key has n ids `stride` apart in each of `slabs` slabs `slab_ids` apart, at a random
// phase of the dispatch cycle: domain 0 owns `on` consecutive XCDs of every `period`, and a key
// is requested once one of its ids lands there.  The ids fall on at most period / gap phases,
// gap = gcd of the period and the strides in use.
static double domain_key_share(size_t n,
                               size_t stride,
                               size_t slabs,
                               size_t slab_ids,
                               size_t on,
                               size_t period) {
  if (on >= period || n == 0 || slabs == 0) return 1.0;
  size_t gap = period;
  if (n > 1) gap = std::gcd(gap, stride);
  if (slabs > 1) gap = std::gcd(gap, slab_ids);
  const size_t phases  = std::min(n * slabs, period / gap);
  const size_t covered = on + (phases - 1) * std::min(gap, on);
  return std::min(static_cast<double>(covered) / static_cast<double>(period), 1.0);
}

cache_hit_rates_t estimate_l2_hit_cdna5(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config,
                                        const context_t& context,
                                        double* unique_bytes) {
  if (unique_bytes) *unique_bytes = 0.0;
  int wgm                          = context.wgm.wgm;
  const size_t n_cu                = std::max<size_t>(context.n_cu, 1);
  const size_t grid_m              = std::max<size_t>(context.grid_m, 1);
  const size_t grid_n              = std::max<size_t>(context.grid_n, 1);
  const size_t splits              = std::max<size_t>(context.splitting_factor, 1);
  const size_t batch               = std::max<size_t>(problem.batch, 1);
  const size_t cs                  = std::max<size_t>(config.cluster_dim.m, 1);
  const size_t ck                  = std::max<size_t>(config.cluster_dim.n, 1);
  const operand_traffic_t& traffic = context.traffic;
  const double l2_capacity         = static_cast<double>(hardware.l2_domain_capacity());

  // A cluster is dispatched whole and bypasses WGM: cluster ids walk the cluster grid M-first,
  // one column-major slab spanning every cluster row, each cluster covering (on average)
  // grid_m / cx_n rows and grid_n / cy_n columns.
  dim4_t grid        = {splits, grid_m, grid_n, batch};
  size_t wave        = std::min(grid.total(), n_cu);
  double rows_per_wg = 1.0;
  double cols_per_wg = 1.0;
  if (cs * ck > 1) {
    const size_t cx_n = math::safe_ceil_div(grid_m, cs);
    const size_t cy_n = math::safe_ceil_div(grid_n, ck);
    grid              = {1, cx_n, cy_n, batch};
    wgm               = -static_cast<int>(cx_n);
    wave              = std::min({cx_n * cy_n * batch,
                                  clusters_per_wave(hardware, n_cu, cs * ck),
                                  std::max<size_t>(context.num_wgs / (cs * ck), 1)});
    rows_per_wg       = static_cast<double>(grid_m) / static_cast<double>(cx_n);
    cols_per_wg       = static_cast<double>(grid_n) / static_cast<double>(cy_n);
  }
  if (wave == 0 || grid.k == 0 || grid.m == 0 || grid.n == 0 || grid.b == 0) return {};
  if (wgm == 0) wgm = 1;
  // Dispatch rotates the XCDs, so domain 0 runs the ids whose residue lies in [first, first + on)
  // (mod period), first = the first id that lands on XCD 0.
  const size_t period = std::max<size_t>(hardware.NUM_XCD, 1);
  const size_t on     = std::min(hardware.xcd_per_l2_domain(), period);
  const size_t first  = (period - hardware.dispatch_xcd(0)) % period;
  const size_t tail   = wave % period;
  const size_t wrap   = first + on > period ? first + on - period : 0;
  const size_t in_tail =
      std::min(tail > first ? tail - first : 0, on - wrap) + std::min(tail, wrap);
  double requests = static_cast<double>(wave / period * on + in_tail);
  if (!(requests > 0.0)) return {};

  // A slab row (an A row; a B column for column-major slabs) is `width` consecutive tiles in
  // every slab the wave reaches, and a slab column one tile per slab row of a single slab;
  // K splits interleave each tile's ids.
  const bool col_major  = wgm < 0;
  const dim4_t unique   = count_unique_range(grid, wgm, 0, wave);
  const size_t g_rows   = col_major ? grid.n : grid.m;
  const size_t g_cols   = col_major ? grid.m : grid.n;
  const size_t width    = std::min<size_t>(std::abs(wgm), g_cols);
  const size_t mn_tiles = math::safe_ceil_div(wave, grid.k);
  const size_t slab_rows =
      std::min(std::max<size_t>(math::safe_ceil_div(mn_tiles, width), 1), g_rows);
  const size_t slabs =
      std::min(math::safe_ceil_div(mn_tiles, g_rows * width), math::safe_ceil_div(g_cols, width));
  const double row_share = domain_key_share(
      std::min(width, mn_tiles), grid.k, slabs, g_rows * width * grid.k, on, period);
  const double col_share = domain_key_share(slab_rows, width * grid.k, 1, 0, on, period);
  const double planes    = static_cast<double>(unique.k * unique.b);
  const double a_share   = col_major ? col_share : row_share;
  const double b_share   = col_major ? row_share : col_share;
  double a_unique        = std::min(planes * static_cast<double>(unique.m) * a_share, requests);
  double b_unique        = std::min(planes * static_cast<double>(unique.n) * b_share, requests);
  // A cluster workgroup covers several tiles.
  requests *= rows_per_wg * cols_per_wg;
  a_unique *= rows_per_wg;
  b_unique *= cols_per_wg;

  // A request hits when another workgroup of the domain already brought its tile, and the
  // domain keeps the share of the footprint that fits its L2.
  const double a_load = a_unique * traffic.a_iter_bytes;
  const double b_load = b_unique * traffic.b_iter_bytes;
  const double load   = a_load + b_load;
  if (unique_bytes) *unique_bytes = load;
  // Retention by TemporalHint: a regular line (RT, HT) keeps its reuse whatever low-priority fills
  // (NT, LU) arrive, so a low-priority operand does not count against a regular one's residency.
  const bool a_low = l2_low_priority_load(load_temporal_hint(config, false));
  const bool b_low = l2_low_priority_load(load_temporal_hint(config, true));
  auto hit         = [&](double tiles, double own, double other, bool own_low, bool other_low) {
    if (!(own > 0.0)) return 0.0;
    const double competing = (other_low && !own_low) ? 0.0 : other * own / load;
    const double residency = std::min(l2_capacity / (own + competing), 1.0);
    const double share     = std::max(1.0 - tiles / requests, 0.0);
    return residency * share;
  };
  return {hit(a_unique, a_load, b_load, a_low, b_low),
          hit(b_unique, b_load, a_load, b_low, a_low),
          0.0,
          0.0};
}

}  // namespace gemm
}  // namespace origami
