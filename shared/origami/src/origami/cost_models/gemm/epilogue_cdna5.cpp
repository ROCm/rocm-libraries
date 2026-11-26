// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// gfx1250 epilogue of one tile, from the measured epilogue-store curve.  TensileLite writes D
// from VGPRs, one buffer_store of VW elements per lane.  The last tile row (column) takes the
// guarded edge store path when M % MT0 (N % MT1) is nonzero; with a cluster, an M remainder sends
// every tile down the M-edge path.  M-edge tiles store one element per lane, or with AdaptiveGemm
// the widest vector width of the store ladder (full, full / 2, ..., 1; only full and 1 with
// LocalSplitU) that divides M; N-only edge tiles store at full width.  An edge tile costs its
// full-width store plus the measured in-kernel extra time of each edge-path store instruction at
// its width. A K-split partial tile is stored as f32.  With the in-kernel fixup it goes to the flat
// workspace layout and the tile's owner waits for the peers' flags (a DRAM round trip), reads their
// partials with per-lane loads and then stores D.  With parallel (post-kernel) reduction it goes
// through the D store code instead.
epilogue_terms_t cdna5_model::epilogue(const problem_t& problem,
                                       const hardware_t& hardware,
                                       const config_t& config,
                                       const context_t& context) {
  using h                     = heuristic_defaults_t;
  const cdna5_constants_t& c5 = cdna5_model::constants(hardware);
  const double d_bytes        = context.d_bytes;
  // An unset D type (data_type_t::None) has a negative size.
  if (!(d_bytes > 0.0) || !(hardware.mem3_perf_ratio > 0.0)) return {};
  const size_t M = problem.size.m, N = problem.size.n;
  const size_t MT_M = config.mt.m, MT_N = config.mt.n;
  const double lanes     = static_cast<double>(context.wavefront_size);
  const double waves     = static_cast<double>(context.wave_group.m * context.wave_group.n);
  const double dram_peak = hardware.mem3_perf_ratio;  // B/clk
  const double scale     = dram_peak / c5.STORE_REF_DRAM_B_PER_CLK;

  const size_t max_vw  = static_cast<size_t>(std::max(
      1.0, std::floor(static_cast<double>(h::EPILOGUE_BYTES_PER_VECTORIZED_STORE) / d_bytes)));
  const size_t vw_full = std::clamp<size_t>(config.gwvw_d > 0 ? config.gwvw_d : max_vw, 1, max_vw);
  size_t vw_edge       = 1;
  if (tparams(config).adaptive_gemm != 0) {
    if (context.wave_group.k > 1)
      vw_edge = (M % vw_full == 0) ? vw_full : 1;
    else
      for (size_t v = vw_full; v > 1; v /= 2)
        if (M % v == 0) {
          vw_edge = v;
          break;
        }
  }

  const double writers = static_cast<double>(std::max<size_t>(context.active_cus, 1));
  size_t vw            = vw_full;
  // Without SourceSwap a 16 B-lane store instruction writes 32 B into each of 16 columns: each
  // instruction takes longer, the device ceiling is lower (both per TemporalHintD), and a tile
  // column of more than a line that is not a whole number of 128 B lines touches more lines than it
  // fills.  A column under a line shares its lines with the next tiles' columns and costs only the
  // instruction's line requests (nonswap_sep).
  const int d_th        = store_temporal_hint(config);
  auto nonswap_line_eff = [&](double col_bytes) {
    const double line = static_cast<double>(h::CACHE_LINE_BYTES);
    if (std::fmod(col_bytes, line) == 0.0 || col_bytes < line) return 1.0;
    double touched = 0.0;
    for (double o = 0.0; o < line; o += c5.NONSWAP_STORE_COLUMN_BYTES)
      touched += std::ceil((o + col_bytes) / line);
    return (col_bytes / line) / (touched / (line / c5.NONSWAP_STORE_COLUMN_BYTES));
  };
  // Share of a non-SourceSwap instruction's line requests beyond one contiguous run.  Its 32 B
  // column chunks sit M elements apart (D's leading dimension is taken as M): from a pitch of one
  // line up each takes its own line (16 lines, the fitted case); a narrower pitch packs them into
  // fewer lines, down to inst_bytes / 128 B (4) for one contiguous run, which stores like the
  // SourceSwap layout.  Lines are averaged over the instruction start offsets in the matrix
  // (16-column groups, 32 B row blocks).
  auto nonswap_sep = [&](double inst_bytes, double elem_bytes) {
    const double line  = static_cast<double>(h::CACHE_LINE_BYTES);
    const double chunk = c5.NONSWAP_STORE_COLUMN_BYTES;
    const double pitch = static_cast<double>(M) * elem_bytes;
    const int cols     = static_cast<int>(inst_bytes / chunk);
    const double lmin  = inst_bytes / line;
    if (pitch >= line || !(cols > lmin)) return 1.0;
    double lines = 0.0;
    int starts   = 0;
    for (int g = 0; g < 8; ++g)
      for (double r = 0.0; r < pitch; r += chunk, ++starts) {
        double last = -1.0;
        for (int c = 0; c < cols; ++c) {
          const double s     = std::fmod(g * cols * pitch + r, line) + c * pitch;
          const double first = std::max(std::floor(s / line), last + 1.0);
          const double end   = std::floor((s + chunk - 1.0) / line);
          if (end >= first) {
            lines += end - first + 1.0;
            last = end;
          }
        }
      }
    return std::clamp((lines / starts - lmin) / (cols - lmin), 0.0, 1.0);
  };
  // Cycles to store `elems` elements of `elem_bytes` from one WGP at full width (lanes of min(vw x
  // bytes, 16 B)) with TemporalHint `th`; `rows`: the tile's rows (M), 0 outside the D layout;
  // `nonswap_layout`: the non-SourceSwap D layout.
  auto store_clk = [&](double elems, double elem_bytes, size_t rows, bool nonswap_layout, int th) {
    if (!(elems > 0.0)) return 0.0;
    const double lane_bytes = std::min(static_cast<double>(vw) * elem_bytes,
                                       static_cast<double>(h::EPILOGUE_BYTES_PER_VECTORIZED_STORE));
    const double inst_bytes = lanes * lane_bytes;
    const double bytes      = elems * elem_bytes;
    const bool nonswap     = nonswap_layout && lane_bytes == h::EPILOGUE_BYTES_PER_VECTORIZED_STORE;
    const double extra_clk = !nonswap  ? 0.0
                             : th == 1 ? c5.NONSWAP_STORE_EXTRA_CLK_NT
                                       : c5.NONSWAP_STORE_EXTRA_CLK_RT;
    const double peak      = !nonswap  ? c5.STORE_PEAK_B_PER_CLK
                             : th == 1 ? c5.NONSWAP_STORE_PEAK_B_PER_CLK_NT
                                       : c5.NONSWAP_STORE_PEAK_B_PER_CLK_RT;
    // Non-SourceSwap: the extra time and the lower ceiling apply to the share of line requests
    // beyond a run.
    const double sep = nonswap ? nonswap_sep(inst_bytes, elem_bytes) : 1.0;
    const double ceiling =
        sep < 1.0 ? 1.0 / ((1.0 - sep) / c5.STORE_PEAK_B_PER_CLK + sep / peak) : peak;
    // The wave and WGP rates are the WGP's own (the same on the 8- and 6-XCD parts); only the
    // device ceiling scales with the part's DRAM peak.
    const double wgp_rate =  // B/clk
        1.0 / ((c5.STORE_INST_CLK + sep * extra_clk + inst_bytes / c5.STORE_WAVE_B_PER_CLK) /
                   (waves * inst_bytes) +
               1.0 / c5.STORE_WGP_B_PER_CLK);
    double device = soft_min(writers * wgp_rate, ceiling * scale, c5.STORE_SHARPNESS);
    if (nonswap)
      return bytes / (device * nonswap_line_eff(static_cast<double>(rows) * elem_bytes) / writers);
    return bytes / device * writers;
  };
  // Extra time of a wave's edge-path store instruction of `lane_bytes` per lane, cycles.
  auto edge_extra_clk = [&](double lane_bytes) {
    if (lane_bytes <= 2.0) return c5.EDGE_STORE_EXTRA_CLK_2B;
    if (lane_bytes <= 4.0)
      return c5.EDGE_STORE_EXTRA_CLK_2B +
             (lane_bytes - 2.0) / 2.0 * (c5.EDGE_STORE_EXTRA_CLK_4B - c5.EDGE_STORE_EXTRA_CLK_2B);
    if (lane_bytes <= 8.0)
      return c5.EDGE_STORE_EXTRA_CLK_4B +
             (lane_bytes - 4.0) / 4.0 * (c5.EDGE_STORE_EXTRA_CLK_8B - c5.EDGE_STORE_EXTRA_CLK_4B);
    if (lane_bytes <= 16.0)
      return c5.EDGE_STORE_EXTRA_CLK_8B +
             (lane_bytes - 8.0) / 8.0 * (c5.EDGE_STORE_EXTRA_CLK_16B - c5.EDGE_STORE_EXTRA_CLK_8B);
    return c5.EDGE_STORE_EXTRA_CLK_16B;
  };
  const bool cluster_m_edge =
      std::max<size_t>(config.cluster_dim.m, 1) * std::max<size_t>(config.cluster_dim.n, 1) > 1 &&
      M % MT_M != 0;

  // m_path: the tile stores through the M-edge path (last row, or any tile of a cluster kernel with
  // an M remainder); otherwise a tile narrower than MT_N takes the full-width N-edge path.
  auto tile_clk = [&](size_t tm, size_t tn, bool m_path) {
    const double elems = static_cast<double>(tm) * static_cast<double>(tn);
    vw                 = vw_full;
    const bool nonswap = config.has_tensile_params() && !config.tensile().source_swap;
    const size_t ew    = m_path ? vw_edge : vw_full;
    // The D store path with elements of `elem_bytes`: the full-width store in the kernel's lane
    // layout plus, on an edge path, the extra time of each edge-path store instruction (lanes of
    // min(ew x bytes, 16 B)).
    auto store_path_clk = [&](double elem_bytes) {
      double clk = store_clk(elems, elem_bytes, tm, nonswap, d_th);
      if (m_path || tn < MT_N) {
        const double lane_bytes =
            std::min(static_cast<double>(ew) * elem_bytes,
                     static_cast<double>(h::EPILOGUE_BYTES_PER_VECTORIZED_STORE));
        const double insts = elems / (lanes * waves * (lane_bytes / elem_bytes));
        clk += insts * (m_path ? edge_extra_clk(lane_bytes) : c5.EDGE_STORE_EXTRA_CLK_N);
      }
      return clk;
    };
    const double t = store_path_clk(d_bytes);
    if (context.splitting_factor <= 1) return t;
    const double ws = static_cast<double>(h::EPILOGUE_WORKSPACE_BYTES_PER_ELEM);
    // Parallel reduction: every split workgroup stores its f32 partial through the D store code
    // (SrdD moved to the split's slice of the workspace): D's lane layout, edge paths and temporal
    // hint, 4 B elements.
    if (context.reduction_strategy == reduction_t::parallel) return store_path_clk(ws);
    // In-kernel linear fixup (Components/StreamK.py storeBranchesCommon): peers store their f32
    // partials in the flat workspace layout and set a device-scope flag; the tile's owner then,
    // once per peer, busy-polls that flag, resets it and reads the partial with per-lane loads, and
    // finally stores D.  The fixup is priced by the partial's store and the F - 1 partial reads (no
    // fixed time per partial is charged); only the tiles' owners read, so they share the per-lane
    // L2 peak.
    const double n_partials = static_cast<double>(context.splitting_factor - 1);
    const double partials   = n_partials * elems * ws;
    const double owners  = std::max(writers / static_cast<double>(context.splitting_factor), 1.0);
    const double read_bw = std::min(c5.LANE_LOAD_WGP_B_PER_CLK, c5.LANE_LOAD_L2_B_PER_CLK / owners);
    return store_clk(elems, ws, 0, false, 0) + (partials / read_bw + t);
  };

  // A grid within one wave finishes with its slowest tile; a longer one averages the tile types.
  const size_t mr = (M % MT_M) ? M % MT_M : MT_M, nr = (N % MT_N) ? N % MT_N : MT_N;
  const bool me = M % MT_M != 0, ne = N % MT_N != 0;
  const double fm =
      (me && context.grid_m > 1) ? 1.0 / static_cast<double>(context.grid_m) : (me ? 1.0 : 0.0);
  const double fn =
      (ne && context.grid_n > 1) ? 1.0 / static_cast<double>(context.grid_n) : (ne ? 1.0 : 0.0);
  const double t_in = (fm < 1.0 && fn < 1.0) ? tile_clk(MT_M, MT_N, cluster_m_edge) : 0.0;
  const double t_m  = (me && fn < 1.0) ? tile_clk(mr, MT_N, true) : 0.0;
  const double t_n  = (ne && fm < 1.0) ? tile_clk(MT_M, nr, cluster_m_edge) : 0.0;
  const double t_mn = (me && ne) ? tile_clk(mr, nr, true) : 0.0;
  const double L_epilogue =
      (context.num_output_tiles <= context.n_cu)
          ? std::max({t_in, t_m, t_n, t_mn})
          : (1 - fm) * (1 - fn) * t_in + fm * (1 - fn) * t_m + (1 - fm) * fn * t_n + fm * fn * t_mn;
  return {L_epilogue, 0.0};
}

}  // namespace gemm
}  // namespace origami
