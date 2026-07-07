/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// origami::comm — analytical communication cost model
//
// wg_tile latency computation. All times are in **GPU cycles**.
//
// Two first principles drive this file:
//
//   1. Bottleneck (roofline) per iteration. A workgroup drives many functional
//      units at once (VMEM issue, TCP, L2, MALL, HBM, xGMI, VALU). They run in
//      parallel, so the time for one iteration is set by the *slowest* unit,
//      not the sum — hence T_wlt = max over per-FU times. compute_iter_times
//      turns each FU's cache-line/op count into a cycle count by dividing by
//      that FU's own throughput; bottleneck() names the winner for reporting.
//
//   2. Software pipelining. A WG streams its tile in num_iters iterations,
//      overlapping the load of iter i+1 with the store of iter i. So the read
//      path is exposed only once (T_prologue, filling the pipe), the write
//      path drains only once (T_epilogue), and in steady state every iteration
//      costs one bottleneck T_wlt:
//
//        T_total = T_prologue + (num_iters - 1) × T_wlt + T_epilogue + T_sync
//
//      T_sync is the once-per-tile producer/consumer handshake, off the
//      bandwidth critical path. With large num_iters the (n-1)·T_wlt term
//      dominates (bandwidth-bound); with num_iters=1 the prologue/epilogue/sync
//      constants dominate (latency-bound small messages).
#pragma once

#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace origami::comm {

// ─── iter_times_t: per-FU one-iteration cycle counts ───────────────
/**
 * @brief Per-functional-unit cycle counts for a single pipelined iteration.
 *
 * One cycle count per functional unit for a single iteration. They are held
 * side by side (not summed) precisely because the units overlap: max_cycles()
 * is the roofline bottleneck for the iteration, and bottleneck() reports which
 * unit is binding so a caller can see *why* a transfer is slow.
 */
struct iter_times_t {
  double vmem       = 0.0;  ///< VMEM (vector-memory) instruction-issue time.
  double tcp        = 0.0;  ///< TCP (L1 vector cache) bandwidth time.
  double l2         = 0.0;  ///< L2/TCC bandwidth time (scaled by active CUs per XCD).
  double mall       = 0.0;  ///< MALL (last-level cache) bandwidth time.
  double hbm_read   = 0.0;  ///< Local HBM read bandwidth time.
  double hbm_write  = 0.0;  ///< Local HBM write bandwidth time.
  double xgmi_read  = 0.0;  ///< Remote xGMI read time (latency/MSHR-bound).
  double xgmi_write = 0.0;  ///< Remote xGMI write time (concentration-bound).
  double valu       = 0.0;  ///< VALU (vector-ALU) lane-op time, e.g. reductions.

  /**
   * @brief Roofline bottleneck cycle count for the iteration.
   *
   * @return double Largest per-functional-unit cycle count; the units overlap,
   *         so the iteration is paced by the slowest one.
   */
  constexpr double max_cycles() const noexcept {
    return std::max({vmem, tcp, l2, mall, hbm_read, hbm_write, xgmi_read, xgmi_write, valu});
  }

  /**
   * @brief Name of the binding functional unit for the iteration.
   *
   * Reports which unit has the largest cycle count so a caller can see *why* a
   * transfer is slow.
   *
   * @return std::string_view The FU name with the largest cycle count.
   */
  constexpr std::string_view bottleneck() const noexcept {
    double best_v           = vmem;
    std::string_view best_k = "vmem";
    auto consider           = [&](double v, std::string_view k) {
      if (v > best_v) {
        best_v = v;
        best_k = k;
      }
    };
    consider(tcp, "tcp");
    consider(l2, "l2");
    consider(mall, "mall");
    consider(hbm_read, "hbm_read");
    consider(hbm_write, "hbm_write");
    consider(xgmi_read, "xgmi_read");
    consider(xgmi_write, "xgmi_write");
    consider(valu, "valu");
    return best_k;
  }
};

/**
 * @brief Per-functional-unit time (in cycles) for one inner-loop iteration.
 *
 * The pattern is the same for every unit — time = work / throughput — but the
 * *throughput* term is where the physics lives:
 *   • bandwidth units (TCP/L2/MALL/HBM) divide cache-line bytes by a per-CU
 *     bytes/cycle rate, with HBM and L2 rates first adjusted for how many CUs
 *     are concurrently active (the contention models from hardware.hpp);
 *   • VMEM divides issued instructions by the issue rate (an issue ceiling,
 *     independent of bandwidth);
 *   • xGMI read is latency-limited (MSHR cap), xGMI write is concentration-
 *     limited — see the two blocks below;
 *   • VALU divides lane-ops by the VALU rate.
 * Output is in cycles because every hardware field is already cycle-native.
 *
 * @param work Per-functional-unit work counts (cache lines, instructions, ops).
 * @param system GPU + fabric hardware description (@see origami::comm::system_t).
 * @param bw_per_wg Per-workgroup share of link bandwidth (bytes/cycle).
 * @param active_cus Number of concurrently active CUs (drives contention scaling).
 * @param heur Tunable heuristic parameters (defaults to DEFAULT_HEURISTICS).
 * @param primitive Optional collective context for the xGMI-write ramp; nullopt
 *        falls back to the default concentration k.
 * @return iter_times_t Per-functional-unit cycle counts for the iteration.
 */
iter_times_t compute_iter_times(const functional_unit_work_t& work,
                                const system_t& system,
                                double bw_per_wg,
                                int active_cus,
                                const heuristics_t& heur             = DEFAULT_HEURISTICS,
                                std::optional<primitive_t> primitive = std::nullopt);

/**
 * @brief Geometry of one workgroup tile for the pipelined-iteration count.
 *
 * Describes the tile compute_wg_tile_latency streams. cachelines and elements
 * are the totals the iteration model consumes directly; the optional shape
 * refines the iteration count for strided tiles (each row is walked separately,
 * so its partial final line cannot merge with the next row). When shape is
 * absent the tile is treated as one flat contiguous byte run.
 */
struct wg_tile_geometry_t {
  std::size_t cachelines;                            ///< Total cache lines in the tile.
  std::size_t elements;                              ///< Total elements in the tile.
  std::optional<tile_shape_t> shape = std::nullopt;  ///< Present -> strided per-row walk.

  /// @brief Build from a tile shape, deriving cachelines/elements (clamped to >= 1).
  /// @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
  static wg_tile_geometry_t from_shape(const tile_shape_t& tile, std::size_t cacheline_bytes) {
    return {std::max<std::size_t>(tile.cachelines(cacheline_bytes), 1),
            std::max<std::size_t>(tile.elements(), 1),
            tile};
  }
};

/**
 * @brief Pipelined iteration count for a WG tile and elements per iteration.
 *
 * How many pipelined iterations a WG tile takes, and how many elements each
 * iteration reduces (the latter only matters for the VALU term). The walk has
 * to respect the tile's memory layout:
 *   contiguous : the tile is one flat byte run, so it is simply chopped into
 *                cl_per_iter-line iterations — ceil(cachelines / cl_per_iter).
 *   strided    : each of the m rows must be walked separately (a row's partial
 *                final line cannot be merged with the next row), so the count
 *                is m × iters_per_row. This is the iteration-level consequence
 *                of the same contiguity penalty tile_shape_t.cachelines models.
 *
 * @param geometry WG tile geometry (totals plus the optional shape that, when
 *        present and strided, drives the per-row walk).
 * @param cl_per_iter Cache lines transferred per pipelined iteration.
 * @param cacheline_bytes Hardware cache-line size (hardware_t::cacheline_bytes).
 * @return std::pair<std::size_t, std::size_t> {num_iters, elements_per_iter};
 *         both are >= 1.
 */
std::pair<std::size_t, std::size_t> iter_counts_from_tile(const wg_tile_geometry_t& geometry,
                                                          std::size_t cl_per_iter,
                                                          std::size_t cacheline_bytes);

/**
 * @brief Loop-invariant context shared across every wg_tile latency query.
 *
 * compute_wg_tile_latency is called once per (timestep, link) while these inputs
 * stay fixed for the whole collective, so they are grouped into one view rather
 * than threaded through individually. Holds non-owning references; the referents
 * must outlive the calls.
 */
struct latency_context_t {
  const comm_config_t& config;  ///< Comm kernel config (load width, WG count, ...).
  const system_t& system;       ///< GPU + fabric hardware.
  const heuristics_t& heur             = DEFAULT_HEURISTICS;  ///< Tunable heuristics.
  std::optional<primitive_t> primitive = std::nullopt;  ///< Collective context for the xGMI ramp.
};

/**
 * @brief Full wg_tile transfer latency for one timestep, in cycles.
 *
 * Composes resolve_work_graph + compute_iter_times into the software-pipelined
 * loop model: T_total = T_prologue + (num_iters - 1) × T_wlt + T_epilogue +
 * T_sync.
 *
 * @param work_graph Ordered communication ops for this rank's timestep.
 * @param geometry WG tile geometry (cache lines, elements, optional shape).
 * @param bw_per_wg Per-workgroup share of link bandwidth (bytes/cycle).
 * @param active_cus Number of concurrently active CUs (drives contention scaling).
 * @param ctx Loop-invariant context (config, system, heuristics, collective).
 * @return wg_tile_latency_breakdown_t Per-stage and per-FU cycle breakdown,
 *         including total cycles and the clock used for any cycle→time step.
 */
wg_tile_latency_breakdown_t compute_wg_tile_latency(const std::vector<op_t>& work_graph,
                                                    const wg_tile_geometry_t& geometry,
                                                    double bw_per_wg,
                                                    int active_cus,
                                                    const latency_context_t& ctx);

}  // namespace origami::comm
