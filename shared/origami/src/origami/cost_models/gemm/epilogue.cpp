// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

double compute_epilogue_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context,
                                double* scalar_store_fraction) {
  const epilogue_terms_t terms = for_family(
      hardware, [&](auto family) { return family.epilogue(problem, hardware, config, context); });
  if (scalar_store_fraction != nullptr) *scalar_store_fraction = terms.scalar_store_fraction;
  return terms.L_epilogue;
}

epilogue_terms_t base_model::epilogue(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config,
                                      const context_t& context) {
  // In epilogue:
  // 1. ACC -> VGPR
  // 2. Alpha/beta scaling
  // 3. Bias operations
  // 4. Activation functions
  // 5. Accumulator conversions
  // 6. Global memory stores

  // Items 2, 3, 4 are conditionally executed based on the problem.
  // For instance, if Beta=0, we skip a bunch of operations.
  // We skip bias and activation functions if they are not present.
  // Herein, we consider the simplest case for now: Alpha=1, Beta=0, and no bias/activation
  // functions. Skipping items 2, 3, 4, and 5 for now.

  // Extract parameters
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;

  const size_t N_CU = context.n_cu;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  const size_t num_active_cus          = context.active_cus;
  const size_t splitting_factor        = context.splitting_factor;
  const double d_bytes                 = context.d_bytes;
  const size_t grid_m                  = context.grid_m;
  const size_t grid_n                  = context.grid_n;
  const size_t num_output_tiles        = context.num_output_tiles;
  const bool debug                     = context.debug;
  const reduction_t reduction_strategy = context.reduction_strategy;
  const bool is_parallel_reduction     = (reduction_strategy == reduction_t::parallel);

  if (d_bytes == 0.0) return {};

  const size_t WAVEFRONT_SIZE                = context.wavefront_size;
  constexpr size_t STORE_PATTERN_IDEAL       = 0;
  constexpr size_t STORE_PATTERN_NARROW      = 1;
  constexpr size_t STORE_PATTERN_WIDE_SPLIT  = 2;
  constexpr size_t STORE_PATTERN_SCALAR_EDGE = 3;

  const size_t wave_group_m_epi       = context.wave_group.m;
  const size_t wave_group_n_epi       = context.wave_group.n;
  const size_t wave_num_epi           = wave_group_m_epi * wave_group_n_epi;
  const size_t wave_issue_parallelism = std::min(wave_num_epi, hardware.simds_per_cu());
  const double wave_batches =
      std::ceil(static_cast<double>(wave_num_epi) / static_cast<double>(wave_issue_parallelism));

  // ACC->VGPR drain: keep this as a per-wave cost because waves can issue on
  // different SIMDs. Store issue below uses max(per-wave work, total work/SIMDs).
  const double acc_elems_per_thread =
      static_cast<double>(MT_M * MT_N) / (WAVEFRONT_SIZE * wave_num_epi);
  const double mi_reg_per_out = std::max(1.0, data_type_to_bytes(problem.mi_dtype) / 4.0);
  const double reads_per_wave = acc_elems_per_thread * mi_reg_per_out;
  // ACC reads and the bounds checks below are vector-ALU instructions.  Without
  // separate accumulation registers the results are already in VGPRs.
  const double valu_cycles = static_cast<double>(context.valu_cycles);
  const double L_acc_transfer =
      hardware.has_accumulation_registers()
          ? reads_per_wave * heuristic_defaults_t::EPILOGUE_VALU_INSTS_PER_ACC_READ * valu_cycles
          : 0.0;

  const size_t max_isa_store_elems = static_cast<size_t>(
      std::ceil(heuristic_defaults_t::EPILOGUE_BYTES_PER_VECTORIZED_STORE / d_bytes));
  const size_t requested_svw =
      (config.gwvw_d > 0) ? static_cast<size_t>(config.gwvw_d) : max_isa_store_elems;
  const size_t sector_bytes = heuristic_defaults_t::CACHE_LINE_BYTES;

  // TensileLite stores gwvw_d contiguous M elements per lane in both SourceSwap
  // modes, so a store only splits when it is wider than one buffer_store.
  const size_t natural_svw = max_isa_store_elems;

  // Device write bandwidth: mem3 x the occupancy quadratic, or on gfx1250 the epilogue-store
  // curve measured with plain stores: a wave's store instruction of B bytes takes a fixed time
  // plus B at the wave's rate, a WGP's waves share the WGP's rate, and the device meets its
  // store ceiling in a soft minimum.  The rates scale with mem3 as fractions of the measured
  // DRAM peak.  The K-split reduction reads its partials at the read bandwidth and writes D at
  // the store one.
  double store_bw        = hardware.mem3_perf_ratio * context.mem_bw_limited;
  const double reduce_bw = hardware.mem3_perf_ratio * context.write_mem_bw_limited;
  double reduce_write_bw = reduce_bw;

  // Per-CU write bandwidth: total write BW shared among all writers.
  const size_t num_writers = std::max(num_active_cus, static_cast<size_t>(1));
  const double per_cu_store_bw =
      (store_bw > 0.0) ? store_bw / static_cast<double>(num_writers) : 0.0;

  struct wave_store_plan_t {
    double active_elements = 0.0;
    double store_insts     = 0.0;
    double issue_insts     = 0.0;
    double sectors         = 0.0;
    double useful_bytes    = 0.0;
    double sector_bytes    = 0.0;
    size_t pattern         = STORE_PATTERN_IDEAL;
  };

  struct tile_epilogue_plan_t {
    // Per-wave store metrics summed over all waves of the tile (store_insts,
    // issue_insts, sectors, useful_bytes, sector_bytes, pattern).
    wave_store_plan_t store{};
    double acc_read          = 0.0;
    double bounds            = 0.0;
    double store_issue       = 0.0;
    double store_memory      = 0.0;
    double reduction         = 0.0;
    double sector_efficiency = 1.0;
    double active_waves      = 0.0;
    double max_wave_issue    = 0.0;

    double total() const { return acc_read + bounds + store_issue + store_memory + reduction; }
  };

  auto wave_range = [](size_t extent, size_t parts, size_t idx) {
    const size_t begin = (extent * idx) / parts;
    const size_t end   = (extent * (idx + 1)) / parts;
    return std::pair<size_t, size_t>{begin, std::max(begin, end)};
  };

  auto active_extent = [](std::pair<size_t, size_t> r, size_t valid) {
    if (r.first >= valid) return static_cast<size_t>(0);
    return std::min(r.second, valid) - r.first;
  };

  auto compute_wave_store_plan = [&](size_t active_m,
                                     size_t active_n,
                                     bool scalar_path,
                                     double store_elem_bytes) -> wave_store_plan_t {
    wave_store_plan_t plan{};
    const size_t active_elements = active_m * active_n;
    if (active_elements == 0) return plan;

    const size_t axis_natural = std::min(natural_svw, active_m);
    const size_t logical_svw  = scalar_path ? static_cast<size_t>(1) : requested_svw;
    const size_t contiguous_svw =
        scalar_path ? static_cast<size_t>(1) : std::min(logical_svw, axis_natural);
    const size_t split_count =
        scalar_path ? static_cast<size_t>(1) : math::safe_ceil_div(logical_svw, contiguous_svw);

    if (scalar_path) {
      plan.pattern = STORE_PATTERN_SCALAR_EDGE;
    } else if (logical_svw < axis_natural) {
      plan.pattern = STORE_PATTERN_NARROW;
    } else if (logical_svw > axis_natural) {
      plan.pattern = STORE_PATTERN_WIDE_SPLIT;
    } else {
      plan.pattern = STORE_PATTERN_IDEAL;
    }

    const double logical_groups =
        std::ceil(static_cast<double>(active_elements) /
                  (static_cast<double>(WAVEFRONT_SIZE) * static_cast<double>(logical_svw)));
    plan.store_insts = logical_groups * static_cast<double>(split_count);

    // Split stores pay extra address/exec manipulation per sub-store.
    double address_issue = 0.0;
    if (plan.pattern == STORE_PATTERN_WIDE_SPLIT) address_issue = plan.store_insts;
    plan.issue_insts = plan.store_insts + address_issue;

    plan.useful_bytes = static_cast<double>(active_elements) * store_elem_bytes;

    // Count memory sectors per logical store group.  Ideal/narrow paths coalesce
    // the group payload; wide-split paths touch one sector group per contiguous
    // sub-run.
    const double useful_per_group  = plan.useful_bytes / logical_groups;
    const double subrun_bytes      = useful_per_group / static_cast<double>(split_count);
    const double sectors_per_group = static_cast<double>(split_count) *
                                     std::ceil(subrun_bytes / static_cast<double>(sector_bytes));
    plan.sectors = logical_groups * sectors_per_group;

    if (!scalar_path && !tparams(config).source_swap)
      plan.sectors *= heuristic_defaults_t::EPILOGUE_NONSWAP_STORE_TRAFFIC;

    plan.sector_bytes    = plan.sectors * static_cast<double>(sector_bytes);
    plan.active_elements = static_cast<double>(active_elements);
    return plan;
  };

  // Edge tile detection
  const bool has_interior  = (M >= MT_M && N >= MT_N);
  const bool has_m_edge    = (M % MT_M != 0);
  const bool has_n_edge    = (N % MT_N != 0);
  const size_t m_remainder = has_m_edge ? (M % MT_M) : MT_M;
  const size_t n_remainder = has_n_edge ? (N % MT_N) : MT_N;

  auto compute_tile_epilogue =
      [&](size_t tile_m, size_t tile_n, bool scalar_path) -> tile_epilogue_plan_t {
    tile_epilogue_plan_t tile{};
    tile.acc_read        = L_acc_transfer;
    const bool edge_path = scalar_path || tile_m != MT_M || tile_n != MT_N;
    // Split-K main kernels write an f32 partial to workspace (not the final
    // output), for BOTH the in-kernel tree reduction and the separate parallel
    // (PostGSU) reduction.
    const double store_elem_bytes =
        (splitting_factor > 1)
            ? static_cast<double>(heuristic_defaults_t::EPILOGUE_WORKSPACE_BYTES_PER_ELEM)
            : d_bytes;

    double max_wave_store_insts  = 0.0;
    double max_wave_sector_bytes = 0.0;
    for (size_t wg_m = 0; wg_m < wave_group_m_epi; ++wg_m) {
      const auto m_range    = wave_range(MT_M, wave_group_m_epi, wg_m);
      const size_t active_m = active_extent(m_range, tile_m);
      for (size_t wg_n = 0; wg_n < wave_group_n_epi; ++wg_n) {
        const auto n_range    = wave_range(MT_N, wave_group_n_epi, wg_n);
        const size_t active_n = active_extent(n_range, tile_n);
        wave_store_plan_t wave =
            compute_wave_store_plan(active_m, active_n, scalar_path, store_elem_bytes);
        if (wave.active_elements == 0.0) continue;

        tile.active_waves += 1.0;
        tile.store.store_insts += wave.store_insts;
        tile.store.issue_insts += wave.issue_insts;
        tile.store.sectors += wave.sectors;
        tile.store.useful_bytes += wave.useful_bytes;
        tile.store.sector_bytes += wave.sector_bytes;
        tile.max_wave_issue   = std::max(tile.max_wave_issue, wave.issue_insts);
        max_wave_store_insts  = std::max(max_wave_store_insts, wave.store_insts);
        max_wave_sector_bytes = std::max(max_wave_sector_bytes, wave.sector_bytes);
        tile.store.pattern    = std::max(tile.store.pattern, wave.pattern);
      }
    }

    // Critical path over SIMD-issue lanes: max of the serialized-batch bound
    // (wave_batches x max_wave, since waves run in ceil(wave_num/simds) rounds)
    // and the throughput bound (total / wave_issue_parallelism).
    auto critical_path = [&](double max_wave, double total) {
      return std::max(max_wave * wave_batches, total / static_cast<double>(wave_issue_parallelism));
    };

    tile.bounds                       = edge_path
                                            ? heuristic_defaults_t::EPILOGUE_VALU_INSTS_PER_BOUNDS_CHECK * valu_cycles *
                            critical_path(max_wave_store_insts, tile.store.store_insts)
                                            : 0.0;
    const double store_issue_parallel = critical_path(tile.max_wave_issue, tile.store.issue_insts);
    tile.store_issue                  = store_issue_parallel;
    const double store_memory_bytes = critical_path(max_wave_sector_bytes, tile.store.sector_bytes);
    tile.store_memory = (per_cu_store_bw > 0.0) ? store_memory_bytes / per_cu_store_bw : 0.0;
    tile.sector_efficiency =
        (tile.store.sector_bytes > 0.0) ? tile.store.useful_bytes / tile.store.sector_bytes : 1.0;

    // Per-tile K-split reduction (in-kernel: spinlock/tree/atomic).  This still
    // uses byte movement plus fixed sync terms because the reduction path is
    // serialized by inter-WG handoff rather than by D-store coalescing alone.
    if (splitting_factor > 1 && !is_parallel_reduction) {
      size_t n_partials       = splitting_factor - 1;
      double per_cu_reduce_bw = reduce_bw / static_cast<double>(num_output_tiles);
      double partial_bytes    = static_cast<double>(n_partials) * tile_m * tile_n *
                             heuristic_defaults_t::EPILOGUE_WORKSPACE_BYTES_PER_ELEM;
      double store_bytes = static_cast<double>(tile_m) * tile_n * store_elem_bytes;

      double L_poll_wait = store_bytes / per_cu_store_bw;
      double L_sync      = L_poll_wait + static_cast<double>(n_partials) *
                                        (heuristic_defaults_t::EPILOGUE_SALU_OVERHEAD +
                                         2.0 * heuristic_defaults_t::L_BARRIER +
                                         heuristic_defaults_t::EPILOGUE_L_SMEM);

      double L_partial_read  = partial_bytes / per_cu_reduce_bw;
      double L_accumulate    = static_cast<double>(n_partials * tile_m * tile_n) / WAVEFRONT_SIZE;
      double L_partial_write = static_cast<double>(tile_m) * tile_n * d_bytes /
                               (reduce_write_bw / static_cast<double>(num_output_tiles));
      tile.reduction = L_sync + L_partial_read + L_accumulate + L_partial_write;
    }

    return tile;
  };

  // Evaluate all tile types
  tile_epilogue_plan_t epilogue_interior;
  tile_epilogue_plan_t epilogue_n_edge;
  tile_epilogue_plan_t epilogue_m_edge;
  tile_epilogue_plan_t epilogue_corner;
  if (has_interior) epilogue_interior = compute_tile_epilogue(MT_M, MT_N, false);
  if (has_n_edge) epilogue_n_edge = compute_tile_epilogue(MT_M, n_remainder, false);
  if (has_m_edge) epilogue_m_edge = compute_tile_epilogue(m_remainder, MT_N, true);
  if (has_m_edge && has_n_edge)
    epilogue_corner = compute_tile_epilogue(m_remainder, n_remainder, true);

  // Aggregate per-tile-type costs into a representative per-tile epilogue.  At
  // most one M-row and one N-column are partial (edge/scalar) tiles; their weight
  // depends on tile count.  Few tiles (<= N_CU) run concurrently so the slowest
  // sets wall-clock (max); many tiles pipeline, so weight by count-fraction
  // (1/grid per edge direction) so a lone scalar edge can't dominate a big grid.
  const double f_m_edge   = has_m_edge ? 1.0 / static_cast<double>(grid_m) : 0.0;
  const double f_n_edge   = has_n_edge ? 1.0 / static_cast<double>(grid_n) : 0.0;
  const double w_interior = (1.0 - f_m_edge) * (1.0 - f_n_edge);
  const double w_m_edge   = f_m_edge * (1.0 - f_n_edge);
  const double w_n_edge   = (1.0 - f_m_edge) * f_n_edge;
  const double w_corner   = f_m_edge * f_n_edge;
  const double weighted_epilogue =
      w_interior * epilogue_interior.total() + w_m_edge * epilogue_m_edge.total() +
      w_n_edge * epilogue_n_edge.total() + w_corner * epilogue_corner.total();
  const double max_epilogue = std::max({epilogue_interior.total(),
                                        epilogue_n_edge.total(),
                                        epilogue_m_edge.total(),
                                        epilogue_corner.total()});

  const bool few_tiles = (num_output_tiles <= N_CU);
  double L_epilogue    = few_tiles ? max_epilogue : weighted_epilogue;

  // selected_epilogue is only used for the debug term breakdown below.
  tile_epilogue_plan_t selected_epilogue = epilogue_interior;
  if (few_tiles) {
    for (const auto& c : {epilogue_interior, epilogue_m_edge, epilogue_n_edge, epilogue_corner})
      if (c.total() == max_epilogue) selected_epilogue = c;
  } else {
    double best_w                                                 = -1.0;
    const std::pair<const tile_epilogue_plan_t*, double> cands[4] = {
        {&epilogue_interior, w_interior},
        {&epilogue_m_edge, w_m_edge},
        {&epilogue_n_edge, w_n_edge},
        {&epilogue_corner, w_corner}};
    for (const auto& c : cands)
      if (c.second > best_w) {
        best_w            = c.second;
        selected_epilogue = *c.first;
      }
  }

  // Fraction of the representative epilogue that is scalar-edge store work: a
  // serialized per-element predicated loop exposed only when store-bound.
  // Reported so the tile model can gate its amplification on store_exposure.
  auto store_terms = [](const tile_epilogue_plan_t& p) {
    return p.bounds + p.store_issue + p.store_memory;
  };
  double scalar_store_cost = 0.0;
  if (few_tiles) {
    if (selected_epilogue.store.pattern == STORE_PATTERN_SCALAR_EDGE)
      scalar_store_cost = store_terms(selected_epilogue);
  } else {
    scalar_store_cost =
        w_m_edge * store_terms(epilogue_m_edge) + w_corner * store_terms(epilogue_corner);
  }
  const double scalar_store_fraction =
      (L_epilogue > 0.0) ? std::clamp(scalar_store_cost / L_epilogue, 0.0, 1.0) : 0.0;

  if (debug) {
    OLOG_DEBUG("epi_wave_num: " << int(wave_num_epi));
    OLOG_DEBUG("epi_acc_elems_per_thread: " << acc_elems_per_thread);
    OLOG_DEBUG("epi_mi_reg_per_out: " << mi_reg_per_out);
    OLOG_DEBUG("epi_reads_per_wave: " << reads_per_wave);
    OLOG_DEBUG("L_epilogue_interior: " << epilogue_interior.total());
    OLOG_DEBUG("L_epilogue_n_edge: " << epilogue_n_edge.total());
    OLOG_DEBUG("L_epilogue_m_edge: " << epilogue_m_edge.total());
    OLOG_DEBUG("L_epilogue_corner: " << epilogue_corner.total());
    OLOG_DEBUG("epi_requested_svw: " << requested_svw);
    OLOG_DEBUG("epi_natural_svw: " << natural_svw);
    OLOG_DEBUG("epi_max_isa_svw: " << max_isa_store_elems);
    OLOG_DEBUG("epi_store_svw_meta: " << config.gwvw_d);
    OLOG_DEBUG("epi_wave_issue_parallelism: " << wave_issue_parallelism);
    OLOG_DEBUG("epi_wave_batches: " << wave_batches);
    OLOG_DEBUG("epi_store_pattern: " << selected_epilogue.store.pattern);
    OLOG_DEBUG("epi_store_insts: " << selected_epilogue.store.store_insts);
    OLOG_DEBUG("epi_store_issue_insts: " << selected_epilogue.store.issue_insts);
    OLOG_DEBUG("epi_store_sectors: " << selected_epilogue.store.sectors);
    OLOG_DEBUG("epi_store_useful_bytes: " << selected_epilogue.store.useful_bytes);
    OLOG_DEBUG("epi_store_sector_bytes: " << selected_epilogue.store.sector_bytes);
    OLOG_DEBUG("epi_store_sector_efficiency: " << selected_epilogue.sector_efficiency);
    OLOG_DEBUG("epi_active_waves: " << selected_epilogue.active_waves);
    OLOG_DEBUG("epi_L_acc_read: " << selected_epilogue.acc_read);
    OLOG_DEBUG("epi_L_bounds: " << selected_epilogue.bounds);
    OLOG_DEBUG("epi_L_store_issue: " << selected_epilogue.store_issue);
    OLOG_DEBUG("epi_L_store_memory: " << selected_epilogue.store_memory);
    OLOG_DEBUG("epi_L_reduce: " << selected_epilogue.reduction);
  }

  return {L_epilogue, scalar_store_fraction};
}

}  // namespace gemm
}  // namespace origami
