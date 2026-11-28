// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// The bandwidth quadratic is in WGPs of the part it was fit on; other parts reach the same
// fraction of their peak at the same fraction of their CUs.
double cdna5_model::occupancy_curve_cus(const hardware_t& hardware, size_t num_active_cus) {
  double CUs = static_cast<double>(num_active_cus);
  if (hardware.N_CU > 0)
    CUs = CUs * constants(hardware).BW_FIT_WGPS / static_cast<double>(hardware.N_CU);
  return CUs;
}

// A/B memory latency per K-iteration, in four steps:
//   1. L2 hit rates of A and B (no MALL) from the tiles the launch's first wave places on one
//      L2 domain (the XCDs of an AID), against the domain's capacity.
//   2. L2 and DRAM read times: L2 bytes in 128 B lines per K-iteration, DRAM bytes in 256 B
//      blocks, at the measured tensor-DMA (or per-lane) bandwidths.
//   3. Multicast clusters cut the L2 request bytes.  Every peer still receives its whole tile,
//      so delivery_floor bounds the time from below at the measured multicast delivery rates.
//   4. TemporalHint: a low-priority (NT, LU) operand does not count against a regular one's L2
//      residency (estimate_l2_hit_cdna5), and its narrow tensor-DMA rows refetch the rest of the
//      DRAM block (below).
// l2_cycles is the L2 read time and dram_cycles the DRAM read time alone.
memory_terms_t cdna5_model::memory(const problem_t& problem,
                                   const hardware_t& hardware,
                                   const config_t& config,
                                   const context_t& context) {
  using h                     = heuristic_defaults_t;
  const cdna5_constants_t& c5 = cdna5_model::constants(hardware);
  const size_t grid_m         = std::max<size_t>(context.grid_m, 1);
  const size_t grid_n         = std::max<size_t>(context.grid_n, 1);
  const size_t cs             = std::max<size_t>(config.cluster_dim.m, 1);
  const size_t ck             = std::max<size_t>(config.cluster_dim.n, 1);
  const bool multicast = cs * ck > 1 && (!config.has_tensile_params() || tparams(config).multicast);
  const bool tdm       = tparams(config).tdm_inst != 0;

  // 1. L2 hit rates.  Workgroup i runs on XCD dispatch_xcd(i) and the launch's WGM
  //    (context.wgm, from predict_workgroup_mapping) orders the ids into slabs.  WGMXCC is not
  //    modeled: persistent kernels reread their rank from the hardware id, so it has no effect
  //    there, but non-persistent kernels apply it and are priced as if it were 1.  A cluster c
  //    is dispatched whole onto XCD dispatch_xcd(c), two shader engines per XCD, bypassing WGM.
  //    A load hits when another tile on the domain already brought that A row / B column (same
  //    K split and batch); estimate_l2_hit_cdna5 estimates how many distinct ones there are.
  const operand_traffic_t& traffic = context.traffic;
  const double l2_cap              = static_cast<double>(hardware.l2_domain_capacity());
  double load                      = 0.0;
  const cache_hit_rates_t hits = estimate_l2_hit_cdna5(problem, hardware, config, context, &load);
  double H_a                   = std::get<0>(hits);
  double H_b                   = std::get<1>(hits);

  // 2. Read times.  Per WG and K-iteration: DRAM bytes in 256 B blocks over the WG's K range
  //    (context traffic), L2 bytes in 128 B lines per K-iteration; cross-tile line sharing for
  //    small M / N at each granule.
  const bool a_trans   = context.a_transposed;
  const bool b_trans   = context.b_transposed;
  const double mt_m    = static_cast<double>(std::min<size_t>(problem.size.m, config.mt.m));
  const double mt_n    = static_cast<double>(std::min<size_t>(problem.size.n, config.mt.n));
  const double mt_k    = static_cast<double>(std::max<size_t>(config.mt.k, 1));
  const double a_rows  = a_trans ? mt_m : mt_k;
  const double a_row   = (a_trans ? mt_k : mt_m) * context.a_bytes;
  const double b_rows  = b_trans ? mt_k : mt_n;
  const double b_row   = (b_trans ? mt_n : mt_k) * context.b_bytes;
  const double line    = static_cast<double>(h::CACHE_LINE_BYTES);
  const double a_l2_wg = a_rows * std::ceil(a_row / line) * line;
  const double b_l2_wg = b_rows * std::ceil(b_row / line) * line;
  const double cus     = static_cast<double>(context.active_cus);
  double Ld_a_l2       = cus * a_l2_wg * std::min(1.0, problem.size.m * context.a_bytes / 128.0);
  double Ld_b_l2       = cus * b_l2_wg * std::min(1.0, problem.size.n * context.b_bytes / 128.0);
  double Ld_dram       = cus * ((1.0 - H_a) * traffic.a_iter_bytes * traffic.a_cl_share +
                          (1.0 - H_b) * traffic.b_iter_bytes * traffic.b_cl_share);
  // A low-priority (NT, LU) tensor-DMA operand: a row narrower than the DRAM block leaves the rest
  // of the block for the next K-iterations, which L2 keeps only while little other traffic arrives.
  const bool a_low = l2_low_priority_load(load_temporal_hint(config, false));
  const bool b_low = l2_low_priority_load(load_temporal_hint(config, true));
  if (tdm && (tparams(config).tdm_inst & 3) == 3 && (a_low || b_low) && cus > 0.0) {
    const double block = static_cast<double>(c5.DRAM_BLOCK_BYTES);
    const double D1    = cus * (a_l2_wg + b_l2_wg) / static_cast<double>(1 << 20);
    auto refetch       = [&](double row, bool low) {
      double f = 1.0;
      for (int j = 1; low && row > 0.0 && j < static_cast<int>(block / row); ++j)
        f += 1.0 / (1.0 + std::pow(c5.TDM_NT_REFETCH_D0_MIB / (j * D1), c5.TDM_NT_REFETCH_N));
      return f;
    };
    Ld_dram =
        cus * ((1.0 - H_a) * traffic.a_iter_bytes * traffic.a_cl_share * refetch(a_row, a_low) +
               (1.0 - H_b) * traffic.b_iter_bytes * traffic.b_cl_share * refetch(b_row, b_low));
  }

  // Tensor DMA, measured with tensor loads on gfx1250: a row holds a WGP's DMA path for
  // max(row bytes, the level's minimum) / its row rate, and the WGPs' rate meets the level's
  // peak in a hard minimum.  Per-lane loads: L2 scales with the active CUs, DRAM with the
  // occupancy quadratic.
  // Everything below is in engine cycles and bytes per engine cycle (B/clk), the units of
  // hardware.mem1_perf_ratio / mem3_perf_ratio.
  double l2_bw       = hardware.mem1_perf_ratio * cus / static_cast<double>(hardware.N_CU);
  double dram_bw     = hardware.mem3_perf_ratio * context.mem_bw_limited;
  const double block = static_cast<double>(c5.DRAM_BLOCK_BYTES);
  if (tdm && cus > 0.0) {
    const double l2_row_clk = (a_rows * std::max(a_row, block) + b_rows * std::max(b_row, block)) /
                              c5.TDM_L2_ROW_B_PER_CLK;
    const double dram_row_clk =
        (a_rows * std::max(a_row, block) + b_rows * std::max(b_row, block)) /
        c5.TDM_DRAM_ROW_B_PER_CLK;
    l2_bw   = std::min(cus * (a_l2_wg + b_l2_wg) / l2_row_clk, hardware.mem1_perf_ratio);
    dram_bw = std::min(cus * (traffic.a_iter_bytes + traffic.b_iter_bytes) / dram_row_clk,
                       hardware.mem3_perf_ratio);
  }

  // 3. Multicast: a cluster requests each A tile once per cluster row and each B tile once per
  //    cluster column.  The requests set the L2 pressure the K-loop overlaps with its MFMAs, but
  //    the peers wait on each other's loads: delivery into LDS gains only
  //    MULTICAST_L2_GAIN from L2 and loses MULTICAST_DRAM_GAIN from DRAM.
  const double Ld_delivered = Ld_a_l2 + Ld_b_l2;
  if (multicast) {
    Ld_a_l2 *= static_cast<double>(math::safe_ceil_div(grid_n, ck)) / static_cast<double>(grid_n);
    Ld_b_l2 *= static_cast<double>(math::safe_ceil_div(grid_m, cs)) / static_cast<double>(grid_m);
  }
  const double Ld_l2      = Ld_a_l2 + Ld_b_l2;
  double L_mem_l2         = (l2_bw > 0.0) ? Ld_l2 / l2_bw : 0.0;
  const double L_mem_dram = (dram_bw > 0.0) ? Ld_dram / dram_bw : 0.0;
  // A tensor-DMA peer fills its whole tile at the WGP row rate; only the requests meet the peak.
  if (multicast && tdm && cus > 0.0 && Ld_l2 > 0.0) {
    const double l2_row_clk = (a_rows * std::max(a_row, block) + b_rows * std::max(b_row, block)) /
                              c5.TDM_L2_ROW_B_PER_CLK;
    L_mem_l2 = Ld_delivered / std::min(cus * (a_l2_wg + b_l2_wg) / l2_row_clk,
                                       hardware.mem1_perf_ratio * Ld_delivered / Ld_l2);
  }

  memory_terms_t terms;
  terms.L_mem       = std::max(L_mem_l2, L_mem_dram);
  terms.l2_cycles   = L_mem_l2;
  terms.dram_cycles = L_mem_dram;
  if (multicast && Ld_l2 > 0.0 && l2_bw > 0.0)
    terms.delivery_floor =
        std::max(Ld_delivered / l2_bw / std::min(Ld_delivered / Ld_l2, c5.MULTICAST_L2_GAIN),
                 L_mem_dram / c5.MULTICAST_DRAM_GAIN);

  if (context.debug) {
    OLOG_DEBUG("L2HitA: " << H_a);
    OLOG_DEBUG("L2HitB: " << H_b);
    OLOG_DEBUG("Ld_l2: " << Ld_l2);
    OLOG_DEBUG("Ld_dram: " << Ld_dram);
    OLOG_DEBUG("L2BW: " << l2_bw);
    OLOG_DEBUG("DramBW: " << dram_bw);
    OLOG_DEBUG("L_mem_l2: " << L_mem_l2);
    OLOG_DEBUG("L_mem_dram: " << L_mem_dram);
  }
  return terms;
}

}  // namespace gemm
}  // namespace origami
