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

namespace {

using h = heuristic_defaults_t;

// Lockstep factor of a multicast cluster's K-loop of `k_iters` iterations with `peers` workgroups
// along its larger axis: measured lambda for 2, 4 and 8 peers, interpolated in log2(peers), the
// 8-peer value beyond.
double cluster_lockstep_factor(const cdna5_constants_t& c5, double peers, double k_iters) {
  if (peers <= 1.0) return 1.0;
  const double x = std::log2(peers);
  double lam     = c5.CLUSTER_LOCKSTEP_8;
  if (x <= 1.0)
    lam = x * c5.CLUSTER_LOCKSTEP_2;
  else if (x <= 2.0)
    lam = c5.CLUSTER_LOCKSTEP_2;  // 2 and 4 peers fit the same value
  else if (x <= 3.0)
    lam = c5.CLUSTER_LOCKSTEP_2 + (x - 2.0) * (c5.CLUSTER_LOCKSTEP_8 - c5.CLUSTER_LOCKSTEP_2);
  return 1.0 + lam * k_iters / (k_iters + c5.CLUSTER_LOCKSTEP_K0);
}

// Issue interval of a BF16 16x16x32 WMMA per SIMD, cycles.  Uses the 1-wave measurement for
// every wave count; the faster 2- and 4-wave rates are not applied.
double wmma_bf16_issue(const cdna5_constants_t& c5, double) { return c5.WMMA_BF16_CLK; }

// How a workgroup's waves issue BF16 16x16x32 WMMAs: on their own SIMDs, at the measured interval
// for the number of its waves sharing a SIMD.
struct wmma_bf16_t {
  bool applies;     // The kernel computes with BF16 16x16x32 WMMAs.
  double n_mi;      // Matrix instructions of one K-iteration of the tile.
  double wg_simds;  // SIMDs the workgroup's waves occupy.
  double wg_wps;    // Waves of the workgroup per SIMD.
};

wmma_bf16_t wmma_bf16(const tile_args_t& args) {
  const config_t& config    = args.config;
  const dim3_t& waves       = args.context.wave_group;
  const double simds_per_cu = static_cast<double>(args.hardware.simds_per_cu());
  const double waves_per_wg = static_cast<double>(waves.m * waves.n * waves.k);
  return {config.mi.m == 16 && config.mi.n == 16 && config.mi.k == 32 &&
              args.problem.mi_dtype == data_type_t::BFloat16,
          static_cast<double>(compute_number_matrix_instructions(config.mt, config.mi)),
          std::min(simds_per_cu, waves_per_wg),
          std::max(1.0, waves_per_wg / simds_per_cu)};
}

// Soft maximum of two times with sharpness s: the larger one, raised by the part of the smaller
// one that does not overlap it.
double soft_max(double x, double y, double s) {
  const double hi = std::max(x, y), lo = std::min(x, y);
  return (hi > 0.0) ? hi * std::pow(1.0 + std::pow(lo / hi, s), 1.0 / s) : 0.0;
}

}  // namespace

// The scale of the read times comes from the refit (L_mem is the max of the L2 and DRAM times).
memory_terms_t cdna5_model::tile_memory(const tile_args_t& args) {
  const cdna5_fit_t& fit = cdna5_model::fit(args.hardware);
  memory_terms_t terms   = memory(args.problem, args.hardware, args.config, args.context);
  const double mem_scale = fit.MEM_SCALE;
  terms.L_mem *= mem_scale;
  terms.l2_cycles *= mem_scale;
  terms.dram_cycles *= mem_scale;
  terms.delivery_floor *= mem_scale;
  return terms;
}

// BF16 WMMA: the workgroup's waves issue on their own SIMDs at the measured interval for the
// number of its waves sharing a SIMD; co-resident workgroups are priced at the end of the tile
// (finish_tile).  Other instructions spread over the SIMDs the resident waves occupy.
void cdna5_model::compute_iteration(const tile_args_t& args, double L_mi, tile_terms_t& t) {
  const cdna5_constants_t& c5 = constants(args.hardware);
  const cdna5_fit_t& fit      = cdna5_model::fit(args.hardware);
  const double simds_per_cu   = static_cast<double>(args.hardware.simds_per_cu());
  const wmma_bf16_t wmma      = wmma_bf16(args);

  t.per_wave_score = 1.0;
  if (wmma.applies)
    t.L_compute = wmma.n_mi / wmma.wg_simds * wmma_bf16_issue(c5, wmma.wg_wps);
  else
    t.L_compute = L_mi / t.per_wave_score * (simds_per_cu / t.simds_busy);
  t.L_compute += t.L_local_read;

  // Refit: scale of the per-K-iteration compute time.
  t.L_compute *= fit.COMPUTE_SCALE;
}

// The part of a load's own transfer through L2 that the global-read prefetch leaves uncovered (no
// load-to-use latency is charged).  A tile's load issues PGR-1 iterations before the tile is used.
// (Tensor-DMA kernels with PGR >= 2 replace L_latency_iter in main_iterations with the stall of
// the kernel's issue/wait recurrence.)
void cdna5_model::load_stall(const tile_args_t& args, tile_terms_t& t) {
  const long pgr     = args.pgr;
  const double L_mem = t.memory.L_mem;

  const double L_iter = (pgr <= 1) ? L_mem + t.L_compute * t.compute_tile_penalty
                                   : std::max(L_mem, t.L_compute * t.compute_tile_penalty);
  t.L_latency_iter =
      (pgr <= 1) ? 0.0 : std::max(0.0, t.memory.l2_cycles - static_cast<double>(pgr - 1) * L_iter);
}

double cdna5_model::main_iterations(const tile_args_t& args, tile_terms_t& t) {
  const cdna5_constants_t& c5 = constants(args.hardware);
  const config_t& config      = args.config;
  const tensile_params_t& tp  = args.tp;
  const long pgr              = args.pgr;
  const double main_iters     = static_cast<double>(t.num_main_iters);
  const double l2_cycles      = t.memory.l2_cycles;
  const double dram_cycles    = t.memory.dram_cycles;
  const double L_mem_floor    = t.memory.delivery_floor;

  const double L_compute_stream_eff = t.L_compute_stream * t.compute_tile_penalty;
  double L_main                     = (pgr <= 1) ? (t.L_mem_stream + L_compute_stream_eff)
                                                 : std::max(t.L_mem_stream, L_compute_stream_eff);
  // The math overlaps the L2 and the DRAM read times as soft maxima, as measured on a K-loop of
  // tensor loads, LDS reads and WMMAs, and multicast delivery bounds the loop from below.
  if (pgr > 1) {
    const double c = L_compute_stream_eff;
    const double d = dram_cycles * main_iters;
    const double l = l2_cycles * main_iters;
    L_main         = std::max({soft_max(c, l, c5.KLOOP_L2_OVERLAP),
                               soft_max(c, d, c5.KLOOP_DRAM_OVERLAP),
                               L_mem_floor * main_iters});
    // Tensor-DMA load stall, from the kernel's issue/wait recurrence.
    // Code fact (KernelWriter.py: "wait for previous set of global reads", tdmWait(PGR - 1), then
    // the workgroup barrier; in the disassembled gfx1250 library kernels the loop is
    // tensor_load_to_lds ... s_wait_tensorcnt 1 ... s_barrier ... ds_load): an iteration issues the
    // load of the tile PGR iterations ahead and then waits only for the load issued PGR-1
    // iterations earlier; the prologue issues PGR loads back to back.  With e[j] the issue time in
    // iteration j, c_it the wave's own time per iteration (math + the iteration's fixed cost) and a
    // load landing its transfer S after its issue,
    //     e[j+1] = max(e[j], e[j-PGR+1] + S) + c_it,
    // so PGR iterations span one load:  T = max(T_nostall, (c_it + S) / PGR).  Per-lane loads and
    // PGR <= 1 keep the charge of load_stall.
    if (tp.tdm_inst != 0) {
      const double depth = static_cast<double>(pgr);
      const double c1    = t.L_compute * t.compute_tile_penalty;
      const double fixed = c5.KITER_FIXED_CLK;
      const double c_it  = c1 + fixed;
      // Iteration time without a latency stall: what L_main and L_loop_overhead charge per
      // iteration.
      const double T_nostall = std::max({soft_max(c1, l2_cycles, c5.KLOOP_L2_OVERLAP),
                                         soft_max(c1, dram_cycles, c5.KLOOP_DRAM_OVERLAP),
                                         L_mem_floor}) +
                               fixed;
      t.L_latency_iter = std::max(T_nostall, (c_it + l2_cycles) / depth) - T_nostall;
    }
  }
  L_main += t.L_latency_iter * main_iters;
  // A multicast cluster's peers advance through the K-loop in lockstep, so its main loop runs
  // slower than its 1x1 twin's by the measured factor for the peers along the cluster's larger
  // axis.
  if (is_multicast_cluster(config))
    L_main *= cluster_lockstep_factor(
        c5,
        static_cast<double>(std::max(config.cluster_dim.m, config.cluster_dim.n)),
        static_cast<double>(t.k_iters));
  return L_main;
}

// The tail loop moves the whole DepthU window of A and B at its measured per-WGP rate.
double cdna5_model::tail_iteration(const tile_args_t& args, const tile_terms_t&) {
  const cdna5_constants_t& c5 = constants(args.hardware);
  return operand_window_bytes(args.config, args.context) / c5.TAIL_WINDOW_B_PER_CLK;
}

// The measured fixed cost of a K-iteration.
double cdna5_model::loop_overhead(const tile_args_t& args, const tile_terms_t& t) {
  const cdna5_constants_t& c5 = constants(args.hardware);
  return c5.KITER_FIXED_CLK * static_cast<double>(t.k_iters);
}

// A multicast load completes only when every peer has issued it, so cluster peers advance in
// lockstep through a cluster-barrier handshake at each in-loop tensor load, plus one before
// the first load.  The handshake runs behind the iteration's loads and MFMAs: only the part
// of the round trip an iteration cannot cover is exposed.
double cdna5_model::cluster_sync(const tile_args_t& args, const tile_terms_t& t) {
  return (is_multicast_cluster(args.config) && t.k_iters > 0)
             ? (static_cast<double>(t.k_iters) + 1.0) *
                   std::max(0.0, h::L_BARRIER - std::max(t.memory.L_mem, t.L_compute))
             : 0.0;
}

double cdna5_model::depth_waste(const tile_args_t&, const tile_terms_t&) { return 0.0; }

// There is no occupancy exposure; the refit moves the base model's saturation S_dev to S_fit,
// applied as the ratio.
double cdna5_model::epilogue_exposure(const hardware_t& hardware, double wgs_per_cu) {
  const cdna5_fit_t& fit = cdna5_model::fit(hardware);
  return wgs_per_cu > 0.0 ? std::max(1.0, fit.EPILOGUE_OCC_SATURATION / wgs_per_cu) /
                                std::max(1.0, fit.EPILOGUE_OCC_SATURATION_DEV / wgs_per_cu)
                          : 1.0;
}

double cdna5_model::finish_tile(const tile_args_t& args, const tile_terms_t& t) {
  const cdna5_constants_t& c5 = constants(args.hardware);
  const cdna5_fit_t& fit      = cdna5_model::fit(args.hardware);
  const problem_t& problem    = args.problem;
  const hardware_t& hardware  = args.hardware;
  const config_t& config      = args.config;
  const context_t& context    = args.context;
  double L_tile_total         = t.L_tile_total;

  // Work outside the tile model that every tile round repeats, growing with the output tile and
  // one K-iteration of A and B (time vs tile rounds of real kernels).
  {
    const double d_kib = static_cast<double>(config.mt.m) * static_cast<double>(config.mt.n) *
                         context.d_bytes / 1024.0;
    const double op_kib = operand_window_bytes(config, context) / 1024.0;
    L_tile_total += c5.TILE_FIXED_CLK_PER_KIB_D * d_kib + c5.TILE_FIXED_CLK_PER_KIB_OP * op_kib;
  }
  // Rows of A and B that are not 4 B aligned (odd bf16 K) take a slow path, measured per KiB of
  // the DepthU window: a part per tile and a part per K-iteration.
  if ((problem.size.k * static_cast<size_t>(context.a_bits)) % 32 != 0) {
    const double win_kib = operand_window_bytes(config, context) / 1024.0;
    L_tile_total +=
        (c5.ODD_K_CLK_PER_KIB + c5.ODD_K_CLK_PER_KIB_ITER * static_cast<double>(t.k_iters)) *
        win_kib;
  }
  // r workgroups resident on a WGP overlap everything but the WGP's shared LDS port (operand
  // reads and tensor-DMA writes at its bandwidth), matrix pipes and store path, so r of them take
  // one tile's time plus r - 1 such busy times x RESIDENT_BUSY_SCALE (measured up to 4 workgroups
  // per WGP; applied up to MAX_RESIDENT_WGS).
  const wmma_bf16_t wmma = wmma_bf16(args);
  if (wmma.applies && t.wgs_per_cu > 1.0 && t.k_iters > 0) {
    const dim3_t& waves = context.wave_group;
    const double r      = std::min(t.wgs_per_cu, fit.MAX_RESIDENT_WGS);
    const double mt_k   = static_cast<double>(config.mt.k);
    const double lds_b  = (static_cast<double>(config.mt.m) * (static_cast<double>(waves.n) + 1.0) +
                          static_cast<double>(config.mt.n) * (static_cast<double>(waves.m) + 1.0)) *
                         mt_k * context.a_bytes;
    const double busy =
        lds_b / static_cast<double>(hardware_t::get_lds_read_bytes_per_clk(hardware.arch)) +
        wmma.n_mi / wmma.wg_simds * wmma_bf16_issue(c5, wmma.wg_wps * r);
    L_tile_total = (L_tile_total + (r - 1.0) * fit.RESIDENT_BUSY_SCALE *
                                       (busy * static_cast<double>(t.k_iters) + t.L_epilogue)) /
                   r;
  }
  return L_tile_total;
}

// ---------------------------------------------------------------------------
// Fixed costs of a kernel
// ---------------------------------------------------------------------------

// The measured cost outside the K-loop that the tile model leaves out (argument and descriptor
// setup, first load, store drain), which grows with the output tile and with one K-iteration of
// A and B.
double cdna5_model::kernel_fixed_cost(const hardware_t& hardware,
                                      const config_t& config,
                                      const context_t& context) {
  const cdna5_constants_t& c5 = constants(hardware);
  const double d_kib  = static_cast<double>(config.mt.m * config.mt.n) * context.d_bytes / 1024.0;
  const double op_kib = operand_window_bytes(config, context) / 1024.0;
  const double occ    = static_cast<double>(std::max(config.occupancy, 1));
  return c5.KERNEL_FIXED_CLK + c5.KERNEL_FIXED_CLK_PER_KIB_D * d_kib +
         c5.KERNEL_FIXED_CLK_PER_KIB_OP * op_kib +
         c5.KERNEL_FIXED_CLK_PER_LOG2_OCC * std::log2(occ);
}

}  // namespace gemm
}  // namespace origami
