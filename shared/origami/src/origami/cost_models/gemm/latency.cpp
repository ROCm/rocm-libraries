// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/simulator/tensilelite/formocast_simulator.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

namespace {

// Non-temporal A/B loads pay off only on a skinny operand whose K range covers whole cache
// lines.  There the non-temporal variant of a tile is required when the pool has one, and
// everywhere else a non-temporal config is turned away.
rejection_t admit_non_temporal(const problem_t& problem, const config_t& config) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const bool a_trans = problem.a_transpose == transpose_t::T;
  const bool b_trans = problem.b_transpose == transpose_t::T;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const int a_bits = datatype_to_bits(problem.a_dtype);
  const int b_bits = datatype_to_bits(problem.b_dtype);

  constexpr size_t cache_line_bits = heuristic_defaults_t::CACHE_LINE_BYTES * 8;  // 128 B in bits
  const size_t K_mod_128bytes      = K * a_bits % cache_line_bits;
  const size_t MT_K_mod_128bytes   = MT_K * a_bits % cache_line_bits;
  if (K_mod_128bytes == 0 && MT_K_mod_128bytes == 0) {
    // avoid division by 0 if K == 0
    if (M <= MT_M * 2 && !b_trans && ((N * b_bits) / (M * a_bits) > 5)) {
      // Use nontemporal B, if the library has one to use
      if (config.non_temporal_b_available && !(config.cache_hints_b == 4))
        return rejection_t::non_temporal_b_required;
    } else if (N <= MT_N * 2 && a_trans && ((M * a_bits) / (N * b_bits) > 5)) {
      // Use Non Temporal A, if the library has one to use
      if (config.non_temporal_a_available && !(config.cache_hints_a == 4))
        return rejection_t::non_temporal_a_required;
    } else {
      // Never use Non Temporal
      if (config.cache_hints_a || config.cache_hints_b)
        return rejection_t::non_temporal_not_allowed;
    }
  } else if (config.cache_hints_a || config.cache_hints_b) {
    return rejection_t::non_temporal_misaligned;
  }
  return rejection_t::none;
}

// W4A16 decode kernels: dot2 (no MI) WaveSplitK kernels, one wave per workgroup that
// owns MT_M weight rows x MT_N tokens and splits K over its lanes. They stream the
// weights once per token tile, so they are bound by DRAM bandwidth or by the rounds of
// resident waves, whichever is longer. The tile model
// (MI tiles, LDS, workgroup waves) does not describe them and rejects them.
bool is_w4a16_decode(const problem_t& problem, const hardware_t& hardware, const config_t& config) {
  return is_w4a16(problem, hardware) && w4a16_decode_dram_bytes_per_us(hardware).has_value() &&
         config.mi.m == 1 && config.mi.n == 1 && config.mi.k == 64;
}

// The admission rules.  @p heuristic receives the config's heuristic parameters once the config
// is known to be valid, so the caller can reuse them.
rejection_t admit(const problem_t& problem,
                  const hardware_t& hardware,
                  const config_t& config,
                  heuristic_params_t& heuristic) {
  if (!config.is_valid()) return rejection_t::invalid_config;

  // Heuristic-driven kernel rejection (e.g. subtile kernels with small K).
  heuristic = get_heuristic_params(problem, hardware, config);
  if (heuristic.reject) return rejection_t::heuristic;

  // ORIGAMI_GEMM_PICK: force a specific MT size for solution selection.
  const auto& pick = runtime_options::get().gemm_pick;
  if (pick.m > 0 && (config.mt.m != pick.m || config.mt.n != pick.n || config.mt.k != pick.k))
    return rejection_t::forced_pick;

  // The rules below shortcut the analytical model; the simulator scores every remaining config.
  if (config.prediction_mode == prediction_modes_t::simulation) return rejection_t::none;

  // W4A16 decode kernels have their own model (and none of the tile model's rules).
  if (is_w4a16_decode(problem, hardware, config)) return rejection_t::none;

  // Use Dot2 only for M < 3
  if (config.mi.m == 1 && config.mi.n == 1 && config.mi.k == 64 && problem.size.m > 2)
    return rejection_t::dot2_large_m;

  const bool non_temporal_rules =
      for_family(hardware, [](auto family) { return family.non_temporal_admission; });
  return non_temporal_rules ? admit_non_temporal(problem, config) : rejection_t::none;
}

}  // namespace

rejection_t admit(const problem_t& problem, const hardware_t& hardware, const config_t& config) {
  heuristic_params_t heuristic;
  return admit(problem, hardware, config, heuristic);
}

// ---------------------------------------------------------------------------
// Simulation method
// ---------------------------------------------------------------------------

double compute_simulated_latency(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config) {
  // Create Formocast simulator instance
  Formocast formocast;

  // Convert problem_t to Formocast::ProblemInfo
  Formocast::ProblemInfo prob_info;
  prob_info.M              = static_cast<double>(problem.size.m);
  prob_info.N              = static_cast<double>(problem.size.n);
  prob_info.K              = static_cast<double>(problem.size.k);
  prob_info.NumBatches     = static_cast<double>(problem.batch);
  prob_info.bpeA           = static_cast<uint32_t>(datatype_to_bits(problem.a_dtype) / 8);
  prob_info.bpeB           = static_cast<uint32_t>(datatype_to_bits(problem.b_dtype) / 8);
  prob_info.bpeD           = static_cast<uint32_t>(datatype_to_bits(problem.d_dtype) / 8);
  prob_info.bpeCompute     = static_cast<uint32_t>(datatype_to_bits(problem.mi_dtype) / 8);
  prob_info.transA         = (problem.a_transpose == transpose_t::T);
  prob_info.transB         = (problem.b_transpose == transpose_t::T);
  prob_info.swizzleTensorA = tparams(config).swizzle_a;
  prob_info.swizzleTensorB = tparams(config).swizzle_b;
  prob_info.dataType       = problem.mi_dtype;

  // Convert config_t to Formocast::SizeMapping
  Formocast::SizeMapping size_mapping;
  size_mapping.macroTile[0]         = static_cast<int>(config.mt.m);
  size_mapping.macroTile[1]         = static_cast<int>(config.mt.n);
  size_mapping.macroTile[2]         = static_cast<int>(config.mt.k);
  size_mapping.matrixInstruction[0] = static_cast<int>(config.mi.m);
  size_mapping.matrixInstruction[1] = static_cast<int>(config.mi.n);
  size_mapping.matrixInstruction[2] = static_cast<int>(config.mi.k);
  size_mapping.matrixInstruction[3] = 1;  // Default

  // Use depth_u if set, otherwise use mt.k
  size_mapping.depthU = (tparams(config).depth_u > 0) ? tparams(config).depth_u : config.mt.k;

  size_mapping.globalSplitU       = tparams(config).global_split_u;
  size_mapping.globalAccumulation = tparams(config).global_accumulation;
  size_mapping.LocalSplitU        = tparams(config).local_split_u;

  size_mapping.DirectToVgprA = tparams(config).direct_to_vgpr_a;
  size_mapping.DirectToVgprB = tparams(config).direct_to_vgpr_b;
  size_mapping.DirectToLdsA  = tparams(config).direct_to_lds_a;
  size_mapping.DirectToLdsB  = tparams(config).direct_to_lds_b;

  size_mapping.NumLoadsCoalescedA = tparams(config).num_loads_coalesced_a;
  size_mapping.NumLoadsCoalescedB = tparams(config).num_loads_coalesced_b;
  size_mapping.VectorWidthA       = config.vector_width_a;
  size_mapping.VectorWidthB       = config.vector_width_b;

  size_mapping.waveNum      = tparams(config).wave_num;
  size_mapping.waveGroup[0] = static_cast<int>(tparams(config).wave_group_m);
  size_mapping.waveGroup[1] = static_cast<int>(tparams(config).wave_group_n);

  size_mapping.workGroupMapping         = config.workgroup_mapping;
  size_mapping.workGroupMappingXCC      = tparams(config).workgroup_mapping_xcc;
  size_mapping.workGroupMappingXCCGroup = tparams(config).workgroup_mapping_xcc_group;
  size_mapping.globalSplitUCoalesced    = tparams(config).global_split_u_coalesced;
  size_mapping.globalSplitUWorkGroupMappingRoundRobin =
      tparams(config).global_split_u_wgm_round_robin;

  size_mapping.CUOccupancy            = config.occupancy;
  size_mapping.PrefetchGlobalRead     = tparams(config).prefetch_global_read;
  size_mapping.MathClocksUnrolledLoop = tparams(config).math_clocks_unrolled_loop;

  // Set problem, solution, and hardware in Formocast
  formocast.setProblem(prob_info);
  formocast.setSolution(size_mapping);
  formocast.setHardware(hardware.arch);

  // Get predicted performance
  Formocast::PredictedPerformance perf = formocast.predictedPerformance();

  // Return latency in microseconds
  return perf.microSeconds;
}

// ---------------------------------------------------------------------------
// Parallel reduction kernel
// ---------------------------------------------------------------------------

// Compute the latency of the PostGSU parallel reduction kernel.
double compute_parallel_reduction_latency(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          const context_t& context) {
  // Single kernel launch and flat reduction.
  // Each thread reads ALL splitting_factor partials from workspace (f32),
  // accumulates them sequentially, and writes one output element (d_dtype).
  // For small GSU (4/8/16) the reads are fully unrolled; for larger GSU a loop is used.

  // Only applies to parallel reduction with splitting
  if (context.splitting_factor <= 1 || context.reduction_strategy != reduction_t::parallel)
    return 0.0;

  // Extract parameters
  const size_t M               = problem.size.m;
  const size_t N               = problem.size.n;
  const size_t batch           = problem.batch;
  const size_t output_elements = M * N * batch;

  const size_t splitting_factor = context.splitting_factor;
  const double d_bytes          = context.d_bytes;

  if (d_bytes == 0.0) return 0.0;

  // Each thread processes VW output elements.
  const size_t VW = std::max(static_cast<size_t>(1), static_cast<size_t>(4.0 / d_bytes));
  const size_t total_wgs =
      math::safe_ceil_div(output_elements, heuristic_defaults_t::POSTGSU_THREADS_PER_WG * VW);
  const size_t active_wgs = std::min(total_wgs, context.n_cu);
  const size_t timesteps  = math::safe_ceil_div(total_wgs, context.n_cu);

  // Bandwidth based on occupancy of the reduction kernel.  Workspace
  // partials are served at MALL bandwidth (mem2) regardless of the
  // main-kernel cache_hints_d setting; without a MALL they come from DRAM.
  const double bw_per_cu = compute_mem_bw_from_occupancy(hardware, active_wgs);
  const double partials_ratio =
      hardware.has_MALL() ? hardware.mem2_perf_ratio : hardware.mem3_perf_ratio;
  double read_bw = std::max(partials_ratio * bw_per_cu, 1e-12);

  // Total data movement per timestep:
  //   Read:  active_wgs × threads_per_wg × VW × splitting_factor × compute_bytes
  //   Write: active_wgs × threads_per_wg × VW × d_bytes
  double elements_per_ts =
      static_cast<double>(active_wgs) * heuristic_defaults_t::POSTGSU_THREADS_PER_WG * VW;
  double read_bytes_per_ts =
      elements_per_ts * splitting_factor * heuristic_defaults_t::POSTGSU_COMPUTE_BYTES;
  double write_bytes_per_ts = elements_per_ts * d_bytes;

  // Per-timestep latency: read + accumulate + write
  double L_read  = read_bytes_per_ts / read_bw;
  double L_write = write_bytes_per_ts / read_bw;
  // Accumulate: each lane adds splitting_factor - 1 partials into each of its VW
  // outputs, one vector-ALU instruction apiece, and a WG's waves share the CU's SIMDs.
  const size_t waves_per_wg =
      math::safe_ceil_div(heuristic_defaults_t::POSTGSU_THREADS_PER_WG, context.wavefront_size);
  const double waves_per_simd =
      static_cast<double>(math::safe_ceil_div(waves_per_wg, hardware.simds_per_cu()));
  double L_acc = static_cast<double>((splitting_factor - 1) * VW) *
                 static_cast<double>(context.valu_cycles) * waves_per_simd;

  const double L_launch =
      for_family(hardware, [&](auto family) { return family.reduction_launch_overhead(hardware); });
  double L_total = L_launch + (L_read + L_acc + L_write) * timesteps;

  if (context.debug) {
    OLOG_DEBUG("L_parallel_reduce_launch: " << L_launch);
    OLOG_DEBUG("L_parallel_reduce_active_wgs: " << active_wgs);
    OLOG_DEBUG("L_parallel_reduce_timesteps: " << timesteps);
    OLOG_DEBUG("L_parallel_reduce_bw: " << read_bw);
    OLOG_DEBUG("L_parallel_reduce_reads: " << L_read);
    OLOG_DEBUG("L_parallel_reduce_accumulates: " << L_acc);
    OLOG_DEBUG("L_parallel_reduce_writes: " << L_write);
  }

  return L_total;
}

// ---------------------------------------------------------------------------
// Total latency
// ---------------------------------------------------------------------------

namespace {

// Latency of a W4A16 decode kernel, in the cycles of compute_total_latency (2200 per us,
// as KERNEL_LAUNCH_OVERHEAD). problem.size: m weight rows, n tokens, k. The longer of the
// weight stream at the DRAM bandwidth and the rounds of resident waves, plus the launch.
double compute_w4a16_decode_latency(const problem_t& problem,
                                    const hardware_t& hardware,
                                    const config_t& config) {
  const double dram_bytes_per_us = *w4a16_decode_dram_bytes_per_us(hardware);
  // 2200 is not the device clock (hardware.compute_clock_ghz is 2.9 GHz on gfx1150 and
  // gfx1151).  It cancels in the wave term, whose constants are cycles, so all it sets is what
  // the DRAM stream costs against the waves and against the tiles: 2200 cycles per us, 24%
  // less than the device's 2900.  With the device clock the picks that change get slower,
  // measured 1.04x on gfx1151 and 1.08x on gfx1150 on 1-5 token shapes.  The factor makes up
  // for the memory rates of hardware_t, which the tile model is calibrated with and which are
  // wrong on these devices (their DRAM bandwidth is 2.5x too high on gfx1151).  Keep it until
  // those are fixed, then use the device clock.
  constexpr double CYCLES_PER_US = 2200.0;
  const double m                 = static_cast<double>(problem.size.m);
  const double k                 = static_cast<double>(problem.size.k);
  const double depth_u           = static_cast<double>(config.mt.k);
  const double batch             = static_cast<double>(std::max<size_t>(problem.batch, 1));
  const double token_tiles =
      static_cast<double>(math::safe_ceil_div(problem.size.n, std::max<size_t>(config.mt.n, 1)));
  const double waves =
      static_cast<double>(math::safe_ceil_div(problem.size.m, std::max<size_t>(config.mt.m, 1))) *
      token_tiles * batch;
  // The weights (half a byte each) are read once per token tile, at the DRAM bandwidth.
  const double t_dram = token_tiles * m * k * batch * 0.5 / dram_bytes_per_us;
  // The waves run in rounds of the resident ones (one per workgroup), each one K iteration
  // after another (the last one whole when DepthU does not divide K).  The kernel's occupancy
  // counts the workgroups on hardware_t::OCCUPANCY_SIMDS SIMDs, a CU has simds_per_cu of them.
  const double resident = static_cast<double>(resolve_num_cus(problem.num_cus, hardware.N_CU)) *
                          static_cast<double>(hardware.simds_per_cu()) /
                          static_cast<double>(hardware_t::OCCUPANCY_SIMDS) *
                          static_cast<double>(std::max(config.occupancy, 1));
  // The cost of a wave has no term in MT_M x MT_N (the dot products of an iteration).  Most
  // problems are DRAM bound and never reach this term: 9 of 10 measured cases of 30 us or
  // more.  Where the waves do bound the time (small problems), an iteration is mostly the
  // load-to-use latency of its global reads: from 1 to 10 elements per wave the measured time
  // over the predicted one shows no trend (0.72-1.08 on gfx1151, 0.85-1.19 on gfx1150).
  const double wave_cycles = w4a16_params_t::DECODE_WAVE_CYCLES +
                             std::ceil(k / depth_u) * w4a16_params_t::DECODE_ITER_CYCLES;
  const double t_waves = std::ceil(waves / resident) * wave_cycles / CYCLES_PER_US;
  return std::max(t_dram, t_waves) * CYCLES_PER_US + heuristic_defaults_t::KERNEL_LAUNCH_OVERHEAD;
}

// The tile model's total, written once for every family.
template <class Family>
double tile_model_latency(const problem_t& problem,
                          const hardware_t& hardware,
                          const config_t& config,
                          const heuristic_params_t& heuristic) {
  // 1) Setup context (computes grid dims, launch params, WGM, etc.)
  context_t context(problem, hardware, config, heuristic);

  // 2) Latency of a timestep: one K-complete output tile computed on one CU
  double L_timestep = compute_tile_latency(problem, hardware, config, context);

  // 3) Latency for all scheduling rounds.  num_timesteps = WG waves passing
  // through the CUs.  Occupancy hides stalls inside a tile but doesn't cut the
  // round count, so it belongs in the per-tile model, not as a divisor here.
  double timesteps = static_cast<double>(context.num_timesteps);
  // A W4A16 Stream-K grid of at most one workgroup per CU that shares more tiles
  // than workgroups runs tiles / workgroups tiles per workgroup (its partial tiles
  // are charged below), not whole waves: below a few tiles per CU, where data
  // parallel pays a partly idle last wave, counting it as whole waves made Origami
  // pick the data-parallel kernel, measured up to 2x slower on gfx1151.
  // Token-tile grids (one workgroup per SIMD, streamk.cpp) are left out and charged whole
  // waves: two workgroups share a CU's SIMDs, so each runs its tiles at half the rate.
  const bool w4a16 = is_w4a16(problem, hardware);
  if (w4a16 && config.stream_k > 0 && context.num_wgs < context.num_output_tiles &&
      context.num_wgs <= context.n_cu)
    timesteps =
        static_cast<double>(context.num_output_tiles) / static_cast<double>(context.num_wgs);
  double total_latency = L_timestep * timesteps;

  // W4A16 Stream-K partial tiles: the workgroup that finishes a tile waits for, reads and adds
  // the MT_M x MT_N f32 partial of every other workgroup that worked on it, on the critical
  // path. A tile split over s workgroups has s - 1 partials; a grid that shares tiles unevenly
  // has one. Without it, splitting a few large tiles' K over every CU looked as fast as
  // splitting many small ones, and was measured up to 2.3x slower.
  if (w4a16 && config.stream_k > 0 && context.num_wgs != context.num_output_tiles) {
    const double tiles    = static_cast<double>(context.num_output_tiles);
    const double wgs      = static_cast<double>(context.num_wgs);
    const double partials = wgs > tiles ? std::ceil(wgs / tiles) - 1.0
                                        : (context.num_output_tiles % context.num_wgs ? 1.0 : 0.0);
    total_latency += partials * w4a16_params_t::SK_FIXUP_CYCLES_PER_ELEM *
                     static_cast<double>(config.mt.m * config.mt.n);
  }

  // 4) Kernel launch overhead, and what else the kernel pays once outside its tiles
  total_latency += Family::kernel_launch_overhead(hardware);
  total_latency += Family::kernel_fixed_cost(hardware, config, context);

  // 5) Add parallel reduction kernel cost (separate kernel launch, 0 if not parallel)
  double L_parallel_reduce = compute_parallel_reduction_latency(problem, hardware, config, context);
  total_latency += L_parallel_reduce;

  if (context.debug) {
    OLOG_DEBUG("L_parallel_reduce: " << L_parallel_reduce);
    OLOG_DEBUG("total_latency: " << total_latency);
    OLOG_DEBUG("=================================");
  }

  return total_latency;
}

// The analytical method: the decode model for a W4A16 decode kernel, the tile model otherwise.
double analytical_latency(const problem_t& problem,
                          const hardware_t& hardware,
                          const config_t& config,
                          const heuristic_params_t& heuristic) {
  if (is_w4a16_decode(problem, hardware, config))
    return compute_w4a16_decode_latency(problem, hardware, config);
  return for_family(hardware, [&](auto family) {
    return tile_model_latency<decltype(family)>(problem, hardware, config, heuristic);
  });
}

}  // namespace

double compute_analytical_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  return analytical_latency(
      problem, hardware, config, get_heuristic_params(problem, hardware, config));
}

evaluation_t evaluate(const problem_t& problem,
                      const hardware_t& hardware,
                      const config_t& config) {
  heuristic_params_t heuristic;
  const rejection_t rejection = admit(problem, hardware, config, heuristic);
  if (rejection != rejection_t::none) return {rejection, REJECTED_LATENCY};

  // The method's evaluator would look the heuristics up again, so the model is called directly.
  if (config.prediction_mode == prediction_modes_t::simulation)
    return {rejection_t::none, compute_simulated_latency(problem, hardware, config)};
  return {rejection_t::none, analytical_latency(problem, hardware, config, heuristic)};
}

double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config) {
  return evaluate(problem, hardware, config).latency;
}

}  // namespace gemm
}  // namespace origami
