// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "origami/gemm.hpp"
#include "origami/streamk.hpp"
#include "origami/simulator/tensilelite/formocast_simulator.hpp"

namespace origami {

// Forward declaration for internal Formocast latency computation
static double compute_formocast_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config);

double calculate_work_utilization(const problem_t& problem, const config_t& config) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  // Calculate the full dimensions covered by the launched grid of tiles (spatial).
  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);

  // Calculate the full depth covered by the k-loop iterations (temporal).
  const double launched_K =
      static_cast<double>(math::safe_ceil_div(K, MT_K)) * static_cast<double>(MT_K);

  // The utilization is the ratio of the useful problem volume to the total scheduled volume.
  const double useful_volume   = static_cast<double>(M * N * K);
  const double launched_volume = launched_M * launched_N * launched_K;

  if (launched_volume < 1.0) return 1.0;  // Avoid division by zero for tiny/empty problems

  const double utilization = useful_volume / launched_volume;

  return utilization;
}

double calculate_output_utilization(const problem_t& problem,
                                    const config_t& config,
                                    size_t vector_elems = 1) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  // Tiled coverage in M/N
  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);

  // Optional: model vectorization/alignment remainders (e.g., ld/st width)
  // This assumes vectors must be fully inside bounds; tail elements are scalarized.
  const size_t M_vec = (vector_elems > 1) ? math::safe_ceil_div(M, vector_elems) * vector_elems : M;
  const size_t N_vec = (vector_elems > 1) ? math::safe_ceil_div(N, vector_elems) * vector_elems : N;

  const double useful   = static_cast<double>(M_vec) * static_cast<double>(N_vec);
  const double launched = launched_M * launched_N;

  if (launched < 1.0) return 1.0;
  return useful / launched;
}

// Computes the number of active compute units if there is only one timestep and it is partial
// Otherwise, returns hardware.N_CU
std::tuple<size_t, size_t, size_t, size_t> compute_cu_occupancy(const problem_t& problem,
                                                                const hardware_t& hardware,
                                                                const config_t& config,
                                                                grid_selection_t grid_selection,
                                                                size_t max_cus,
                                                                size_t split = 0) {
  // Number of output MTs
  size_t num_mts = streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  size_t num_wgs, num_active_cus, num_timesteps, split_factor;

  if (split)  // if it is given
  {
    split          = split > 1 ? split : 1;
    num_wgs        = num_mts * split;
    num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
    num_timesteps  = math::safe_ceil_div(num_wgs, hardware.N_CU);
    split_factor   = split;

  } else  // as what StreamK predicts
  {
    auto config_with_reduction = config;
    config_with_reduction.reduction_strategy =
        streamk::select_reduction(problem, hardware, config, grid_selection);

    num_wgs = streamk::select_grid_size(
        problem, hardware, config_with_reduction, grid_selection, max_cus);

    // output variables
    num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
    // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
    // That means, we artifically observe one full timesteps, but that is not what actually happens
    // under the hood. From a theoretical point of view, these distributions change all of the
    // computations in Origami. With current implementation, it is hard to capture that
    // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
    // num_timesteps based on the num_mts. Otherwise, we use num_wgs to compute num_timesteps.
    num_timesteps = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, hardware.N_CU)
                                      : math::safe_ceil_div(num_mts, hardware.N_CU);
    split_factor  = math::safe_ceil_div(num_wgs, num_mts);
  }

  return std::make_tuple(num_wgs, num_active_cus, num_timesteps, split_factor);
}

/* ---------------------------------------------------------------------------------------- */
/* Compute-related functions                                                                */
/* ---------------------------------------------------------------------------------------- */
// Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi) {
  // Compute the number of Matrix Instructions required in each dim.
  size_t num_m_instrs = math::safe_ceil_div(mt.m, mi.m);
  size_t num_n_instrs = math::safe_ceil_div(mt.n, mi.n);
  size_t num_k_instrs = math::safe_ceil_div(mt.k, mi.k);

  // Total number of matrix instructions.
  size_t num_matrix_instrs = num_m_instrs * num_n_instrs * num_k_instrs;

  return num_matrix_instrs;
}

// Compute arithmic intensity
double arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

// Computes Emulated arithmetic intensity for TF32 (assumes 3xBF16).
double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 3.0 * 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 3.0 * 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

// Compute cvt overhead in x1 tf32 emulation
// TODO: We can generalize the same routine to cover more GEMMs that perform conversion
static inline double compute_cvt_overhead_x1(const problem_t& problem,
                                             const hardware_t& hardware,
                                             const config_t& config) {
  // In X1 TF32 GEMMs, we do:
  // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
  // v_cvt_pk_bf16_f32  (convert/pack fp32 to bf16)
  // ds_write_b64
  // That is, the extra instructions that we need to account for are the two cvt_pk ops
  // per wavefront tile

  // However, these extra ops should not be added up to the overal tile latency becuase
  // they can be run in parallel to Matix and Memory operations (given they are not dependent).
  // So, We should ideally take L_tile = max{Mem, Comp, Vec (cvt latencies)}.
  // Since, Vec latency is not modeled yet, we somehow model that into the current logic
  // by scaling according to MFMA latencies and putting some heuristics to model the fact
  // that these vector operations can be hidden (read interleaved) with the other memory
  // or MFMA instructions.

  // --- Shorthands -----------------------------------------------------------
  const double MT_M = static_cast<double>(config.mt.m);
  const double MT_N = static_cast<double>(config.mt.n);
  const double MT_K = static_cast<double>(config.mt.k);

  const double MI_M = static_cast<double>(config.mi.m);
  const double MI_N = static_cast<double>(config.mi.n);
  const double MI_K = static_cast<double>(config.mi.k);

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);

  // TODO: Use kernel's actual wavetiles (wavefront's tile size).
  const double wave_tile_m = MT_M / 2.0;
  const double wave_tile_n = MT_N / 2.0;
  const double wave_tile_k = MT_K / MI_K;

  // MFMA count
  const double N_MI     = (wave_tile_m / MI_M) * (wave_tile_n / MI_N) * wave_tile_k;
  const double num_mfma = 1.0 * N_MI;
  // Cycle scale per MI
  const double L_MI        = hardware.get_mi_latency(MI_M, MI_N, MI_K, problem.mi_dtype);
  const double mfma_cycles = num_mfma * L_MI;

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  const double bytesA = wave_tile_m * MT_K * static_cast<double>(a_bytes);
  const double bytesB = wave_tile_n * MT_K * static_cast<double>(b_bytes);

  // 3) Modeled transfer quanta (128B lines)
  //      dsA = bytesA / (128 * MI_M)
  //      dsB = bytesB / (128 * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  const double dsA = (bytesA / 128.0) / MI_M;  // LDS->VGPR for A
  const double dsB = (bytesB / 128.0) / MI_N;  // LDS->VGPR for B
  const double GR  = dsA;                      // Global->LDS reads
  const double LR  = dsA + dsB;                // total DS->VGPR

  // 5) Exposed vs hidden CVT
  // spare MFMA
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  // 2 cvt per each ds_write (this for SS_BSS -- should be revised for other datatypes)
  // Each cvt has a latency of four. It is scaled by the MI Latency
  // Note: change 16.0 based on mi_data_type if we want to generalize this for all
  // casting GEMMs.
  const double cvt = (2.0 * 4.0 / 16.0 * L_MI) * LR;
  // cvt ops are interleaved in main loop and don't stall matrix or memory units.
  // Heuristically, we set
  const double H        = (8.0 / 16.0 * L_MI) * spare_mfma + (4.0 / 16.0) * L_MI * (LR + GR);
  const double overhead = std::max(cvt - H, 0.0);

  return overhead;
}

// Compute cvt overhead in tf32 emulation
double compute_cvt_overhead(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) {
  // Wavefront tile sizes
  // TODO: Use kernel's actual wavetiles (wavefront's tile size).
  const double wave_tile_m = config.mt.m / 2.0;
  const double wave_tile_n = config.mt.n / 2.0;
  const double wave_tile_k = config.mt.k / config.mi.k;

  // MFMA count and cycles
  const double N_MI = (wave_tile_m / config.mi.m) * (wave_tile_n / config.mi.n) * wave_tile_k;

  // TF32 emu: 3× BF16 MI issue slots
  const double num_mfma = 3.0 * static_cast<double>(N_MI);

  // Cycle scale per MI (use BF16 MI latency as the basic timing quantum)
  const double L_MI_bf16 =
      hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, data_type_t::BFloat16);
  // const double mfma_cycles = num_mfma * L_MI_bf16;

  // 2) Bytes (per K-slice), using ceil-div to whole bytes
  auto a_bytes = data_type_to_bytes(problem.a_dtype);
  auto b_bytes = data_type_to_bytes(problem.b_dtype);

  const double bytesA = static_cast<double>(wave_tile_m) * config.mt.k * a_bytes;
  const double bytesB = static_cast<double>(wave_tile_n) * config.mt.k * b_bytes;

  // const double mt_bytesA
  //     = static_cast<double>(MT_M) * MT_K * safe_ceil_div(element_size_A, 8);

  // 3) Modeled transfer quanta (128B lines)
  //      dsA = bytesA / (128 * MI_M)
  //      dsB = bytesB / (128 * MI_N)
  //      GR  = dsA  (global->LDS modeled equal to A-side DS)
  const double dsA = (bytesA / 128.0) / static_cast<double>(config.mi.m);  // LDS->VGPR for A
  const double dsB = (bytesB / 128.0) / static_cast<double>(config.mi.n);  // LDS->VGPR for B
  const double GR  = dsA;                                                  // Global->LDS reads
  const double LR  = dsA + dsB;                                            // total DS->VGPR

  // 4) Heuristic cycle weights (scaled to MI latency).
  //    Preserves your A=104, B=8, C=4 when L_MI_bf16 == 16.
  // 24 vector instructions per 2 ds_reads (16x16x32)
  // 24 vector instructions per 2 ds_reads for A and for B.
  // 3 instructions per fp32 value read; number ds_read * size
  const double A = (104.0 / 16.0) * L_MI_bf16;  // CVT per LR-sized chunk (DS->VGPR)
  const double B = (8.0 / 16.0) * L_MI_bf16;    // hidden per spare MFMA slot
  // MI16: 16 - 4 (12 cycles), for those 4 cycles, VGPRs are locked. 8 cycles to do anything.
  const double C = (4.0 / 16.0) * L_MI_bf16;  // hidden per (LR+GR) slot     // MI16
  // 32 cycles (mfma), 4 cycles, 28, 4 vgpr lock, 24 cycles left.
  // 24: 6 conv instructions, 3 ds_reads, ~6 grs

  // 5) Exposed vs hidden CVT
  const double spare_mfma = std::max(0.0, num_mfma - LR - GR);
  const double cvt        = A * dsA;                         // only DS->VGPR contributes CVT
  const double H          = B * spare_mfma + C * (LR + GR);  // hidden cycles
  const double overhead   = std::max(cvt - H, 0.0);

  // 6) Efficiency
  // const double denom = mfma_cycles + overhead;
  // const double eff   = (denom > 0.0) ? (mfma_cycles / denom) : 1;

  return overhead;
}

// Determine the compute latency per MT_MxMT_NxMT_K Macro Tile (L_MT).
size_t compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  // Compute the number of matrix instructions
  size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  // Latency of a single MT_MxMT_NxMT_k tile is the latency of one MI multiplied by
  // number of MI per MT_MxMT_NxMT_k.
  size_t L_MI = hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);

  // size_t mt_arith = arithmetic_intensity(MT_M, MT_N, MT_K, 2);
  // printf("MT_M:%d MT_N:%d MT_K:%d arith:%d\n", MT_M, MT_N, MT_K, mt_arith);
  // size_t arith = ((M * N * K * 2) / (M * K + N * K + M * N));
  size_t L_MT = L_MI * N_MI;

  return L_MT;
}

/* ---------------------------------------------------------------------------------------- */
/* Memory-related functions                                                                 */
/* ---------------------------------------------------------------------------------------- */
// Check if MT fits in LDS
bool check_lds_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype,
                        data_type_t b_dtype) {
  // A and B size
  auto a_loads_in_bytes = mt.mk() * data_type_to_bytes(a_dtype);
  auto b_loads_in_bytes = mt.nk() * data_type_to_bytes(b_dtype);
  // Size of those in bytes
  auto LDS_usage = a_loads_in_bytes + b_loads_in_bytes;

  if (LDS_usage > hardware.lds_capacity) {
    return false;  // Exceeds LDS capacity
  } else {
    return true;  // Within LDS capacity
  }
}

// Compute limited achievable memory bandwidth based on active CUs
double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus) {
  const double CUs = static_cast<double>(num_active_cus);

  if (num_active_cus > hardware.N_CU) return 1.0;

  const double bw_limited = std::get<0>(hardware.mem_bw_per_wg_coefficients) * CUs * CUs +
                            std::get<1>(hardware.mem_bw_per_wg_coefficients) * CUs +
                            std::get<2>(hardware.mem_bw_per_wg_coefficients);

  return std::min(bw_limited, 1.0);
}

double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       size_t splitting_factor) {
  // Use size_t for dimensions and counts to ensure type safety.
  const size_t workgroups_m     = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t workgroups_n     = math::safe_ceil_div(problem.size.n, config.mt.n);
  const size_t total_workgroups = workgroups_m * workgroups_n;

  // Concurrently executing workgroups are limited by the number of CUs.a
  const size_t concurrent_workgroups = std::min(total_workgroups, hardware.N_CU);
  if (concurrent_workgroups == 0)
    throw std::runtime_error("#Workgroups is zero in estimate l2 hit");

  // Number of CUs that might share the same K-tiles, adjusted for K-splitting.
  // This affects contention on the L2 cache partitions (XCDs).
  const size_t effective_cus =
      math::safe_ceil_div(concurrent_workgroups, splitting_factor * problem.batch);
  const size_t cu_per_xcd =
      std::max(math::safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

  // Initial guess for the L2 tile dimensions (a tile of workgroups).
  size_t l2_tile_n = std::min(static_cast<size_t>(config.workgroup_mapping), workgroups_n);
  size_t l2_tile_m = math::safe_ceil_div(cu_per_xcd, l2_tile_n);

  // Handle wrap-around case: if the tile is taller than the grid, wrap it to be wider.
  if (l2_tile_m > workgroups_m) {
    size_t num_wraps = (l2_tile_m / workgroups_m);
    l2_tile_n += (num_wraps * config.workgroup_mapping);
    l2_tile_m = workgroups_m;
  }

  // Clamp initial tile dimensions to the actual grid size.
  l2_tile_m = std::max(std::min(workgroups_m, l2_tile_m), static_cast<size_t>(1));
  l2_tile_n = std::max(std::min(workgroups_n, l2_tile_n), static_cast<size_t>(1));

  // Calculate memory footprint in bytes.
  const auto a_bytes       = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes       = data_type_to_bytes(problem.b_dtype);
  auto calculate_footprint = [&](auto tile_m, auto tile_n) {
    auto a_footprint = tile_m * config.mt.mk() * a_bytes;
    auto b_footprint = tile_n * config.mt.nk() * b_bytes;
    return a_footprint + b_footprint;
  };

  // Symmetrically shrink the L2 tile until it fits in the L2 cache capacity.
  // This is more robust than shrinking only one dimension.
  while (calculate_footprint(l2_tile_m, l2_tile_n) > hardware.L2_capacity) {
    if (l2_tile_m > 1 && l2_tile_m >= l2_tile_n) {
      l2_tile_m--;
    } else if (l2_tile_n > 1) {
      l2_tile_n--;
    } else {
      // Cannot shrink further.
      break;
    }
  }

  // Uncached reads are the first read of each unique element within the L2 tile.
  const long long uncached_A_reads     = static_cast<long long>(l2_tile_m) * config.mt.mk();
  const long long uncached_B_reads     = static_cast<long long>(l2_tile_n) * config.mt.nk();
  const long long total_uncached_reads = uncached_A_reads + uncached_B_reads;

  // Total reads are the sum of all reads performed by all workgroups in the L2 tile.
  // Matrix A is reused l2_tile_n times, Matrix B is reused l2_tile_m times.
  const long long total_A_reads = uncached_A_reads * l2_tile_n;
  const long long total_B_reads = uncached_B_reads * l2_tile_m;
  const long long total_reads   = std::max(total_A_reads + total_B_reads, 1LL);

  const long long cached_reads = total_reads - total_uncached_reads;

  double l2_hit_rate = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  // Final clamping and logging.

  // Clamp the hit rate to be within a realistic [0, 1] range.
  return std::max(0.0, std::min(l2_hit_rate, 1.0));
}

// Estimate MALL hit-rate
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         size_t num_active_cus,
                         size_t splitting_factor) {
  const size_t workgroups_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t workgroups_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (num_active_cus == 0) throw std::runtime_error("Number of Active CUs was 0");

  // --- Initial Tile Sizing based on Concurrency ---
  // Use ceiling division for a more accurate initial guess.
  size_t mall_tile_m =
      math::safe_ceil_div(num_active_cus, static_cast<size_t>(config.workgroup_mapping));
  size_t mall_tile_n = std::min(static_cast<size_t>(config.workgroup_mapping), workgroups_n);

  // Handle wrap-around case if the tile is taller than the grid.
  if (mall_tile_m > workgroups_m) {
    size_t num_wraps = mall_tile_m / workgroups_m;
    mall_tile_n += (num_wraps * config.workgroup_mapping);
    mall_tile_m = workgroups_m;
  }

  // Clamp initial tile dimensions to the actual grid size.
  mall_tile_m = std::max(std::min(workgroups_m, mall_tile_m), static_cast<size_t>(1));
  mall_tile_n = std::max(std::min(workgroups_n, mall_tile_n), static_cast<size_t>(1));

  // --- CRITICAL: Shrink tile to fit into MALL Capacity ---
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);

  auto calculate_footprint = [&](auto tile_m, auto tile_n) {
    auto a_footprint = tile_m * config.mt.mk() * a_bytes;
    auto b_footprint = tile_n * config.mt.nk() * b_bytes;
    return a_footprint + b_footprint;
  };

  // --- Calculate Hit Rate based on the final, capacity-aware tile size ---
  const long long uncached_A_reads     = static_cast<long long>(mall_tile_m) * config.mt.mk();
  const long long uncached_B_reads     = static_cast<long long>(mall_tile_n) * config.mt.nk();
  const long long total_uncached_reads = uncached_A_reads + uncached_B_reads;

  const long long total_A_reads = uncached_A_reads * mall_tile_n;
  const long long total_B_reads = uncached_B_reads * mall_tile_m;
  const long long total_reads   = std::max(total_A_reads + total_B_reads, 1LL);

  const long long cached_reads = total_reads - total_uncached_reads;

  double mall_hit_rate = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  // Clamp the final result to the valid [0, 1] range.
  return std::max(0.0, std::min(mall_hit_rate, 1.0));
}

/**
 * @brief L2 hit rate from a global (problem-wide) perspective using the refactored API.
 * Computes in BYTES to correctly handle differing A/B dtypes.
 */
double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t l2_capacity_bytes) {
  // --- Hardware Parameters (as requested, defined locally) ---
  // You would normally get l2_capacity_bytes from your hardware_t struct.
  if (l2_capacity_bytes == 0) throw std::runtime_error("L2 Capacity is zero");

  // 1. Calculate the grid dimensions in terms of macro-tiles
  const size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (grid_m == 0 || grid_n == 0)
    throw std::runtime_error("estimate_l2_hit grid dimensions can not be zero");

  // 2. Calculate the working set size for one full pass of global reuse
  // This is the data needed by one full column of CUs (for A) and one full row (for B).
  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  const double b_bytes = data_type_to_bytes(problem.b_dtype);

  const double a_working_set           = static_cast<double>(grid_m * config.mt.mk()) * a_bytes;
  const double b_working_set           = static_cast<double>(grid_n * config.mt.nk()) * b_bytes;
  const double total_working_set_bytes = a_working_set + b_working_set;

  // 3. CRUCIAL: Check if the working set fits in the L2 cache.
  // If it doesn't, the global reuse pattern is broken by capacity misses,
  // and the hit rate will be very low.
  if (total_working_set_bytes > l2_capacity_bytes) {
    // Return a floor value for the hit rate. The exact value can be tuned,
    // but it should be low to indicate that the ideal reuse is not possible.
    return 0.1;  // 10% hit rate
  }

  // 4. If it fits, calculate the idealized global hit rate
  // Total reads if nothing was cached
  const double total_A_reads = static_cast<double>(grid_m * grid_n * config.mt.mk());
  const double total_B_reads = static_cast<double>(grid_m * grid_n * config.mt.nk());

  // Uncached reads are the first-time fetches for each row/column
  const double uncached_A_reads =
      static_cast<double>(grid_m * config.mt.mk());  // One full column fetches A
  const double uncached_B_reads =
      static_cast<double>(grid_n * config.mt.nk());  // One full row fetches B

  const double total_reads = total_A_reads + total_B_reads;
  if (total_reads == 0) return 1.0;  // No reads, perfect hit rate.

  const double cached_reads =
      (total_A_reads - uncached_A_reads) + (total_B_reads - uncached_B_reads);

  return cached_reads / total_reads;
}

inline size_t round_up_mul(size_t x, size_t m) { return (x + m - 1) / m * m; }

size_t round_elements_to_128B(size_t elements, size_t element_size_bits) {
  const size_t transaction_bits = 128u * 8u;  // 1024 bits = 128 bytes
  const size_t g                = std::gcd(element_size_bits, transaction_bits);
  const size_t E_block          = transaction_bits / g;  // elements per 128B-aligned chunk
  return round_up_mul(elements, E_block);
}

// Determine the memory latency
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              size_t num_active_cus,
                              size_t splitting_factor) {
  
  bool debug = runtime_options::get().debug_enabled;

  // Extract parameters from structured types
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  const auto a_bits  = datatype_to_bits(problem.a_dtype);
  const auto b_bits  = datatype_to_bits(problem.b_dtype);
  size_t batch       = problem.batch;

  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  heuristic_params_t heuristic = get_heuristic_params(problem, hardware, config);

  // 1) Estimate L2 hit-rate
  double H_mem_l2 = estimate_l2_hit(problem, hardware, config, splitting_factor);

  // Global cap on L2 hit-rate (prevents impossible cache residency claims)
  // (Assumes capacity is given in KiB, convert to bytes)
  double H_mem_l2_global =
      compute_l2_hit_rate_global(problem, hardware, config, hardware.L2_capacity * 1024);

  H_mem_l2 = std::min(H_mem_l2, H_mem_l2_global);

  if (H_mem_l2 == 0) { H_mem_l2 = heuristic.l2_min_hit_rate_default; }

  // 2) Estimate mall hit-rate
  double H_mem_mall =
      hardware.has_MALL()
          ? estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor)
          : 0.0;  // MALL is not supported, so we emulate every read as a miss

  // 3) Total loads are loads from A and loads from B
  size_t Ld_A_value = a_trans ? MT_M * round_elements_to_128B(MT_K, a_bits)
                              : round_elements_to_128B(MT_M, a_bits) * MT_K;
  size_t Ld_B_value = b_trans ? round_elements_to_128B(MT_N, b_bits) * MT_K
                              : MT_N * round_elements_to_128B(MT_K, b_bits);
  auto Ld_CU_bytes  = (Ld_A_value * a_bytes)    // A Bytes
                     + (Ld_B_value * b_bytes);  // B Bytes

  // Logic for block scaled datatypes (Assuming BS=32 and 8-bit scales)
  // TODO This is technically wrong, need separate flag to enable MX so we can differentiate FP8
  // and MX8
  if (a_bits < 8 && problem.a_mx_block_size != 0) {
    // Number of scales per tile
    size_t num_scales_A = math::safe_ceil_div(config.mt.mk(), problem.a_mx_block_size);
    Ld_CU_bytes += num_scales_A;  // One Byte per scale
  }
  if (b_bits < 8 && problem.b_mx_block_size != 0) {
    // Number of scales per tile
    size_t num_scales_B = math::safe_ceil_div(config.mt.nk(), problem.b_mx_block_size);
    Ld_CU_bytes += num_scales_B;  // One Byte per scale
  }

  // 4) total loads by all CUs
  double total_Ld = Ld_CU_bytes * static_cast<double>(num_active_cus);

  // 5) mem_l2‐limited factor (simple linear model)
  double mem_l2_bw_limited =
      static_cast<double>(num_active_cus) / static_cast<double>(hardware.N_CU);
  double limited_mem_l2_bw = (hardware.mem1_perf_ratio * mem_l2_bw_limited);

  // 6) mem_l2 latency
  double L_mem_mem_l2 = (limited_mem_l2_bw > 0) ? (total_Ld / (limited_mem_l2_bw)) : 0.0;

  // 7) mem_mall‐limited from occupancy (Can't Issue enough load/stores)
  double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);

  // 8) loads that reach each level
  double Ld_mem_mall =
      hardware.has_MALL()
          ? (1.0 - H_mem_l2) * total_Ld
          : 0.0;  // MALL is not supported, we emulate it by saying there are zero loads to MALL
  double Ld_mem_dram = (1.0 - H_mem_mall) * Ld_mem_mall;

  // 9) enforce whole‐problem minimum loads when we can fit M/N in the CUs.
  // Calculate the tile of workgroups that can run concurrently (logic from estimate_mall_hit).
  size_t grid_m = math::safe_ceil_div(problem.size.m, MT_M);
  size_t grid_n = math::safe_ceil_div(problem.size.n, MT_N);
  size_t mall_m =
      math::safe_ceil_div(num_active_cus, static_cast<size_t>(config.workgroup_mapping));
  size_t mall_n = std::min(static_cast<size_t>(config.workgroup_mapping), grid_n);
  // Handle wrap-around case
  if (mall_m > grid_m) {
    size_t num_wraps = (mall_m / grid_m);
    mall_n += (num_wraps * config.workgroup_mapping);
    mall_m = grid_m;
  }
  // Clamp tile dimensions
  mall_m = std::max(std::min(grid_m, mall_m), static_cast<size_t>(1));
  mall_n = std::max(std::min(grid_n, mall_n), static_cast<size_t>(1));
  // This is the minimum unique bytes needed from HBM to feed the concurrent workgroups.
  double concurrent_batches =
      std::min(static_cast<double>(problem.batch),
               std::max(static_cast<double>(num_active_cus) / (grid_m * grid_n), 1.));
  double min_load = static_cast<double>((mall_m * config.mt.mk() * a_bytes) +
                                        (mall_n * config.mt.nk() * b_bytes)) *
                    concurrent_batches;  // Apply batching to the minimum load itself.
  // The actual loads cannot be less than this physical minimum.
  Ld_mem_dram = std::max(Ld_mem_dram, min_load);
  Ld_mem_mall = std::max(Ld_mem_mall, min_load);

  // 10) mem_mall latency
  double limited_mem_mall_bw = (hardware.mem2_perf_ratio * bw_limited);
  double L_mem_mem_mall = (limited_mem_mall_bw > 0) ? (Ld_mem_mall / limited_mem_mall_bw) : 0.0;

  // 11) mem_dram latency
  double limited_mem_bw = (hardware.mem3_perf_ratio * bw_limited);
  double L_mem_mem_dram = (limited_mem_bw > 0) ? (Ld_mem_dram / limited_mem_bw) : 0.0;
  L_mem_mem_dram += heuristic.main_memory_load_latency;

  // 12) pick the worst‐case bound
  double L_mem = std::max({L_mem_mem_l2 * heuristic.weight_mem_l2,
                           L_mem_mem_mall * heuristic.weight_mem_mall,
                           L_mem_mem_dram * heuristic.weight_mem_dram});

  if(debug)
  {
    OLOG_DEBUG("Ld_CU_bytes: " << Ld_CU_bytes);
    OLOG_DEBUG("total_Ld: " << total_Ld);
    OLOG_DEBUG("H_mem_l2: " << H_mem_l2);
    OLOG_DEBUG("H_mem_l2_global: " << H_mem_l2_global);
    OLOG_DEBUG("H_mem_mall: " << H_mem_mall);
    OLOG_DEBUG("Ld_mem_dram: " << Ld_mem_dram);
    OLOG_DEBUG("Ld_mem_mall: " << Ld_mem_mall);
    OLOG_DEBUG("bw_limited: " << bw_limited);
    OLOG_DEBUG("L_mem_mem_mall: " << L_mem_mem_mall);
    OLOG_DEBUG("L_mem_mem_dram: " << L_mem_mem_dram);
    OLOG_DEBUG("L_mem: " << L_mem);
    OLOG_DEBUG("grid_m: " << int(grid_m));
    OLOG_DEBUG("grid_n: " << int(grid_n));
    OLOG_DEBUG("mall_m: " << int(mall_m));
    OLOG_DEBUG("mall_n: " << int(mall_n));
    OLOG_DEBUG("config.workgroup_mapping: " << int(config.workgroup_mapping));
  }
  return L_mem;
}

/* ---------------------------------------------------------------------------------------- */
/* Tile-related functions                                                                   */
/* ---------------------------------------------------------------------------------------- */
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            size_t num_active_cus,
                            size_t splitting_factor) {
  
  bool debug = runtime_options::get().debug_enabled;

  // Extract parameters from structured types
  const size_t K = problem.size.k;
  size_t batch   = problem.batch;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const auto a_bits  = datatype_to_bits(problem.a_dtype);
  const auto b_bits  = datatype_to_bits(problem.b_dtype);
  const auto d_bytes = data_type_to_bytes(problem.d_dtype);

  heuristic_params_t heuristic = get_heuristic_params(problem, hardware, config);

  // 1) Compute per-tile latencies
  double L_compute = compute_mt_compute_latency(problem, hardware, config);

  double L_mem =
      compute_memory_latency(problem, hardware, config, num_active_cus, splitting_factor);

  // TODO Does work utilization need to be 128-byte rounded for a cache line?
  double utilization        = calculate_work_utilization(problem, config);
  double output_utilization = calculate_output_utilization(problem, config, 1UL);
  // The effective latency per useful operation increases as utilization drops.
  // This penalty affects BOTH compute and memory bounds for the tile's core work.
  double effective_tile_penalty = (utilization > 1e-9) ? (1.0 / (utilization)) : 1.0;
  double output_utilization_penalty =
      (output_utilization > 1e-9) ? (1.0 / (output_utilization)) : 1.0;

  // 2) Work-group setup & iteration latencies
  double L_WG_setup = 1;  // WG_setup_Latency

  // 3) Prologue and Epilogue latencies
  // Prologue and Epilogue overhead are reduced with higher occupancy kernels.
  int grid_m = static_cast<int>(math::safe_ceil_div(problem.size.m, MT_M));
  int grid_n = static_cast<int>(math::safe_ceil_div(problem.size.n, MT_N));
  size_t real_occupancy =
      std::min(std::max(config.occupancy, static_cast<int>(1)),
               static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor,
                                                    hardware.N_CU)));  // Number of WGs per CU.
  double occupancy_factor = pow(heuristic.occupancy_decay_base, real_occupancy);

  // 3-1) Prologue: set as memory latency
  double L_prologue = L_mem;
  L_prologue *= effective_tile_penalty;
  L_prologue *= occupancy_factor;

  // 3-2) Epilogue: writes from all active CUs with limited bandwidth
  double mem_bw_occ         = compute_mem_bw_from_occupancy(hardware, num_active_cus);
  double mem_bw_occ_limited = hardware.mem3_perf_ratio * mem_bw_occ;
  // Round to cache line (128B) for memory system alignment
  int d_bits = datatype_to_bits(problem.d_dtype);
  size_t MT_M_rounded = (d_bits > 0) ? round_elements_to_128B(MT_M, static_cast<size_t>(d_bits))
                                     : round_elements_to_128B(MT_M, 16);  // fallback to 16-bit

  // Each block can be independently calculated and reordered
  epilogue_components_t epilogue_comp = {};

  // Block 1: Initial memory write latency
  epilogue_comp.initial_memory_write = (static_cast<double>(num_active_cus / splitting_factor) *
                                        MT_M_rounded * MT_N * d_bytes) /
                                       mem_bw_occ_limited;

  // Block 2: One compute iteration in the epilogue
  epilogue_comp.compute_iteration = L_compute * effective_tile_penalty;

  if(debug)
  {
    OLOG_DEBUG("mem_bw_occ: " << mem_bw_occ);
    OLOG_DEBUG("mem_bw_occ_limited: " << mem_bw_occ_limited);
    OLOG_DEBUG("utilization: " << utilization);
    OLOG_DEBUG("output_utilization: " << output_utilization);
    OLOG_DEBUG("effective_tile_penalty: " << effective_tile_penalty);
    OLOG_DEBUG("output_utilization_penalty: " << output_utilization_penalty);
    OLOG_DEBUG("config.occupancy: " << config.occupancy);
    OLOG_DEBUG("real_occupancy: " << real_occupancy);
    OLOG_DEBUG("num_active_cus: " << int(num_active_cus));
    OLOG_DEBUG("splitting_factor: " << int(splitting_factor));
  }

  // Block 3: K-split reduction (if applicable)
  if (splitting_factor > 1) {
    size_t n_partials = splitting_factor - 1;

    // Only the reduction CU reads from all splits.
    double partial_read_bytes =
        grid_m * grid_n * n_partials * MT_M_rounded * MT_N * static_cast<double>(d_bytes);

    // All CUs write (once for each partial, and once by the reduction CU for the output.)
    double partial_write_bytes =
        grid_m * grid_n * MT_M_rounded * MT_N * static_cast<double>(d_bytes);

    double partial_readwrite_bytes = partial_read_bytes + partial_write_bytes;

    // 64 Threads active in a SIMD. Exposed to at least latency of reducing splitting_factor
    // tiles.
    double partial_adds =
        (static_cast<double>(config.mt.mn()) * static_cast<double>(splitting_factor)) / (64);

    double L_reduce                      = partial_readwrite_bytes / (mem_bw_occ_limited);
    epilogue_comp.k_split_reduction      = L_reduce + partial_adds;
    epilogue_comp.k_split_overhead_const = heuristic.k_split_reduction_overhead;
    if(debug)
    {
        OLOG_DEBUG("partial_read_bytes: " << partial_read_bytes);
        OLOG_DEBUG("partial_write_bytes: " << partial_write_bytes);
        OLOG_DEBUG("partial_readwrite_bytes: " << partial_readwrite_bytes);
        OLOG_DEBUG("partial_adds: " << partial_adds);
        OLOG_DEBUG("L_reduce: " << L_reduce);
    }
  }

  // Block 4: K-padding penalty (if applicable)
  double problem_k_quant = 0.0;
  if (K % MT_K != 0) {
    problem_k_quant = static_cast<double>(K % MT_K) / static_cast<double>(K);
    epilogue_comp.k_padding      = problem_k_quant * heuristic.k_padding_penalty;
  }

  double L_epilogue = compose_epilogue(epilogue_comp, heuristic, occupancy_factor);

  // 4) Single-tile latency (apply penalty after finding the bottleneck)
  // tf32 emu has some more overhead
  double L_cvt = 0;
  if ((problem.mi_dtype == data_type_t::XFloat32) &&
      (hardware.arch == hardware_t::architecture_t::gfx950)) {
    L_cvt = compute_cvt_overhead(problem, hardware, config);
  } else if ((a_bits == 32) && (b_bits == 32) && (problem.mi_dtype == data_type_t::BFloat16) &&
             (hardware.arch == hardware_t::architecture_t::gfx950))  // SS_BSS on GFX950
  {
    L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
  }
  double L_tile_single =
      std::max(L_compute * heuristic.weight_compute, L_mem * heuristic.weight_memory);
  L_tile_single *= heuristic.main_loop_efficiency;
  L_tile_single *= effective_tile_penalty;
  L_tile_single += L_cvt;

  // 5) Number of K-iterations (excluding epilogue), at least 1
  const long k_per_split = static_cast<long>(math::safe_ceil_div(K, splitting_factor));
  long num_iter =
      std::max(static_cast<long>(math::safe_ceil_div(static_cast<size_t>(k_per_split), MT_K) - 1),
               static_cast<long>(1));

  // 6) Total tile latency
  double L_tile_total = L_tile_single * static_cast<double>(num_iter);
  L_tile_total += heuristic.weight_prologue * L_prologue;
  L_tile_total += heuristic.weight_epilogue * L_epilogue;
  L_tile_total += heuristic.weight_wg_setup * L_WG_setup;
  L_tile_total += heuristic.weight_loop_overhead * static_cast<double>(num_iter);

  // Apply final tile total weight
  L_tile_total *= heuristic.weight_tile_total;
  
  if(debug)
  {
    OLOG_DEBUG("L_mem: " << L_mem);
    OLOG_DEBUG("L_compute: " << L_compute);
    OLOG_DEBUG("L_cvt: " << L_cvt);
    OLOG_DEBUG("k_per_split: " << k_per_split);
    OLOG_DEBUG("num_iter: " << int(num_iter));
    OLOG_DEBUG("problem_k_quant: " << problem_k_quant);
    OLOG_DEBUG("L_prologue: " << L_prologue);
    OLOG_DEBUG("L_tile_single: " << L_tile_single);
    OLOG_DEBUG("L_epilogue: " << L_epilogue);
    OLOG_DEBUG("L_tile_total: " << L_tile_total);
  }
  return L_tile_total;
}

double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                size_t num_active_cus,
                                size_t splitting_factor) {
  // Assume latency of a timestep is latency of a single K-complete output tile computed on one CU.
  double L_timestep =
      compute_tile_latency(problem, hardware, config, num_active_cus, splitting_factor);

  return L_timestep;
}

double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus) {
  assert(config.is_valid());
  bool debug = runtime_options::get().debug_enabled;

  // Use Formocast simulation model if prediction_mode is set to simulation
  if (config.prediction_mode == prediction_modes_t::simulation) {
    return compute_formocast_latency(problem, hardware, config);
  }

  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  bool a_trans = problem.a_transpose == transpose_t::T;
  bool b_trans = problem.b_transpose == transpose_t::T;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;
  size_t MT_K = config.mt.k;
  size_t MI_M = config.mi.m;
  size_t MI_N = config.mi.n;
  size_t MI_K = config.mi.k;

  const int a_bits  = datatype_to_bits(problem.a_dtype);
  const int b_bits  = datatype_to_bits(problem.b_dtype);
  const int a_bytes = data_type_to_bytes(problem.a_dtype);

  // 0) Short-circuit
  // We don't need to compute latency for all MTs. With this, we can shortcut.
  bool shortCircuit = true;
  if (shortCircuit) {
    // When problem dimensions are small enough that we can fit them in one tile, we should do
    // so. This short circuit condition also decreases selection latency when problems are very
    // small :)
    // TODO 256 and 256 here should be largest M and N tile dimensions in library
    if (M <= 256 && N <= 256 && K < 1024 && batch != 1 && (MT_M < M || MT_N < N))
      return std::numeric_limits<double>::max();

    // Use Dot2 only for M < 3
    if (MI_M == 1 && MI_N == 1 && MI_K == 64 && M > 2) return std::numeric_limits<double>::max();

    size_t K_mod_128bytes    = K * a_bits % 1024;
    size_t MT_K_mod_128bytes = MT_K * a_bits % 1024;
    if (K_mod_128bytes == 0 && MT_K_mod_128bytes == 0) {
      // avoid division by 0 if K == 0
      if (M <= MT_M * 2 && !b_trans && ((N * b_bits) / (M * a_bits) > 5)) {
        // Use nontemporal B
        if (!(config.cache_hints_b == 4)) { return std::numeric_limits<double>::max(); }
      } else if (N <= MT_N * 2 && a_trans && ((M * a_bits) / (N * b_bits) > 5)) {
        // Use Non Temporal A
        if (!(config.cache_hints_a == 4)) { return std::numeric_limits<double>::max(); }
      } else {
        // Never use Non Temporal
        if (config.cache_hints_a || config.cache_hints_b) {
          return std::numeric_limits<double>::max();
        }
      }
    } else if (config.cache_hints_a || config.cache_hints_b) {
      return std::numeric_limits<double>::max();
    }
  }
  if(debug)
  {
    OLOG_DEBUG("======== Origami Debug Info ========");
    OLOG_DEBUG("Problem size: " << int(M) << "x" << int(N) << "x" << int(K));
    OLOG_DEBUG("batch: " << int(batch));
    OLOG_DEBUG("Macrotile: " << int(MT_M) << "x" << int(MT_N) << "x" << int(MT_K));
    OLOG_DEBUG("MatrixInstruction: " << int(MI_M) << "x" << int(MI_N) << "x" << int(MI_K));
    OLOG_DEBUG("Element size A (bits): " << int(a_bits));
    OLOG_DEBUG("Element size B (bits): " << int(b_bits));
    OLOG_DEBUG("cache_hints_a: " << int(config.cache_hints_a));
    OLOG_DEBUG("cache_hints_b: " << int(config.cache_hints_b));
    OLOG_DEBUG("problem mxa_block_size: " << int(problem.a_mx_block_size));
    OLOG_DEBUG("problem mxb_block_size: " << int(problem.b_mx_block_size));
  }
  // 1-1) To compute the latency, use default WGM. And WGM can't be greater than one
  int defaultWGM =
      batch > 1 ? 1 : static_cast<int>(ceil(std::sqrt(hardware.N_CU / hardware.NUM_XCD)));
  auto config_with_default_wgm              = config;
  config_with_default_wgm.workgroup_mapping = std::max(defaultWGM, 1);

  // 1-2) Find CU occupancy
  auto [num_wgs, num_active_cus, num_timesteps, splitting_factor] = compute_cu_occupancy(
      problem, hardware, config_with_default_wgm, config_with_default_wgm.grid_selection, max_cus);

  // 2) Compute latency of a timestep
  double L_timestep = compute_timestep_latency(
      problem, hardware, config_with_default_wgm, num_active_cus, splitting_factor);

  // Compute latency for all timesteps and return it as the latency for the MT/problem
  double total_latency = L_timestep * num_timesteps;
  if (debug)
  {
    OLOG_DEBUG("num_timesteps: " << num_timesteps);
    OLOG_DEBUG("total_latency: " << total_latency);
    OLOG_DEBUG("=================================");
  }
  return total_latency;
}

static double compute_formocast_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config) {
  // Create Formocast simulator instance
  Formocast formocast;

  // Convert problem_t to Formocast::ProblemInfo
  Formocast::ProblemInfo prob_info;
  prob_info.M = static_cast<double>(problem.size.m);
  prob_info.N = static_cast<double>(problem.size.n);
  prob_info.K = static_cast<double>(problem.size.k);
  prob_info.NumBatches = static_cast<double>(problem.batch);
  prob_info.bpeA = static_cast<uint32_t>(datatype_to_bits(problem.a_dtype) / 8);
  prob_info.bpeB = static_cast<uint32_t>(datatype_to_bits(problem.b_dtype) / 8);
  prob_info.bpeD = static_cast<uint32_t>(datatype_to_bits(problem.d_dtype) / 8);
  prob_info.bpeCompute = static_cast<uint32_t>(datatype_to_bits(problem.mi_dtype) / 8);
  prob_info.transA = (problem.a_transpose == transpose_t::T);
  prob_info.transB = (problem.b_transpose == transpose_t::T);
  prob_info.swizzleTensorA = config.tensile().swizzle_a;
  prob_info.swizzleTensorB = config.tensile().swizzle_b;
  prob_info.dataType = problem.mi_dtype;

  // Convert config_t to Formocast::SizeMapping
  Formocast::SizeMapping size_mapping;
  size_mapping.macroTile[0] = static_cast<int>(config.mt.m);
  size_mapping.macroTile[1] = static_cast<int>(config.mt.n);
  size_mapping.macroTile[2] = static_cast<int>(config.mt.k);
  size_mapping.matrixInstruction[0] = static_cast<int>(config.mi.m);
  size_mapping.matrixInstruction[1] = static_cast<int>(config.mi.n);
  size_mapping.matrixInstruction[2] = static_cast<int>(config.mi.k);
  size_mapping.matrixInstruction[3] = 1;  // Default

  // Use depth_u if set, otherwise use mt.k
  size_mapping.depthU = (config.tensile().depth_u > 0) ? config.tensile().depth_u : config.mt.k;

  size_mapping.globalSplitU = config.tensile().global_split_u;
  size_mapping.globalAccumulation = config.tensile().global_accumulation;
  size_mapping.LocalSplitU = config.tensile().local_split_u;

  size_mapping.grvwA = config.grvw_a;
  size_mapping.grvwB = config.grvw_b;
  size_mapping.gwvwD = config.gwvw_d;
  size_mapping.gwvwC = config.gwvw_d;  // Typically same as D

  size_mapping.DirectToVgprA = config.tensile().direct_to_vgpr_a;
  size_mapping.DirectToVgprB = config.tensile().direct_to_vgpr_b;
  size_mapping.DirectToLdsA = config.tensile().direct_to_lds_a;
  size_mapping.DirectToLdsB = config.tensile().direct_to_lds_b;

  size_mapping.NumLoadsCoalescedA = config.tensile().num_loads_coalesced_a;
  size_mapping.NumLoadsCoalescedB = config.tensile().num_loads_coalesced_b;
  size_mapping.VectorWidthA = config.vector_width_a;
  size_mapping.VectorWidthB = config.vector_width_b;

  size_mapping.waveNum = config.tensile().wave_num;
  size_mapping.waveGroup[0] = config.tensile().wave_group_m;
  size_mapping.waveGroup[1] = config.tensile().wave_group_n;

  size_mapping.workGroupMapping = config.workgroup_mapping;
  size_mapping.workGroupMappingXCC = config.tensile().workgroup_mapping_xcc;
  size_mapping.workGroupMappingXCCGroup = config.tensile().workgroup_mapping_xcc_group;
  size_mapping.globalSplitUCoalesced = config.tensile().global_split_u_coalesced;
  size_mapping.globalSplitUWorkGroupMappingRoundRobin = config.tensile().global_split_u_wgm_round_robin;

  size_mapping.CUOccupancy = config.occupancy;
  size_mapping.PrefetchGlobalRead = config.tensile().prefetch_global_read;
  size_mapping.MathClocksUnrolledLoop = config.tensile().math_clocks_unrolled_loop;

  // Set problem, solution, and hardware in Formocast
  formocast.setProblem(prob_info);
  formocast.setSolution(size_mapping);
  formocast.setHardware(hardware.arch);

  // Get predicted performance
  Formocast::PredictedPerformance perf = formocast.predictedPerformance();

  // Return latency in microseconds
  return perf.microSeconds;
}

}  // namespace origami
