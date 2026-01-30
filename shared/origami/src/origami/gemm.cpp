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
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "origami/gemm.hpp"
#include "origami/streamk.hpp"

namespace origami {

/* ---------------------------------------------------------------------------------------- */
/* Helper functions                                                                         */
/* ---------------------------------------------------------------------------------------- */
// Calculate work utilization
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

// Calculate output utilization
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

// Computes the launch parameters for the kernel
std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection,
    size_t max_cus) {
  reduction_t reduction_strategy =
      streamk::select_reduction(problem, hardware, config, grid_selection);
  auto config_with_reduction               = config;
  config_with_reduction.reduction_strategy = reduction_strategy;
  size_t num_wgs =
      streamk::select_grid_size(problem, hardware, config_with_reduction, grid_selection, max_cus);

  // output variables
  size_t num_MT_M       = math::safe_ceil_div(problem.size.m, config.mt.m);
  size_t num_MT_N       = math::safe_ceil_div(problem.size.n, config.mt.n);
  size_t num_mts        = num_MT_M * num_MT_N;
  size_t num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
  // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
  // That means, we artifically observe one full timesteps, but that is not what actually happens
  // under the hood. From a theoretical point of view, these distributions change all of the
  // computations in Origami. With current implementation, it is hard to capture that
  // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
  // num_timesteps based on the num_mts. Otherwise, we use num_wgs to compute num_timesteps.
  size_t num_timesteps    = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, hardware.N_CU)
                                              : math::safe_ceil_div(num_mts, hardware.N_CU);
  size_t splitting_factor = math::safe_ceil_div(num_wgs, num_mts);

  return std::make_tuple(
      reduction_strategy, num_wgs, num_active_cus, num_timesteps, splitting_factor);
}

// Check if MT fits in LDS
bool check_lds_capacity(const hardware_t& hardware,
                        const dim3_t& mt,
                        const data_type_t a_dtype,
                        const data_type_t b_dtype) {
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

// Round elements to 128B
size_t round_elements_to_128B(size_t elements, size_t element_size_bits) {
  auto round_up_mul = [](size_t x, size_t m) { return (x + m - 1) / m * m; };

  const size_t transaction_bits = 128u * 8u;  // 1024
  const size_t g                = std::gcd(element_size_bits, transaction_bits);
  const size_t E_block          = transaction_bits / g;  // elements per 128B-aligned chunk
  return round_up_mul(elements, E_block);
}

// Compute L2 tile dimensions
std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           size_t grid_m,
                                           size_t grid_n,
                                           size_t num_active_cus,
                                           size_t splitting_factor,
                                           size_t wgm_value) {
  // Number of CUs that might share the same K-tiles, adjusted for K-splitting.
  const size_t effective_cus =
      math::safe_ceil_div(num_active_cus, splitting_factor * problem.batch);
  const size_t cu_per_xcd =
      std::max(math::safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

  // Initial guess for the L2 tile dimensions (a tile of workgroups).
  size_t l2_tile_n = std::min(wgm_value, grid_n);
  size_t l2_tile_m = math::safe_ceil_div(cu_per_xcd, l2_tile_n);

  // Handle wrap-around case: if the tile is taller than the grid, wrap it to be wider.
  if (l2_tile_m > grid_m) {
    size_t num_wraps = (l2_tile_m / grid_m);
    l2_tile_n += (num_wraps * wgm_value);
    l2_tile_m = grid_m;
  }

  // Clamp initial tile dimensions to the actual grid size.
  l2_tile_m = std::max(std::min(grid_m, l2_tile_m), static_cast<size_t>(1));
  l2_tile_n = std::max(std::min(grid_n, l2_tile_n), static_cast<size_t>(1));

  // Calculate memory footprint in bytes.
  const auto a_bytes       = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes       = data_type_to_bytes(problem.b_dtype);
  auto calculate_footprint = [&](auto tile_m, auto tile_n) {
    auto a_footprint = tile_m * config.mt.mk() * a_bytes;
    auto b_footprint = tile_n * config.mt.nk() * b_bytes;
    return a_footprint + b_footprint;
  };

  // Symmetrically shrink the L2 tile until it fits in the L2 cache capacity.
  while (calculate_footprint(l2_tile_m, l2_tile_n) > hardware.L2_capacity) {
    if (l2_tile_m > 1 && l2_tile_m >= l2_tile_n) {
      l2_tile_m--;
    } else if (l2_tile_n > 1) {
      l2_tile_n--;
    } else {
      break;
    }
  }

  return {l2_tile_m, l2_tile_n};
}

// Compute MALL tile dimensions
std::pair<size_t, size_t> compute_mall_tiles(const problem_t& problem,
                                             const hardware_t& hardware,
                                             const config_t& config,
                                             size_t grid_m,
                                             size_t grid_n,
                                             size_t num_active_cus,
                                             size_t wgm_value) {
  // Initial Tile Sizing based on Concurrency
  size_t mall_tile_m = math::safe_ceil_div(num_active_cus, wgm_value);
  size_t mall_tile_n = std::min(wgm_value, grid_n);

  // Handle wrap-around case if the tile is taller than the grid.
  if (mall_tile_m > grid_m) {
    size_t num_wraps = mall_tile_m / grid_m;
    mall_tile_n += (num_wraps * wgm_value);
    mall_tile_m = grid_m;
  }

  // Clamp initial tile dimensions to the actual grid size.
  mall_tile_m = std::max(std::min(grid_m, mall_tile_m), static_cast<size_t>(1));
  mall_tile_n = std::max(std::min(grid_n, mall_tile_n), static_cast<size_t>(1));

  return {mall_tile_m, mall_tile_n};
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
double compute_cvt_overhead_x1(const problem_t& problem,
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

  size_t L_MT = L_MI * N_MI;

  return L_MT;
}

/* ---------------------------------------------------------------------------------------- */
/* Memory-related functions                                                                 */
/* ---------------------------------------------------------------------------------------- */
// Estimate L2 hit rate
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       const context_t& context) {
  // Fetch pre-computed L2 tile dimensions from context
  const size_t l2_tile_m = context.l2_tiles_m;
  const size_t l2_tile_n = context.l2_tiles_n;

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

  // Clamp the hit rate to be within a realistic [0, 1] range.
  return std::max(0.0, std::min(l2_hit_rate, 1.0));
}

// Estimate MALL hit-rate
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         const context_t& context) {
  // Fetch pre-computed MALL tile dimensions from context
  const size_t mall_tile_m = context.mall_tiles_m;
  const size_t mall_tile_n = context.mall_tiles_n;

  // Calculate Hit Rate based on the tile size
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
                                  const context_t& context) {
  // Extract parameters
  const size_t l2_capacity_bytes = hardware.L2_capacity * 1024;

  const size_t grid_m = context.grid_m;
  const size_t grid_n = context.grid_n;

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

// Determine the memory latency
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context) {
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

  const size_t num_active_cus = context.active_cus;
  const double bw_limited     = context.mem_bw_limited;

  // 1) Estimate L2 hit-rate
  double H_mem1 = estimate_l2_hit(problem, hardware, config, context);

  // Global cap on L2 hit-rate (prevents impossible cache residency claims)
  // (Assumes capacity is given in KiB, convert to bytes)
  double H_mem1_global = compute_l2_hit_rate_global(problem, hardware, config, context);

  H_mem1 = std::min(H_mem1, H_mem1_global);

  if (H_mem1 == 0) { H_mem1 = 0.5; }

  // 2) Estimate mall hit-rate
  double H_mem2 =
      hardware.has_MALL()
          ? estimate_mall_hit(problem, hardware, config, context)
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

  // 5) mem1‐limited factor (simple linear model)
  double mem1_bw_limited = static_cast<double>(num_active_cus) / static_cast<double>(hardware.N_CU);
  double limited_mem1_bw = (hardware.mem1_perf_ratio * mem1_bw_limited);

  // 6) mem1 latency
  double L_mem_mem1 = (limited_mem1_bw > 0) ? (total_Ld / (limited_mem1_bw)) : 0.0;

  // 7) mem2‐limited from occupancy (Can't Issue enough load/stores)

  // 8) loads that reach each level
  double Ld_mem2 =
      hardware.has_MALL()
          ? (1.0 - H_mem1) * total_Ld
          : 0.0;  // MALL is not supported, we emulate it by saying there are zero loads to MALL
  double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

  // 9) enforce whole‐problem minimum loads when we can fit M/N in the CUs.
  // Fetch pre-computed MALL tile dimensions from context.
  const size_t grid_m = context.grid_m;
  const size_t grid_n = context.grid_n;
  const size_t mall_m = context.mall_tiles_m;
  const size_t mall_n = context.mall_tiles_n;
  // This is the minimum unique bytes needed from HBM to feed the concurrent workgroups.
  double concurrent_batches =
      std::min(static_cast<double>(problem.batch),
               std::max(static_cast<double>(num_active_cus) / (grid_m * grid_n), 1.));
  double min_load = static_cast<double>((mall_m * config.mt.mk() * a_bytes) +
                                        (mall_n * config.mt.nk() * b_bytes)) *
                    concurrent_batches;  // Apply batching to the minimum load itself.
  // The actual loads cannot be less than this physical minimum.
  Ld_MEM  = std::max(Ld_MEM, min_load);
  Ld_mem2 = std::max(Ld_mem2, min_load);

  // 10) mem2 latency
  double limited_mem2_bw = (hardware.mem2_perf_ratio * bw_limited);
  double L_mem_mem2      = (limited_mem2_bw > 0) ? (Ld_mem2 / limited_mem2_bw) : 0.0;

  // 11) MEM latency
  double limited_mem_bw = (hardware.mem3_perf_ratio * bw_limited);
  double L_mem_MEM      = (limited_mem_bw > 0) ? (Ld_MEM / limited_mem_bw) : 0.0;
  L_mem_MEM += 200;  // Load Latency

  // 12) pick the worst‐case bound
  double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

  return L_mem;
}

/* ---------------------------------------------------------------------------------------- */
/* Tile-related functions                                                                   */
/* ---------------------------------------------------------------------------------------- */
// Determine the epilogue latency of a single tile
double compute_epilogue_latency(const problem_t& problem,
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

  const size_t MT_M    = config.mt.m;
  const size_t MT_N    = config.mt.n;
  const size_t d_bytes = data_type_to_bytes(problem.d_dtype);

  const reduction_t reduction_strategy = context.reduction_strategy;
  const size_t num_active_cus          = context.active_cus;
  const size_t splitting_factor        = context.splitting_factor;
  const int grid_m                     = context.grid_m;
  const int grid_n                     = context.grid_n;
  const double mem_bw_occ_limited      = context.mem_bw_limited;

  // Constants
  constexpr double cycles_per_acc_read        = 8.0;
  constexpr double acc_read_parallelism       = 0.9;
  constexpr double cycles_per_bounds_check    = 6.0;
  constexpr double scalar_store_penalty       = 2.;
  constexpr size_t threads_per_wave           = 64;
  constexpr size_t bytes_per_vectorized_store = 16;  // buffer_store_dwordx4 = 16 bytes
  constexpr size_t cache_line_bytes           = 128;
  constexpr double cycles_per_sync            = 100.0;

  // Common setup
  const double cycles_per_second = hardware.compute_clock_ghz * 1e9;
  const size_t total_mfmas =
      math::safe_ceil_div(MT_M, config.mi.m) * math::safe_ceil_div(MT_N, config.mi.n);
  const size_t MT_M_rounded_128bytes =
      round_elements_to_128B(MT_M, datatype_to_bits(problem.a_dtype));
  const size_t elements_per_vectorized_store = bytes_per_vectorized_store / d_bytes;
  const size_t elements_per_cache_line       = math::safe_ceil_div(cache_line_bytes, d_bytes);
  const double alignment_penalty             = (M % elements_per_cache_line != 0) ? 1.3 : 1.0;

  // Helper: Convert cycles to seconds
  auto cycles_to_time = [&](double cycles) { return cycles / cycles_per_second; };

  // Helper: compute reduction overhead for a given tile size
  auto compute_reduction_overhead = [&](size_t tile_m, size_t tile_n) -> double {
    if (splitting_factor <= 1 || config.reduction_strategy == reduction_t::parallel) return 0.0;

    size_t tile_m_128bytes = round_elements_to_128B(tile_m, datatype_to_bits(problem.d_dtype));
    size_t tile_elements   = tile_m_128bytes * tile_n;
    size_t n_partials      = splitting_factor - 1;

    // IN-KERNEL REDUCTION (spinlock, tree, atomic)
    // 1. Sync overhead: Finishing WG spins on flags
    double L_sync = cycles_to_time(n_partials * cycles_per_sync);

    // 2. Partial READ: Load all partials from workspace
    double partial_read_bytes = n_partials * tile_elements * d_bytes;
    double L_partial_read     = partial_read_bytes / mem_bw_occ_limited;

    // 3. Accumulation: v_add_f32 for each element × each partial
    double L_accumulate = cycles_to_time(n_partials * tile_elements / threads_per_wave);

    // 4. ACC write-back: Results written back to ACC
    double L_acc_writeback =
        cycles_to_time(total_mfmas * cycles_per_acc_read * acc_read_parallelism);

    return L_sync + L_partial_read + L_accumulate + L_acc_writeback;
  };

  // Check if we have edge tiles
  bool has_interior  = (M / MT_M > 1 && N / MT_N > 1);
  bool has_m_edge    = (M % MT_M != 0);
  bool has_n_edge    = (N % MT_N != 0);
  size_t m_remainder = has_m_edge ? (M % MT_M) : MT_M;
  size_t n_remainder = has_n_edge ? (N % MT_N) : MT_N;

  // TYPE 1: Interior (NonEdge) - Full tile, no bounds checking
  double L_interior = 0.0;
  if (has_interior) {
    // ACC transfer overhead
    size_t acc_reads      = total_mfmas;
    double L_acc_transfer = cycles_to_time(acc_reads * cycles_per_acc_read * acc_read_parallelism);

    // Bounds checking overhead: one check per store instruction
    double L_edge_check = 0.0;

    // Store bandwidth (full tile)
    size_t store_bytes = MT_M_rounded_128bytes * MT_N * d_bytes;
    double L_store     = static_cast<double>(store_bytes) * alignment_penalty / mem_bw_occ_limited;

    // Reduction overhead for interior tile
    double L_reduce = compute_reduction_overhead(MT_M, MT_N);

    L_interior = L_acc_transfer + L_edge_check + L_store + L_reduce;
  }

  // TYPE 2: N-Edge - Uses vectorized store path with bounds checking
  double L_n_edge = 0.0;
  if (has_n_edge) {
    // Same ACC reads as interior (with vectorized store)
    size_t acc_reads      = total_mfmas;
    double L_acc_transfer = cycles_to_time(acc_reads * cycles_per_acc_read * acc_read_parallelism);

    // Bounds checking overhead: one check per store instruction
    size_t total_elements_n = MT_M * n_remainder;
    size_t store_instructions =
        math::safe_ceil_div(total_elements_n, threads_per_wave * elements_per_vectorized_store);
    double edge_check_cycles = store_instructions * cycles_per_bounds_check;
    double L_edge_check      = cycles_to_time(edge_check_cycles);

    // Store bandwidth (smaller tile)
    size_t store_bytes = MT_M * n_remainder * d_bytes;
    double L_store     = static_cast<double>(store_bytes) * alignment_penalty / mem_bw_occ_limited;

    // Reduction overhead for N-edge tile
    double L_reduce = compute_reduction_overhead(MT_M, n_remainder);

    L_n_edge = L_acc_transfer + L_edge_check + L_store + L_reduce;
  }

  // TYPE 3: M-Edge - Uses scalar store path with bounds checking
  double L_m_edge = 0.0;
  if (has_m_edge) {
    // Scalar store path has 2× MORE ACC reads (multiple passes/batches)
    size_t acc_reads      = 2 * total_mfmas;
    double L_acc_transfer = cycles_to_time(acc_reads * cycles_per_acc_read * acc_read_parallelism);

    // Per-element bounds checking (divided by threads_per_wave for SIMD parallelism)
    size_t total_elements_m   = m_remainder * MT_N;
    double store_instructions = math::safe_ceil_div(total_elements_m, threads_per_wave);
    double edge_check_cycles  = store_instructions * cycles_per_bounds_check;
    double L_edge_check       = cycles_to_time(edge_check_cycles);

    // Store bandwidth (smaller tile, scalar stores are less efficient)
    size_t store_bytes = m_remainder * MT_N * d_bytes;
    double L_store = static_cast<double>(store_bytes) * scalar_store_penalty * alignment_penalty /
                     mem_bw_occ_limited;

    // Reduction overhead for M-edge tile
    double L_reduce = compute_reduction_overhead(m_remainder, MT_N);

    L_m_edge = L_acc_transfer + L_edge_check + L_store + L_reduce;
  }

  // TYPE 4: Corner - Both M and N edges
  double L_corner = 0.0;
  if (has_m_edge && has_n_edge) {
    // Scalar store path with 2× more ACC reads (same as M-edge)
    size_t acc_reads      = 2 * total_mfmas;
    double L_acc_transfer = cycles_to_time(acc_reads * cycles_per_acc_read * acc_read_parallelism);

    // Per-element bounds checking (divided by threads_per_wave for SIMD parallelism)
    size_t total_elements_corner = m_remainder * n_remainder;
    double store_instructions    = math::safe_ceil_div(total_elements_corner, threads_per_wave);
    double edge_check_cycles     = store_instructions * cycles_per_bounds_check;
    double L_edge_check          = cycles_to_time(edge_check_cycles);

    // Store bandwidth (smallest tile, scalar stores)
    size_t store_bytes = m_remainder * n_remainder * d_bytes;
    double L_store = static_cast<double>(store_bytes) * scalar_store_penalty * alignment_penalty /
                     mem_bw_occ_limited;

    // Reduction overhead for corner tile
    double L_reduce = compute_reduction_overhead(m_remainder, n_remainder);

    L_corner = L_acc_transfer + L_edge_check + L_store + L_reduce;
  }

  // CRITICAL PATH: maximum of all tile types (including their reduction overhead)
  double L_epilogue = std::max({L_interior, L_n_edge, L_m_edge, L_corner});

  // OCCUPANCY ADJUSTMENT: Higher occupancy reduces overhead (empirical)
  // size_t batch = problem.batch;
  // size_t real_occupancy =
  //     std::min(std::max(config.occupancy, static_cast<int>(1)),
  //              static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor,
  //                                                   hardware.N_CU)));
  // L_epilogue = L_epilogue * pow(0.95, real_occupancy);

  return L_epilogue;
}

// Determine the total latency for a single tile
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context) {
  // Extract parameters from structured types
  const size_t K = problem.size.k;
  size_t batch   = problem.batch;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const auto a_bits  = datatype_to_bits(problem.a_dtype);
  const auto b_bits  = datatype_to_bits(problem.b_dtype);
  const auto d_bytes = data_type_to_bytes(problem.d_dtype);

  const size_t grid_m             = context.grid_m;
  const size_t grid_n             = context.grid_n;
  const size_t splitting_factor   = context.splitting_factor;
  const size_t mem_bw_occ_limited = context.mem_bw_limited;

  // 1) Compute per-tile latencies
  // 1-1) Compute latency for the tile
  double L_compute = compute_mt_compute_latency(problem, hardware, config);

  // 1-2) Compute latency for the memory
  double L_mem = compute_memory_latency(problem, hardware, config, context);

  // 1-3) Adjust based on work utilization
  // TODO Does work utilization need to be 128-byte rounded for a cache line?
  // The effective latency per useful operation increases as utilization drops.
  // This penalty affects BOTH compute and memory bounds for the tile's core work.
  double utilization            = calculate_work_utilization(problem, config);
  double effective_tile_penalty = (utilization > 1e-9) ? (1.0 / (utilization)) : 1.0;

  // 2) Work-group setup & iteration latencies
  double L_WG_setup = 1;

  // 3) Prologue: modeled as scaled memory latency
  // 3-1) Compute the prologue latency
  double L_prologue = 1.5 * L_mem;  // 1.5 chosen emprically
  // 3-2) Adjust based on occupancy
  size_t real_occupancy =
      std::min(std::max(config.occupancy, static_cast<int>(1)),
               static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor,
                                                    hardware.N_CU)));  // Number of WGs per CU.
  L_prologue = L_prologue * pow(0.95, real_occupancy);                 // Factor chosen empirically
  L_prologue *= effective_tile_penalty;

  // 4) MainLoop
  // 4-0) tf32 emu has some more overhead
  double L_cvt = 0;
  if ((problem.mi_dtype == data_type_t::XFloat32) &&
      (hardware.arch == hardware_t::architecture_t::gfx950)) {
    L_cvt = compute_cvt_overhead(problem, hardware, config);
  } else if ((a_bits == 32) && (b_bits == 32) && (problem.mi_dtype == data_type_t::BFloat16) &&
             (hardware.arch == hardware_t::architecture_t::gfx950))  // SS_BSS on GFX950
  {
    L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
  }
  // 4-1) Look up main_loop_efficiency from hardware map
  double main_loop_efficiency = 1.0;
  if (config.custom_mainloop_scheduling) {
    main_loop_efficiency = hardware.get_adjusted_main_loop_efficiency(problem.a_transpose,
                                                                      problem.b_transpose,
                                                                      config.mt.m,
                                                                      config.mt.n,
                                                                      config.mt.k,
                                                                      problem.mi_dtype);
  }
  // 4-2) Single-tile latency (apply penalty after finding the bottleneck)
  double L_tile_single =
      (std::max(L_compute, L_mem) * main_loop_efficiency * effective_tile_penalty) + L_cvt;
  // 4-3) Number of K-iterations
  const long k_per_split = static_cast<long>(math::safe_ceil_div(K, splitting_factor));
  long num_iter =
      std::max(static_cast<long>(math::safe_ceil_div(static_cast<size_t>(k_per_split), MT_K) - 1),
               static_cast<long>(1));
  // 4-4) Main loop latency
  double L_main_loop = L_tile_single * static_cast<double>(num_iter);

  // 5) Epilogue: writes from all active CUs with limited bandwidth
  double L_epilogue = compute_epilogue_latency(problem, hardware, config, context);

  // 6) Adjustments
  // 6-1) TailLoop: Zero Padding in the K dimension on last iteration
  if (K % MT_K != 0) {
    const double problem_k_quant = static_cast<double>(K % MT_K) / static_cast<double>(K);
    // Scale by remainder proportion of problem.
    // 50k cycle penalty if have to zero pad all except 1.
    L_epilogue += problem_k_quant * 50000;
  }

  // 7) Total tile latency
  double L_tile_total = L_WG_setup + L_prologue + L_main_loop + L_epilogue;

  return L_tile_total;
}

// Compute the latency for a single timestep
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context) {
  // Assume latency of a timestep is latency of a single K-complete output tile computed on one CU.
  double L_timestep = compute_tile_latency(problem, hardware, config, context);

  return L_timestep;
}

// Compute the latency for the parallel reduction kernel
double compute_parallel_reduction_latency(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          const context_t& context) {
  // Models a separate reduction kernel that:
  // 1. Reads all partials for each tile (splitting_factor × tile_elements)
  // 2. Reduces using log2(N) tree approach
  // 3. Writes final output
  // This kernel launches num_tiles WGs, processed in ceil(num_tiles/N_CU) timesteps.

  // Extract parameters
  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  const size_t M     = problem.size.m;
  const size_t N     = problem.size.n;
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  const auto d_bytes = data_type_to_bytes(problem.d_dtype);

  const size_t grid_m                  = context.grid_m;
  const size_t grid_n                  = context.grid_n;
  const size_t tile_elements           = context.tile_elements;
  const size_t splitting_factor        = context.splitting_factor;
  const reduction_t reduction_strategy = context.reduction_strategy;

  if (splitting_factor == 1 || reduction_strategy != reduction_t::parallel) return 0.0;

  // Constants
  constexpr size_t threads_per_wave              = 64;
  constexpr double kernel_launch_overhead_cycles = 10000;
  const double cycles_per_second                 = hardware.compute_clock_ghz * 1e9;

  // Common setup
  const size_t num_tiles = grid_m * grid_n * splitting_factor;
  const size_t tile_elements_128bytes =
      round_elements_to_128B(MT_M, datatype_to_bits(problem.d_dtype)) * MT_N;

  // Launch overhead
  double L_launch = kernel_launch_overhead_cycles / cycles_per_second;

  // Reduction kernel has num_tiles WGs (1 WG per output tile)
  size_t reduction_active_cus = std::min(num_tiles, static_cast<size_t>(hardware.N_CU));
  double mem_bw               = compute_mem_bw_from_occupancy(hardware, reduction_active_cus);
  double mem_bw_limited       = hardware.mem3_perf_ratio * mem_bw;

  // READ: splitting_factor partials per tile
  double read_bytes_per_tile = splitting_factor * tile_elements_128bytes * d_bytes;
  double L_read              = read_bytes_per_tile / mem_bw_limited;

  // COMPUTE: log2(splitting_factor) reduction steps
  size_t reduction_steps =
      static_cast<size_t>(std::ceil(std::log2(static_cast<double>(splitting_factor))));
  double adds_per_tile_cycles = tile_elements * reduction_steps / threads_per_wave;
  double L_compute            = adds_per_tile_cycles / cycles_per_second;

  // WRITE: 1 output tile (output dtype)
  double write_bytes_per_tile = tile_elements_128bytes * d_bytes;
  double L_write              = write_bytes_per_tile / mem_bw_limited;

  double L_total = L_launch + L_read + L_compute + L_write;

  return L_total;
}

// Compute the total latency for a problem
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus) {
  assert(config.is_valid());

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
    // Use Dot2 only for M < 3
    bool isDot2 = MI_M == 1 && MI_N == 1 && MI_K == 64;
    if (isDot2 && M > 2) return std::numeric_limits<double>::max();

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

  // 0. Setup
  context_t context(problem, hardware, config);

  // 1. Compute latency of a timestep
  double L_timestep = compute_timestep_latency(problem, hardware, config, context);

  // 2. Compute latency for all timesteps and return it as the latency for the MT/problem
  double total_latency = L_timestep * context.num_timesteps;

  // 3. Add parallel reduction kernel latency (if applicable)
  total_latency += compute_parallel_reduction_latency(problem, hardware, config, context);

  // 4. Customized heuristics
  // These are quantifying effects that don't work in the current math.
  // THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
  bool heuristics = get_runtime_options(config).heuristics_enabled;

  if (heuristics) {
    if (MT_M == 64 && MT_N == 32 && MT_K == 32 && !b_trans && a_bits == 16) {
      total_latency = total_latency * 10;
    }

    bool tf32_emu = ((problem.mi_dtype == data_type_t::XFloat32) &&
                     (hardware.arch == hardware_t::architecture_t::gfx950));

    //  Heuristics for TF32
    if (tf32_emu) {
      double bytes_per_element = a_bytes;
      double arith             = emulated_tf32_arithmetic_intensity(M, N, K, bytes_per_element);
      double compute_threshold = 1000;  // threshold empirically determined.

      // The kernel for this is more optimized (Custom kernel NT)
      if ((!a_trans && b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
        if (arith < compute_threshold)
          total_latency = total_latency * 0.6;
        else
          total_latency = total_latency * 0.4;
      }

      // The kernel for this is more optimized (Custom kernel NN)
      if ((!a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
        if (arith < compute_threshold)
          total_latency = total_latency * 0.8;
        else
          total_latency = total_latency * 0.4;
      }

      // The kernel for this is more optimized (Custom kernel TN)
      if ((a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
        if (arith < compute_threshold)
          total_latency = total_latency * 0.8;
        else
          total_latency = total_latency * 0.4;
      }

      // Bias large DU where K-dimension is large and M and N are small.
      if ((K >= (M * 16) && K >= (N * 16)) && (MT_K >= 128)) {
        total_latency = total_latency * 0.5;
      }
    }
  }

  return total_latency;
}

}  // namespace origami
