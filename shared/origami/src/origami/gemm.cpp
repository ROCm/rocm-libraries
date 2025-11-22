// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
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
auto lds128_penalty(size_t dim_elems, size_t element_bits) {
  const size_t bytes = dim_elems * math::safe_ceil_div(element_bits, 8);
  const size_t mod   = bytes % 128;
  if (mod == 0) return 1.0;
  const double frac = double(mod) / 128.0;                // 0..1
  const double base = (element_bits <= 16) ? 1.1 : 1.35;  // BF16/FP16 < FP32
  return 1.0 + base * frac;                               // up to ~2.35x worst case
}

double calculate_work_utilization(const problem_t& problem, const config_t& config) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  if (M == 0 || N == 0 || K == 0 || MT_M == 0 || MT_N == 0 || MT_K == 0) return 1.0;

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

  if (M == 0 || N == 0 || MT_M == 0 || MT_N == 0) return 1.0;

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

// Computes the number of active compute units if there is only one wave and it is partial
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

  size_t num_wgs, num_active_cus, numWaves, splitFactor;

  if (split)  // if it is given
  {
    split          = split > 1 ? split : 1;
    num_wgs        = num_mts * split;
    num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
    numWaves       = math::safe_ceil_div(num_wgs, hardware.N_CU);
    splitFactor    = split;

    if (hardware_t::is_debug_enabled()) { hardware.log_debug("reduction type", "Origami"); }
  } else  // as what StreamK predicts
  {
    reduction_t rt = streamk::select_reduction(problem, hardware, config, grid_selection);

    num_wgs = streamk::select_grid_size(problem, hardware, config, grid_selection, max_cus);

    // output variables
    num_active_cus = num_wgs < hardware.N_CU ? num_wgs : hardware.N_CU;
    // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
    // That means, we artifically observe one full wave, but that is not what actually happens
    // under the hood. From a theoretical point of view, these distributions change all of the
    // computations in Origami. With current implementation, it is hard to capture that
    // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
    // numWaves based on the num_mts. Otherwise, we use num_wgs to compute numWaves.
    numWaves    = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, hardware.N_CU)
                                    : math::safe_ceil_div(num_mts, hardware.N_CU);
    splitFactor = math::safe_ceil_div(num_wgs, num_mts);
  }

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("num_mts", num_mts);
    hardware.log_debug("num_wgs", num_wgs);
    hardware.log_debug("num_active_cus", num_active_cus);
    hardware.log_debug("numWaves", numWaves);
    hardware.log_debug("splitFactor", splitFactor);
    hardware.log_debug("max_cus", max_cus);
  }

  return std::make_tuple(num_wgs, num_active_cus, numWaves, splitFactor);
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

  return numerator / denominator;
}

// Computes Emulated arithmetic intensity for TF32 (assumes 3xBF16).
double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  // Numerator: 3.0 * 2.0 * m * n * k
  // Denominator: (m*n + n*k + m*k) * bytes_per_element
  double numerator   = 3.0 * 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;

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
  // per wave tile

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

  // TODO: Use kernel's actual wavetiles.
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
static inline double compute_cvt_overhead(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config) {
  // Wave tile sizes
  // TODO: Use kernel's actual wavetiles.
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
  int a_bytes = data_type_to_bytes(problem.a_dtype);
  int b_bytes = data_type_to_bytes(problem.b_dtype);

  const double bytesA =
      static_cast<double>(wave_tile_m) * config.mt.k * math::safe_ceil_div(a_bytes, 8);
  const double bytesB =
      static_cast<double>(wave_tile_n) * config.mt.k * math::safe_ceil_div(b_bytes, 8);

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
  size_t a_loads_in_bytes = mt.mk() * data_type_to_bytes(a_dtype);
  size_t b_loads_in_bytes = mt.nk() * data_type_to_bytes(b_dtype);
  // Size of those in bytes
  size_t LDS_usage = a_loads_in_bytes + b_loads_in_bytes;

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
  const size_t effective_cus = math::safe_ceil_div(concurrent_workgroups, splitting_factor);
  const size_t cu_per_xcd =
      std::max(math::safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

  // Initial guess for the L2 tile dimensions (a tile of workgroups).
  size_t WGM = std::ceil(std::sqrt(hardware.N_CU / hardware.NUM_XCD));
  size_t l2_tile_n = std::min(static_cast<size_t>(WGM), workgroups_n);
  size_t l2_tile_m = math::safe_ceil_div(cu_per_xcd, l2_tile_n);

  // Handle wrap-around case: if the tile is taller than the grid, wrap it to be wider.
  if (l2_tile_m > workgroups_m) {
    size_t num_wraps = (l2_tile_m / workgroups_m);
    l2_tile_n += (num_wraps * WGM);
    l2_tile_m = workgroups_m;
  }

  // Clamp initial tile dimensions to the actual grid size.
  l2_tile_m = std::max(std::min(workgroups_m, l2_tile_m), static_cast<size_t>(1));
  l2_tile_n = std::max(std::min(workgroups_n, l2_tile_n), static_cast<size_t>(1));

  // Calculate memory footprint in bytes.
  const size_t a_bytes     = static_cast<size_t>(data_type_to_bytes(problem.a_dtype));
  const size_t b_bytes     = static_cast<size_t>(data_type_to_bytes(problem.b_dtype));
  auto calculate_footprint = [&](size_t tile_m, size_t tile_n) {
    size_t a_footprint = tile_m * config.mt.mk() * a_bytes;
    size_t b_footprint = tile_n * config.mt.nk() * b_bytes;
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
  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("L2Tile_M", l2_tile_m);
    hardware.log_debug("L2Tile_N", l2_tile_n);
    hardware.log_debug("TotalWorkgroups", total_workgroups);
    hardware.log_debug("ConcurrentWorkgroups", concurrent_workgroups);
  }

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
  size_t WGM = std::ceil(std::sqrt(hardware.N_CU / hardware.NUM_XCD));
  size_t mall_tile_m =
      math::safe_ceil_div(num_active_cus, static_cast<size_t>(WGM));
  size_t mall_tile_n = std::min(static_cast<size_t>(WGM), workgroups_n);

  // Handle wrap-around case if the tile is taller than the grid.
  if (mall_tile_m > workgroups_m) {
    size_t num_wraps = mall_tile_m / workgroups_m;
    mall_tile_n += (num_wraps * WGM);
    mall_tile_m = workgroups_m;
  }

  // Clamp initial tile dimensions to the actual grid size.
  mall_tile_m = std::max(std::min(workgroups_m, mall_tile_m), static_cast<size_t>(1));
  mall_tile_n = std::max(std::min(workgroups_n, mall_tile_n), static_cast<size_t>(1));

  // --- CRITICAL: Shrink tile to fit into MALL Capacity ---
  const size_t a_bytes = static_cast<size_t>(data_type_to_bytes(problem.a_dtype));
  const size_t b_bytes = static_cast<size_t>(data_type_to_bytes(problem.b_dtype));

  auto calculate_footprint = [&](size_t tile_m, size_t tile_n) {
    size_t a_footprint = tile_m * config.mt.mk() * a_bytes;
    size_t b_footprint = tile_n * config.mt.nk() * b_bytes;
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

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("MallTile_M", mall_tile_m);
    hardware.log_debug("MallTile_N", mall_tile_n);
    hardware.log_debug("MallFootprint_Bytes", calculate_footprint(mall_tile_m, mall_tile_n));
  }

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
  const double a_bytes = static_cast<double>(data_type_to_bytes(problem.a_dtype));
  const double b_bytes = static_cast<double>(data_type_to_bytes(problem.b_dtype));

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
  const size_t transaction_bits = 128u * 8u;  // 1024
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
  // Extract parameters from structured types
  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  const auto a_bits  = data_type_to_bits(problem.a_dtype);
  const auto b_bits  = data_type_to_bits(problem.b_dtype);
  size_t batch       = problem.batch;

  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  // 1) Estimate L2 hit-rate
  double H_mem1 = estimate_l2_hit(problem, hardware, config, splitting_factor);

  // Global cap on L2 hit-rate (prevents impossible cache residency claims)
  // (Assumes capacity is given in KiB, convert to bytes)
  double H_mem1_global =
      compute_l2_hit_rate_global(problem, hardware, config, hardware.L2_capacity * 1024);

  H_mem1 = std::min(H_mem1, H_mem1_global);

  if (H_mem1 == 0) { H_mem1 = 0.5; }

  // 2) Estimate mall hit-rate
  double H_mem2 = estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor);

  // 3) Total loads are loads from A and loads from B
  size_t MT_M_rounded_128bytes = round_elements_to_128B(MT_M, a_bits);
  size_t MT_N_rounded_128bytes = round_elements_to_128B(MT_N, a_bits);
  size_t MT_K_rounded_128bytes = round_elements_to_128B(MT_K, a_bits);

  if (!a_trans && !b_trans) {
    MT_N_rounded_128bytes = MT_N;
    MT_K_rounded_128bytes = MT_K;
  } else if (a_trans && !b_trans) {
    MT_M_rounded_128bytes = MT_M;
    MT_N_rounded_128bytes = MT_N;
  } else if (!a_trans && b_trans) {
    MT_K_rounded_128bytes = MT_K;
  }

  size_t Ld_A_value  = MT_M_rounded_128bytes * MT_K_rounded_128bytes;
  size_t Ld_B_value  = MT_N_rounded_128bytes * MT_K_rounded_128bytes;
  size_t Ld_CU_bytes = (Ld_A_value * static_cast<size_t>(a_bytes))     // A Bytes
                       + (Ld_B_value * static_cast<size_t>(b_bytes));  // B Bytes

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
  double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);

  // 8) loads that reach each level
  double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
  double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

  // 9) enforce whole‐problem minimum loads when we can fit M/N in the CUs.
  // Calculate the tile of workgroups that can run concurrently (logic from estimate_mall_hit).
  size_t WGM = std::ceil(std::sqrt(hardware.N_CU / hardware.NUM_XCD));
  size_t grid_m = math::safe_ceil_div(problem.size.m, MT_M);
  size_t grid_n = math::safe_ceil_div(problem.size.n, MT_N);
  size_t mall_m =
      math::safe_ceil_div(num_active_cus, static_cast<size_t>(WGM));
  size_t mall_n = std::min(static_cast<size_t>(WGM), grid_n);
  // Handle wrap-around case
  if (mall_m > grid_m) {
    size_t num_wraps = (mall_m / grid_m);
    mall_n += (num_wraps * WGM);
    mall_m = grid_m;
  }
  // Clamp tile dimensions
  mall_m = std::max(std::min(grid_m, mall_m), static_cast<size_t>(1));
  mall_n = std::max(std::min(grid_n, mall_n), static_cast<size_t>(1));
  // This is the minimum unique bytes needed from HBM to feed the concurrent workgroups.
  double min_load = static_cast<double>((mall_m * config.mt.mk() * static_cast<size_t>(a_bytes)) +
                                        (mall_n * config.mt.nk() * static_cast<size_t>(b_bytes))) *
                    batch;  // Apply batching to the minimum load itself.
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

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("mem1_perf_ratio", hardware.mem1_perf_ratio);
    hardware.log_debug("mem2_perf_ratio", hardware.mem2_perf_ratio);
    hardware.log_debug("mem3_perf_ratio", hardware.mem3_perf_ratio);
    hardware.log_debug("mem_bw_per_wg_coefficients(0)",
                       std::get<0>(hardware.mem_bw_per_wg_coefficients));
    hardware.log_debug("mem_bw_per_wg_coefficients(1)",
                       std::get<1>(hardware.mem_bw_per_wg_coefficients));
    hardware.log_debug("mem_bw_per_wg_coefficients(2)",
                       std::get<2>(hardware.mem_bw_per_wg_coefficients));
    hardware.log_debug("H_mem1 (mem1 hit ratio)", H_mem1);
    hardware.log_debug("H_mem2 (mem2 hit ratio)", H_mem2);
    hardware.log_debug("Total Load (bytes)", total_Ld);
    hardware.log_debug("Ld_mem2 (bytes)", Ld_mem2);
    hardware.log_debug("Ld_MEM (bytes)", Ld_MEM);
    hardware.log_debug("L_mem_mem1 (cycles)", L_mem_mem1);
    hardware.log_debug("L_mem_mem2 (cycles)", L_mem_mem2);
    hardware.log_debug("L_mem_MEM (cycles)", L_mem_MEM);
    hardware.log_debug("MT_K % 128 bytes", MT_K * static_cast<size_t>(b_bytes) % 128);
    hardware.log_debug("MT_M % 128 bytes", MT_M * static_cast<size_t>(a_bytes) % 128);
    hardware.log_debug("MT_N % 128 bytes", MT_N * static_cast<size_t>(b_bytes) % 128);
    hardware.log_debug(
        "MT_N % 128 + MT_M % 128 bytes",
        (MT_M * static_cast<size_t>(a_bytes) % 128) + MT_N * static_cast<size_t>(b_bytes) % 128);
    hardware.log_debug(
        "MT_N % 64 + MT_M % 64 bytes",
        (MT_M * static_cast<size_t>(a_bytes) % 64) + MT_N * static_cast<size_t>(b_bytes) % 64);
    hardware.log_debug("MT_K % 64 bytes", MT_K * static_cast<size_t>(b_bytes) % 64);
    hardware.log_debug("MT_M % 64 bytes", MT_M * static_cast<size_t>(a_bytes) % 64);
    hardware.log_debug("MT_N % 64 bytes", MT_N * static_cast<size_t>(b_bytes) % 64);
    hardware.log_debug("Tile Arithmetic Intensity",
                       MT_M * MT_N * MT_K / (MT_M * MT_K + MT_N * MT_K));
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
  // Extract parameters from structured types
  const size_t K = problem.size.k;
  size_t batch   = problem.batch;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const auto a_bits    = data_type_to_bits(problem.a_dtype);
  const auto b_bits    = data_type_to_bits(problem.b_dtype);
  const size_t a_bytes = static_cast<size_t>(data_type_to_bytes(problem.a_dtype));
  const size_t d_bytes = static_cast<size_t>(data_type_to_bytes(problem.d_dtype));

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

  // 3) Prologue: 2.2× memory latency
  double L_prologue = 1.5 * L_mem;  // 1.5 chosen emprically

  // L_compute *= std::max(L_compute, L_LDS);

  // 4) Epilogue: writes from all active CUs with limited bandwidth
  double mem_bw_occ            = compute_mem_bw_from_occupancy(hardware, num_active_cus);
  double mem_bw_occ_limited    = hardware.mem3_perf_ratio * mem_bw_occ;
  size_t MT_M_rounded_128bytes = round_elements_to_128B(MT_M, data_type_to_bits(problem.a_dtype));

  double L_epilogue = (static_cast<double>(num_active_cus / splitting_factor) *
                       MT_M_rounded_128bytes * MT_N * static_cast<double>(d_bytes)) /
                      mem_bw_occ_limited;
  // One compute iteration happens in the prologue
  L_epilogue += L_compute * effective_tile_penalty;
  // Epilogue and Prologue overhead are reduced with higher occupancy kernels.
  int grid_m = static_cast<int>(math::safe_ceil_div(problem.size.m, MT_M));
  int grid_n = static_cast<int>(math::safe_ceil_div(problem.size.n, MT_N));

  size_t real_occupancy =
      std::min(std::max(config.occupancy,static_cast<int>(1)),
               static_cast<int>(math::safe_ceil_div(grid_m * grid_n * batch * splitting_factor,
                                                    hardware.N_CU)));  // Number of WGs per CU.

  L_prologue = L_prologue * pow(0.95, real_occupancy);  // Factor chosen empirically
  L_epilogue = L_epilogue * pow(0.95, real_occupancy);  // Factor chosen empirically
  // 4') K-split reductions are globally coherent, we need to write and read split-1 MT_M*MT_N
  // tiles to coherent memory
  if (splitting_factor > 1) {
    size_t n_partials = splitting_factor - 1;

    // Only the reduction CU reads from all splits.
    double partial_read_bytes =
        grid_m * grid_n * n_partials * MT_M_rounded_128bytes * MT_N * static_cast<double>(d_bytes);

    // All CUs write (once for each partial, and once by the reduction CU for the output.)
    double partial_write_bytes =
        grid_m * grid_n * MT_M_rounded_128bytes * MT_N * static_cast<double>(d_bytes);

    double partial_readwrite_bytes = partial_read_bytes + partial_write_bytes;

    // 64 Threads active in a SIMD. Exposed to at least latency of reducing splitting_factor
    // tiles.
    double partial_adds =
        (static_cast<double>(config.mt.mn()) * static_cast<double>(splitting_factor)) / (64);
    // Things have to be written to memory
    double mem_bw_occ         = compute_mem_bw_from_occupancy(hardware, num_active_cus);
    double mem_bw_occ_limited = hardware.mem3_perf_ratio * mem_bw_occ;

    double L_reduce = partial_readwrite_bytes / (mem_bw_occ_limited);
    L_epilogue += L_reduce + partial_adds + 10000;
  }
  // 4'') tf32 emu has some more overhead
  double L_cvt = 0;
  if ((problem.mi_dtype == data_type_t::XFloat32) &&
      (hardware.arch == hardware_t::architecture_t::gfx950)) {
    L_cvt = compute_cvt_overhead(problem, hardware, config);
  } else if ((a_bits == 32) && (b_bits == 32) && (problem.mi_dtype == data_type_t::BFloat16) &&
             (hardware.arch == hardware_t::architecture_t::gfx950))  // SS_BSS on GFX950
  {
    L_cvt = compute_cvt_overhead_x1(problem, hardware, config);
  }

  // 5) Single-tile latency (always additive)
  // Calculate the fraction of the work that is useful (not padding).

  // 5) Single-tile latency (apply penalty after finding the bottleneck)
  double L_tile_single = (std::max(L_compute, L_mem) * effective_tile_penalty) + L_cvt;
  L_prologue *= effective_tile_penalty;
  // 6) Number of K-iterations (excluding epilogue), at least 1
  // long num_iter = static_cast<long>(((K + MT_K - 1) / MT_K)) - 1;
  // num_iter      = std::ceil(num_iter / splitting_factor);
  // num_iter      = std::max(num_iter, 1L);
  const long k_per_split = static_cast<long>(math::safe_ceil_div(K, splitting_factor));
  long num_iter =
      std::max(static_cast<long>(math::safe_ceil_div(static_cast<size_t>(k_per_split), MT_K) - 1),
               static_cast<long>(1));
  // Zero Padding in the K dimension on last iteration
  if (K % MT_K != 0) {
    const double problem_k_quant = static_cast<double>(K % MT_K) / static_cast<double>(K);
    L_epilogue += problem_k_quant * 50000;  // Scale by remainder proportion of problem. 50k cycle
                                            // penalty if have to zero pad all except 1.
                                            //(Scale Determined Empirically)
  }
  // L_epilogue *= output_utilization_penalty;

  // 7) Total tile latency
  double L_tile_total =
      (L_tile_single * static_cast<double>(num_iter)) + L_prologue + L_epilogue * 2 + L_WG_setup +
      (500 * static_cast<double>(
                 num_iter));  // 7 instructions (each with 4 cycles) at the end of the loop

  if (MT_K == 1024) { L_prologue = L_prologue * 100; }

  if (hardware_t::is_debug_enabled()) {
    double problem_k_quant = ((K % MT_K) / (double)K);
    hardware.log_debug("Iteration Compute Latency", L_compute);
    hardware.log_debug("L_mem", L_mem);
    hardware.log_debug("L_cvt", L_cvt);
    hardware.log_debug("L_tile_single", L_tile_single);
    hardware.log_debug("num_iter", num_iter);
    hardware.log_debug("L_prologue", L_prologue);
    hardware.log_debug("L_epilogue", L_epilogue);
    hardware.log_debug("L_tile_total", L_tile_total);
    hardware.log_debug("Effective Tile Peanlty", effective_tile_penalty);
    hardware.log_debug("Problem K quant", problem_k_quant);
    hardware.log_debug("K quant overhead", (problem_k_quant * 50000));
    hardware.log_debug("Problem Tiile Quant", utilization);
    hardware.log_debug("Real Occupancy", utilization);
    hardware.log_debug("Output Utilization Penalty", output_utilization_penalty);
    hardware.log_debug("Output Utilization", output_utilization);
    std::string bound_source;
    if (L_compute >= L_mem) {
      L_tile_single = L_compute + L_cvt;
      bound_source  = "Compute";
    } else {
      L_tile_single = L_mem + L_cvt;
      bound_source  = "Memory";
    }
    hardware.log_debug("Iteration Bound",
                       bound_source + " (" + std::to_string(L_tile_single) + ")");

    hardware.log_debug("K % MT_K", K % MT_K);
  }

  return L_tile_total;
}

// Computes the latency per K-complete MT wave
// A wave is defined as : The time it takes for one CU to complete one K-complete output tile
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                size_t num_active_cus,
                                size_t splitting_factor) {
  // Assume latency of a wave is latency of a single k-complete output tile.
  double L_wave = compute_tile_latency(problem, hardware, config, num_active_cus, splitting_factor);

  return L_wave;
}

// Compute the total latency of a gemm based on the latency of one wave multiplied by the number of
// waves A wave is defined as : The time it takes for one CU to complete one K-complete output tile
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus) {
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

  const int a_bits  = data_type_to_bits(problem.a_dtype);
  const int b_bits  = data_type_to_bits(problem.b_dtype);
  const int a_bytes = data_type_to_bytes(problem.a_dtype);
  const int d_bytes = data_type_to_bytes(problem.d_dtype);

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug(
        "Problem_Size",
        std::to_string(int(M)) + "x" + std::to_string(int(N)) + "x" + std::to_string(int(K)));
    hardware.log_debug("Macro_Tile",
                       std::to_string(int(MT_M)) + "x" + std::to_string(int(MT_N)) + "x" +
                           std::to_string(int(MT_K)));
    hardware.log_debug("Element Size A (bits)", a_bits);
    hardware.log_debug("Element Size B (bits)", b_bits);
  }

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

    // Override dot2 instruction with vector lane widths
    if (MI_N == 0 && MI_M == 0 && MI_K == 0) {
      // We only use Dot2 for NN layout where M < 3
      if (M > 2 || a_trans || b_trans) return std::numeric_limits<double>::max();
      MI_M = 1;
      MI_N = 1;
      MI_K = 64;
    }

    if (batch == 1) {
      size_t K_mod_128bytes    = K * a_bytes % 128;
      size_t MT_K_mod_128bytes = MT_K * a_bytes % 128;
      if (K_mod_128bytes == 0 && MT_K_mod_128bytes == 0 && batch == 1) {
        // avoid division by 0 if K == 0
        if (M <= MT_M * 2 && !b_trans && ((N * b_bits) / (M * a_bits) > 5)) {
          // Use nontemporal B
          if (!(config.non_temporal_b == 4)) { return std::numeric_limits<double>::max(); }
        } else if (N <= MT_N * 2 && a_trans && ((M * a_bits) / (N * b_bits) > 5)) {
          // Use Non Temporal A
          if (!(config.non_temporal_a == 4)) { return std::numeric_limits<double>::max(); }
        } else {
          // Never use Non Temporal
          if (config.non_temporal_a || config.non_temporal_b) {
            return std::numeric_limits<double>::max();
          }
        }
      } else if (config.non_temporal_a || config.non_temporal_b) {
        return std::numeric_limits<double>::max();
      }
    }
  }

  // 1-1) config.workgroup_mapping
  // auto WGM = std::max(config.workgroup_mapping, 1);  // WGM can't be less than one.

  // 1-2) Find CU occupancy
  auto [num_wgs, num_active_cus, numWaves, splitting_factor] =
      compute_cu_occupancy(problem, hardware, config, grid_selection_t::k_split_aware, max_cus);

  // 2) Compute latency of a wave
  // Compute latency of a wave
  double L_wave =
      compute_timestep_latency(problem, hardware, config, num_active_cus, splitting_factor);

  // Compute latency for all waves and return it as the latency for the MT/problem
  double total_latency = L_wave * numWaves;

  if (MT_M == 64 && MT_N == 32 && MT_K == 32 && !b_trans && a_bits == 16) {
    total_latency = total_latency * 10;
  }

  // 3) Customized heuristics
  // TODO These are quantifying effects that don't work in the current math.
  // TODO THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
  bool heuristics = hardware_t::is_heuristics_enabled();

  const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS");
  heuristics      = !(env && std::string(env) == "0");

  // heuristics = 0;
  //  Heuristics for TF32
  bool tf32_emu = ((problem.mi_dtype == data_type_t::XFloat32) &&
                   (hardware.arch == hardware_t::architecture_t::gfx950));
  if (tf32_emu && heuristics) {
    double bytes_per_element = static_cast<double>(a_bytes);
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
    if ((K >= (M * 16) && K >= (N * 16)) && (MT_K >= 128)) { total_latency = total_latency * 0.5; }
  }

  if (hardware_t::is_debug_enabled()) {
    hardware.log_debug("Total_latency (with heuristics)", total_latency);
    hardware.log_debug("non_temporal_a", config.non_temporal_a);
    hardware.log_debug("non_temporal_b", config.non_temporal_b);
    hardware.log_debug("kernel_occupancy", config.occupancy);
    hardware.log_debug("splitting_factor", splitting_factor);
    hardware.log_debug("Input Tile Size A", MT_M * MT_K);
    hardware.log_debug("Input Tile Size B", MT_N * MT_K);
    hardware.log_debug("Output Tile Size", MT_M * MT_N);
    hardware.log_debug("Tile M/N", MT_M / MT_N);
    hardware.log_debug("Tile N/M", MT_N / MT_M);
    hardware.log_debug("Problem M/N", MT_M / MT_N);
    hardware.log_debug("Problem N/M", MT_N / MT_M);
    size_t occupancy_percent = num_active_cus / hardware.N_CU;
    hardware.log_debug("Peak theoretical GFLOPs based on occupancy", 1300 * occupancy_percent);
    if (hardware_t::is_debug_enabled()) { hardware.print_debug_info(); }
  }

  return total_latency;
}

// Extract analytical metrics from a GEMM computation and return as map
std::unordered_map<std::string, std::string> extract_analytical_metrics(const hardware_t& hardware,
                                                                        const problem_t& problem,
                                                                        const config_t& config) {
  // Clear any previous debug info
  hardware.clear_debug();

  // Enable metrics collection mode to bypass environment variable check
  hardware.set_metrics_collection_mode(true);

  // Extract basic parameters for forced logging
  size_t M                = problem.size.m;
  size_t N                = problem.size.n;
  size_t K                = problem.size.k;
  size_t batch            = problem.batch;
  bool a_trans            = problem.a_transpose == transpose_t::T;
  bool b_trans            = problem.b_transpose == transpose_t::T;
  size_t element_size_A   = data_type_to_bits(problem.a_dtype);
  size_t element_size_B   = data_type_to_bits(problem.b_dtype);
  size_t element_size_out = data_type_to_bits(problem.d_dtype);
  data_type_t mi_datatype = problem.mi_dtype;
  size_t a_mx_block_size  = problem.a_mx_block_size;
  size_t b_mx_block_size  = problem.b_mx_block_size;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;
  size_t MT_K = config.mt.k;
  size_t MI_M = config.mi.m;
  size_t MI_N = config.mi.n;
  size_t MI_K = config.mi.k;
  int WGM     = config.workgroup_mapping;

  // Log basic problem and configuration parameters
  hardware.log_debug(
      "problem_size",
      std::to_string(int(M)) + "x" + std::to_string(int(N)) + "x" + std::to_string(int(K)));
  hardware.log_debug("macro_tile",
                     std::to_string(int(MT_M)) + "x" + std::to_string(int(MT_N)) + "x" +
                         std::to_string(int(MT_K)));
  hardware.log_debug("micro_tile",
                     std::to_string(int(MI_M)) + "x" + std::to_string(int(MI_N)) + "x" +
                         std::to_string(int(MI_K)));
  hardware.log_debug("element_size_a_bits", element_size_A);
  hardware.log_debug("element_size_b_bits", element_size_B);
  hardware.log_debug("element_size_out_bits", element_size_out);
  hardware.log_debug("batch_size", batch);
  hardware.log_debug("a_transpose", a_trans ? "true" : "false");
  hardware.log_debug("b_transpose", b_trans ? "true" : "false");
  hardware.log_debug("workgroup_mapping", WGM);
  hardware.log_debug("a_mx_block_size", a_mx_block_size);
  hardware.log_debug("b_mx_block_size", b_mx_block_size);

  // Run compute_total_latency which will populate debug_info with all intermediate values
  // since we enabled metrics collection mode
  double total_latency = compute_total_latency(problem, hardware, config, hardware.N_CU);

  // Disable metrics collection mode
  hardware.set_metrics_collection_mode(false);

  // Add the final latency to metrics
  hardware.log_debug_force("final_total_latency", total_latency);

  // Return all collected metrics
  return hardware.get_analytical_metrics();
}

// Extract analytical metrics and export to CSV
void extract_analytical_metrics_csv(const hardware_t& hardware,
                                    const problem_t& problem,
                                    const config_t& config,
                                    const std::string& filename) {
  // Extract metrics
  extract_analytical_metrics(hardware, problem, config);

  // Export to CSV
  hardware.extract_analytical_metrics_csv(filename);
}

}  // namespace origami
