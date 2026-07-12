// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <tuple>

#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "origami/attention.hpp"

namespace origami {
namespace attention {

/* ======================================================================================== */
/* Utility functions                                                                        */
/* ======================================================================================== */

double calculate_work_utilization(const problem_t& problem, const config_t& config) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);
  const double launched_K =
      static_cast<double>(math::safe_ceil_div(K, MT_K)) * static_cast<double>(MT_K);

  const double useful_volume   = static_cast<double>(M * N * K);
  const double launched_volume = launched_M * launched_N * launched_K;

  if (launched_volume < 1.0) return 1.0;
  return useful_volume / launched_volume;
}

double calculate_output_utilization(const problem_t& problem,
                                    const config_t& config,
                                    size_t vector_elems = 1) {
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  if (MT_M <= 0 || MT_N <= 0) return 1.0;

  const double launched_M =
      static_cast<double>(math::safe_ceil_div(M, MT_M)) * static_cast<double>(MT_M);
  const double launched_N =
      static_cast<double>(math::safe_ceil_div(N, MT_N)) * static_cast<double>(MT_N);

  const size_t M_vec = (vector_elems > 1) ? math::safe_ceil_div(M, vector_elems) * vector_elems : M;
  const size_t N_vec = (vector_elems > 1) ? math::safe_ceil_div(N, vector_elems) * vector_elems : N;

  const double useful   = static_cast<double>(M_vec) * static_cast<double>(N_vec);
  const double launched = launched_M * launched_N;

  if (launched < 1.0) return 1.0;
  return useful / launched;
}

std::tuple<size_t, size_t, size_t, size_t> compute_cu_occupancy(const problem_t& problem,
                                                                const hardware_t& hardware,
                                                                const config_t& config,
                                                                grid_selection_t grid_selection,
                                                                size_t max_cus,
                                                                size_t split = 0) {
  OLOG_DEBUG("compute_cu_occupancy: problem size (M=" << problem.size.m
             << ", N=" << problem.size.n << ", K=" << problem.size.k
             << "), config MT (M=" << config.mt.m << ", N=" << config.mt.n
             << ", K=" << config.mt.k << ")");

  // For flash attention: total work = batch * q_heads * grid_M * grid_N tiles
  size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);
  size_t num_tiles = grid_m * grid_n * problem.batch * problem.q_heads;

  size_t split_factor = (split > 1) ? split : 1;
  size_t num_wgs = num_tiles * split_factor;

  size_t effective_cus = std::min(max_cus, hardware.N_CU);
  size_t num_active_cus = std::min(num_wgs, effective_cus);
  size_t num_timesteps = math::safe_ceil_div(num_wgs, effective_cus);

  return std::make_tuple(num_wgs, num_active_cus, num_timesteps, split_factor);
}

/* ======================================================================================== */
/* Compute-related functions                                                                */
/* ======================================================================================== */

size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi) {
  size_t num_m_instrs = math::safe_ceil_div(mt.m, mi.m);
  size_t num_n_instrs = math::safe_ceil_div(mt.n, mi.n);
  size_t num_k_instrs = math::safe_ceil_div(mt.k, mi.k);
  return num_m_instrs * num_n_instrs * num_k_instrs;
}

double arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  double numerator   = 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;
  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element) {
  double numerator   = 3.0 * 2.0 * m * n * k;
  double denominator = (m * n + n * k + m * k) * bytes_per_element;
  if (denominator == 0) return 0.0;
  return numerator / denominator;
}

size_t compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  size_t L_MI = hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);
  return L_MI * N_MI;
}

/* ======================================================================================== */
/* Memory-related functions                                                                 */
/* ======================================================================================== */

bool check_rf_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype) {
  // RF size: Q[BLOCK_M, HEAD_DIM] in F16, K and V[BLOCK_N, HEAD_DIM] in F16, O[BLOCK_M, HEAD_DIM] in F32, and P[BLOCK_M, BLOCK_N] in F32 for 2 WGs
  size_t rf_usage = 2 * (mt.mk() * data_type_to_bytes(a_dtype) +
                    hardware.parallel_mi_cu * 2 * mt.nk() * data_type_to_bytes(a_dtype) +
                    (mt.mk() + mt.mn()) * 4);
  return rf_usage <= hardware.rf_capacity;
}

bool check_lds_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype) {
  auto lds_usage = 2 * std::max(mt.mk(), mt.nk()) * data_type_to_bytes(a_dtype); // max(Q, K), V reuses the LDS space for K
  return lds_usage <= hardware.lds_capacity;
}

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
  const size_t workgroups_m     = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t workgroups_n     = math::safe_ceil_div(problem.size.n, config.mt.n);
  const size_t total_workgroups = workgroups_m * workgroups_n;

  const size_t concurrent_workgroups = std::min(total_workgroups, hardware.N_CU);
  if (concurrent_workgroups == 0)
    return 0.0;

  const size_t effective_cus =
      math::safe_ceil_div(concurrent_workgroups, splitting_factor * problem.batch);
  const size_t cu_per_xcd =
      std::max(math::safe_ceil_div(effective_cus, hardware.NUM_XCD), static_cast<size_t>(1));

  size_t l2_tile_n = std::min(static_cast<size_t>(std::max(config.workgroup_mapping, 1)), workgroups_n);
  size_t l2_tile_m = math::safe_ceil_div(cu_per_xcd, l2_tile_n);

  if (l2_tile_m > workgroups_m) {
    size_t num_wraps = (l2_tile_m / workgroups_m);
    l2_tile_n += (num_wraps * std::max(config.workgroup_mapping, 1));
    l2_tile_m = workgroups_m;
  }

  l2_tile_m = std::max(std::min(workgroups_m, l2_tile_m), static_cast<size_t>(1));
  l2_tile_n = std::max(std::min(workgroups_n, l2_tile_n), static_cast<size_t>(1));

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);
  auto calculate_footprint = [&](size_t tile_m, size_t tile_n) {
    auto a_footprint = tile_m * config.mt.mk() * a_bytes;
    auto b_footprint = tile_n * config.mt.nk() * b_bytes;
    return a_footprint + b_footprint;
  };

  while (calculate_footprint(l2_tile_m, l2_tile_n) > hardware.L2_capacity) {
    if (l2_tile_m > 1 && l2_tile_m >= l2_tile_n) {
      l2_tile_m--;
    } else if (l2_tile_n > 1) {
      l2_tile_n--;
    } else {
      break;
    }
  }

  const long long uncached_A_reads     = static_cast<long long>(l2_tile_m) * config.mt.mk();
  const long long uncached_B_reads     = static_cast<long long>(l2_tile_n) * config.mt.nk();
  const long long total_uncached_reads = uncached_A_reads + uncached_B_reads;

  const long long total_A_reads = uncached_A_reads * l2_tile_n;
  const long long total_B_reads = uncached_B_reads * l2_tile_m;
  const long long total_reads   = std::max(total_A_reads + total_B_reads, 1LL);

  const long long cached_reads = total_reads - total_uncached_reads;
  double l2_hit_rate = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  return std::max(0.0, std::min(l2_hit_rate, 1.0));
}

double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         size_t num_active_cus,
                         size_t splitting_factor) {
  const size_t workgroups_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t workgroups_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (num_active_cus == 0) return 0.0;

  size_t mall_tile_m =
      math::safe_ceil_div(num_active_cus, static_cast<size_t>(std::max(config.workgroup_mapping, 1)));
  size_t mall_tile_n = std::min(static_cast<size_t>(std::max(config.workgroup_mapping, 1)), workgroups_n);

  if (mall_tile_m > workgroups_m) {
    size_t num_wraps = mall_tile_m / workgroups_m;
    mall_tile_n += (num_wraps * std::max(config.workgroup_mapping, 1));
    mall_tile_m = workgroups_m;
  }

  mall_tile_m = std::max(std::min(workgroups_m, mall_tile_m), static_cast<size_t>(1));
  mall_tile_n = std::max(std::min(workgroups_n, mall_tile_n), static_cast<size_t>(1));

  const long long uncached_A_reads     = static_cast<long long>(mall_tile_m) * config.mt.mk();
  const long long uncached_B_reads     = static_cast<long long>(mall_tile_n) * config.mt.nk();
  const long long total_uncached_reads = uncached_A_reads + uncached_B_reads;

  const long long total_A_reads = uncached_A_reads * mall_tile_n;
  const long long total_B_reads = uncached_B_reads * mall_tile_m;
  const long long total_reads   = std::max(total_A_reads + total_B_reads, 1LL);

  const long long cached_reads = total_reads - total_uncached_reads;
  double mall_hit_rate = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

  return std::max(0.0, std::min(mall_hit_rate, 1.0));
}

double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t l2_capacity_bytes) {
  if (l2_capacity_bytes == 0) return 0.0;

  const size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (grid_m == 0 || grid_n == 0) return 1.0;

  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  const double b_bytes = data_type_to_bytes(problem.b_dtype);

  const double a_working_set           = static_cast<double>(grid_m * config.mt.mk()) * a_bytes;
  const double b_working_set           = static_cast<double>(grid_n * config.mt.nk()) * b_bytes;
  const double total_working_set_bytes = a_working_set + b_working_set;

  if (total_working_set_bytes > l2_capacity_bytes) {
    return 0.1;
  }

  const double total_A_reads = static_cast<double>(grid_m * grid_n * config.mt.mk());
  const double total_B_reads = static_cast<double>(grid_m * grid_n * config.mt.nk());

  const double uncached_A_reads = static_cast<double>(grid_m * config.mt.mk());
  const double uncached_B_reads = static_cast<double>(grid_n * config.mt.nk());

  const double total_reads = total_A_reads + total_B_reads;
  if (total_reads == 0) return 1.0;

  const double cached_reads =
      (total_A_reads - uncached_A_reads) + (total_B_reads - uncached_B_reads);
  return cached_reads / total_reads;
}

inline size_t round_up_mul(size_t x, size_t m) { return (x + m - 1) / m * m; }

size_t round_elements_to_128B(size_t elements, size_t element_size_bits) {
  const size_t transaction_bits = 128u * 8u;
  const size_t g                = std::gcd(element_size_bits, transaction_bits);
  const size_t E_block          = transaction_bits / g;
  return round_up_mul(elements, E_block);
}

/* ======================================================================================== */
/* Flash Attention stage latency functions                                                  */
/* ======================================================================================== */

// VALU throughput: elements processed per cycle per CU.
// CDNA3 (gfx942/gfx950): 4 SIMDs x 16 lanes / 4-cycle issue = 16 elem/cyc
// RDNA3+ (gfx1100/gfx1201): 2 SIMDs x 32 lanes / 4-cycle issue = 16 elem/cyc
static constexpr double VALU_ELEMENTS_PER_CYCLE = 16.0;

// LDS bandwidth per CU in bytes/cycle (128B/cycle for one 128B transaction)
static constexpr double LDS_BW_BYTES_PER_CYCLE = 128.0;

// Number of workgroups per CU in the 2-WG flash attention pipeline
static constexpr size_t FA_WGS_PER_CU = 2;

/**
 * @brief Q tile load latency: load Q[MT_M x H_DIM] from HBM.
 *
 * Q tiles are unique per workgroup (no inter-WG reuse), so all loads go to HBM.
 * Latency = total bytes across all CUs / HBM bandwidth.
 */
static double q_tile_load_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t num_active_cus) {
  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  double bytes_per_cu = static_cast<double>(config.mt.m) * static_cast<double>(config.mt.k) * a_bytes;
  double total_bytes = bytes_per_cu * static_cast<double>(num_active_cus);

  double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);
  double effective_bw = hardware.mem3_perf_ratio * bw_limited;
  if (effective_bw <= 0.0) return 0.0;

  return total_bytes / effective_bw;
}

/**
 * @brief KV tile load latency: load K or V [MT_N x MT_K] through the memory hierarchy.
 *
 * KV tiles are shared across Q-row workgroups, so they benefit from L2/MALL caching.
 * Uses the multi-level cache model (L2 -> MALL -> HBM) with hit-rate estimation.
 * Adds LDS round-trip overhead (write + first read) since KV tiles go through LDS.
 */
static double kv_tile_load_latency(const problem_t& problem,
                                   const hardware_t& hardware,
                                   const config_t& config,
                                   size_t num_active_cus,
                                   size_t splitting_factor) {
  const double b_bytes = data_type_to_bytes(problem.b_dtype);

  // KV tile size per CU: MT_N x MT_K elements
  double Ld_CU_bytes = static_cast<double>(config.mt.n) * static_cast<double>(config.mt.k) * b_bytes;
  double total_Ld = Ld_CU_bytes * static_cast<double>(num_active_cus);

  // L2 hit rate
  double H_l2 = estimate_l2_hit(problem, hardware, config, splitting_factor);
  double H_l2_global =
      compute_l2_hit_rate_global(problem, hardware, config, hardware.L2_capacity * 1024);
  H_l2 = std::min(H_l2, H_l2_global);
  if (H_l2 <= 0.0) H_l2 = 0.5;

  // MALL hit rate
  double H_mall = hardware.has_MALL()
      ? estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor)
      : 0.0;

  // Bandwidth-limited factor
  double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);

  // L2 latency (all loads hit L2 first)
  double l2_bw_factor = static_cast<double>(num_active_cus) / static_cast<double>(hardware.N_CU);
  double L_l2 = (hardware.mem1_perf_ratio * l2_bw_factor > 0.0)
      ? total_Ld / (hardware.mem1_perf_ratio * l2_bw_factor)
      : 0.0;

  // Loads that miss L2
  double Ld_mall = hardware.has_MALL() ? (1.0 - H_l2) * total_Ld : 0.0;
  double Ld_dram = hardware.has_MALL() ? (1.0 - H_mall) * Ld_mall : (1.0 - H_l2) * total_Ld;

  // MALL latency
  double L_mall = (hardware.has_MALL() && hardware.mem2_perf_ratio * bw_limited > 0.0)
      ? Ld_mall / (hardware.mem2_perf_ratio * bw_limited)
      : 0.0;

  // HBM latency
  double L_dram = (hardware.mem3_perf_ratio * bw_limited > 0.0)
      ? Ld_dram / (hardware.mem3_perf_ratio * bw_limited)
      : 0.0;

  // Worst-case across memory levels
  double L_mem = std::max({L_l2, L_mall, L_dram});

  // LDS overhead: KV tiles go HBM -> LDS -> RF.
  // The write to and first read from LDS are on the critical path.
  L_mem += 2.0 * Ld_CU_bytes / LDS_BW_BYTES_PER_CYCLE;

  return L_mem;
}

/**
 * @brief WGMMA0 compute latency: S = Q * K^T.
 *
 * GEMM dimensions: M=MT_M, N=MT_N, K=MT_K (head dim tile).
 * Uses macro-tile compute latency (matrix instruction count * MI latency).
 */
static double wgmma0_compute_latency(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config) {
  // S = Q * K^T : [MT_M x MT_K] * [MT_K x MT_N] -> [MT_M x MT_N]
  // This maps directly to the standard MT compute latency
  size_t N_MI = compute_number_matrix_instructions(config.mt, config.mi);
  OLOG_DEBUG("  WGMMA0 MI: " << config.mi.m << ", " << config.mi.n << ", " << config.mi.k);
  size_t L_MI = hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);
  return static_cast<double>(L_MI) * static_cast<double>(N_MI);
}

/**
 * @brief Softmax compute latency on the score matrix S[MT_M x MT_N].
 *
 * Online softmax involves:
 *   1. Row-max reduction (MT_M * MT_N + MT_N elements)
 *   2. Exponential (MT_M * MT_N elements)
 *   3. Scale by exp(old_max - new_max) (MT_M * MT_N elements)
 *   4. Row-sum accumulation via FMA (MT_M * MT_N elements)
 * Total vector operations: ~4 * MT_M * MT_N + MT_N elements.
 */
static double softmax_compute_latency(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config) {
  double elements = static_cast<double>(config.mt.m) * static_cast<double>(config.mt.n);
  double reduction_elements = static_cast<double>(config.mt.n);

  // 4 passes over elements (max, exp, scale, fma) plus row reduction overhead
  double total_ops = 4.0 * elements + reduction_elements;
  return total_ops / VALU_ELEMENTS_PER_CYCLE;
}

/**
 * @brief WGMMA1 compute latency: O = P * V.
 *
 * GEMM dimensions: M=MT_M, N=H_DIM(=MT_K), K=K_SEQ_TILE(=MT_N).
 * The output O has shape [MT_M x H_DIM], using P[MT_M x MT_N] and V[MT_N x H_DIM].
 */
static double wgmma1_compute_latency(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config) {
  // O = P * V : [MT_M x MT_N] * [MT_N x MT_K] -> [MT_M x MT_K]
  // GEMM: M=MT_M, N=MT_K (head dim), K=MT_N (sequence tile)
  size_t num_mi_m = math::safe_ceil_div(config.mt.m, config.mi.m);
  size_t num_mi_n = math::safe_ceil_div(config.mt.k, config.mi.n);
  size_t num_mi_k = math::safe_ceil_div(config.mt.n, config.mi.k);

  size_t N_MI = num_mi_m * num_mi_n * num_mi_k;
  size_t L_MI = hardware.get_mi_latency(config.mi.m, config.mi.n, config.mi.k, problem.mi_dtype);
  return static_cast<double>(L_MI) * static_cast<double>(N_MI);
}

/**
 * @brief Output tile write latency: write O[MT_M x H_DIM] to HBM.
 *
 * Output tiles bypass the L2 cache and write directly to HBM.
 */
static double output_tile_write_latency(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config,
                                        size_t num_active_cus) {
  const double d_bytes = data_type_to_bytes(problem.d_dtype);

  // Output tile: O[MT_M x H_DIM] = O[MT_M x MT_K]
  double bytes_per_cu = static_cast<double>(config.mt.m) * static_cast<double>(config.mt.k) * d_bytes;
  double total_bytes = bytes_per_cu * static_cast<double>(num_active_cus);

  double bw_limited = compute_mem_bw_from_occupancy(hardware, num_active_cus);
  double effective_bw = hardware.mem3_perf_ratio * bw_limited;
  if (effective_bw <= 0.0) return 0.0;

  return total_bytes / effective_bw;
}

/* ======================================================================================== */
/* Memory latency (KV tile load through cache hierarchy)                                    */
/* ======================================================================================== */

double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              size_t num_active_cus,
                              size_t splitting_factor) {
  // For flash attention, memory latency is dominated by KV tile loads
  return kv_tile_load_latency(problem, hardware, config, num_active_cus, splitting_factor);
}

/* ======================================================================================== */
/* Tile and timestep latency (flash attention pipeline)                                     */
/* ======================================================================================== */

/**
 * @brief Compute the flash attention pipeline cycle latency.
 *
 * The 2-WG pipelined schedule has 5 stages (t0-t4). Each stage overlaps
 * work from two workgroups. The pipeline cycle time is the maximum stage latency.
 *
 *        prologue   t0                    t1                    t2            t3          t4          epilogue
 *       +----------+---------------------+--------------------+--------------+-----------+-----------+-----------+
 * WG 0: | [LD Q K] | [WGMMA0 + Softmax] | [LD V]            | [WGMMA1]     | [ST O]    | [LD Q K]  |           |
 * WG 1: |          | [LD Q K]            | [WGMMA0 + Softmax]| [LD V]       | [WGMMA1]  | [ST O]    | [ST O]    |
 *       +----------+---------------------+--------------------+--------------+-----------+-----------+-----------+
 */
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            size_t num_active_cus,
                            size_t splitting_factor) {
  bool debug = runtime_options::get().debug_enabled;

  // Compute individual stage latencies
  double L_ld_q    = q_tile_load_latency(problem, hardware, config, num_active_cus);
  double L_ld_kv   = kv_tile_load_latency(problem, hardware, config, num_active_cus, splitting_factor);
  double L_wgmma0  = wgmma0_compute_latency(problem, hardware, config);
  double L_softmax = softmax_compute_latency(problem, hardware, config);
  double L_wgmma1  = wgmma1_compute_latency(problem, hardware, config);
  double L_st_o    = output_tile_write_latency(problem, hardware, config, num_active_cus);

  if (debug) {
    // Compute cache hit rates for logging (same logic as in kv_tile_load_latency)
    double H_l2 = estimate_l2_hit(problem, hardware, config, splitting_factor);
    double H_l2_global = compute_l2_hit_rate_global(problem, hardware, config, hardware.L2_capacity * 1024);
    H_l2 = std::min(H_l2, H_l2_global);
    if (H_l2 <= 0.0) H_l2 = 0.5;
    double H_mall = hardware.has_MALL()
        ? estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor)
        : 0.0;

    OLOG_DEBUG("  --- Stage Latencies (loop cycle) ---");
    OLOG_DEBUG("    KV tile load: MT_N=" << config.mt.n << " MT_K=" << config.mt.k
               << " bytes=" << (config.mt.n * config.mt.k * 2)
               << " l2_hit=" << H_l2 << " mall_hit=" << H_mall
               << " latency=" << L_ld_kv << " cycles");
    OLOG_DEBUG("    WGMMA0 (Q*K^T): MT_M=" << config.mt.m << " MT_N=" << config.mt.n << " MT_K=" << config.mt.k
               << " MI=(" << config.mi.m << "," << config.mi.n << "," << config.mi.k << ")"
               << " latency=" << L_wgmma0 << " cycles");
    OLOG_DEBUG("    Softmax: elements=" << (config.mt.m * config.mt.n)
               << " latency=" << L_softmax << " cycles");
    OLOG_DEBUG("    WGMMA1 (P*V): remapped dims M=" << config.mt.m << " N=" << config.mt.k << " K=" << config.mt.n
               << " MI=(" << config.mi.m << "," << config.mi.n << "," << config.mi.k << ")"
               << " latency=" << L_wgmma1 << " cycles");
    OLOG_DEBUG("    Output write: bytes=" << (config.mt.m * config.mt.k * 2)
               << " latency=" << L_st_o << " cycles");
  }

  // Composite latencies
  double L_compute_s = L_wgmma0 + L_softmax;  // S = QK^T + softmax
  double L_load_qk   = L_ld_q + L_ld_kv;      // Load Q and K tiles (sequential)

  // Pipeline stages (each is max of two concurrent WG operations)
  double t0 = std::max(L_compute_s, L_load_qk);    // WG0: compute S, WG1: load Q+K
  double t1 = std::max(L_ld_kv, L_compute_s);       // WG0: load V,   WG1: compute S
  double t2 = std::max(L_wgmma1, L_ld_kv);          // WG0: compute O, WG1: load V
  double t3 = std::max(L_st_o, L_wgmma1);           // WG0: store O,  WG1: compute O
  double t4 = std::max(L_load_qk, L_st_o);          // WG0: load Q+K, WG1: store O

  if (debug) {
    OLOG_DEBUG("  --- Pipeline Stages ---");
    OLOG_DEBUG("    t0 (max(compute_s, load_qk)): " << t0 << " cycles");
    OLOG_DEBUG("    t1 (max(load_v, compute_s)): " << t1 << " cycles");
    OLOG_DEBUG("    t2 (max(compute_o, load_v)): " << t2 << " cycles");
    OLOG_DEBUG("    t3 (max(store_o, compute_o)): " << t3 << " cycles");
    OLOG_DEBUG("    t4 (max(load_qk, store_o)): " << t4 << " cycles");
  }

  // Pipeline cycle = max of all stage latencies (critical path)
  double loop_cycle = std::max({t0, t1, t2, t3, t4});

  if (debug) {
    OLOG_DEBUG("  Loop cycle (critical path): " << loop_cycle << " cycles");
  }

  return loop_cycle;
}

double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                size_t num_active_cus,
                                size_t splitting_factor) {
  return compute_tile_latency(problem, hardware, config, num_active_cus, splitting_factor);
}

/* ======================================================================================== */
/* Total latency                                                                            */
/* ======================================================================================== */

/**
 * @brief Compute total flash attention latency.
 *
 * Total = prologue + (loop_cycle * num_iters) + epilogue
 *
 * Where:
 *   - prologue = ld_q + ld_k (initial tile loads before pipeline starts)
 *   - loop_cycle = max(t0, t1, t2, t3, t4) (pipelined steady-state)
 *   - epilogue = st_o (final output tile write after pipeline drains)
 *   - num_iters = ceil(batch * grid_M * grid_N * grid_K / N_CU)
 */
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus) {
  assert(config.is_valid());
  bool debug = runtime_options::get().debug_enabled;

  OLOG_DEBUG("=== Attention compute_total_latency START ===");
  OLOG_DEBUG("Evaluating config: MT=(" << config.mt.m << "," << config.mt.n << "," << config.mt.k
             << ") MI=(" << config.mi.m << "," << config.mi.n << "," << config.mi.k << ")");
  OLOG_DEBUG("Problem: M=" << problem.size.m << " N=" << problem.size.n
             << " K=" << problem.size.k << " batch=" << problem.batch
             << " q_heads=" << problem.q_heads);
  /*
             << " kv_heads=" << problem.kv_heads
             << " head_dim=" << problem.head_dim);
  */
  OLOG_DEBUG("Hardware: N_CU=" << hardware.N_CU << " max_cus=" << max_cus);

  // Problem dimensions: M=Q_SEQ, N=K_SEQ, K=H_DIM
  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;
  const size_t batch = problem.batch;
  const size_t q_heads = problem.q_heads;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  // Grid dimensions (tiles in each dimension)
  const size_t grid_m = math::safe_ceil_div(M, MT_M);
  const size_t grid_n = math::safe_ceil_div(N, MT_N);
  const size_t grid_k = math::safe_ceil_div(K, MT_K);

  OLOG_DEBUG("Grid: M=" << M << "/" << MT_M << "=" << grid_m
             << " N=" << N << "/" << MT_N << "=" << grid_n
             << " K=" << K << "/" << MT_K << "=" << grid_k
             << " total_tiles=" << (grid_m * grid_n * grid_k));

  const size_t wgs_per_cu = 2;

  // CU occupancy (simplified for flash attention)
  size_t num_active_cus = std::min(max_cus, hardware.N_CU);
  size_t splitting_factor = 1;

  // --- Prologue: initial Q and K tile loads (sequential) ---
  double L_ld_q = q_tile_load_latency(problem, hardware, config, num_active_cus);
  OLOG_DEBUG("  Q tile load: MT_M=" << config.mt.m << " MT_K=" << config.mt.k
             << " bytes=" << (config.mt.m * config.mt.k * 2)
             << " latency=" << L_ld_q << " cycles");

  double L_ld_k = kv_tile_load_latency(problem, hardware, config, num_active_cus, splitting_factor);
  OLOG_DEBUG("  K tile load (prologue): MT_N=" << config.mt.n << " MT_K=" << config.mt.k
             << " bytes=" << (config.mt.n * config.mt.k * 2)
             << " latency=" << L_ld_k << " cycles");

  double L_prologue = L_ld_q + L_ld_k;
  OLOG_DEBUG("  Prologue total: " << L_prologue << " cycles");

  // --- Loop: pipelined steady-state cycle ---
  double L_loop = compute_tile_latency(problem, hardware, config,
                                           num_active_cus, splitting_factor);
  OLOG_DEBUG("  Loop cycle latency: " << L_loop << " cycles");

  // --- Epilogue: final output tile write ---
  double L_epilogue = output_tile_write_latency(problem, hardware, config, num_active_cus);
  OLOG_DEBUG("  Epilogue (output write): " << L_epilogue << " cycles");

  // --- Number of pipeline iterations ---
  // Total work = batch * grid_M * grid_N * grid_K tiles distributed across N_CU CUs
  const size_t adjustment_factor = 8; // adjusting amount of work for prolog or epilogue bound tile sizes
  double total_tiles = static_cast<double>(adjustment_factor) *
                       static_cast<double>(batch) *
                       static_cast<double>(q_heads) *
                       static_cast<double>(grid_m) *
                       static_cast<double>(grid_n) *
                       static_cast<double>(grid_k);
  double num_iters = std::ceil(total_tiles / (static_cast<double>(wgs_per_cu) * static_cast<double>(num_active_cus)));
  num_iters = std::max(num_iters, 1.0);

  // --- Total latency ---
  double total_latency = L_prologue + L_loop * num_iters + L_epilogue;

  OLOG_DEBUG("Pipeline summary: prologue=" << L_prologue << " loop=" << L_loop
             << " epilogue=" << L_epilogue << " num_iters=" << num_iters
             << " total=" << total_latency << " cycles");

  if (debug) {
    OLOG_DEBUG("======== Flash Attention Debug Info ========");
    OLOG_DEBUG("Problem: Q_SEQ=" << M << " K_SEQ=" << N << " H_DIM=" << K);
    OLOG_DEBUG("Batch: " << batch << " q_heads=" << q_heads);
    OLOG_DEBUG("Macrotile: " << MT_M << "x" << MT_N << "x" << MT_K);
    OLOG_DEBUG("Grid: " << grid_m << "x" << grid_n << "x" << grid_k);
    OLOG_DEBUG("Total tiles: " << total_tiles << " (with adjustment_factor=" << adjustment_factor << ")");
    OLOG_DEBUG("Active CUs: " << num_active_cus << " WGs per CU: " << wgs_per_cu);
    OLOG_DEBUG("L_prologue: " << L_prologue);
    OLOG_DEBUG("L_loop: " << L_loop);
    OLOG_DEBUG("L_epilogue: " << L_epilogue);
    OLOG_DEBUG("num_iters: " << num_iters);
    OLOG_DEBUG("total_latency: " << total_latency);
    OLOG_DEBUG("============================================");
  }

  return total_latency;
}

}  // namespace attention
}  // namespace origami
