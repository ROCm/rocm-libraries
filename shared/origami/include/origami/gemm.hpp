// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <utility>
#include <vector>
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Per-operand cache hit rates for L1, L2, and MALL.
 *
 * Tuple layout:
 *   (H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B)
 */
using cache_hit_rates_t = std::tuple<double, double, double, double, double, double>;

/**
 * @brief Context for kernel execution.
 *
 * Holds derived/computed values for a GEMM kernel execution.
 * This struct bundles grid dimensions, occupancy info, and other
 * values computed from problem, config, and hardware.
 */
struct context_t {
  /// Heuristic parameters (cached to avoid repeated lookups).
  heuristic_params_t heuristic{};

  /// Element sizes (fractional for sub-byte types like FP4).
  double a_bytes = 0.0;
  double b_bytes = 0.0;
  double d_bytes = 0.0;

  /// Grid dimensions.
  size_t grid_m           = 0;
  size_t grid_n           = 0;
  size_t num_output_tiles = 0;

  /// Launch parameters.
  reduction_t reduction_strategy = reduction_t::none;
  size_t splitting_factor        = 0;
  size_t num_wgs                 = 0;
  size_t num_timesteps           = 0;

  /// Hardware-derived values.
  size_t active_cus           = 0;
  double mem_bw_limited       = 0.0;
  double write_mem_bw_limited = 0.0;

  /// Tile-derived values.
  size_t k_per_split       = 0;
  size_t k_iters           = 0;
  size_t tile_elements     = 0;
  double output_tile_bytes = 0.0;

  /// Workgroup mapping parameters.
  workgroup_mapping_t wgm{0, 8, 1};

  /// Occupancy-derived decay factor.
  size_t real_occupancy   = 0;
  double occupancy_factor = 1.0;

  /// Debug flag (cached from runtime_options to avoid repeated singleton lookups).
  bool debug = false;

  /// Default constructor.
  context_t() = default;

  /**
   * @brief Constructor from config, problem, and hardware.
   *
   * @param problem Problem description (M, N, K, etc.)
   * @param hardware Hardware characteristics (@see origami::hardware_t)
   * @param config Kernel configuration.
   */
  context_t(const problem_t& problem, const hardware_t& hardware, const config_t& config);

  /**
   * @brief Check if the context is valid.
   *
   * @return bool True if the context is valid, false otherwise.
   */
  bool is_valid() const;
};

/**
 * @brief calculate the work utilization which is the ratio of the useful problem volume to the
 * total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
double calculate_work_utilization(const problem_t& problem, const config_t& config);

/**
 * @brief calculate the output utilization which is the ratio of the useful problem volume to the
 * total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param vector_elems elements in the vector.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
double calculate_output_utilization(const problem_t& problem,
                                    const config_t& config,
                                    size_t vector_elems);

/**
 * @brief This function rounds the number of elements up to the smallest value whose total size
 * (given the element bit-width) is an exact multiple of a 128-byte memory transaction.
 *
 * @param elements Macro tile dimension
 * @param element_size_bits size in bits
 * @return size_t
 */
size_t round_elements_to_128B(size_t elements, size_t element_size_bits);

/**
 * @brief Fast WGM prediction based on last-XCD L2 working set minimization.
 *
 * Evaluates a small set of WGM candidates and picks the one that minimizes
 * the L2 working set for the last XCD in the first timestep.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param splitting_factor K-split factor.
 * @return workgroup_mapping_t Predicted workgroup mapping.
 */
workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const config_t& config,
                                              size_t grid_m,
                                              size_t grid_n,
                                              size_t splitting_factor);
/**
 * @brief Computes the launch parameters for the kernel.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection Different algorithms to select the grid size for kernel execution.
 * @param max_cus maximum number of CU's
 * @return tuple<reduction_t, size_t, size_t, size_t, size_t>
 *         (reduction_strategy, num_wgs, num_active_cus, num_timesteps, split_factor)
 */
std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection,
    size_t max_cus);

/**
 * @brief Check if MT fits in LDS
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro tile dimensions
 * @param a_dtype Data type of operand A
 * @param b_dtype Data type of operand B
 * @return bool True if MT fits in LDS, false otherwise
 */
bool check_lds_capacity(const hardware_t& hardware,
                        const dim3_t& mt,
                        const data_type_t& a_dtype,
                        const data_type_t& b_dtype);

/**
 * @brief Compute limited achievable memory bandwidth based on active CUs
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param num_active_cus number of CU's
 * @return double memory bandwidth
 */
double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus);

/**
 * @brief Compute MALL tile dimensions: how many concurrent workgroup tiles fit in MALL.
 *
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param active_cus Number of active compute units.
 * @param wgm_value Workgroup mapping slab width.
 * @return std::pair<size_t, size_t> (mall_tile_m, mall_tile_n).
 */
std::pair<size_t, size_t> compute_mall_tiles(size_t grid_m,
                                             size_t grid_n,
                                             size_t active_cus,
                                             size_t wgm_value);

/**
 * @brief Compute L2 tile dimensions: how many tiles share one XCD's L2, shrunk to fit capacity.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param active_cus Number of active compute units.
 * @param splitting_factor K-split factor.
 * @param wgm_value Workgroup mapping slab width.
 * @return std::pair<size_t, size_t> (l2_tile_m, l2_tile_n).
 */
std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           size_t grid_m,
                                           size_t grid_n,
                                           size_t active_cus,
                                           size_t splitting_factor,
                                           size_t wgm_value);

/**
 * @brief Map a linear workgroup ID to 4D tile coordinates (k, m, n, b) using WGM slab ordering.
 *
 * Dispatch order: k (innermost) -> mn (WGM slab ordering) -> b (outermost).
 * Within the mn space, slabs of width min(wgm, grid.n) are laid out N-first, then M.
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm_mapping Workgroup mapping parameters (wgmxcc, wgm).
 * @param id Linear workgroup ID.
 * @return dim4_t 4D tile coordinate.
 */
dim4_t wgm_to_grid(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping, size_t id);

/**
 * @brief Count unique tile coordinates touched by a contiguous range of workgroup IDs.
 *
 * Computes how many distinct rows (m), columns (n), K-splits (k), and batches (b)
 * appear in the range [start, start+count) under raw dispatch order (no WGMXCC).
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm WGM slab width (absolute value).
 * @param start First workgroup ID in the range.
 * @param count Number of workgroups in the range.
 * @return dim4_t Unique tile counts in each dimension.
 */
dim4_t count_unique_range(const dim4_t& grid, int wgm, size_t start, size_t count);

/**
 * @brief Count unique tiles for a specific XCD during a specific timestep.
 *
 * With wgmxcc, XCD x in timestep ts sees cus_per_xcd consecutive tiles
 * in raw dispatch order. Without wgmxcc, falls back to even division.
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm_mapping Workgroup mapping parameters (wgmxcc, wgm).
 * @param N_CU Total number of CUs.
 * @param num_xcd Number of XCDs.
 * @param xcd_id XCD index (0-based).
 * @param timestep_id Timestep index (0-based).
 * @return dim4_t Unique tile counts in each dimension.
 */
dim4_t count_unique_tiles(const dim4_t& grid,
                          const workgroup_mapping_t& wgm_mapping,
                          size_t N_CU,
                          size_t num_xcd,
                          size_t xcd_id,
                          size_t timestep_id);

/**
 * @brief Count unique tiles for an entire timestep (all XCDs combined).
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm_mapping Workgroup mapping parameters (wgmxcc, wgm).
 * @param N_CU Total number of CUs.
 * @param timestep_id Timestep index (0-based).
 * @return dim4_t Unique tile counts in each dimension.
 */
dim4_t count_unique_tiles_timestep(const dim4_t& grid,
                                   const workgroup_mapping_t& wgm_mapping,
                                   size_t N_CU,
                                   size_t timestep_id);

/**
 * @brief Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K
 * tile.
 *
 * @param mt Macro tile dimensions
 * @param mi Micro tile dimensions
 * @return size_t Number of matrix instructions
 */
size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi);

/**
 * @brief Compute arithmetic intensity.
 *
 * @param m problem size M
 * @param n problem size N
 * @param k problem size K
 * @param bytes_per_element bytes per element
 * @return double arithmetic intensity.
 */
double arithmetic_intensity(double m, double n, double k, double bytes_per_element);

/**
 * @brief Emulated tf32 arithmetic intensity.
 *
 * @param m problem size M
 * @param n problem size N
 * @param k problem size K
 * @param bytes_per_element bytes per element
 * @return double arithmetic intensity.
 */
double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element);

/**
 * @brief Compute TF32 X1 conversion overhead (SS_BSS path).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_cvt_overhead_x1(const problem_t& problem,
                               const hardware_t& hardware,
                               const config_t& config);

/**
 * @brief Compute TF32 X3 conversion overhead.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_cvt_overhead(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config);

/**
 * @brief Compute the latency to process a single macro-tile for the given problem and hardware.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return size_t Latency in cycles.
 */
size_t compute_mt_compute_latency(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config);

/**
 * @brief A linear-estimation method for estimating L2-hitrate.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Predicted L2-hitrate.
 */
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       const context_t& context);

/**
 * @brief Estimate the MALL-hitrate (last-level cache.)
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Predicted MALL-hitrate.
 */
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         const context_t& context);

/**
 * @brief Estimate per-operand L1, L2, and MALL hit rates using the analytical model.
 *
 * Computes unique tiles counts for the first two timesteps using WGM,
 * then estimates temporal reuse from T0->T1 overlap. Extrapolates to all timesteps.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return cache_hit_rates_t
 */
cache_hit_rates_t estimate_cache_hit_rates(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           const context_t& context);

/**
 * @brief L2 hit rate from a global (problem-wide) perspective using the refactored API.
 *        Computes in BYTES to correctly handle differing A/B dtypes.
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param l2_capacity_bytes l2 capacity in bytes
 * @return double
 */
double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t l2_capacity_bytes);

/**
 * @brief Determine the memory latency per MT_M x MT_N x MT_K Macro Tile (L_MT).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Latency in cycles.
 */
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context);

/**
 * @brief Compute the epilogue latency for a single tile.
 *
 * Models the cost of writing output (or workspace partials) after the main loop:
 * ACC->VGPR transfer, edge-tile bounds checking, global memory stores, and
 * in-kernel serial reduction (if applicable).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Epilogue latency in cycles.
 */
double compute_epilogue_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context);

/**
 * @brief Computes the latency to compute a K-COMPLETE tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Latency in cycles.
 */
double compute_tile_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            const context_t& context);

/**
 * @brief Computes the latency per K-complete macro-tile timestep.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Latency in cycles.
 */
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                const context_t& context);

/**
 * @brief Compute the latency of the parallel reduction kernel (separate kernel launch).
 *
 * For parallel reduction, the GEMM kernel writes partials to workspace and a second
 * kernel is launched to read all partials, accumulate, and write the final output.
 * This function estimates the latency of that second kernel including launch overhead.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Latency in cycles (0 if no parallel reduction).
 */
double compute_parallel_reduction_latency(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          const context_t& context);

/**
 * @brief Compute the total latency of a gemm based on the latency of one timestep multiplied by the
 * number of timesteps. (@see compute_timestep_latency)
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param max_cus
 * @return double Latency in cycles.
 */
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus);

}  // namespace origami
