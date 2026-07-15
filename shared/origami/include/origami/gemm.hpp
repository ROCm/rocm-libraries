// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/types.hpp"
#include "origami/origami_export.h"

namespace origami {
namespace gemm {

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
struct ORIGAMI_EXPORT context_t {
  /// Heuristic parameters, looked up lazily and memoized on first use via
  /// @ref get_heuristic. Coarse detail levels (timesteps / compute proxy) never
  /// need them, so deferring the lookup keeps early-pruned configs from paying
  /// for the (allocating) database lookup + struct copy.
  mutable std::optional<heuristic_params_t> heuristic_cache;

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
  /// Number of compute units usable for this problem. Equals hardware.N_CU
  /// unless problem.num_cus caps it lower (@see origami::problem_t::num_cus).
  /// This is the effective CU count the whole model schedules against.
  size_t n_cu                 = 0;
  size_t active_cus           = 0;
  double mem_bw_limited       = 0.0;
  double write_mem_bw_limited = 0.0;

  /// Tile-derived values.
  size_t k_per_split       = 0;
  size_t k_iters           = 0;
  size_t tile_elements     = 0;
  double output_tile_bytes = 0.0;

  /// Occupancy-derived value. The decay factor itself (occupancy_factor) needs
  /// the heuristic, so it is derived lazily at the full level.
  size_t real_occupancy = 0;

  /// Debug flag (cached from runtime_options to avoid repeated singleton lookups).
  bool debug = false;

  /// Leveled latency components, filled progressively as finer detail levels run
  /// and reused across the levels that share this context (hierarchical
  /// carry-over). An unset optional means "not computed at this level yet".
  /// Mutable because memoization is logically const; a context is per-config and
  /// used single-threaded, so the in-place fill is race-free. These double as a
  /// per-config feature record for any future ML export.
  mutable double                             occupancy_factor = 1.0;  // [L3] decay factor
  mutable std::optional<double>              utilization;             // [L2] cheap
  mutable std::optional<double>              effective_tile_penalty;  // [L2] cheap
  mutable std::optional<workgroup_mapping_t> wgm;                     // [L3] lazily predicted WGM
  mutable std::optional<cache_hit_rates_t>   cache_rates;             // [L3] per-operand hit rates
  mutable std::optional<double>              L_comp_stream;           // [L1] per-iteration compute arm
  mutable std::optional<double>              L_mem_stream;            // [L3] per-iteration memory arm (detailed)
  mutable std::optional<double>              L_prologue;              // [L3] fill cost
  mutable std::optional<double>              L_comp;                  // [L3] whole-tile compute total (L_comp_stream * num_iter)
  mutable std::optional<double>              L_cvt;                   // [L3] dtype-conversion overhead
  mutable std::optional<double>              L_mem;                   // [L3] whole-tile memory total (L_mem_stream * num_iter)
  mutable std::optional<double>              L_epilogue;              // [L3] epilogue/drain (stores + reduction)
  mutable std::optional<double>              L_tile_single;           // [L3] steady per-iteration latency
  mutable std::optional<double>              L_parallel_reduction;    // [L3] PostGSU reduction kernel
  mutable std::optional<double>              L_total;                 // [L3] full analytical latency

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

  /**
   * @brief Lazily predict (and memoize) the workgroup mapping.
   *
   * WGM prediction is the most expensive part of building a context and is only
   * needed by the detailed cache/memory model and the full level, so it is
   * deferred here instead of being computed eagerly. The result is cached in
   * @ref wgm and reused on subsequent calls.
   */
  const workgroup_mapping_t& get_wgm(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config) const;

  /**
   * @brief Lazily look up (and memoize) the heuristic parameters.
   *
   * The heuristics database lookup allocates and copies a sizable struct, but the
   * coarse detail levels never read it. Deferring it here (mirroring @ref get_wgm)
   * means a config pruned before the memory/full level never pays for it. The
   * result is cached in @ref heuristic_cache and reused on subsequent calls.
   */
  const heuristic_params_t& get_heuristic(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config) const;
};

/**
 * @brief calculate the work utilization which is the ratio of the useful problem volume to the
 * total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
ORIGAMI_EXPORT double calculate_work_utilization(const problem_t& problem, const config_t& config);

/**
 * @brief calculate the output utilization which is the ratio of the useful problem volume to the
 * total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param vector_elems elements in the vector.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
ORIGAMI_EXPORT double calculate_output_utilization(const problem_t& problem,
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
ORIGAMI_EXPORT size_t round_elements_to_128B(size_t elements, size_t element_size_bits);

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
ORIGAMI_EXPORT workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
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
 * @return tuple<reduction_t, size_t, size_t, size_t, size_t>
 *         (reduction_strategy, num_wgs, num_active_cus, num_timesteps, split_factor)
 */
ORIGAMI_EXPORT std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection);

/**
 * @brief Check if MT fits in LDS
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro tile dimensions
 * @param a_dtype Data type of operand A
 * @param b_dtype Data type of operand B
 * @return bool True if MT fits in LDS, false otherwise
 */
ORIGAMI_EXPORT bool check_lds_capacity(const hardware_t& hardware,
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
ORIGAMI_EXPORT double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus);

/**
 * @brief Compute MALL tile dimensions: how many concurrent workgroup tiles fit in MALL.
 *
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param active_cus Number of active compute units.
 * @param wgm_value Workgroup mapping slab width.
 * @return std::pair<size_t, size_t> (mall_tile_m, mall_tile_n).
 */
ORIGAMI_EXPORT std::pair<size_t, size_t> compute_mall_tiles(size_t grid_m,
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
ORIGAMI_EXPORT std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
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
ORIGAMI_EXPORT dim4_t wgm_to_grid(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping, size_t id);

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
ORIGAMI_EXPORT dim4_t count_unique_range(const dim4_t& grid, int wgm, size_t start, size_t count);

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
ORIGAMI_EXPORT dim4_t count_unique_tiles(const dim4_t& grid,
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
ORIGAMI_EXPORT dim4_t count_unique_tiles_timestep(const dim4_t& grid,
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
ORIGAMI_EXPORT size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi);

/**
 * @brief Compute arithmetic intensity.
 *
 * @param m problem size M
 * @param n problem size N
 * @param k problem size K
 * @param bytes_per_element bytes per element
 * @return double arithmetic intensity.
 */
ORIGAMI_EXPORT double arithmetic_intensity(double m, double n, double k, double bytes_per_element);

/**
 * @brief Emulated tf32 arithmetic intensity.
 *
 * @param m problem size M
 * @param n problem size N
 * @param k problem size K
 * @param bytes_per_element bytes per element
 * @return double arithmetic intensity.
 */
ORIGAMI_EXPORT double emulated_tf32_arithmetic_intensity(double m, double n, double k, double bytes_per_element);

/**
 * @brief Compute TF32 X1 conversion overhead (SS_BSS path).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_cvt_overhead_x1(const problem_t& problem,
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
ORIGAMI_EXPORT double compute_cvt_overhead(const problem_t& problem,
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
ORIGAMI_EXPORT size_t compute_mt_compute_latency(const problem_t& problem,
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
ORIGAMI_EXPORT double estimate_l2_hit(const problem_t& problem,
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
ORIGAMI_EXPORT double estimate_mall_hit(const problem_t& problem,
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
ORIGAMI_EXPORT cache_hit_rates_t estimate_cache_hit_rates(const problem_t& problem,
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
ORIGAMI_EXPORT double compute_l2_hit_rate_global(const problem_t& problem,
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
ORIGAMI_EXPORT double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              const context_t& context);

/**
 * @brief Core memory-latency math from precomputed per-operand cache hit rates.
 *
 * Everything in @ref compute_memory_latency below the hit-rate estimation. Lets a
 * caller inject the cache fidelity (detailed estimate vs a flat proxy) so the
 * leveled entry points share one implementation. Does not touch the context's
 * cost record.
 *
 * @param rates Per-operand L1/L2/MALL hit rates to use.
 */
ORIGAMI_EXPORT double compute_memory_latency_from_rates(const problem_t& problem,
                                         const hardware_t& hardware,
                                         const config_t& config,
                                         const context_t& context,
                                         const cache_hit_rates_t& rates);

/**
 * @brief Per-tile memory latency at a given detail @p Level.
 *
 * One entry point parameterized over the detail level, instead of a family of
 * `_coarse` / `_detailed` siblings. The level selects the cache fidelity:
 *   - Level >= 3: full detail -- the analytical cache model (predicts the WGM and
 *     memoizes into the context's cost record). Delegates to the non-template
 *     @ref compute_memory_latency.
 *   - Level == 2: a flat ~0.5 hit-rate proxy (no WGM / detailed cache, uncached).
 *   - Level <= 1: a cheaper proxy still (no caching modeled).
 *
 * To add a dedicated routine for another level (e.g. a bespoke level-1 memory
 * model), extend the `if constexpr` chain below -- the runtime dispatch (a
 * `switch` mapping the phase's runtime level to one of these instantiations)
 * lives in the cost-model adapter.
 *
 * The coarse levels never write @ref context_t::L_mem_stream, so a coarse value
 * can never poison the full level's detailed result.
 */
template <std::size_t Level>
inline double compute_memory_latency(const problem_t& problem,
                                     const hardware_t& hardware,
                                     const config_t& config,
                                     const context_t& context) {
  if constexpr (Level >= 3) {
    // Calls the non-template (detailed) overload: Level is non-deducible, so an
    // un-templated call never re-selects this template -- no recursion.
    return compute_memory_latency(problem, hardware, config, context);
  } else {
    // Coarser levels use a flat per-operand hit-rate proxy that sharpens with the
    // level. Add a `Level == N` branch here for a bespoke per-level routine.
    constexpr double        rate = (Level >= 2) ? 0.5 : 0.0;
    const cache_hit_rates_t rates{rate, rate, rate, rate, rate, rate};
    return compute_memory_latency_from_rates(problem, hardware, config, context, rates);
  }
}

/**
 * @brief Context-free quick epilogue proxy: store latency of one interior tile.
 *
 * The cheapest possible epilogue estimate -- just the cost of storing a single
 * MT_M x MT_N output tile -- so the coarse (context-free) scoring levels can
 * account for the write cost before a context is built. Ignores edge/corner
 * tiles, split-K reduction, ACC->VGPR transfer, and alignment; active CU count
 * and bandwidth are approximated from the output-tile count.
 *
 * The caller supplies @p num_output_tiles, so a caller that has already derived
 * the grid (e.g. the level-1 proxy) does not recompute it.
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro-tile dimensions.
 * @param d_dtype Output element data type.
 * @param num_output_tiles grid_m * grid_n * batch.
 * @return double Approximate tile-store latency in cycles.
 */
ORIGAMI_EXPORT double compute_epilogue_latency_quick(const hardware_t& hardware,
                                                      const dim3_t& mt,
                                                      data_type_t d_dtype,
                                                      size_t num_output_tiles);

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
ORIGAMI_EXPORT double compute_epilogue_latency(const problem_t& problem,
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
ORIGAMI_EXPORT double compute_tile_latency(const problem_t& problem,
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
ORIGAMI_EXPORT double compute_timestep_latency(const problem_t& problem,
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
ORIGAMI_EXPORT double compute_parallel_reduction_latency(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          const context_t& context);

/**
 * @brief Compute the total analytical (estimation) latency of a GEMM.
 *
 * The analytical estimation path: fast-rejection gate, then the per-timestep
 * latency summed over all timesteps plus the parallel-reduction cost. Simulation
 * is a separate model with its own entry (@see compute_formocast_latency).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config);

/**
 * @brief Full analytical estimation latency from a prebuilt context.
 *
 * Single source of truth for the estimation latency: sums the per-timestep
 * latency over all timesteps plus the parallel-reduction cost. The
 * four/five-argument compute_total_latency builds a context and delegates here;
 * the leveled cost model reuses a carried context to avoid rebuilding it.
 *
 * @note The caller is responsible for fast_reject; this routine does NOT re-run
 *       it. Both internal entry points already filter upstream (the leveled model
 *       at level 0, compute_total_latency before building the context).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Prebuilt execution context for (problem, hardware, config).
 * @return double Latency in cycles (max if rejected/short-circuited).
 */
ORIGAMI_EXPORT double estimation_latency_from_context(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config,
                                      const context_t& context);

/**
 * @brief Cheap, context-free fast rejection: the single collection point for
 *        every structural disqualification rule for a GEMM config.
 *
 * Used by the leveled model's level 0 (no execution context built) and to gate
 * the full estimation / simulation paths. This is a performance gate ("this
 * kernel is not worth scoring"), distinct from feasibility ("can it run").
 *
 * @return true if the config should be rejected before any real scoring.
 */
ORIGAMI_EXPORT bool fast_reject(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config);

}  // namespace gemm
}  // namespace origami
