// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <tuple>

#include "origami/core/config.hpp"
#include "origami/core/data_type.hpp"
#include "origami/core/dimensions.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"

namespace origami {
namespace attention {

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
 * @brief Computes the number of active compute units if there is only one wave and it is partial,
 * Otherwise, returns hardware.N_CU
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection Different algorithms to select the grid size for kernel execution.
 * @param max_cus maximum number of CU's
 * @param split split
 * @return tuple<size_t, size_t, size_t, size_t> tuple(num_wgs, num_active_cus, numWaves,
 * splitFactor)
 */
ORIGAMI_EXPORT std::tuple<size_t, size_t, size_t, size_t> compute_cu_occupancy(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection,
    size_t max_cus,
    size_t split);

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
ORIGAMI_EXPORT double emulated_tf32_arithmetic_intensity(double m,
                                                         double n,
                                                         double k,
                                                         double bytes_per_element);

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
 * @brief Check if MT fits in RF
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro tile dimensions
 * @param a_dtype Data type of operand A
 * @return bool True if MT fits in RF, false otherwise
 */
ORIGAMI_EXPORT bool check_rf_capacity(const hardware_t& hardware, dim3_t mt, data_type_t a_dtype);

/**
 * @brief Check if MT fits in LDS
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro tile dimensions
 * @param a_dtype Data type of operand A
 * @return bool True if MT fits in LDS, false otherwise
 */
ORIGAMI_EXPORT bool check_lds_capacity(const hardware_t& hardware, dim3_t mt, data_type_t a_dtype);

/**
 * @brief Compute limited achievable memory bandwidth based on active CUs
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param num_active_cus number of CU's
 * @return double memory bandwidth
 */
ORIGAMI_EXPORT double compute_mem_bw_from_occupancy(const hardware_t& hardware,
                                                    size_t num_active_cus);

/**
 * @brief A linear-estimation method for estimating L2-hitrate.
 *
 * @todo Parameterize this based on the space-filling curve algos.
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param splitting_factor
 * @return double Predicted L2-hitrate.
 */
ORIGAMI_EXPORT double estimate_l2_hit(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config,
                                      std::size_t splitting_factor);

/**
 * @brief Estimate the MALL-hitrate (last-level cache.)
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Predicted MALL-hitrate.
 */
ORIGAMI_EXPORT double estimate_mall_hit(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config,
                                        std::size_t num_active_cus,
                                        std::size_t splitting_factor);

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
 * @brief This function rounds the number of elements up to the smallest value whose total size
 * (given the element bit-width) is an exact multiple of a 128-byte memory transaction.
 *
 * @param elements Macro tile dimension
 * @param element_size_bits size in bits
 * @return size_t
 */
ORIGAMI_EXPORT size_t round_elements_to_128B(size_t elements, size_t element_size_bits);

/**
 * @brief Determine the memory latency per MT_M x MT_N x MT_K Macro Tile (L_MT).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_memory_latency(const problem_t& problem,
                                             const hardware_t& hardware,
                                             const config_t& config,
                                             std::size_t num_active_cus,
                                             std::size_t splitting_factor);

/**
 * @brief Compute the flash attention pipeline cycle latency.
 *
 * The 2-WG pipelined schedule has 5 stages, each overlapping work from two workgroups. The
 * pipeline cycle time is the maximum stage latency.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_tile_latency(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           std::size_t num_active_cus,
                                           std::size_t splitting_factor);

/**
 * @brief Computes the latency per K-complete macro-tile timestep.
 * A timestep is defined as the time it takes for one set of concurrent
 * K-complete output tiles to be computed on one or more CUs. Typically,
 * this is simply the time it takes for one CU to complete one K-complete
 * output tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_timestep_latency(const problem_t& problem,
                                               const hardware_t& hardware,
                                               const config_t& config,
                                               std::size_t num_active_cus,
                                               std::size_t splitting_factor);

/**
 * @brief Compute total flash attention latency.
 *
 * Total = prologue + (loop_cycle * num_iters) + epilogue
 *
 * Where:
 *   - prologue = ld_q + ld_k (initial tile loads before pipeline starts)
 *   - loop_cycle = max(t0, t1, t2, t3, t4) (pipelined steady-state, @see compute_tile_latency)
 *   - epilogue = st_o (final output tile write after pipeline drains)
 *   - num_iters = ceil(8 * batch * q_heads * grid_M * grid_N * grid_K / (2 * active CUs))
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
ORIGAMI_EXPORT double compute_total_latency(const problem_t& problem,
                                            const hardware_t& hardware,
                                            const config_t& config);

}  // namespace attention
}  // namespace origami
