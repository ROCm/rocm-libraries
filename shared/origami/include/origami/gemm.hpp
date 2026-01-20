/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <vector>
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief calculate the work utilization which is the ratio of the useful problem volume to the total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
double calculate_work_utilization(const problem_t& problem, const config_t& config);

/**
 * @brief calculate the output utilization which is the ratio of the useful problem volume to the total scheduled volume.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param vector_elems elements in the vector.
 * @return double ratio of the useful problem volume to the total scheduled volume.
 */
double calculate_output_utilization(const problem_t& problem, const config_t& config, size_t vector_elems);

/**
 * @brief Computes the number of active compute units if there is only one wave and it is partial, Otherwise, returns hardware.N_CU
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection Different algorithms to select the grid size for kernel execution.
 * @param max_cus maximum number of CU's
 * @param split split
 * @return tuple<size_t, size_t, size_t, size_t> tuple(num_wgs, num_active_cus, numWaves, splitFactor)
 */
std::tuple<size_t, size_t, size_t, size_t> compute_cu_occupancy(const problem_t& problem,
                                                                const hardware_t& hardware,
                                                                const config_t& config,
                                                                grid_selection_t grid_selection,
                                                                size_t max_cus,
                                                                size_t split);

/**
 * @brief Compute limited achievable memory bandwidth based on active CUs
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param num_active_cus number of CU's
 * @return double memory bandwidth
 */
double compute_mem_bw_from_occupancy(const hardware_t& hardware, size_t num_active_cus);

/**
 * @brief This function rounds the number of elements up to the smallest value whose total size (given the element bit-width) is an exact multiple of a 128-byte memory transaction.
 *
 * @param elements Macro tile dimension
 * @param element_size_bits size in bits
 * @return size_t
 */
size_t round_elements_to_128B(size_t elements, size_t element_size_bits);

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
 * @brief Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K
 * tile.
 *
 * @param mt Macro tile dimensions
 * @param mi Micro tile dimensions
 * @return size_t Number of matrix instructions
 */
size_t compute_number_matrix_instructions(dim3_t mt, dim3_t mi);

/**
 * @brief Compute TF32 conversion overhead.
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
 * @brief Check if MT fits in LDS
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param mt Macro tile dimensions
 * @param a_dtype Data type of operand A
 * @param b_dtype Data type of operand B
 * @return bool True if MT fits in LDS, false otherwise
 */
bool check_lds_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype,
                        data_type_t b_dtype);

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
double estimate_l2_hit(const problem_t& problem,
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
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         std::size_t num_active_cus,
                         std::size_t splitting_factor);

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
double compute_memory_latency(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config,
                              std::size_t num_active_cus,
                              std::size_t splitting_factor);

/**
 * @brief Computes the latency to compute a K-COMPLETE tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
double compute_tile_latency(const problem_t& problem,
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
double compute_timestep_latency(const problem_t& problem,
                                const hardware_t& hardware,
                                const config_t& config,
                                std::size_t num_active_cus,
                                std::size_t splitting_factor);

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
