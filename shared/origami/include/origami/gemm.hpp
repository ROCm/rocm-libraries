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
static inline double compute_cvt_overhead(const problem_t& problem,
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
 * @brief Compute the amount of data loaded from A to produce a MT_MxMT_NxMT_K tile.
 *
 * @param MT_M Macro tile dimension M
 * @param MT_K Macro tile dimension K
 * @return size_t Amount of data loaded from A
 */
size_t compute_A_loads(size_t MT_M, size_t MT_K);

/**
 * @brief Compute the amount of data loaded from B to produce a MT_MxMT_NxMT_K tile.
 *
 * @param MT_N Macro tile dimension N
 * @param MT_K Macro tile dimension K
 * @return size_t Amount of data loaded from B
 */
size_t compute_B_loads(size_t MT_N, size_t MT_K);

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
