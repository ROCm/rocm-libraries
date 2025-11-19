// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {
/* ---------------------------------------------------------------------------------------- */
/* Compute-related functions                                                                */
/* ---------------------------------------------------------------------------------------- */
// Compute the number of matrix instructions required to compute a single MT_MXMT_NXMT_K tile.
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

/* ---------------------------------------------------------------------------------------- */
/* Memory-related functions                                                                 */
/* ---------------------------------------------------------------------------------------- */
// Check if MT fits in LDS
bool check_lds_capacity(const hardware_t& hardware,
                        dim3_t mt,
                        data_type_t a_dtype,
                        data_type_t b_dtype);

// Compute the amount of data loaded from A to produce a MT_MxMT_NxMT_K tile.
size_t compute_A_loads(size_t MT_M, size_t MT_K);

// Compute the amount of data loaded from B to produce a MT_MxMT_NxMT_K tile.
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

/* ---------------------------------------------------------------------------------------- */
/* Tile-related functions                                                                   */
/* ---------------------------------------------------------------------------------------- */
// Computes the latency to compute a K-COMPLETE tile.
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
 * @brief Computes the latency per K-complete MT wave.
 * A wave is defined as the time it takes for one CU to complete one
 * K-complete output tile
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param num_active_cus
 * @param splitting_factor
 * @return double Latency in cycles.
 */
double compute_wave_latency(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config,
                            std::size_t num_active_cus,
                            std::size_t splitting_factor);

/**
 * @brief Compute the total latency of a gemm based on the latency of one wave multiplied by the
 * number of waves A wave is defined as the time it takes for one CU to complete one K-complete
 * output tile.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @return double Latency in cycles.
 */
double compute_total_latency(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             size_t max_cus);

// Extract analytical metrics from a GEMM computation and export to CSV
void extract_analytical_metrics_csv(const hardware_t& hardware,
                                    const problem_t& problem,
                                    const config_t& config,
                                    const std::string& filename);

// Extract analytical metrics from a GEMM computation and export to JSON
void extract_analytical_metrics_json(const hardware_t& hardware,
                                     const problem_t& problem,
                                     const config_t& config,
                                     const std::string& filename);

// Extract analytical metrics from a GEMM computation and return as map
std::unordered_map<std::string, std::string> extract_analytical_metrics(const hardware_t& hardware,
                                                                        const problem_t& problem,
                                                                        const config_t& config);

}  // namespace origami
