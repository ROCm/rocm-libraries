// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

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
 * @brief Data type operand A has in LDS
 *
 * The type of A in global memory, unless the kernel converts A before writing it to LDS (a
 * W4A16 kernel dequantizes its int4 A to the MI type).
 *
 * @param problem Problem characteristics (@see origami::problem_t)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @return data_type_t Type of A's elements in LDS
 */
ORIGAMI_EXPORT data_type_t lds_dtype_a(const problem_t& problem, const hardware_t& hardware);

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
 * @brief LDS->VGPR read time of one MT_K-deep K-iteration of a tile.
 *
 * Mirrors TensileLite's local reads: every wave reads its own M (or N) slice, so
 * A is read once per wave column and B once per wave row.  An operand whose M/N
 * axis is contiguous in LDS is read VectorWidth elements per lane, i.e.
 * LoopIters * MIWaveTile * MIInputPerThread / VW reads per wave.  Unroll-major and
 * LDS-transpose operands read along K at full width; DirectToVgpr operands skip LDS.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param full_width Charge every read at full bank width (the VectorWidth-independent floor).
 * @return double LDS cycles per K-iteration, in the units of compute_mt_compute_latency.
 */
ORIGAMI_EXPORT double compute_local_read_latency(const problem_t& problem,
                                                 const hardware_t& hardware,
                                                 const config_t& config,
                                                 bool full_width = false);

}  // namespace gemm
}  // namespace origami
