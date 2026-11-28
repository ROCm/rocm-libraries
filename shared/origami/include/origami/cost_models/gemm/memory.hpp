// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Bytes of operands A and B that one workgroup reads from memory.
 *
 * Rows are rounded up to whole memory transactions.
 */
struct operand_traffic_t {
  /// Bytes of A one workgroup reads over its whole K range.
  double a_tile_bytes = 0.0;

  /// Bytes of B one workgroup reads over its whole K range.
  double b_tile_bytes = 0.0;

  /// Bytes of A one workgroup reads per K iteration (a_tile_bytes / k_iters).
  double a_iter_bytes = 0.0;

  /// Bytes of B one workgroup reads per K iteration (b_tile_bytes / k_iters).
  double b_iter_bytes = 0.0;

  /// K iterations one workgroup runs (at least 1).
  double k_iters = 1.0;

  /// Fraction of a memory transaction that the problem's M extent of A fills, at most 1.
  double a_cl_share = 1.0;

  /// Fraction of a memory transaction that the problem's N extent of B fills, at most 1.
  double b_cl_share = 1.0;
};

}  // namespace origami

namespace origami {
namespace gemm {

struct context_t;

/**
 * @brief Memory side of one K-iteration of a tile.
 *
 * Families whose K-loop overlaps the L2 and the DRAM read times separately also report the two
 * and the multicast delivery bound; the others leave them zero.
 */
struct memory_terms_t {
  double L_mem          = 0.0;  ///< Time to read A and B for one K-iteration, in cycles.
  double l2_cycles      = 0.0;  ///< L2 read time alone, in cycles.
  double dram_cycles    = 0.0;  ///< DRAM read time alone, in cycles.
  double delivery_floor = 0.0;  ///< Lower bound on the time from multicast delivery, in cycles.
};

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
 * @brief Compute limited achievable memory bandwidth based on active CUs
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param num_active_cus number of CU's
 * @return double memory bandwidth
 */
ORIGAMI_EXPORT double compute_mem_bw_from_occupancy(const hardware_t& hardware,
                                                    size_t num_active_cus);

}  // namespace gemm
}  // namespace origami
