// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/cost_models/gemm/tile.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

/**
 * @brief Epilogue of one tile.
 */
struct epilogue_terms_t {
  double L_epilogue            = 0.0;  ///< Time to store the tile, in cycles.
  double scalar_store_fraction = 0.0;  ///< Share of that time spent in scalar edge stores.
};

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
                                               const context_t& context,
                                               double* scalar_store_fraction = nullptr);

}  // namespace gemm
}  // namespace origami
