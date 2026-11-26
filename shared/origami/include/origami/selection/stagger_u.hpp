// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include "origami/core/config.hpp"
#include "origami/core/mapping.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Select the best staggerU (maximizing cache hits) for the given tile size.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration
 * @param skGrid StreamK grid size
 * @param wgm Workgroup mapping
 * @return staggerU_t Best predicted staggerU parameters (staggerUMapping, staggerU,
 * staggerUStrideShift).
 */
ORIGAMI_EXPORT staggerU_t select_staggerU(const problem_t& problem,
                                          const hardware_t& hardware,
                                          const config_t& config,
                                          size_t skGrid,
                                          int32_t wgm);

}  // namespace origami
