// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/core/config.hpp"
#include "origami/core/mapping.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Select the best workgroup mapping (maximizing cache hits) for the given tile size.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param skGrid StreamK grid size.
 * @return workgroup_mapping_t Best predicted workgroup mapping parameters (wgmxccsplitk,
 * wgmxccchunk, wgmxcc, wgm).
 */
ORIGAMI_EXPORT workgroup_mapping_t select_workgroup_mapping(const problem_t& problem,
                                                            const hardware_t& hardware,
                                                            const config_t& config,
                                                            size_t skGrid);

}  // namespace origami
