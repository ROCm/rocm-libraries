// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Given a latency, compute the achieved throughput in gflops.
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param problem Problem description (M, N, K, etc.)
 * @param latency Kernel latency.
 * @return double Throughput in gflops/s.
 */
ORIGAMI_EXPORT double compute_perf_gflops(const hardware_t& hardware,
                                          const problem_t& problem,
                                          const double latency);

}  // namespace origami
