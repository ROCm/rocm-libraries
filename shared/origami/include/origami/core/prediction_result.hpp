// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/core/config.hpp"

namespace origami {

/**
 * @brief Latency prediction result given kernel configuration.
 *
 * Combines a configuration with its estimated latency.
 */
struct prediction_result_t {
  /// Predicted latency in cycles.
  double latency;

  /// Configuration the latency was predicted for.
  config_t config;
};

}  // namespace origami
