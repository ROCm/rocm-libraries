// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/core/dimensions.hpp"
#include "origami/origami_export.h"

namespace origami {

/**
 * @brief An environment variable Origami reads.
 *
 * Every variable is named ORIGAMI_<OPTION>.  The variables that were once named
 * ANALYTICAL_GEMM_<OPTION> are still read under that name when the ORIGAMI_ one is not set.
 */
struct env_var_t {
  const char* name;    ///< Name of the variable
  const char* legacy;  ///< Former name, read when @ref name is unset; nullptr if there is none

  /// Value of the variable, or nullptr when neither name is set.
  ORIGAMI_EXPORT const char* read() const;
};

/**
 * @brief The environment variables Origami reads.
 */
namespace env {

/// "1" enables debug logging.
inline constexpr env_var_t DEBUG{"ORIGAMI_DEBUG", "ANALYTICAL_GEMM_DEBUG"};

/// "0" disables heuristics; they are on otherwise.
inline constexpr env_var_t HEURISTICS{"ORIGAMI_HEURISTICS", "ANALYTICAL_GEMM_HEURISTICS"};

/// Relative latency band within which ranking breaks ties; 0 (the default) turns it off.
inline constexpr env_var_t HEURISTICS_VARIANCE{"ORIGAMI_HEURISTICS_VARIANCE",
                                               "ANALYTICAL_GEMM_HEURISTICS_VARIANCE"};

/// Macro tile "MxNxK" that GEMM selection is forced to, e.g. "128x128x64".
inline constexpr env_var_t GEMM_PICK{"ORIGAMI_GEMM_PICK", "ANALYTICAL_GEMM_PICK"};

/// File the logger writes to; a ".csv" extension selects CSV, anything else text.
inline constexpr env_var_t LOG_FILE{"ORIGAMI_LOG_FILE", nullptr};

}  // namespace env

/**
 * @brief Runtime options for controlling debug, heuristics, and other behaviors.
 *
 * Provides programmatic access to runtime configuration options that can be
 * set either programmatically or via environment variables (@see origami::env).
 */
struct ORIGAMI_EXPORT runtime_options {
  /// Enable debug logging (reads from ORIGAMI_DEBUG env var)
  bool debug_enabled;

  /// Enable heuristics (reads from ORIGAMI_HEURISTICS env var)
  bool heuristics_enabled;

  /// Heuristics variance threshold (reads from ORIGAMI_HEURISTICS_VARIANCE env var)
  double heuristics_variance;

  /// Force a specific MT size for solution selection (reads from ORIGAMI_GEMM_PICK env var).
  /// Format: "MxNxK" e.g. "128x128x64". When set, non-matching configs get max latency.
  dim3_t gemm_pick{0, 0, 0};

  /**
   * @brief Constructor with explicit values (does not read from environment).
   */
  runtime_options(bool debug, bool heuristics, double variance);

  /**
   * @brief Get the global runtime options instance.
   *
   * Inline to prevent ODR violations when included in multiple shared libraries.
   * Static local variable ensures only one instance exists across all translation units. (PR#1862)
   */
  static inline runtime_options& get() {
    static runtime_options instance;
    return instance;
  }

  /**
   * @brief Read debug setting from environment variable.
   * @return true if ORIGAMI_DEBUG is set to "1", false otherwise
   */
  static bool read_debug_from_env();

  /**
   * @brief Read heuristics setting from environment variable.
   * @return false if ORIGAMI_HEURISTICS is set to "0", true otherwise (including when it
   * is not set)
   */
  static bool read_heuristics_from_env();

  /**
   * @brief Read heuristics variance from environment variable.
   * @return double Variance value from ORIGAMI_HEURISTICS_VARIANCE, or 0.0 (variance
   * disabled) if it is not set or is not a finite non-negative number
   */
  static double read_heuristics_variance_from_env();

  /**
   * @brief Read GEMM pick MT size from environment variable.
   * @return dim3_t MT size from ORIGAMI_GEMM_PICK, or {0,0,0} if not set
   */
  static dim3_t read_gemm_pick_from_env();

  /**
   * @brief Update runtime options from environment variables.
   */
  void update_from_env();

 private:
  /**
   * @brief Default constructor that reads from environment variables.
   *
   * This is made private because it should only be used through the static get() member.
   */
  runtime_options();
};

}  // namespace origami
