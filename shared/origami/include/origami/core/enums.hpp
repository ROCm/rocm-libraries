// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Grid selection algorithms for StreamK.
 *
 * Different algorithms to select the grid size for kernel execution.
 */
enum class grid_selection_t : std::uint32_t {
  number_of_cus        = 0,  ///< Use number of compute units
  min_resources        = 1,  ///< Use minimum required resources
  energy_aware         = 2,  ///< Energy-aware selection
  reduction_cost_aware = 3,  ///< Reduction cost-aware selection
  data_parallel        = 4,  ///< Data parallel approach
  analytical           = 5,  ///< Analytical model-based selection
  k_split_aware        = 6,  ///< K-split aware selection
  count,                     ///< Count of Grid selection algos
  none = 0xFFFFFFFFu         ///< Explicitly invalid
};

/**
 * @brief The tile-scheduling mode a kernel launches in.
 *
 * Picks between the SK3 static work-assignment sub-path and the SK4
 * dynamic per-XCD work-queue sub-path inside a single SK5 kernel launch.
 */
enum class hybrid_mode_t : std::uint32_t {
  static_ = 0,        ///< SK3 static work-assignment sub-path
  dynamic = 1,        ///< SK4 dynamic per-XCD work-queue sub-path
  count,              ///< Count of hybrid modes
  none = 0xFFFFFFFFu  ///< Explicitly invalid
};

/**
 * @brief Convert hybrid_mode_t to string.
 *
 * @param mode Tile-scheduling mode
 * @return std::string String representation of the mode
 */
ORIGAMI_EXPORT std::string hybrid_mode_to_string(hybrid_mode_t mode);

/**
 * @brief Origami model types for performance prediction.
 *
 * Specifies which cost model to use for latency computation.
 */
enum class model_t : std::uint32_t {
  gemm      = 0,      ///< GEMM model for matrix multiplication
  attention = 1,      ///< Attention model for Flash Attention
  count,              ///< Count of model types
  none = 0xFFFFFFFFu  ///< Explicitly invalid
};

/**
 * @brief Prediction mode types for latency estimation.
 *
 * Different approaches for predicting kernel performance.
 */
enum class prediction_modes_t : std::uint32_t {
  estimation = 0,     ///< Fast analytical estimation-based prediction (typically faster)
  simulation = 1,     ///< Slow simulation-like prediction (typically more accurate)
  count,              ///< Count of prediction modes
  none = 0xFFFFFFFFu  ///< Explicitly invalid
};

/**
 * @brief Reduction strategy types for StreamK.
 *
 * Different algorithms for reduction operations in StreamK.
 */
enum class reduction_t : std::uint32_t {
  spinlock = 0,       ///< Spinlock-based reduction
  tree     = 1,       ///< Tree-based reduction
  parallel = 2,       ///< Parallel reduction
  atomic   = 3,       ///< Atomic Add-based reduction
  count,              ///< Count of reduction types
  none = 0xFFFFFFFFu  ///< Explicitly invalid / no reduction
};

/**
 * @brief Convert integer to reduction_t enum.
 *
 * @param rt Integer value to convert
 * @return reduction_t Corresponding reduction type
 */
inline constexpr reduction_t int_to_reduction_t(int rt) { return static_cast<reduction_t>(rt); }

/**
 * @brief Target backend types for kernel execution.
 *
 * Different backends that kernels can target.
 */
enum class target_t : std::uint32_t {
  generic           = 0,  ///< Generic backend (backend agnostic, not supported yet)
  tensilelite       = 1,  ///< hipBLASLt (tensilelite) backend
  rocroller         = 2,  ///< hipBLASLt (rocroller) backend
  triton            = 3,  ///< Triton backend
  composable_kernel = 4,  ///< Composable Kernel backend (Not supported yet)
  count,                  ///< Count of target types
  none = 0xFFFFFFFFu      ///< Explicitly invalid
};

/**
 * @brief Indicates whether a matrix is supplied in transposed or not.
 */
enum class transpose_t {
  T,  ///< Transposed
  N,  ///< Not transposed

  Count
};

}  // namespace origami
