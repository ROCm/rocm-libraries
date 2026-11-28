// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace origami {

/**
 * @brief Struct to define various workgroup mapping parameters.
 *
 * Contains all the parameters needed to describe various workgroup mapping parameters.
 */
struct workgroup_mapping_t {
  /// Split-K factor for K-Coherent reorder (0 or 1 = disabled).
  std::size_t wgmxccsplitk = 0;

  /// Workgroup mapping chunk size.
  std::size_t wgmxccchunk = 0;

  /// Number of XCCs the workgroup ids are remapped across.
  std::size_t wgmxcc = 8;

  /// Workgroup mapping slab width (negative = column-major).
  int32_t wgm = 1;
};

/**
 * @brief Struct to define various staggerU parameters.
 *
 * Contains all the parameters needed to describe various staggerU parameters.
 */
struct staggerU_t {
  /// StaggerU mapping size.
  std::size_t staggerUMapping = 0;

  /// StaggerU size.
  std::size_t staggerU = 0;

  /// StaggerUStrideShift size.
  std::size_t staggerUStrideShift = 0;
};

}  // namespace origami
