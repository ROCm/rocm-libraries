// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace origami {

/**
 * @brief A compact 3-D dimension triple (M, N, K).
 *
 * Provides convenient accessors for common GEMM tiling parameters
 * and helpers like mnk() for volume.
 */
struct dim3_t {
  /// M dimension (rows).
  std::size_t m;

  /// N dimension (columns).
  std::size_t n;

  /// K dimension (reduction).
  std::size_t k;

  constexpr bool operator==(const dim3_t& o) const noexcept {
    return m == o.m && n == o.n && k == o.k;
  }

  constexpr bool operator!=(const dim3_t& o) const noexcept { return !(*this == o); }

  /// @return Product m*n.
  constexpr std::size_t mn() const noexcept { return m * n; }

  /// @return Product m*k.
  constexpr std::size_t mk() const noexcept { return m * k; }

  /// @return Product n*k.
  constexpr std::size_t nk() const noexcept { return n * k; }

  /// @return Product m*n*k.
  constexpr std::size_t mnk() const noexcept { return m * n * k; }
};

/**
 * @brief 4-dimensional size/coordinate: (k, m, n, b).
 *
 * Used for tile coordinates and unique tile counts across the GEMM grid.
 */
struct dim4_t {
  /// K dimension (reduction / split).
  std::size_t k = 0;

  /// M dimension (rows).
  std::size_t m = 0;

  /// N dimension (columns).
  std::size_t n = 0;

  /// B dimension (batch).
  std::size_t b = 0;

  constexpr bool operator==(const dim4_t& o) const noexcept {
    return k == o.k && m == o.m && n == o.n && b == o.b;
  }

  constexpr bool operator!=(const dim4_t& o) const noexcept { return !(*this == o); }

  /// @return Product m*n.
  constexpr std::size_t mn() const noexcept { return m * n; }

  /// @return Product m*n*k.
  constexpr std::size_t mnk() const noexcept { return m * n * k; }

  /// @return Product k*m*n*b.
  constexpr std::size_t total() const noexcept { return k * m * n * b; }
};

}  // namespace origami
