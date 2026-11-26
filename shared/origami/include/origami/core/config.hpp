// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>

#include "origami/backends/backend.hpp"
#include "origami/backends/tensilelite.hpp"
#include "origami/core/dimensions.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/math.hpp"
#include "origami/core/runtime_options.hpp"

namespace origami {

/**
 * @brief Full kernel configuration (tile shape + execution parameters).
 *
 * Holds the geometric tile sizes along with occupancy,
 * work-group mapping (WGM), and cache-control hints.
 */
struct config_t {
  /// Macro tile and matrix-instruction shape.
  dim3_t mt{0, 0, 0};
  dim3_t mi{0, 0, 0};

  /// Main loop optimization flag (indicates use of any optimized kernel variant)
  bool hand_optimized_main_loop = false;

  /// Whether this kernel uses the subtile implementation (UseSubtileImpl).
  bool subtile = false;

  /// Occupancy (number of wavefronts resident per CU).
  int occupancy = -1;

  /// Reorder workgroup id for L2 reuse.
  int workgroup_mapping = 0;

  /// Whether operand A is accessed with cache-flags.
  int cache_hints_a = 0;

  /// Whether operand B is accessed with cache-flags.
  int cache_hints_b = 0;

  /// Whether output D is accessed with cache-flags (streaming/non-temporal).
  int cache_hints_d = 0;

  /// Filled by memoize_non_temporal_availability() when the candidate list is
  /// built. True when this macro tile has a non-temporal read of that operand
  /// in the list. rank_configs() reads these and does not rescan.
  bool non_temporal_a_available = false;
  bool non_temporal_b_available = false;

  /// Workspace size parameters.
  std::size_t workspace_size            = 0;
  std::size_t workspace_size_per_elem_c = 0;

  /// Stream-K mode selector (0 disables stream-K; 1 to 5 enables stream-K).
  int stream_k = 5;

  /// Reduction strategy.
  reduction_t reduction_strategy = reduction_t::none;

  /// Prediction mode for latency estimation.
  prediction_modes_t prediction_mode = prediction_modes_t::estimation;

  /// Target backend for kernel execution.
  target_t target = target_t::tensilelite;

  /// Grid selection algorithm.
  grid_selection_t grid_selection = grid_selection_t::k_split_aware;

  /// Index of config, not used by Origami but can be used by the user
  std::size_t index = 0;

  /// Global read vector width for matrix A (elements per load)
  std::size_t grvw_a = 1;

  /// Global read vector width for matrix B (elements per load)
  std::size_t grvw_b = 1;

  /// Global write vector width for matrix D (elements per store)
  std::size_t gwvw_d = 1;

  /// TensileLite VectorWidthA: contiguous M elements a lane owns across adjacent
  /// MFMA blocks.  Sets the LDS read width only when A is M-contiguous in LDS.
  int vector_width_a = 1;

  /// TensileLite VectorWidthB: contiguous N elements a lane owns across adjacent
  /// MFMA blocks.  Sets the LDS read width only when B is N-contiguous in LDS.
  int vector_width_b = 1;

  /// cluster dimensions
  dim3_t cluster_dim{1, 1, 1};

  /// Backend-specific parameters (type should match target).
  /// Use tensile() accessor to get/set Tensile-specific params.
  backend_params_t backend{};

  /// Get mutable reference to Tensile params. Initializes if not already set.
  tensile_params_t& tensile() {
    if (!std::holds_alternative<tensile_params_t>(backend)) { backend = tensile_params_t{}; }
    return std::get<tensile_params_t>(backend);
  }

  /// Get const reference to Tensile params. Throws if not set.
  const tensile_params_t& tensile() const { return std::get<tensile_params_t>(backend); }

  /// Check if Tensile params are currently set.
  bool has_tensile_params() const noexcept {
    return std::holds_alternative<tensile_params_t>(backend);
  }

  /// The fields that identify a kernel, apart from the backend parameters.  operator== and
  /// hash() are both defined over this list.  occupancy, the non-temporal availability flags,
  /// the workspace sizes, grid_selection and index are not part of it.
  auto identity() const noexcept {
    return std::make_tuple(mt.m,
                           mt.n,
                           mt.k,
                           mi.m,
                           mi.n,
                           mi.k,
                           hand_optimized_main_loop,
                           subtile,
                           cache_hints_a,
                           cache_hints_b,
                           cache_hints_d,
                           stream_k,
                           workgroup_mapping,
                           static_cast<std::uint32_t>(reduction_strategy),
                           static_cast<std::uint32_t>(prediction_mode),
                           static_cast<std::uint32_t>(target),
                           grvw_a,
                           grvw_b,
                           gwvw_d,
                           vector_width_a,
                           vector_width_b,
                           cluster_dim.m,
                           cluster_dim.n,
                           cluster_dim.k);
  }

  /// Equality over identity() and the backend parameters.
  bool operator==(const config_t& o) const noexcept {
    return identity() == o.identity() && backend == o.backend;
  }

  /// @return Hash over the fields operator== compares.
  std::size_t hash() const {
    std::size_t seed =
        std::apply([](const auto&... field) { return math::hash_combine(field...); }, identity());
    // Hash backend-specific parameters if present. The visitor pattern allows
    // automatic handling of any backend type that provides a hash() method,
    // while std::monostate (no backend params) is a no-op.
    std::visit(
        [&seed](const auto& params) {
          if constexpr (!std::is_same_v<std::decay_t<decltype(params)>, std::monostate>) {
            math::hash_combine(seed, params.hash());
          }
        },
        backend);
    return seed;
  }

  /// Throw std::runtime_error unless is_valid().
  void validate() const {
    if (!is_valid()) { throw std::runtime_error("Invalid config_t"); }
  }

  /// @return True when the tile, matrix-instruction and cluster dimensions and the occupancy are
  /// all positive.
  bool is_valid() const {
    return mt.m > 0 && mt.n > 0 && mt.k > 0 && mi.m > 0 && mi.n > 0 && mi.k > 0 && occupancy > 0 &&
           cluster_dim.m > 0 && cluster_dim.n > 0 && cluster_dim.k > 0;
  }
};

/**
 * @brief Get runtime options (always uses global singleton).
 *
 * @param config Configuration struct (unused, kept for API compatibility)
 * @return const runtime_options& Reference to runtime options singleton
 */
inline const runtime_options& get_runtime_options(const config_t& config) {
  (void)config;  // Unused parameter - kept for API compatibility
  return runtime_options::get();
}

}  // namespace origami

namespace std {

/**
 * @brief std::hash for origami::config_t, so it can key unordered containers.
 *
 * Inline so that several shared libraries including this header do not violate the ODR.
 */
template <>
struct hash<origami::config_t> {
  inline std::size_t operator()(const origami::config_t& config) const noexcept {
    return config.hash();
  }
};

}  // namespace std
