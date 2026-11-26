// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <utility>

#include "origami/core/enums.hpp"

namespace origami {

/**
 * @brief Tag types, one per cost model, that select a performance model at compile time.
 *
 * Code that works for any cost model is a template over one of these tags and reads what differs
 * between cost models from cost_model_traits.
 */
namespace cost_model {

struct attention {};  ///< Flash attention (model_t::attention).
struct gemm {};       ///< Matrix multiplication (model_t::gemm).

/**
 * @brief Call @p fn with the tag of @p model.
 *
 * This is the one place a run-time model becomes a compile-time tag.  @p fn must return the same
 * type for every tag; a model that names no cost model takes the gemm tag.
 */
template <class Fn>
constexpr decltype(auto) dispatch(model_t model, Fn&& fn) {
  switch (model) {
    case model_t::attention: return std::forward<Fn>(fn)(attention{});
    case model_t::gemm:
    default: return std::forward<Fn>(fn)(gemm{});
  }
}

}  // namespace cost_model

/**
 * @brief Why a cost model does not admit a kernel configuration.
 *
 * A rejected configuration is not scored: ranking gives it the maximum latency and drops it.
 * A cost model reports the first of its rules that fails; cost_model_traits<W>::admit gives the
 * order.
 */
enum class rejection_t {
  none,  ///< The configuration is admitted

  // Capacity limits.
  lds_capacity,  ///< The tiles do not fit in local data share (LDS) memory
  rf_capacity,   ///< The tiles do not fit in the register file (RF)

  // Rules of the GEMM cost model.
  invalid_config,           ///< A tile, matrix-instruction or cluster dimension, or occupancy, is 0
  heuristic,                ///< A registered heuristic rejects the kernel for this problem
  forced_pick,              ///< ORIGAMI_GEMM_PICK forces another macro tile
  dot2_large_m,             ///< A Dot2 kernel is kept only for M <= 2
  non_temporal_b_required,  ///< A non-temporal B variant of the tile exists and this is not it
  non_temporal_a_required,  ///< A non-temporal A variant of the tile exists and this is not it
  non_temporal_not_allowed,  ///< The problem's shape does not call for non-temporal loads
  non_temporal_misaligned    ///< K or MT_K is not a whole number of cache lines
};

/// Latency given to a configuration that its cost model does not admit.
inline constexpr double REJECTED_LATENCY = std::numeric_limits<double>::max();

/**
 * @brief What a cost model says about one configuration.
 */
struct evaluation_t {
  rejection_t rejection = rejection_t::none;  ///< Why it is not admitted, or none
  double latency        = REJECTED_LATENCY;   ///< Predicted latency; REJECTED_LATENCY if rejected
};

/**
 * @brief Convert rejection_t to the text used in logs.
 *
 * @param rejection Rejection reason
 * @return const char* Description of the reason
 */
constexpr const char* to_string(rejection_t rejection) noexcept {
  switch (rejection) {
    case rejection_t::none: return "none";
    case rejection_t::lds_capacity: return "LDS capacity exceeded";
    case rejection_t::rf_capacity: return "Register File (RF) capacity exceeded";
    case rejection_t::invalid_config: return "Invalid configuration";
    case rejection_t::heuristic: return "Rejected by a heuristic";
    case rejection_t::forced_pick: return "ORIGAMI_GEMM_PICK forces another macro tile";
    case rejection_t::dot2_large_m: return "Dot2 is kept only for M <= 2";
    case rejection_t::non_temporal_b_required: return "The non-temporal B variant is required";
    case rejection_t::non_temporal_a_required: return "The non-temporal A variant is required";
    case rejection_t::non_temporal_not_allowed: return "Non-temporal loads are not allowed";
    case rejection_t::non_temporal_misaligned: return "Non-temporal loads need whole cache lines";
    default: return "Invalid";
  }
}

/**
 * @brief What a cost model provides to the code that is generic over cost models.
 *
 * A cost model specializes this template for its tag.  Every specialization provides:
 *  - `name`: the cost model's name,
 *  - `model`: the model_t value the tag stands for,
 *  - `static rejection_t admit(const problem_t&, const hardware_t&, const config_t&)`: the
 *    first rule the configuration fails, or rejection_t::none,
 *  - `static double latency(const problem_t&, const hardware_t&, const config_t&)`: the
 *    predicted latency, in cycles, of an admitted configuration,
 *  - `static evaluation_t evaluate(const problem_t&, const hardware_t&, const config_t&)`: both
 *    at once, which is what ranking calls.
 */
template <class Model>
struct cost_model_traits;

}  // namespace origami
