// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include "origami/core/config.hpp"
#include "origami/core/enums.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"

namespace origami {

/**
 * @brief Tag types, one per method of predicting a latency, that select it at compile time.
 */
namespace method {

struct analytical {};  ///< Closed-form estimation (prediction_modes_t::estimation).
struct simulation {};  ///< Kernel simulation (prediction_modes_t::simulation).

/**
 * @brief Call @p fn with the tag of @p mode.
 *
 * This is the one place a run-time prediction mode becomes a compile-time tag.  @p fn must
 * return the same type for every tag; a mode that names no method takes the analytical tag.
 */
template <class Fn>
constexpr decltype(auto) dispatch(prediction_modes_t mode, Fn&& fn) {
  switch (mode) {
    case prediction_modes_t::simulation: return std::forward<Fn>(fn)(simulation{});
    case prediction_modes_t::estimation:
    default: return std::forward<Fn>(fn)(analytical{});
  }
}

/**
 * @brief Customization point: the latency a cost model predicts with one method.
 *
 * A cost model provides a method by specializing this template for its own tag and the method's
 * tag, with one static member:
 *
 * @code
 * template <>
 * struct method::evaluator<cost_model::gemm, method::analytical> {
 *   static double latency(const problem_t&, const hardware_t&, const config_t&);
 * };
 * @endcode
 *
 * Every cost model must specialize the analytical method.  The primary template is what a cost
 * model gets for a method it does not provide: it evaluates the analytical one instead.
 */
template <class Model, class Method>
struct evaluator {
  /**
   * @brief Latency of @p config for @p problem on @p hardware, in the method's unit.
   */
  static double latency(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config) {
    static_assert(!std::is_same_v<Method, analytical>,
                  "a cost model must specialize method::evaluator<Model, method::analytical>");
    return evaluator<Model, analytical>::latency(problem, hardware, config);
  }
};

/**
 * @brief Latency of @p config under the method its prediction_mode names.
 *
 * Methods need not share a unit: for GEMM the analytical method returns cycles and the simulation
 * returns microseconds.  Compare latencies of one method only.
 *
 * @tparam Model Model tag (@see origami::cost_model)
 * @param problem Problem description
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration; config.prediction_mode selects the method
 */
template <class Model>
double latency(const problem_t& problem, const hardware_t& hardware, const config_t& config) {
  return dispatch(config.prediction_mode, [&](auto tag) {
    return evaluator<Model, decltype(tag)>::latency(problem, hardware, config);
  });
}

}  // namespace method
}  // namespace origami
