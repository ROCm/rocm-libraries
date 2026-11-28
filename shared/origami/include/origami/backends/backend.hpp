// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <variant>

#include "origami/backends/tensilelite.hpp"
#include "origami/core/enums.hpp"

namespace origami {

/// Variant holding backend-specific parameters.
/// std::monostate represents no backend-specific params (default).
using backend_params_t = std::variant<std::monostate, tensile_params_t>;

/**
 * @brief Tag types, one per kernel backend, that select backend behaviour at compile time.
 *
 * Code that differs between backends is a template over one of these tags.  Its primary
 * template holds the behaviour the backends share and a specialization holds what one backend
 * does differently.
 */
namespace backend {

struct composable_kernel {};
struct generic {};
struct rocroller {};
struct tensilelite {};
struct triton {};

/**
 * @brief Call @p fn with the tag of @p target.
 *
 * This is the one place a run-time target becomes a compile-time tag.  @p fn must return the
 * same type for every tag; a target that names no backend takes the generic tag.
 */
template <class Fn>
constexpr decltype(auto) dispatch(target_t target, Fn&& fn) {
  switch (target) {
    case target_t::composable_kernel: return std::forward<Fn>(fn)(composable_kernel{});
    case target_t::rocroller: return std::forward<Fn>(fn)(rocroller{});
    case target_t::tensilelite: return std::forward<Fn>(fn)(tensilelite{});
    case target_t::triton: return std::forward<Fn>(fn)(triton{});
    case target_t::generic:
    default: return std::forward<Fn>(fn)(generic{});
  }
}

}  // namespace backend

/**
 * @brief What is known about a backend at compile time.
 *
 * Every specialization provides:
 *  - `name`: the backend's name,
 *  - `target`: the target_t value the tag stands for,
 *  - `params_type`: the alternative of backend_params_t that carries the backend's parameters
 *    (std::monostate when it has none).
 */
template <class Backend>
struct backend_traits;

template <>
struct backend_traits<backend::composable_kernel> {
  static constexpr const char* name = "composable_kernel";
  static constexpr target_t target  = target_t::composable_kernel;
  using params_type                 = std::monostate;
};

template <>
struct backend_traits<backend::generic> {
  static constexpr const char* name = "generic";
  static constexpr target_t target  = target_t::generic;
  using params_type                 = std::monostate;
};

template <>
struct backend_traits<backend::rocroller> {
  static constexpr const char* name = "rocroller";
  static constexpr target_t target  = target_t::rocroller;
  using params_type                 = std::monostate;
};

template <>
struct backend_traits<backend::tensilelite> {
  static constexpr const char* name = "tensilelite";
  static constexpr target_t target  = target_t::tensilelite;
  using params_type                 = tensile_params_t;
};

template <>
struct backend_traits<backend::triton> {
  static constexpr const char* name = "triton";
  static constexpr target_t target  = target_t::triton;
  using params_type                 = std::monostate;
};

}  // namespace origami
