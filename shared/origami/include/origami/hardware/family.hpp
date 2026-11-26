// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "origami/hardware/architecture.hpp"
#include "origami/hardware/hardware.hpp"

namespace origami {

/**
 * @brief Tag types, one per family, that select a model specialization at compile time.
 *
 * A model is a template over one of these tags.  Its primary template holds the behaviour the
 * families share and a specialization holds what one family does differently.  The families
 * themselves (@ref family_t, @ref family_of) are defined in architecture.hpp.
 */
namespace arch {

struct cdna2 {};
struct cdna3 {};
struct cdna4 {};
struct cdna5 {};
struct rdna3 {};
struct rdna4 {};

/**
 * @brief Call @p fn with the tag of @p family.
 *
 * This is the one place a run-time family becomes a compile-time tag.  @p fn must return the
 * same type for every tag; an unknown family takes the cdna3 tag.
 */
template <class Fn>
constexpr decltype(auto) dispatch(family_t family, Fn&& fn) {
  switch (family) {
    case family_t::cdna2: return std::forward<Fn>(fn)(cdna2{});
    case family_t::cdna4: return std::forward<Fn>(fn)(cdna4{});
    case family_t::cdna5: return std::forward<Fn>(fn)(cdna5{});
    case family_t::rdna3: return std::forward<Fn>(fn)(rdna3{});
    case family_t::rdna4: return std::forward<Fn>(fn)(rdna4{});
    case family_t::cdna3:
    default: return std::forward<Fn>(fn)(cdna3{});
  }
}

/**
 * @brief Call @p fn with the tag of the family of @p hardware.
 */
template <class Fn>
decltype(auto) dispatch(const hardware_t& hardware, Fn&& fn) {
  return dispatch(hardware.family(), std::forward<Fn>(fn));
}

}  // namespace arch
}  // namespace origami
