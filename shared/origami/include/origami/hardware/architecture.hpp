// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string_view>

namespace origami {

/**
 * @brief Supported GPU architectures.
 *
 * The order fixes the integer value of every enumerator, which callers may have stored, so a
 * new architecture goes last, before Count.  Every architecture has one row in
 * @ref ARCHITECTURES, in this order.
 */
enum class architecture_t {
  gfx90a,
  gfx942,
  gfx950,
  gfx1200,
  gfx1201,
  gfx1170,
  gfx1171,
  gfx1100,
  gfx1101,
  gfx1150,
  gfx1151,
  gfx1152,
  gfx1153,
  gfx1250,
  Count  ///< Number of architectures; also the value of an unrecognized architecture.
};

/**
 * @brief Architecture family: the generation whose behaviour a model is written for.
 *
 * Models are specialized per family, not per architecture.  An architecture belongs to the
 * family whose instruction set and memory hierarchy it shares.
 */
enum class family_t {
  cdna2,
  cdna3,
  cdna4,
  cdna5,
  rdna3,
  rdna4,
  unknown  ///< Family of an unrecognized architecture.
};

/**
 * @brief Identity of one architecture: its name and its family.
 */
struct architecture_info {
  architecture_t arch;    ///< Architecture this row describes
  std::string_view name;  ///< Name as HIP reports it, without feature suffixes (e.g. "gfx942")
  family_t family;        ///< Family the architecture belongs to
};

/**
 * @brief The architecture table: one row per architecture, in enumeration order.
 *
 * Name conversion and family lookup both read this table, so adding an architecture is one
 * enumerator and one row here, plus its rows in the traits and constants tables.
 */
// clang-format off
inline constexpr architecture_info ARCHITECTURES[] = {
    // architecture            name       family
    {architecture_t::gfx90a,  "gfx90a",  family_t::cdna2},
    {architecture_t::gfx942,  "gfx942",  family_t::cdna3},
    {architecture_t::gfx950,  "gfx950",  family_t::cdna4},
    {architecture_t::gfx1200, "gfx1200", family_t::rdna4},
    {architecture_t::gfx1201, "gfx1201", family_t::rdna4},
    {architecture_t::gfx1170, "gfx1170", family_t::rdna4},
    {architecture_t::gfx1171, "gfx1171", family_t::rdna4},
    {architecture_t::gfx1100, "gfx1100", family_t::rdna3},
    {architecture_t::gfx1101, "gfx1101", family_t::rdna3},
    {architecture_t::gfx1150, "gfx1150", family_t::rdna3},
    {architecture_t::gfx1151, "gfx1151", family_t::rdna3},
    {architecture_t::gfx1152, "gfx1152", family_t::rdna3},
    {architecture_t::gfx1153, "gfx1153", family_t::rdna3},
    {architecture_t::gfx1250, "gfx1250", family_t::cdna5},
};
// clang-format on

/**
 * @brief Whether @p rows holds exactly one row per architecture, in enumeration order.
 *
 * A per-architecture table that passes this check can be indexed by the architecture's value.
 *
 * @tparam Row Row type with an `arch` member
 */
template <class Row, std::size_t N>
constexpr bool has_one_row_per_architecture(const Row (&rows)[N]) noexcept {
  if (N != static_cast<std::size_t>(architecture_t::Count)) { return false; }
  for (std::size_t i = 0; i < N; ++i) {
    if (static_cast<std::size_t>(rows[i].arch) != i) { return false; }
  }
  return true;
}

static_assert(has_one_row_per_architecture(ARCHITECTURES),
              "ARCHITECTURES needs one row per architecture_t enumerator, in the same order");

/**
 * @brief Whether @p arch is one of the supported architectures.
 */
constexpr bool is_known_architecture(architecture_t arch) noexcept {
  return static_cast<std::size_t>(arch) < static_cast<std::size_t>(architecture_t::Count);
}

/**
 * @brief Architecture with the given name.
 *
 * @param name Architecture name (e.g. "gfx90a", "gfx942")
 * @return architecture_t The architecture, or architecture_t::Count if the name is not recognized
 */
constexpr architecture_t architecture_from_name(std::string_view name) noexcept {
  for (const architecture_info& info : ARCHITECTURES) {
    if (info.name == name) { return info.arch; }
  }
  return architecture_t::Count;
}

/**
 * @brief Name of an architecture (e.g. for logging).
 *
 * @param arch Architecture
 * @return std::string_view The architecture's name, or "unknown" if it is not a supported one
 */
constexpr std::string_view architecture_name(architecture_t arch) noexcept {
  return is_known_architecture(arch) ? ARCHITECTURES[static_cast<std::size_t>(arch)].name
                                     : std::string_view("unknown");
}

/**
 * @brief Family of an architecture.
 *
 * @param arch Architecture
 * @return family_t The architecture's family, or family_t::unknown if it is not a supported one
 */
constexpr family_t family_of(architecture_t arch) noexcept {
  return is_known_architecture(arch) ? ARCHITECTURES[static_cast<std::size_t>(arch)].family
                                     : family_t::unknown;
}

/**
 * @brief Name of a family (e.g. for logging).
 */
constexpr std::string_view family_name(family_t family) noexcept {
  switch (family) {
    case family_t::cdna2: return "cdna2";
    case family_t::cdna3: return "cdna3";
    case family_t::cdna4: return "cdna4";
    case family_t::cdna5: return "cdna5";
    case family_t::rdna3: return "rdna3";
    case family_t::rdna4: return "rdna4";
    default: return "unknown";
  }
}

}  // namespace origami
