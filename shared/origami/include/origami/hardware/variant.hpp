// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "origami/hardware/architecture.hpp"

namespace origami {

/**
 * @brief Chip variant: a part of an architecture that has its own measured constants.
 *
 * Variants share the instruction set and traits of their architecture and differ only in
 * the constants row they select.  @ref resolve_variant is the one place that derives a
 * variant from raw device identifiers.
 */
enum class variant_t {
  standard,       ///< The architecture's primary part; the only variant of most architectures
  gfx950_id75a8,  ///< gfx950 part with PCI device id 0x75a8
  gfx1250_6xcd    ///< gfx1250 6-XCD part: 192 WGPs, or 96 WGPs in a DPX partition
};

/**
 * @brief Name of a variant (e.g. for logging).
 */
constexpr std::string_view variant_name(variant_t variant) noexcept {
  switch (variant) {
    case variant_t::standard: return "standard";
    case variant_t::gfx950_id75a8: return "gfx950_id75a8";
    case variant_t::gfx1250_6xcd: return "gfx1250_6xcd";
    default: return "unknown";
  }
}

/**
 * @brief Variant of a device, from its architecture and raw identifiers.
 *
 * gfx950 is told apart by PCI device id: low 16 bits @c 0x75a8 of the chip id (as reported
 * by e.g. @c hipDeviceAttributePciChipId) select @ref variant_t::gfx950_id75a8; a missing or
 * any other id is the standard part.
 *
 * gfx1250 is told apart by WGP count: 192 WGPs, or 96 in a DPX partition, are the 6-XCD
 * part.  No partition of the 8-XCD part has these counts; 64 and 32 WGPs are ambiguous and
 * count as the standard 8-XCD part, as does a count of 0.
 *
 * Every other architecture has only the standard variant.
 *
 * @param arch Architecture of the device
 * @param pci_chip_id PCI chip id, if known; used for gfx950 only
 * @param n_cu Compute-unit (WGP) count of the device; used for gfx1250 only
 * @return variant_t The device's variant
 */
constexpr variant_t resolve_variant(architecture_t arch,
                                    std::optional<int> pci_chip_id = std::nullopt,
                                    std::size_t n_cu               = 0) noexcept {
  switch (arch) {
    case architecture_t::gfx950: {
      const bool is_id75a8 =
          pci_chip_id.has_value() && (static_cast<unsigned>(*pci_chip_id) & 0xFFFFu) == 0x75a8u;
      return is_id75a8 ? variant_t::gfx950_id75a8 : variant_t::standard;
    }
    case architecture_t::gfx1250: {
      const bool is_6xcd = n_cu == 192 || n_cu == 96;
      return is_6xcd ? variant_t::gfx1250_6xcd : variant_t::standard;
    }
    default: return variant_t::standard;
  }
}

}  // namespace origami
