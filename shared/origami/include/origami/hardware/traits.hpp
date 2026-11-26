// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/hardware/architecture.hpp"

namespace origami {

/**
 * @brief Structural facts of one architecture: what the hardware is, as opposed to the
 *        measured constants of @ref architecture_constants.
 */
struct architecture_traits {
  architecture_t arch;  ///< Architecture this row describes

  // Execution
  std::size_t wavefront_size;       ///< Threads per wavefront TensileLite builds by default
  std::size_t simd_lanes;           ///< Lanes per SIMD
  std::size_t simds_per_cu;         ///< SIMD units per compute unit
  bool has_accumulation_registers;  ///< Matrix results land in separate accumulation registers
                                    ///< that the epilogue has to copy out
  bool has_native_tf32;             ///< Has TF32 matrix instructions (no emulation needed)

  // Memory
  std::size_t lds_read_bytes_per_clk;  ///< LDS read bandwidth per CU in bytes per clock
  std::size_t l1_capacity;             ///< L1 data cache per CU in bytes
  std::size_t cache_line_bytes;        ///< L2 cache-line size in bytes
  bool has_mall;                       ///< Has a MALL (Memory Attached Last Level) cache

  // Topology
  std::size_t default_num_xcds;            ///< XCD count without a runtime query; 0 if none
  std::size_t max_xcds_per_l2;             ///< Most XCDs that share one L2 cache
  std::size_t cus_per_reported_processor;  ///< Physical CUs per HIP multiProcessorCount unit
  bool dispatch_starts_at_last_xcd;        ///< Dispatch starts at the last XCD, not XCD 0
};

/**
 * @brief The traits table: one row per architecture, in enumeration order.
 *
 * Wavefront and SIMD: CDNA2-4 run wave64 on four SIMD16 per CU, RDNA wave32 on two SIMD32,
 * and gfx1250 wave32 on four SIMD32.  gfx1250 keeps matrix results in its unified VGPR file,
 * so it has no accumulation registers to copy out.
 *
 * LDS: 32 four-byte banks (128 B/clk) through CDNA3 and on RDNA; CDNA4 doubles the bank
 * count and read bandwidth to 256 B/clk, and gfx1250 keeps 64 banks.  gfx1250 has no L1
 * capacity because its tensor-DMA loads bypass L1.
 *
 * Topology: gfx1250 shares one L2 between the up to 4 XCDs of an AID, and its dispatch id 0
 * currently lands on the last XCD (XCC 7 on all GPUs measured; to be fixed with
 * ClusterAIDRemapping).  On RDNA, HIP runs in WGP mode, where multiProcessorCount counts
 * WGPs of 2 CUs each; CDNA and gfx1250 report the count Origami models.  The default XCD
 * counts are a fallback for architectures that predate the runtime XCC query
 * (hipDeviceAttributeNumberOfXccs, HIP 7.0+); a new architecture should leave the column at
 * 0 and rely on the query.
 *
 * The columns are the fields of @ref architecture_traits, in order.
 */
// clang-format off
inline constexpr architecture_traits ARCHITECTURE_TRAITS[] = {
    // architecture           wave  lanes  simds  accum  tf32   lds  l1         line  mall   xcds  /l2  cus  last
    {architecture_t::gfx90a,  64,   16,    4,     true,  false, 128, 16 * 1024, 128,  true,  1,    1,   1,   false},
    {architecture_t::gfx942,  64,   16,    4,     true,  true,  128, 32 * 1024, 128,  true,  8,    1,   1,   false},
    {architecture_t::gfx950,  64,   16,    4,     true,  false, 256, 32 * 1024, 128,  true,  8,    1,   1,   false},
    {architecture_t::gfx1200, 32,   32,    2,     true,  false, 128, 64 * 1024, 128,  true,  1,    1,   2,   false},
    {architecture_t::gfx1201, 32,   32,    2,     true,  false, 128, 64 * 1024, 128,  true,  1,    1,   2,   false},
    {architecture_t::gfx1170, 32,   32,    2,     true,  false, 128, 64 * 1024, 128,  false, 1,    1,   2,   false},
    {architecture_t::gfx1171, 32,   32,    2,     true,  false, 128, 64 * 1024, 128,  false, 1,    1,   2,   false},
    {architecture_t::gfx1100, 32,   32,    2,     true,  false, 128, 32 * 1024, 128,  true,  1,    1,   2,   false},
    {architecture_t::gfx1101, 32,   32,    2,     true,  false, 128, 32 * 1024, 128,  true,  1,    1,   2,   false},
    {architecture_t::gfx1150, 32,   32,    2,     true,  false, 128, 32 * 1024, 128,  false, 1,    1,   2,   false},
    {architecture_t::gfx1151, 32,   32,    2,     true,  false, 128, 32 * 1024, 128,  true,  1,    1,   2,   false},
    {architecture_t::gfx1152, 32,   32,    2,     true,  false, 128, 32 * 1024, 128,  false, 1,    1,   2,   false},
    {architecture_t::gfx1153, 32,   32,    2,     true,  false, 128, 32 * 1024, 128,  false, 1,    1,   2,   false},
    {architecture_t::gfx1250, 32,   32,    4,     false, false, 256, 0,         128,  false, 8,    4,   1,   true},
};

/**
 * @brief Traits assumed for an unrecognized architecture.
 */
inline constexpr architecture_traits UNKNOWN_ARCHITECTURE_TRAITS =
    {architecture_t::Count,   32,   32,    4,     true,  false, 128, 32 * 1024, 128,  false, 0,    1,   1,   false};
// clang-format on

static_assert(has_one_row_per_architecture(ARCHITECTURE_TRAITS),
              "ARCHITECTURE_TRAITS needs one row per architecture_t enumerator, in the same order");

/**
 * @brief Traits of an architecture.
 *
 * @param arch Architecture
 * @return const architecture_traits& The architecture's row, or @ref UNKNOWN_ARCHITECTURE_TRAITS
 *         if it is not a supported one
 */
constexpr const architecture_traits& traits_of(architecture_t arch) noexcept {
  return is_known_architecture(arch) ? ARCHITECTURE_TRAITS[static_cast<std::size_t>(arch)]
                                     : UNKNOWN_ARCHITECTURE_TRAITS;
}

}  // namespace origami
