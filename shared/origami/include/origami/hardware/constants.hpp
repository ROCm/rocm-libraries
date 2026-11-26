// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <tuple>

#include "origami/hardware/architecture.hpp"
#include "origami/hardware/traits.hpp"
#include "origami/hardware/variant.hpp"

namespace origami {

/**
 * @brief MALL value for those architectures that do not support it.
 */
inline constexpr double NO_MALL_AVAILABLE = 1.21875121875121875122 * 1000;

/**
 * @brief Architecture-specific constants for memory and compute characteristics.
 */
struct architecture_constants {
  double mem1_perf_ratio;      ///< Memory level 1 performance ratio (microbenchmarked)
  double mem2_perf_ratio;      ///< Memory level 2 performance ratio (microbenchmarked)
  double mem3_perf_ratio;      ///< Memory level 3 performance ratio (microbenchmarked)
  std::size_t parallel_mi_cu;  ///< Number of parallel matrix instructions per compute unit
  std::tuple<double, double, double>
      mem_bw_per_wg_coefficients;  ///< Memory bandwidth coefficients per workgroup
  double mem_clock_ratio;          ///< Memory clock ratio relative to compute clock
  std::size_t simds_per_cu;  ///< SIMD units per Compute Unit (CDNA: 4 SIMD16, RDNA3/4: 2 SIMD32)
  std::size_t l1_capacity;   ///< L1 data cache capacity per CU in bytes (shared across all SIMDs)

  constexpr architecture_constants(double mem1_perf_ratio,
                                   double mem2_perf_ratio,
                                   double mem3_perf_ratio,
                                   std::size_t parallel_mi_cu,
                                   std::tuple<double, double, double> mem_bw_per_wg_coefficients,
                                   double mem_clock_ratio,
                                   std::size_t simds_per_cu,
                                   std::size_t l1_capacity)
      : mem1_perf_ratio(mem1_perf_ratio)
      , mem2_perf_ratio(mem2_perf_ratio)
      , mem3_perf_ratio(mem3_perf_ratio)
      , parallel_mi_cu(parallel_mi_cu)
      , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
      , mem_clock_ratio(mem_clock_ratio)
      , simds_per_cu(simds_per_cu)
      , l1_capacity(l1_capacity) {}
};

/**
 * @brief Measured constants of one (architecture, variant) pair.
 *
 * The fields have the meaning of the @ref architecture_constants fields of the same name.
 */
struct constants_row {
  architecture_t arch;  ///< Architecture this row describes
  variant_t variant;    ///< Variant of the architecture this row describes
  double mem1_perf_ratio;
  double mem2_perf_ratio;
  double mem3_perf_ratio;
  std::size_t parallel_mi_cu;
  std::tuple<double, double, double> mem_bw_per_wg_coefficients;
  double mem_clock_ratio;
  std::size_t wgps_measured;  ///< WGP count mem1 and mem3 were measured at and scale from;
                              ///< 0 if they do not depend on the device's WGP count
};

/**
 * @brief gfx1250 6-XCD part: L2 read bandwidth of a 96-WGP partition, the measured 14.75 TB/s
 *        times the refit ratio (see @ref ARCHITECTURE_CONSTANTS).
 */
inline constexpr double GFX1250_6XCD_MEM1 = 14.75 * (41301.81761089743 / 41930.0);

/**
 * @brief gfx1250 6-XCD part: DRAM read bandwidth of a 96-WGP partition, the measured 8.637 TB/s
 *        times the refit ratio, scaled by memory/engine clock.
 */
inline constexpr double GFX1250_6XCD_MEM3 =
    8.637 * (18382.976995668938 / 19725.454545454544) * 1.65 / 2.4;

/**
 * @brief The constants table: one or more rows per architecture, architectures in enumeration
 *        order and each architecture's standard variant first.
 *
 * The values are obtained through microbenchmarking.
 *
 * gfx950: the standard row is the id75a0 part; the id75a8 part has its own row.
 *
 * gfx1250, standard 8-XCD part (256 WGPs, 1.9 GHz memory): peak tensor-DMA read bandwidth
 * L2 41.93 TB/s, sustained DRAM 17.13 TB/s.  6-XCD part (1.65 GHz memory), measured on a
 * 96-WGP DPX/NPS2 partition: L2 14.75 TB/s, DRAM 8.64 TB/s; both scale with the WGPs of the
 * partition (192 WGPs unmeasured).  They are refit on the 145-size N-sweep of this part: the
 * fit moved the base model's peaks (L2 41.93, DRAM 19.725 TB/s) to 41.302 and 18.383 TB/s,
 * applied as those ratios.  On both parts the quadratic is the fraction of the DRAM peak
 * reached vs active WGPs in 256-WGP units (compute_mem_bw_from_occupancy rescales to the
 * part's N_CU), mem3 is the DRAM peak scaled by memory/engine clock, and mem_clock_ratio is
 * engine/memory clock.
 *
 * gfx1170 and gfx1171 are not measured; they use the gfx1201 values without MALL.
 */
// clang-format off
inline constexpr constants_row ARCHITECTURE_CONSTANTS[] = {
    // architecture, variant,
    //   mem1               mem2                              mem3               MI/CU  bandwidth coefficients            clock ratio  WGPs
    {architecture_t::gfx90a, variant_t::standard,
         5.5,               1.21875121875121875122 * 1.2,     1.2,               4,     {0, 0.03, 0},                     1.5,         0},
    {architecture_t::gfx942, variant_t::standard,
         17,                1.21875121875121875122 * 6,       4,                 4,     {0, 0.015, 0},                    1.5,         0},
    {architecture_t::gfx950, variant_t::standard,
         17,                1.21875121875121875122 * 7,       6,                 4,     {-0.0000194, 0.008772, 0.007898}, 1.5,         0},
    {architecture_t::gfx950, variant_t::gfx950_id75a8,
         8.5,               1.21875121875121875122 * 3.6,     2.55,              4,     {-0.000098, 0.02011, 0},          1.5,         0},
    {architecture_t::gfx1200, variant_t::standard,
         3.28,              1.21875121875121875122 * 1.45,    0.280,             2,     {0, 0.31, 0},                     1.5,         0},
    {architecture_t::gfx1201, variant_t::standard,
         5.74,              1.21875121875121875122 * 2.41,    0.464,             2,     {0, 0.17, 0},                     1.5,         0},
    {architecture_t::gfx1170, variant_t::standard,
         5.74,              NO_MALL_AVAILABLE,                0.464,             2,     {0, 0.17, 0},                     1.5,         0},
    {architecture_t::gfx1171, variant_t::standard,
         5.74,              NO_MALL_AVAILABLE,                0.464,             2,     {0, 0.17, 0},                     1.5,         0},
    {architecture_t::gfx1100, variant_t::standard,
         7.12,              1.21875121875121875122 * 3.48,    0.732,             2,     {0, 0.11, 0},                     1.5,         0},
    {architecture_t::gfx1101, variant_t::standard,
         4.86133,           1.21875121875121875122 * 1.28305, 0.420583,          2,     {0, 0.13169, 0},                  1.5,         0},
    {architecture_t::gfx1150, variant_t::standard,
         1.497,             NO_MALL_AVAILABLE,                0.0300383,         2,     {0, 0.18, 0},                     1.5,         0},
    {architecture_t::gfx1151, variant_t::standard,
         2.47,              1.21875121875121875122 * 0.93,    0.215,             2,     {0, 0.22, 0},                     1.5,         0},
    {architecture_t::gfx1152, variant_t::standard,
         0.849,             NO_MALL_AVAILABLE,                0.096,             4,     {0, 0.13, 0},                     1.5,         0},
    {architecture_t::gfx1153, variant_t::standard,
         0.240,             NO_MALL_AVAILABLE,                0.066,             2,     {0, 0.19, 0},                     1.5,         0},
    {architecture_t::gfx1250, variant_t::standard,
         41.93,             NO_MALL_AVAILABLE,                17.13 * 1.9 / 2.4, 4,     {-1.81e-5, 8.43e-3, -1.42e-3},    2.4 / 1.9,   0},
    {architecture_t::gfx1250, variant_t::gfx1250_6xcd,
         GFX1250_6XCD_MEM1, NO_MALL_AVAILABLE,                GFX1250_6XCD_MEM3, 4,     {-1.447e-5, 7.683e-3, -1.314e-3}, 2.4 / 1.65,  96},
};
// clang-format on

/**
 * @brief Row of the constants table for a variant of an architecture.
 *
 * @param arch Architecture
 * @param variant Variant of the architecture
 * @return const constants_row* The variant's row; the architecture's standard row if it has no
 *         row for @p variant; nullptr if the architecture has no rows
 */
constexpr const constants_row* find_constants_row(architecture_t arch, variant_t variant) noexcept {
  const constants_row* standard = nullptr;
  for (const constants_row& row : ARCHITECTURE_CONSTANTS) {
    if (row.arch != arch) { continue; }
    if (row.variant == variant) { return &row; }
    if (row.variant == variant_t::standard) { standard = &row; }
  }
  return standard;
}

/**
 * @brief Whether every architecture has a standard row in the constants table.
 */
constexpr bool has_standard_constants_for_every_architecture() noexcept {
  for (const architecture_info& info : ARCHITECTURES) {
    const constants_row* row = find_constants_row(info.arch, variant_t::standard);
    if (row == nullptr || row->variant != variant_t::standard) { return false; }
  }
  return true;
}

static_assert(has_standard_constants_for_every_architecture(),
              "ARCHITECTURE_CONSTANTS needs a standard row for every architecture");

/**
 * @brief Constants of a variant of an architecture.
 *
 * Joins the variant's measured row with the SIMD count and L1 capacity of the architecture's
 * traits.  A row measured at a WGP count has its mem1 and mem3 scaled to the @p n_cu WGPs of
 * the device.  An unrecognized architecture gets all-zero measured constants.
 *
 * @param arch Architecture
 * @param variant Variant of the architecture (see @ref resolve_variant)
 * @param n_cu Compute-unit (WGP) count of the device; used only by rows that scale with it
 * @return architecture_constants Constants for the variant
 */
constexpr architecture_constants constants_of(architecture_t arch,
                                              variant_t variant = variant_t::standard,
                                              std::size_t n_cu  = 0) noexcept {
  const architecture_traits& traits = traits_of(arch);
  const constants_row* row          = find_constants_row(arch, variant);
  if (row == nullptr) {
    return {0, 0, 0, 0, std::make_tuple(0, 0, 0), 0, traits.simds_per_cu, traits.l1_capacity};
  }
  architecture_constants constants{row->mem1_perf_ratio,
                                   row->mem2_perf_ratio,
                                   row->mem3_perf_ratio,
                                   row->parallel_mi_cu,
                                   row->mem_bw_per_wg_coefficients,
                                   row->mem_clock_ratio,
                                   traits.simds_per_cu,
                                   traits.l1_capacity};
  if (row->wgps_measured != 0) {
    const double partition    = static_cast<double>(n_cu) / static_cast<double>(row->wgps_measured);
    constants.mem1_perf_ratio = row->mem1_perf_ratio * partition;
    constants.mem3_perf_ratio = row->mem3_perf_ratio * partition;
  }
  return constants;
}

}  // namespace origami
