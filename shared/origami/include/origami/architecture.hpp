/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// origami — GPU architecture identity and microbenchmarked constants.
//
// This is the HIP-free foundation of the hardware model: the architecture enum,
// its string<->enum mappings, and the per-architecture constants table, all
// pure constexpr C++. It is deliberately split out of hardware.hpp (which pulls
// in <hip/hip_runtime.h> for live device introspection) so that consumers which
// only need to *name* an architecture or read its calibrated constants — most
// notably origami::comm — can do so without a ROCm toolchain.
//
// hardware.hpp includes this header and re-exposes everything here as members
// of hardware_t (hardware_t::architecture_t, hardware_t::get_arch_constants,
// ...), so existing call sites are unaffected.
#pragma once

#include <cstddef>
#include <string_view>
#include <tuple>

namespace origami {

/**
 * @brief Enumeration of supported GPU architectures.
 *
 */
enum class architecture_t {
  gfx90a,
  gfx942,
  gfx950,
  gfx1201,
  gfx1100,
  gfx1150,
  gfx1151,
  gfx1152,
  gfx1153,
  gfx1250,
  Count
};

/**
 * @brief Convert architecture name string to architecture_t enum.
 *
 * @param str Architecture name as string (e.g., "gfx90a", "gfx942")
 * @return architecture_t Corresponding enum value, or Count if not recognized
 */
constexpr architecture_t arch_name_to_enum(std::string_view str) noexcept {
  if (str == "gfx90a") return architecture_t::gfx90a;
  if (str == "gfx942") return architecture_t::gfx942;
  if (str == "gfx950") return architecture_t::gfx950;
  if (str == "gfx1201") return architecture_t::gfx1201;
  if (str == "gfx1100") return architecture_t::gfx1100;
  if (str == "gfx1150") return architecture_t::gfx1150;
  if (str == "gfx1151") return architecture_t::gfx1151;
  if (str == "gfx1152") return architecture_t::gfx1152;
  if (str == "gfx1153") return architecture_t::gfx1153;
  if (str == "gfx1250") return architecture_t::gfx1250;
  return architecture_t::Count;
}

/**
 * @brief Convert architecture_t to string (e.g. for logging).
 *
 * @param a Architecture enum value
 * @return std::string_view Corresponding string value
 */
constexpr std::string_view arch_enum_to_name(architecture_t a) noexcept {
  switch (a) {
    case architecture_t::gfx90a: return "gfx90a";
    case architecture_t::gfx942: return "gfx942";
    case architecture_t::gfx950: return "gfx950";
    case architecture_t::gfx1201: return "gfx1201";
    case architecture_t::gfx1100: return "gfx1100";
    case architecture_t::gfx1150: return "gfx1150";
    case architecture_t::gfx1151: return "gfx1151";
    case architecture_t::gfx1152: return "gfx1152";
    case architecture_t::gfx1153: return "gfx1153";
    case architecture_t::gfx1250: return "gfx1250";
    default: return "unknown";
  }
}

/**
 * MALL value for those architectures that do not support it.
 * The value '1000' is just a big number.
 */
inline constexpr double NO_MALL_AVAILABLE = 1.21875121875121875122 * 1000;

/**
 * @brief Architecture-specific constants for memory and compute characteristics.
 *
 */
struct architecture_constants {
  double mem1_perf_ratio;
  double mem2_perf_ratio;
  double mem3_perf_ratio;
  std::size_t parallel_mi_cu;  ///< Number of parallel matrix instructions per compute unit
  std::tuple<double, double, double>
      mem_bw_per_wg_coefficients;  ///< Memory bandwidth coefficients per workgroup
  double mem_clock_ratio;          ///< Memory clock ratio relative to compute clock

  constexpr architecture_constants(double mem1_perf_ratio,
                                   double mem2_perf_ratio,
                                   double mem3_perf_ratio,
                                   std::size_t parallel_mi_cu,
                                   std::tuple<double, double, double> mem_bw_per_wg_coefficients,
                                   double mem_clock_ratio)  // Obtained through microbenchmarking
      : mem1_perf_ratio(mem1_perf_ratio)
      , mem2_perf_ratio(mem2_perf_ratio)
      , mem3_perf_ratio(mem3_perf_ratio)
      , parallel_mi_cu(parallel_mi_cu)
      , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
      , mem_clock_ratio(mem_clock_ratio) {}
};

/**
 * @brief Get architecture-specific constants for a given architecture.
 *
 * Returns the pre-configured constants (memory performance ratios, bandwidth
 * coefficients, etc.) for the specified architecture. These values are
 * determined through microbenchmarking.
 *
 * @param arch Architecture enum value
 * @return architecture_constants Constants for the specified architecture
 */
constexpr architecture_constants get_arch_constants(architecture_t arch) {
  switch (arch) {
    case architecture_t::gfx90a:
      return {5.5, 1.21875121875121875122 * 1.2, 1.2, 4, std::make_tuple(0, 0.03, 0), 1.5};
    case architecture_t::gfx942:
      return {17, 1.21875121875121875122 * 6, 4, 4, std::make_tuple(0, 0.015, 0), 1.5};
    case architecture_t::gfx950:
      return {17,
              1.21875121875121875122 * 7,
              6,
              4,
              std::make_tuple(-0.000013, 0.007070, 0.027355),
              1.5};
    case architecture_t::gfx1201:
      return {5.74, 1.21875121875121875122 * 2.41, 0.464, 2, std::make_tuple(0, 0.17, 0), 1.5};
    case architecture_t::gfx1100:
      return {7.12, 1.21875121875121875122 * 3.48, 0.732, 2, std::make_tuple(0, 0.11, 0), 1.5};
    case architecture_t::gfx1150:
      // AMD Strix Point iGPU
      return {1.497, NO_MALL_AVAILABLE, 0.077, 16, std::make_tuple(0, 0.18, 0), 1.5};
    case architecture_t::gfx1151:
      // AMD Strix Halo iGPU
      return {2.47, 1.21875121875121875122 * 0.93, 0.215, 2, std::make_tuple(0, 0.22, 0), 1.5};
    case architecture_t::gfx1152:
      // AMD Radeon 840M iGPU
      return {0.849, NO_MALL_AVAILABLE, 0.096, 4, std::make_tuple(0, 0.13, 0), 1.5};
    case architecture_t::gfx1153:
      // AMD Radeon 820M iGPU
      return {0.240, NO_MALL_AVAILABLE, 0.066, 2, std::make_tuple(0, 0.19, 0), 1.5};
    case architecture_t::gfx1250: {
      // TODO: Update with real gfx1250 constants when available
      auto c                       = get_arch_constants(architecture_t::gfx950);
      c.mem2_perf_ratio            = NO_MALL_AVAILABLE;
      c.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.016, 0);
      return c;
    }
    default: return {0, 0, 0, 0, std::make_tuple(0, 0, 0), 0};
  }
}

}  // namespace origami
