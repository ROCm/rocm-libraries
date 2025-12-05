/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include <hip/hip_runtime.h>

#include "origami/types.hpp"

namespace origami {
/**
 * @brief Represents hardware characteristics and capabilities of GPU architectures.
 *
 */
class hardware_t {
 public:
  /**
   * @brief Enumeration of supported GPU architectures.
   *
   */
  enum class architecture_t { gfx90a, gfx942, gfx950, gfx1201, gfx1100, gfx1151, Count };

  /**
   * @brief Convert architecture name string to architecture_t enum.
   *
   * @param str Architecture name as string (e.g., "gfx90a", "gfx942")
   * @return architecture_t Corresponding enum value, or Count if not recognized
   */
  static constexpr architecture_t arch_name_to_enum(std::string_view str) noexcept {
    if (str == "gfx90a") return architecture_t::gfx90a;
    if (str == "gfx942") return architecture_t::gfx942;
    if (str == "gfx950") return architecture_t::gfx950;
    if (str == "gfx1201") return architecture_t::gfx1201;
    if (str == "gfx1100") return architecture_t::gfx1100;
    if (str == "gfx1151") return architecture_t::gfx1151;
    return architecture_t::Count;
  }

  /**
   * @brief Architecture-specific constants for memory and compute characteristics.
   *
   */
  struct architecture_constants {
    size_t num_xcds;  ///< Number of XCDs (XCD = XGMI Complex Die)
    double mem1_perf_ratio;
    double mem2_perf_ratio;
    double mem3_perf_ratio;
    size_t parallel_mi_cu;  ///< Number of parallel matrix instructions per compute unit
    std::tuple<double, double, double>
        mem_bw_per_wg_coefficients;  ///< Memory bandwidth coefficients per workgroup
    double mem_clock_ratio;          ///< Memory clock ratio relative to compute clock

    constexpr architecture_constants(size_t num_xcds,
                                     double mem1_perf_ratio,
                                     double mem2_perf_ratio,
                                     double mem3_perf_ratio,
                                     size_t parallel_mi_cu,
                                     std::tuple<double, double, double> mem_bw_per_wg_coefficients,
                                     double mem_clock_ratio)  // Obtained through microbenchmarking
        : num_xcds(num_xcds)
        , mem1_perf_ratio(mem1_perf_ratio)
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
  static constexpr architecture_constants get_arch_constants(architecture_t arch) {
    switch (arch) {
      case architecture_t::gfx90a:
        return {1, 5.5, 1.21875121875121875122 * 1.2, 1.2, 4, std::make_tuple(0, 0.03, 0), 1.5};
      case architecture_t::gfx942:
        return {8, 17, 1.21875121875121875122 * 6, 4, 4, std::make_tuple(0, 0.015, 0), 1.5};
      case architecture_t::gfx950:
        return {8, 17, 1.21875121875121875122 * 7, 6, 4, std::make_tuple(0, 0.008, 0), 1.5};
      case architecture_t::gfx1201:
        return {1, 5.74, 1.21875121875121875122 * 2.41, 0.464, 2, std::make_tuple(0, 0.17, 0), 1.5};
      case architecture_t::gfx1100:
        return {1, 7.12, 1.21875121875121875122 * 3.48, 0.732, 2, std::make_tuple(0, 0.11, 0), 1.5};
      case architecture_t::gfx1151:
        return {1, 2.47, 1.21875121875121875122 * 0.93, 0.215, 2, std::make_tuple(0, 0.22, 0), 1.5};
      default: return {0, 0, 0, 0, 0, std::make_tuple(0, 0, 0), 0};
    }
  }

  /**
   * @brief Map of matrix instruction latencies by architecture.
   *
   */
  static const std::unordered_map<architecture_t, std::unordered_map<matrix_instruction, size_t>>
      INSTRUCTION_MAP;

  architecture_t arch;  ///< GPU architecture type
  size_t N_CU;          ///< Number of Compute Units
  size_t lds_capacity;  ///< Capacity of Local Data Share (LDS) in bytes
  double mem1_perf_ratio;
  double mem2_perf_ratio;
  double mem3_perf_ratio;
  size_t L2_capacity;        ///< Capacity of L2 cache in bytes
  size_t CU_per_L2;          ///< Number of compute units per L2 cache domain
  double compute_clock_ghz;  ///< Compute clock frequency in GHz
  size_t parallel_mi_cu;     ///< Number of parallel matrix instructions per compute unit
  std::tuple<double, double, double>
      mem_bw_per_wg_coefficients;  ///< Memory bandwidth coefficients per workgroup
  size_t NUM_XCD;                  ///< Number of XCDs (XGMI Complex Die)

  /**
   * @brief Construct hardware_t with explicit parameters.
   *
   * @param arch GPU architecture type
   * @param N_CU Number of compute units
   * @param lds_capacity LDS capacity in bytes
   * @param NUM_XCD Number of XCDs
   * @param mem1_perf_ratio Memory level 1 performance ratio
   * @param mem2_perf_ratio Memory level 2 performance ratio
   * @param mem3_perf_ratio Memory level 3 performance ratio
   * @param L2_capacity L2 cache capacity in bytes
   * @param compute_clock_ghz Compute clock frequency in GHz
   * @param parallel_mi_cu Number of parallel matrix instructions per CU
   * @param mem_bw_per_wg_coefficients Memory bandwidth coefficients per workgroup
   */
  hardware_t(architecture_t arch,
             size_t N_CU,
             size_t lds_capacity,
             size_t NUM_XCD,
             double mem1_perf_ratio,
             double mem2_perf_ratio,
             double mem3_perf_ratio,
             size_t L2_capacity,
             double compute_clock_ghz,
             size_t parallel_mi_cu,
             std::tuple<double, double, double> mem_bw_per_wg_coefficients);

  /**
   * @brief Construct hardware_t from HIP device properties.
   *
   * Automatically determines architecture and extracts hardware parameters
   * from the provided HIP device properties structure.
   *
   * @param properties HIP device properties structure
   */
  hardware_t(hipDeviceProp_t properties);

  /**
   * @brief Copy constructor.
   *
   * @param other Another hardware_t instance to copy from
   */
  hardware_t(const hardware_t& other);

  /**
   * @brief Create hardware_t instance from HIP device properties.
   *
   *
   * @param properties HIP device properties structure
   * @return hardware_t Configured hardware instance
   */
  static hardware_t get_hardware_for_properties(hipDeviceProp_t properties);

  /**
   * @brief Create hardware_t instance for a specific HIP device.
   *
   * Queries the specified HIP device and creates a hardware_t instance
   * with the appropriate architecture and parameters.
   *
   * @param deviceId HIP device ID
   * @return hardware_t Configured hardware instance for the device
   */
  static hardware_t get_hardware_for_device(int deviceId);

  /**
   * @brief Check if the hardware described by properties is supported.
   *
   * Determines whether the GPU architecture represented by the device
   * properties is supported by the analytical model.
   *
   * @param properties HIP device properties structure
   * @return true if the architecture is supported, false otherwise
   */
  static bool is_hardware_supported(hipDeviceProp_t properties);

  /**
   * @brief Print hardware details to stdout.
   *
   */
  void print() const;

  /**
   * @brief Get matrix instruction latency for given instruction parameters.
   *
   *
   * @param MI_M Matrix instruction M dimension
   * @param MI_N Matrix instruction N dimension
   * @param MI_K Matrix instruction K dimension
   * @param mi_input_type Input data type for the matrix instruction
   * @return size_t Instruction latency in cycles, or 0 if not found
   */
  size_t get_mi_latency(size_t MI_M, size_t MI_N, size_t MI_K, data_type_t mi_input_type) const;

 private:
  /**
   * @brief Extract substring before the first colon character.
   *
   * Helper function used for parsing architecture names from device
   * property strings (e.g., extracting "gfx90a" from "gfx90a:...").
   *
   * @param input Input string to parse
   * @return std::string Substring before the first colon, or entire string if no colon found
   */
  static std::string get_before_first_colon(const std::string& input);
};
}  // namespace origami
