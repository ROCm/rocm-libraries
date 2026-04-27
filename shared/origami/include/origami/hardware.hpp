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

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

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
  static constexpr architecture_t arch_name_to_enum(std::string_view str) noexcept {
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
  static constexpr std::string_view arch_enum_to_name(architecture_t a) noexcept {
    switch (a) {
      case architecture_t::gfx90a: return "gfx90a";
      case architecture_t::gfx942: return "gfx942";
      case architecture_t::gfx950: return "gfx950";
      case architecture_t::gfx1201: return "gfx1201";
      case architecture_t::gfx1100: return "gfx1100";
      case architecture_t::gfx1151: return "gfx1151";
      default: return "unknown";
    }
  }

  /**
   * @brief Architecture-specific constants for memory and compute characteristics.
   *
   */
  struct architecture_constants {
    double mem1_perf_ratio;
    double mem2_perf_ratio;
    double mem3_perf_ratio;
    size_t parallel_mi_cu;  ///< Number of parallel matrix instructions per compute unit
    std::tuple<double, double, double>
        mem_bw_per_wg_coefficients;  ///< Memory bandwidth coefficients per workgroup
    double mem_clock_ratio;          ///< Memory clock ratio relative to compute clock

    constexpr architecture_constants(double mem1_perf_ratio,
                                     double mem2_perf_ratio,
                                     double mem3_perf_ratio,
                                     size_t parallel_mi_cu,
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
   * MALL value for those architectures that do not support it.
   * The value '1000' is just a big number.
   */
  static constexpr double NO_MALL_AVAILABLE = 1.21875121875121875122 * 1000;

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
      case architecture_t::gfx1250:
        // TODO: Update this, but for now using gfx950 values
        return {17, 1.21875121875121875122 * 7, 6, 4, std::make_tuple(0, 0.008, 0), 1.5};
      default: return {0, 0, 0, 0, std::make_tuple(0, 0, 0), 0};
    }
  }

  /**
   * @brief Map of matrix instruction latencies by architecture.
   *
   * Inline to prevent ODR violations when included in multiple shared libraries.
   * This ensures only one definition exists across all translation units. (PR#1862)
   */
  static inline const std::unordered_map<architecture_t,
                                         std::unordered_map<matrix_instruction, size_t>>
      INSTRUCTION_MAP = {
          // clang-format off
        {architecture_t::gfx90a,
         {
             // F32
             {matrix_instruction(32, 32, 2, data_type_t::Float), 64}, // v_mfma_f32_32x32x2_f32
             {matrix_instruction(32, 32, 1, data_type_t::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
             {matrix_instruction(16, 16, 4, data_type_t::Float), 32}, // v_mfma_f32_16x16x4_f32
             {matrix_instruction(16, 16, 1, data_type_t::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
             {matrix_instruction(4, 4, 1, data_type_t::Float), 8}, // v_mfma_f32_4x4x1_16b_f32

             // F64
             {matrix_instruction(16, 16, 4, data_type_t::Double), 32}, // v_mfma_f64_16x16x4_f64
             {matrix_instruction(4, 4, 4, data_type_t::Double), 16}, // v_mfma_f64_4x4x4_4b_f64

             // TODO ComplexFloat
             // TODO ComplexDouble

             // F16
             {matrix_instruction(32, 32, 4, data_type_t::Half), 64}, // v_mfma_f32_32x32x4_2b_f16
             {matrix_instruction(32, 32, 8, data_type_t::Half), 64}, // v_mfma_f32_32x32x8_f16
             {matrix_instruction(16, 16, 4, data_type_t::Half), 32}, // v_mfma_f32_16x16x4_4b_f16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32}, // v_mfma_f32_16x16x16_f16
             {matrix_instruction(4, 4, 4, data_type_t::Half), 8}, // v_mfma_f32_4x4x4_16b_f16

             // BF16
             {matrix_instruction(32, 32, 4, data_type_t::BFloat16), 64}, // v_mfma_f32_32x32x4_2b_bf16
             {matrix_instruction(32, 32, 8, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
             {matrix_instruction(16, 16, 4, data_type_t::BFloat16), 32}, // v_mfma_f32_16x16x4_4b_bf16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
             {matrix_instruction(4, 4, 4, data_type_t::BFloat16), 8}, // v_mfma_f32_4x4x4_16b_bf16

             // I8
             {matrix_instruction(32, 32, 8, data_type_t::Int8), 64}, // v_mfma_f32_32x32x16_f8
             {matrix_instruction(32, 32, 4, data_type_t::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32}, // v_mfma_f32_16x16x32_i8
             {matrix_instruction(16, 16, 4, data_type_t::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
             {matrix_instruction(4, 4, 4, data_type_t::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8

             // XF32
             {matrix_instruction(32, 32, 8, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x8_bf16 * 3
             {matrix_instruction(32, 32, 16, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x16_bf16 * 3
             {matrix_instruction(16, 16, 16, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
             {matrix_instruction(16, 16, 32, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
         }},
        {architecture_t::gfx942,
         {
             // F32
             {matrix_instruction(32, 32, 2, data_type_t::Float), 64}, // v_mfma_f32_32x32x2_f32
             {matrix_instruction(32, 32, 1, data_type_t::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
             {matrix_instruction(16, 16, 4, data_type_t::Float), 32}, // v_mfma_f32_16x16x4_f32
             {matrix_instruction(16, 16, 1, data_type_t::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
             {matrix_instruction(4, 4, 1, data_type_t::Float), 8}, // v_mfma_f32_4x4x1_16b_f32

             // F64
             {matrix_instruction(16, 16, 4, data_type_t::Double), 32}, // v_mfma_f64_16x16x4_f64
             {matrix_instruction(4, 4, 4, data_type_t::Double), 16}, // v_mfma_f64_4x4x4_4b_f64

             // TODO ComplexFloat
             // TODO ComplexDouble

             // F16
             {matrix_instruction(32, 32, 4, data_type_t::Half), 64}, // v_mfma_f32_32x32x4_2b_f16
             {matrix_instruction(32, 32, 8, data_type_t::Half), 32}, // v_mfma_f32_32x32x8_f16
             {matrix_instruction(16, 16, 4, data_type_t::Half), 32}, // v_mfma_f32_16x16x4_4b_f16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 16}, // v_mfma_f32_16x16x16_f16
             {matrix_instruction(4, 4, 4, data_type_t::Half), 8}, // v_mfma_f32_4x4x4_16b_f16

             // BF16
             {matrix_instruction(32, 32, 4, data_type_t::BFloat16), 64}, // v_mfma_f32_32x32x4_2b_bf16
             {matrix_instruction(32, 32, 8, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
             {matrix_instruction(16, 16, 4, data_type_t::BFloat16), 32}, // v_mfma_f32_16x16x4_4b_bf16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
             {matrix_instruction(4, 4, 4, data_type_t::BFloat16), 8}, // v_mfma_f32_4x4x4_16b_bf16

             // F8
             {matrix_instruction(32, 32, 16, data_type_t::Float8_fnuz), 32}, // v_mfma_f32_32x32x16_f8
             {matrix_instruction(16, 16, 32, data_type_t::Float8_fnuz), 16}, // v_mfma_f32_16x16x32_f8

             // BF8
             {matrix_instruction(32, 32, 16, data_type_t::BFloat8_fnuz), 32}, // v_mfma_f32_32x32x16_bf8
             {matrix_instruction(16, 16, 32, data_type_t::BFloat8_fnuz), 16}, // v_mfma_f32_16x16x32_bf8

             // F8B8
             {matrix_instruction(32, 32, 16, data_type_t::Float8BFloat8_fnuz), 32}, // v_mfma_f32_32x32x16_f8_bf8
             {matrix_instruction(16, 16, 32, data_type_t::Float8BFloat8_fnuz), 16}, // v_mfma_f32_16x16x32_f8_bf8

             // B8F8
             {matrix_instruction(32, 32, 16, data_type_t::BFloat8Float8_fnuz), 32}, // v_mfma_f32_32x32x16_bf8_f8
             {matrix_instruction(16, 16, 32, data_type_t::BFloat8Float8_fnuz), 16}, // v_mfma_f32_16x16x32_bf8_f8

             // I8
             {matrix_instruction(32, 32, 16, data_type_t::Int8), 32}, // v_mfma_f32_32x32x16_f8
             {matrix_instruction(32, 32, 4, data_type_t::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
             {matrix_instruction(16, 16, 32, data_type_t::Int8), 16}, // v_mfma_f32_16x16x32_i8
             {matrix_instruction(16, 16, 4, data_type_t::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
             {matrix_instruction(4, 4, 4, data_type_t::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8

             // XF32
             {matrix_instruction(32, 32, 4, data_type_t::XFloat32), 32}, // v_mfma_f32_32x32x4_xf32
             {matrix_instruction(16, 16, 32, data_type_t::XFloat32), 16}, // v_mfma_f32_16x16x8_xf32
         }},
        {architecture_t::gfx950,
         {
             // F32
             {matrix_instruction(32, 32, 2, data_type_t::Float), 64}, // v_mfma_f32_32x32x2_f32
             {matrix_instruction(32, 32, 1, data_type_t::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
             {matrix_instruction(16, 16, 4, data_type_t::Float), 32}, // v_mfma_f32_16x16x4_f32
             {matrix_instruction(16, 16, 1, data_type_t::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
             {matrix_instruction(4, 4, 1, data_type_t::Float), 8}, // v_mfma_f32_4x4x1_16b_f32

             // F64
             {matrix_instruction(16, 16, 4, data_type_t::Double), 64}, // v_mfma_f64_16x16x4_f64
             {matrix_instruction(4, 4, 4, data_type_t::Double), 16}, // v_mfma_f64_4x4x4_4b_f64

             // TODO ComplexFloat
             // TODO ComplexDouble

             // F16
             {matrix_instruction(32, 32, 8, data_type_t::Half), 32}, // v_mfma_f32_32x32x8_f16
             {matrix_instruction(32, 32, 16, data_type_t::Half), 32}, // v_mfma_f32_32x32x16_f16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 16}, // v_mfma_f32_16x16x16_f16
             {matrix_instruction(16, 16, 32, data_type_t::Half), 16}, // v_mfma_f32_16x16x32_f16

             // BF16
             {matrix_instruction(32, 32, 8, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
             {matrix_instruction(32, 32, 16, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x16_bf16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
             {matrix_instruction(16, 16, 32, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16

             // F8
             {matrix_instruction(32, 32, 64, data_type_t::Float8), 64}, // v_mfma_f32_32x32x64_f8
             {matrix_instruction(32, 32, 16, data_type_t::Float8), 32}, // v_mfma_f32_32x32x16_f8
             {matrix_instruction(16, 16, 128, data_type_t::Float8), 32}, // v_mfma_f32_16x16x128_f8
             {matrix_instruction(16, 16, 32, data_type_t::Float8), 16}, // v_mfma_f32_16x16x32_f8

             // BF8
             {matrix_instruction(32, 32, 64, data_type_t::BFloat8), 64}, // v_mfma_f32_32x32x64_bf8
             {matrix_instruction(32, 32, 16, data_type_t::BFloat8), 32}, // v_mfma_f32_32x32x16_bf8
             {matrix_instruction(16, 16, 128, data_type_t::BFloat8), 32}, // v_mfma_f32_16x16x128_bf8
             {matrix_instruction(16, 16, 32, data_type_t::BFloat8), 16}, // v_mfma_f32_16x16x32_bf8

             // F8B8
             {matrix_instruction(32, 32, 64, data_type_t::Float8BFloat8), 64}, // v_mfma_f32_32x32x64_f8_bf8
             {matrix_instruction(32, 32, 16, data_type_t::Float8BFloat8), 32}, // v_mfma_f32_32x32x16_f8_bf8
             {matrix_instruction(16, 16, 128, data_type_t::Float8BFloat8), 32}, // v_mfma_f32_16x16x128_f8_bf8
             {matrix_instruction(16, 16, 32, data_type_t::Float8BFloat8), 16}, // v_mfma_f32_16x16x32_f8_bf8

             // B8F8
             {matrix_instruction(32, 32, 64, data_type_t::BFloat8Float8), 64}, // v_mfma_f32_32x32x64_bf8_f8
             {matrix_instruction(32, 32, 16, data_type_t::BFloat8Float8), 32}, // v_mfma_f32_32x32x16_bf8_f8
             {matrix_instruction(16, 16, 128, data_type_t::BFloat8Float8), 32}, // v_mfma_f32_16x16x128_bf8_f8
             {matrix_instruction(16, 16, 32, data_type_t::BFloat8Float8), 16}, // v_mfma_f32_16x16x32_bf8_f8

             // I8
             {matrix_instruction(32, 32, 16, data_type_t::Int8), 32}, // v_mfma_f32_32x32x16_f8
             {matrix_instruction(32, 32, 4, data_type_t::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
             {matrix_instruction(16, 16, 32, data_type_t::Int8), 16}, // v_mfma_f32_16x16x32_i8
             {matrix_instruction(16, 16, 4, data_type_t::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
             {matrix_instruction(4, 4, 4, data_type_t::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8

             // XF32
             {matrix_instruction(32, 32, 8, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x8_bf16 * 3
             {matrix_instruction(32, 32, 16, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x16_bf16 * 3
             {matrix_instruction(16, 16, 16, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
             {matrix_instruction(16, 16, 32, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3

             // F6
             {matrix_instruction(32, 32, 64, data_type_t::Float6), 32}, // v_mfma_f32_32x32x64_f6
             {matrix_instruction(16, 16, 128, data_type_t::Float6), 16}, // v_mfma_f32_16x16x128_f6

             // BF6
             {matrix_instruction(32, 32, 64, data_type_t::BFloat6), 32}, // v_mfma_f32_32x32x64_bf6
             {matrix_instruction(16, 16, 128, data_type_t::BFloat6), 16}, // v_mfma_f32_16x16x128_bf6

             // F4
             {matrix_instruction(32, 32, 64, data_type_t::Float4), 32}, // v_mfma_f32_32x32x64_f4
             {matrix_instruction(16, 16, 128, data_type_t::Float4), 16}, // v_mfma_f32_16x16x128_f4

             // DOT2
             {matrix_instruction(1, 1, 64, data_type_t::Half), 16}, // V_DOT2_F32_F16
             {matrix_instruction(1, 1, 64, data_type_t::BFloat16), 16}, // V_DOT2_F32_BF16
         }},
        {architecture_t::gfx1201,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 16}, // v_wmma_f16_16x16x16_f16/v_wmma_f32_16x16x16_f16

             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_wmma_bf16_16x16x16_bf16/v_wmma_f32_16x16x16_bf16

             // F8
             {matrix_instruction(16, 16, 16, data_type_t::Float8), 8}, // v_wmma_f32_16x16x16_fp8_fp8

             // F8B8
             {matrix_instruction(16, 16, 16, data_type_t::Float8BFloat8), 8}, // v_wmma_f32_16x16x16_fp8_bf8

             // B8F8
             {matrix_instruction(16, 16, 16, data_type_t::BFloat8Float8), 8}, // v_wmma_f32_16x16x16_bf8_fp8

             // B8
             {matrix_instruction(16, 16, 16, data_type_t::BFloat8), 8}, // v_wmma_f32_16x16x16_bf8_bf8

             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 8}, // v_wmma_i32_16x16x16_iu8

             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 8}, // v_wmma_i32_16x16x16_iu4
             {matrix_instruction(16, 16, 32, data_type_t::Int4), 8}, // v_wmma_i32_16x16x32_iu4
         }},
        {architecture_t::gfx1100,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
         }},
        {architecture_t::gfx1150,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
         }},
        {architecture_t::gfx1151,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
         }},
        {architecture_t::gfx1152,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
         }},
        {architecture_t::gfx1153,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
         }},
      };
  // clang-format on

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
   * @brief Construct hardware_t using architecture constants and a clock frequency.
   *
   * Computes memory performance ratios from the provided architecture constants
   * and compute clock, then delegates to the full constructor.
   *
   * @param arch GPU architecture type
   * @param N_CU Number of compute units
   * @param lds_capacity LDS capacity in bytes
   * @param constants Architecture-specific constants
   * @param num_xcds Number of XCDs — provided separately from constants so that
   *                 it can come from a runtime query or a known-architecture table
   * @param L2_capacity L2 cache capacity in bytes
   * @param compute_clock_ghz Compute clock frequency in GHz
   * @param memory_clock_ghz Memory clock frequency in GHz
   */
  hardware_t(architecture_t arch,
             size_t N_CU,
             size_t lds_capacity,
             const architecture_constants& constants,
             size_t num_xcds,
             size_t L2_capacity,
             double compute_clock_ghz,
             double memory_clock_ghz);

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
   * @param properties HIP device properties structure
   * @param num_xcds_override If non-zero, use this XCD count instead of
   *                          the hardcoded default. Passed by
   *                          get_hardware_for_device() after a runtime query.
   * @return hardware_t Configured hardware instance
   */
  static hardware_t get_hardware_for_properties(hipDeviceProp_t properties,
                                                size_t num_xcds_override = 0);

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
   * @brief Create hardware_t instance for a specific HIP device using
   *        caller-provided properties.
   *
   * Same as @ref get_hardware_for_device(int) but uses the supplied
   * `hipDeviceProp_t` instead of re-querying via `hipGetDeviceProperties`.
   * Callers that have already adjusted fields on `prop` (for example,
   * overriding `multiProcessorCount` with
   * `hipDeviceAttributePhysicalMultiProcessorCount` on multi-XCC
   * architectures) should use this overload so those adjustments are
   * preserved. The runtime XCC query
   * (`hipDeviceAttributeNumberOfXccs` on HIP 7+) is still performed
   * against `deviceId`.
   *
   * @warning `prop` must describe the same physical device as `deviceId`,
   *          aside from intentional field-level overrides the caller has
   *          applied (e.g. swapping `multiProcessorCount` for the
   *          physical MP count). Passing a `prop` from one device together
   *          with a `deviceId` for a different device produces an
   *          internally-inconsistent `hardware_t` — the XCC count will
   *          come from `deviceId` while CU count, clocks, LDS/L2 capacity,
   *          and architecture all come from `prop`. This is not checked.
   *
   * @param deviceId HIP device ID used to query the XCC count
   * @param prop     Caller-owned device properties to model from; must
   *                 correspond to `deviceId`
   * @return hardware_t Configured hardware instance for the device
   */
  static hardware_t get_hardware_for_device(int deviceId,
                                            hipDeviceProp_t const& prop);

  /**
   * @brief Create hardware_t instance for a specific architecture with specified parameters.
   *
   * Creates a hardware instance using the specified architecture and hardware parameters.
   * Useful for analytical modeling when actual hardware is not available.
   *
   * @param arch Architecture enum value
   * @param N_CU Number of compute units
   * @param lds_capacity LDS capacity in bytes
   * @param L2_capacity L2 cache capacity in bytes
   * @param compute_clock_khz Compute clock in KHz
   * @return hardware_t Configured hardware instance
   * @throws std::runtime_error if architecture is not supported
   */
  static hardware_t get_hardware_for_arch(architecture_t arch,
                                          size_t N_CU,
                                          size_t lds_capacity,
                                          size_t L2_capacity,
                                          int compute_clock_khz);

  /**
   * @brief Get the default (hardcoded) XCD count for a known architecture.
   *
   * Legacy fallback table for architectures that predate the runtime XCC
   * query (hipDeviceAttributeNumberOfXccs, HIP 7.0+). Do NOT add new
   * architectures here — new hardware should rely solely on the runtime
   * query via get_hardware_for_device(). Throws for architectures not in
   * the table.
   *
   * @param arch Architecture enum value
   * @return Number of XCDs for the architecture
   * @throws std::runtime_error if the architecture has no hardcoded default
   */
  static size_t get_default_num_xcds(architecture_t arch);

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

  /**
   * @brief Get valid matrix instruction dimensions for a given datatype.
   *
   * Returns a list of valid matrix instruction dimensions (M, N, K) for
   * the specified datatype on the current hardware architecture. Multiple
   * dimensions may be available for the same datatype.
   *
   * @param mi_input_type Input data type for the matrix instruction
   * @return std::vector<dim3_t> List of valid dimensions for the datatype
   */
  std::vector<dim3_t> get_valid_matrix_instructions(data_type_t mi_input_type) const;

  /**
   * @brief Get recommended matrix instruction dimensions for a given datatype.
   *
   * Returns the single best matrix instruction dimension for the specified datatype
   * based on throughput (M*N*K/latency). If multiple instructions are available,
   * returns the one with the highest throughput.
   *
   * @param mi_input_type Input data type for the matrix instruction
   * @return dim3_t Recommended dimension for the datatype. Returns {0,0,0} if not supported.
   */
  dim3_t get_recommended_matrix_instruction(data_type_t mi_input_type) const;

  /**
   * @brief Check if the architecture has MALL (Memory Attached Last Level).
   *
   * @return true if the architecture has MALL, false otherwise
   */
  bool has_MALL() const;

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
