// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "origami/math.hpp"
#include "origami/types.hpp"

namespace origami {
class hardware_t {
 public:
  enum class architecture_t { gfx90a, gfx942, gfx950, gfx1201, gfx1100, gfx1151, Count };

  static constexpr architecture_t arch_name_to_enum(std::string_view str) noexcept {
    if (str == "gfx90a") return architecture_t::gfx90a;
    if (str == "gfx942") return architecture_t::gfx942;
    if (str == "gfx950") return architecture_t::gfx950;
    if (str == "gfx1201") return architecture_t::gfx1201;
    if (str == "gfx1100") return architecture_t::gfx1100;
    if (str == "gfx1151") return architecture_t::gfx1151;
    return architecture_t::Count;
  }

  struct architecture_constants {
    size_t num_xcds;
    double mem1_perf_ratio;
    double mem2_perf_ratio;
    double mem3_perf_ratio;
    size_t parallel_mi_cu;
    std::tuple<double, double, double> mem_bw_per_wg_coefficients;
    double mem_clock_ratio;

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

  inline static const std::unordered_map<architecture_t,
                                         std::unordered_map<matrix_instruction, size_t>>
      INSTRUCTION_MAP = {
          {hardware_t::architecture_t::gfx90a,
           {
               // F32
               {matrix_instruction(32, 32, 2, data_type_t::Float), 64},  // v_mfma_f32_32x32x2_f32
               {matrix_instruction(32, 32, 1, data_type_t::Float),
                64},  // v_mfma_f32_32x32x1_2b_f32
               {matrix_instruction(16, 16, 4, data_type_t::Float), 32},  // v_mfma_f32_16x16x4_f32
               {matrix_instruction(16, 16, 1, data_type_t::Float),
                32},                                                  // v_mfma_f32_16x16x1_4b_f32
               {matrix_instruction(4, 4, 1, data_type_t::Float), 8},  // v_mfma_f32_4x4x1_16b_f32
                                                                      // F64
               {matrix_instruction(16, 16, 4, data_type_t::Double), 32},  // v_mfma_f64_16x16x4_f64
               {matrix_instruction(4, 4, 4, data_type_t::Double), 16},    // v_mfma_f64_4x4x4_4b_f64
                                                                          // TODO ComplexFloat
                                                                          // TODO ComplexDouble
                                                                          // F16
               {matrix_instruction(32, 32, 4, data_type_t::Half), 64},  // v_mfma_f32_32x32x4_2b_f16
               {matrix_instruction(32, 32, 8, data_type_t::Half), 64},  // v_mfma_f32_32x32x8_f16
               {matrix_instruction(16, 16, 4, data_type_t::Half), 32},  // v_mfma_f32_16x16x4_4b_f16
               {matrix_instruction(16, 16, 16, data_type_t::Half), 32},  // v_mfma_f32_16x16x16_f16
               {matrix_instruction(4, 4, 4, data_type_t::Half), 8},      // v_mfma_f32_4x4x4_16b_f16
                                                                         // BF16
               {matrix_instruction(32, 32, 4, data_type_t::BFloat16),
                64},  // v_mfma_f32_32x32x4_2b_bf16
               {matrix_instruction(32, 32, 8, data_type_t::BFloat16),
                32},  // v_mfma_f32_32x32x8_bf16
               {matrix_instruction(16, 16, 4, data_type_t::BFloat16),
                32},  // v_mfma_f32_16x16x4_4b_bf16
               {matrix_instruction(16, 16, 16, data_type_t::BFloat16),
                16},  // v_mfma_f32_16x16x16_bf16
               {matrix_instruction(4, 4, 4, data_type_t::BFloat16),
                8},                                                     // v_mfma_f32_4x4x4_16b_bf16
                                                                        // I8
               {matrix_instruction(32, 32, 8, data_type_t::Int8), 64},  // v_mfma_f32_32x32x16_f8
               {matrix_instruction(32, 32, 4, data_type_t::Int8), 64},  // v_mfma_i32_32x32x4_2b_i8
               {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_mfma_f32_16x16x32_i8
               {matrix_instruction(16, 16, 4, data_type_t::Int8), 32},   // v_mfma_i32_16x16x4_4b_i8
               {matrix_instruction(4, 4, 4, data_type_t::Int8), 8},      // v_mfma_i32_4x4x4_16b_i8
                                                                         // XF32
               {matrix_instruction(32, 32, 8, data_type_t::XFloat32),
                96},  // v_mfma_f32_32x32x8_bf16 * 3
               {matrix_instruction(32, 32, 16, data_type_t::XFloat32),
                96},  // v_mfma_f32_32x32x16_bf16 * 3
               {matrix_instruction(16, 16, 16, data_type_t::XFloat32),
                48},  // v_mfma_f32_16x16x16_bf16 * 3
               {matrix_instruction(16, 16, 32, data_type_t::XFloat32),
                48},  // v_mfma_f32_16x16x16_bf16 * 3
           }},
          {hardware_t::architecture_t::gfx942,
           {
               // F32
               {matrix_instruction(32, 32, 2, data_type_t::Float), 64},  // v_mfma_f32_32x32x2_f32
               {matrix_instruction(32, 32, 1, data_type_t::Float),
                64},  // v_mfma_f32_32x32x1_2b_f32
               {matrix_instruction(16, 16, 4, data_type_t::Float), 32},  // v_mfma_f32_16x16x4_f32
               {matrix_instruction(16, 16, 1, data_type_t::Float),
                32},                                                  // v_mfma_f32_16x16x1_4b_f32
               {matrix_instruction(4, 4, 1, data_type_t::Float), 8},  // v_mfma_f32_4x4x1_16b_f32
                                                                      // F64
               {matrix_instruction(16, 16, 4, data_type_t::Double), 32},  // v_mfma_f64_16x16x4_f64
               {matrix_instruction(4, 4, 4, data_type_t::Double), 16},    // v_mfma_f64_4x4x4_4b_f64
                                                                          // TODO ComplexFloat
                                                                          // TODO ComplexDouble
                                                                          // F16
               {matrix_instruction(32, 32, 4, data_type_t::Half), 64},  // v_mfma_f32_32x32x4_2b_f16
               {matrix_instruction(32, 32, 8, data_type_t::Half), 32},  // v_mfma_f32_32x32x8_f16
               {matrix_instruction(16, 16, 4, data_type_t::Half), 32},  // v_mfma_f32_16x16x4_4b_f16
               {matrix_instruction(16, 16, 16, data_type_t::Half), 16},  // v_mfma_f32_16x16x16_f16
               {matrix_instruction(4, 4, 4, data_type_t::Half), 8},      // v_mfma_f32_4x4x4_16b_f16
                                                                         // BF16
               {matrix_instruction(32, 32, 4, data_type_t::BFloat16),
                64},  // v_mfma_f32_32x32x4_2b_bf16
               {matrix_instruction(32, 32, 8, data_type_t::BFloat16),
                32},  // v_mfma_f32_32x32x8_bf16
               {matrix_instruction(16, 16, 4, data_type_t::BFloat16),
                32},  // v_mfma_f32_16x16x4_4b_bf16
               {matrix_instruction(16, 16, 16, data_type_t::BFloat16),
                16},  // v_mfma_f32_16x16x16_bf16
               {matrix_instruction(4, 4, 4, data_type_t::BFloat16),
                8},  // v_mfma_f32_4x4x4_16b_bf16
                     // F8
               {matrix_instruction(32, 32, 16, data_type_t::Float8_fnuz),
                32},  // v_mfma_f32_32x32x16_f8
               {matrix_instruction(16, 16, 32, data_type_t::Float8_fnuz),
                16},  // v_mfma_f32_16x16x32_f8
                      // BF8
               {matrix_instruction(32, 32, 16, data_type_t::BFloat8_fnuz),
                32},  // v_mfma_f32_32x32x16_bf8
               {matrix_instruction(16, 16, 32, data_type_t::BFloat8_fnuz),
                16},  // v_mfma_f32_16x16x32_bf8
                      // F8B8
               {matrix_instruction(32, 32, 16, data_type_t::Float8BFloat8_fnuz),
                32},  // v_mfma_f32_32x32x16_f8_bf8
               {matrix_instruction(16, 16, 32, data_type_t::Float8BFloat8_fnuz),
                16},  // v_mfma_f32_16x16x32_f8_bf8
                      // B8F8
               {matrix_instruction(32, 32, 16, data_type_t::BFloat8Float8_fnuz),
                32},  // v_mfma_f32_32x32x16_bf8_f8
               {matrix_instruction(16, 16, 32, data_type_t::BFloat8Float8_fnuz),
                16},  // v_mfma_f32_16x16x32_bf8_f8
                      // I8
               {matrix_instruction(32, 32, 16, data_type_t::Int8), 32},  // v_mfma_f32_32x32x16_f8
               {matrix_instruction(32, 32, 4, data_type_t::Int8), 64},   // v_mfma_i32_32x32x4_2b_i8
               {matrix_instruction(16, 16, 32, data_type_t::Int8), 16},  // v_mfma_f32_16x16x32_i8
               {matrix_instruction(16, 16, 4, data_type_t::Int8), 32},   // v_mfma_i32_16x16x4_4b_i8
               {matrix_instruction(4, 4, 4, data_type_t::Int8), 8},      // v_mfma_i32_4x4x4_16b_i8
                                                                         // XF32
               {matrix_instruction(32, 32, 4, data_type_t::XFloat32),
                32},  // v_mfma_f32_32x32x4_xf32
               {matrix_instruction(16, 16, 32, data_type_t::XFloat32),
                16},  // v_mfma_f32_16x16x8_xf32
           }},
          {hardware_t::architecture_t::gfx950,
           {
               // F32
               {matrix_instruction(32, 32, 2, data_type_t::Float), 64},  // v_mfma_f32_32x32x2_f32
               {matrix_instruction(32, 32, 1, data_type_t::Float),
                64},  // v_mfma_f32_32x32x1_2b_f32
               {matrix_instruction(16, 16, 4, data_type_t::Float), 32},  // v_mfma_f32_16x16x4_f32
               {matrix_instruction(16, 16, 1, data_type_t::Float),
                32},                                                  // v_mfma_f32_16x16x1_4b_f32
               {matrix_instruction(4, 4, 1, data_type_t::Float), 8},  // v_mfma_f32_4x4x1_16b_f32
                                                                      // F64
               {matrix_instruction(16, 16, 4, data_type_t::Double), 64},  // v_mfma_f64_16x16x4_f64
               {matrix_instruction(4, 4, 4, data_type_t::Double), 16},    // v_mfma_f64_4x4x4_4b_f64
                                                                          // TODO ComplexFloat
                                                                          // TODO ComplexDouble
                                                                          // F16
               {matrix_instruction(32, 32, 8, data_type_t::Half), 32},    // v_mfma_f32_32x32x8_f16
               {matrix_instruction(32, 32, 16, data_type_t::Half), 32},   // v_mfma_f32_32x32x16_f16
               {matrix_instruction(16, 16, 16, data_type_t::Half), 16},   // v_mfma_f32_16x16x16_f16
               {matrix_instruction(16, 16, 32, data_type_t::Half), 16},   // v_mfma_f32_16x16x32_f16
                                                                          // BF16
               {matrix_instruction(32, 32, 8, data_type_t::BFloat16),
                32},  // v_mfma_f32_32x32x8_bf16
               {matrix_instruction(32, 32, 16, data_type_t::BFloat16),
                32},  // v_mfma_f32_32x32x16_bf16
               {matrix_instruction(16, 16, 16, data_type_t::BFloat16),
                16},  // v_mfma_f32_16x16x16_bf16
               {matrix_instruction(16, 16, 32, data_type_t::BFloat16),
                16},  // v_mfma_f32_16x16x16_bf16
                      // F8
               {matrix_instruction(32, 32, 64, data_type_t::Float8), 64},  // v_mfma_f32_32x32x64_f8
               {matrix_instruction(32, 32, 16, data_type_t::Float8), 32},  // v_mfma_f32_32x32x16_f8
               {matrix_instruction(16, 16, 128, data_type_t::Float8),
                32},  // v_mfma_f32_16x16x128_f8
               {matrix_instruction(16, 16, 32, data_type_t::Float8), 16},  // v_mfma_f32_16x16x32_f8
                                                                           // BF8
               {matrix_instruction(32, 32, 64, data_type_t::BFloat8),
                64},  // v_mfma_f32_32x32x64_bf8
               {matrix_instruction(32, 32, 16, data_type_t::BFloat8),
                32},  // v_mfma_f32_32x32x16_bf8
               {matrix_instruction(16, 16, 128, data_type_t::BFloat8),
                32},  // v_mfma_f32_16x16x128_bf8
               {matrix_instruction(16, 16, 32, data_type_t::BFloat8),
                16},  // v_mfma_f32_16x16x32_bf8
                      // F8B8
               {matrix_instruction(32, 32, 64, data_type_t::Float8BFloat8),
                64},  // v_mfma_f32_32x32x64_f8_bf8
               {matrix_instruction(32, 32, 16, data_type_t::Float8BFloat8),
                32},  // v_mfma_f32_32x32x16_f8_bf8
               {matrix_instruction(16, 16, 128, data_type_t::Float8BFloat8),
                32},  // v_mfma_f32_16x16x128_f8_bf8
               {matrix_instruction(16, 16, 32, data_type_t::Float8BFloat8),
                16},  // v_mfma_f32_16x16x32_f8_bf8
                      // B8F8
               {matrix_instruction(32, 32, 64, data_type_t::BFloat8Float8),
                64},  // v_mfma_f32_32x32x64_bf8_f8
               {matrix_instruction(32, 32, 16, data_type_t::BFloat8Float8),
                32},  // v_mfma_f32_32x32x16_bf8_f8
               {matrix_instruction(16, 16, 128, data_type_t::BFloat8Float8),
                32},  // v_mfma_f32_16x16x128_bf8_f8
               {matrix_instruction(16, 16, 32, data_type_t::BFloat8Float8),
                16},  // v_mfma_f32_16x16x32_bf8_f8
                      // I8
               {matrix_instruction(32, 32, 16, data_type_t::Int8), 32},  // v_mfma_f32_32x32x16_f8
               {matrix_instruction(32, 32, 4, data_type_t::Int8), 64},   // v_mfma_i32_32x32x4_2b_i8
               {matrix_instruction(16, 16, 32, data_type_t::Int8), 16},  // v_mfma_f32_16x16x32_i8
               {matrix_instruction(16, 16, 4, data_type_t::Int8), 32},   // v_mfma_i32_16x16x4_4b_i8
               {matrix_instruction(4, 4, 4, data_type_t::Int8), 8},      // v_mfma_i32_4x4x4_16b_i8
                                                                         // XF32
               {matrix_instruction(32, 32, 8, data_type_t::XFloat32),
                96},  // v_mfma_f32_32x32x8_bf16 * 3
               {matrix_instruction(32, 32, 16, data_type_t::XFloat32),
                96},  // v_mfma_f32_32x32x16_bf16 * 3
               {matrix_instruction(16, 16, 16, data_type_t::XFloat32),
                48},  // v_mfma_f32_16x16x16_bf16 * 3
               {matrix_instruction(16, 16, 32, data_type_t::XFloat32),
                48},  // v_mfma_f32_16x16x16_bf16
                      // * 3 F6
               {matrix_instruction(32, 32, 64, data_type_t::Float6), 32},  // v_mfma_f32_32x32x64_f6
               {matrix_instruction(16, 16, 128, data_type_t::Float6),
                16},  // v_mfma_f32_16x16x128_f6
                      // BF6
               {matrix_instruction(32, 32, 64, data_type_t::BFloat6),
                32},  // v_mfma_f32_32x32x64_bf6
               {matrix_instruction(16, 16, 128, data_type_t::BFloat6),
                16},  // v_mfma_f32_16x16x128_bf6
                      // F4
               {matrix_instruction(32, 32, 64, data_type_t::Float4), 32},  // v_mfma_f32_32x32x64_f4
               {matrix_instruction(16, 16, 128, data_type_t::Float4),
                16},                                                   // v_mfma_f32_16x16x128_f4
                                                                       // DOT2
               {matrix_instruction(1, 1, 64, data_type_t::Half), 16},  // V_DOT2_F32_F16
               {matrix_instruction(1, 1, 64, data_type_t::BFloat16), 16},  // V_DOT2_F32_BF16
           }},
          {hardware_t::architecture_t::gfx1201,
           {
               // F16
               {matrix_instruction(16, 16, 16, data_type_t::Half),
                16},  // v_wmma_f16_16x16x16_f16/v_wmma_f32_16x16x16_f16
                      // BF16
               {matrix_instruction(16, 16, 16, data_type_t::BFloat16),
                16},  // v_wmma_bf16_16x16x16_bf16/v_wmma_f32_16x16x16_bf16
                      // F8
               {matrix_instruction(16, 16, 16, data_type_t::Float8),
                8},  // v_wmma_f32_16x16x16_fp8_fp8
                     // F8B8
               {matrix_instruction(16, 16, 16, data_type_t::Float8BFloat8),
                8},  // v_wmma_f32_16x16x16_fp8_bf8
                     // B8F8
               {matrix_instruction(16, 16, 16, data_type_t::BFloat8Float8),
                8},  // v_wmma_f32_16x16x16_bf8_fp8
                     // B8
               {matrix_instruction(16, 16, 16, data_type_t::BFloat8),
                8},  // v_wmma_f32_16x16x16_bf8_bf8
                     // I8
               {matrix_instruction(16, 16, 16, data_type_t::Int8), 8},  // v_wmma_i32_16x16x16_iu8
                                                                        // I4
               {matrix_instruction(16, 16, 16, data_type_t::Int4), 8},  // v_wmma_i32_16x16x16_iu4
               {matrix_instruction(16, 16, 32, data_type_t::Int4), 8},  // v_wmma_i32_16x16x32_iu4
           }},
          {hardware_t::architecture_t::gfx1100,
           {
               // F16
               {matrix_instruction(16, 16, 16, data_type_t::Half),
                32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
                      // BF16
               {matrix_instruction(16, 16, 16, data_type_t::BFloat16),
                32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
                      // I8
               {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
                                                                         // I4
               {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
           }},
          {hardware_t::architecture_t::gfx1151,
           {
               // F16
               {matrix_instruction(16, 16, 16, data_type_t::Half),
                32},  // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16
                      // BF16
               {matrix_instruction(16, 16, 16, data_type_t::BFloat16),
                32},  // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16
                      // I8
               {matrix_instruction(16, 16, 16, data_type_t::Int8), 32},  // v_wmma_i32_16x16x16_iu8
                                                                         // I4
               {matrix_instruction(16, 16, 16, data_type_t::Int4), 16},  // v_wmma_i32_16x16x16_iu4
           }}};

  architecture_t arch;
  size_t N_CU;          // Number of Compute Units
  size_t lds_capacity;  // Capacity of LDS
  double mem1_perf_ratio;
  double mem2_perf_ratio;
  double mem3_perf_ratio;
  size_t L2_capacity;  // Capacity of L2 in bytes
  size_t CU_per_L2;    // Number of compute units per L2 domain
  double compute_clock_ghz;
  size_t parallel_mi_cu;  // The number of parallel MI in a CU
  std::tuple<double, double, double> mem_bw_per_wg_coefficients;
  size_t NUM_XCD;

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
             std::tuple<double, double, double> mem_bw_per_wg_coefficients)
      : arch(arch)
      , N_CU(N_CU)
      , lds_capacity(lds_capacity)
      , mem1_perf_ratio(mem1_perf_ratio)
      , mem2_perf_ratio(mem2_perf_ratio)
      , mem3_perf_ratio(mem3_perf_ratio)
      , L2_capacity(L2_capacity)
      , CU_per_L2(N_CU / NUM_XCD)
      , compute_clock_ghz(compute_clock_ghz)
      , parallel_mi_cu(parallel_mi_cu)
      , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
      , NUM_XCD(NUM_XCD) {}

  hardware_t(hipDeviceProp_t properties) : hardware_t(get_hardware_for_properties(properties)) {}

  hardware_t(const hardware_t& other)
      : arch(other.arch)
      , N_CU(other.N_CU)
      , lds_capacity(other.lds_capacity)
      , mem1_perf_ratio(other.mem1_perf_ratio)
      , mem2_perf_ratio(other.mem2_perf_ratio)
      , mem3_perf_ratio(other.mem3_perf_ratio)
      , L2_capacity(other.L2_capacity)
      , CU_per_L2(other.CU_per_L2)
      , compute_clock_ghz(other.compute_clock_ghz)
      , parallel_mi_cu(other.parallel_mi_cu)
      , mem_bw_per_wg_coefficients(other.mem_bw_per_wg_coefficients)
      , NUM_XCD(other.NUM_XCD) {}

  static hardware_t get_hardware_for_properties(hipDeviceProp_t properties) {
    auto arch_name = get_before_first_colon(properties.gcnArchName);
    auto arch_enum = arch_name_to_enum(arch_name);
    if (arch_enum == architecture_t::Count) {
      throw std::runtime_error(
          std::string("Attempting to retrieve hardware constants for unsupported architecture: ") +
          std::string(arch_name));
    }
    auto constants = get_arch_constants(arch_enum);
    return hardware_t(
        arch_enum,
        properties.multiProcessorCount,
        properties.sharedMemPerBlock,
        constants.num_xcds,
        1e9 * constants.mem1_perf_ratio / properties.clockRate,
        1e9 * constants.mem2_perf_ratio / (properties.memoryClockRate * constants.mem_clock_ratio),
        1e9 * constants.mem3_perf_ratio / properties.memoryClockRate,
        properties.l2CacheSize,
        properties.clockRate / 1e6,
        constants.parallel_mi_cu,
        constants.mem_bw_per_wg_coefficients);
  }

  static hardware_t get_hardware_for_device(int deviceId) {
    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, deviceId);
    if (e) { throw std::runtime_error(hipGetErrorString(e)); }
    return get_hardware_for_properties(prop);
  }

  static bool is_hardware_supported(hipDeviceProp_t properties) {
    auto arch_name = get_before_first_colon(properties.gcnArchName);
    auto arch_enum = arch_name_to_enum(arch_name);
    return arch_enum != architecture_t::Count;
  }

  // Function to print hardware details
  void print() const {
    std::cout << "================== Hardware Configuration ==================\n";
    std::cout << "Number of CUs (N_CU)      : " << N_CU << "\n";
    std::cout << "LDS capacity              : " << lds_capacity << " bytes\n";
    std::cout << "mem1_perf_ratio           : " << mem1_perf_ratio << "\n";
    std::cout << "mem2_perf_ratio           : " << mem2_perf_ratio << "\n";
    std::cout << "mem3_perf_ratio           : " << mem3_perf_ratio << "\n";
    std::cout << "L2 Cache capacity         : " << L2_capacity << " bytes\n";
    std::cout << "CUs per L2 domain         : " << CU_per_L2 << "\n";
    std::cout << "Compute clock (GHz)       : " << compute_clock_ghz << "\n";
    std::cout << "Parallel MI/CU            : " << parallel_mi_cu << "\n";
    std::cout << "Number of XCDs (NUM_XCD)  : " << NUM_XCD << "\n";
    std::cout << "mem_bw_per_wg_coefficients: " << std::get<0>(mem_bw_per_wg_coefficients) << ", "
              << std::get<1>(mem_bw_per_wg_coefficients) << ", "
              << std::get<2>(mem_bw_per_wg_coefficients) << "\n\n";

    std::cout << "------------------ Instruction Map -------------------------\n";
    // Loop over the instruction_map and print each entry
    for (const auto& kv : INSTRUCTION_MAP.at(arch)) {
      const auto& key  = kv.first;
      const auto& L_MI = kv.second;

      std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N
                << ", MI_K=" << key.MI_K << ", mi_input_type=" << to_string(key.mi_input_type)
                << " bytes\n"
                << "  -> Latency (L_MI): " << L_MI << "\n";
    }
    std::cout << "===========================================================\n";
  }
  // Debug tracking info
  mutable std::unordered_map<std::string, std::string> debug_info;

  // Metrics collection mode (bypasses environment variable check)
  mutable bool metrics_collection_mode = false;

  static bool is_debug_enabled() {
    static bool debugEnvVar = read_debug_env_var();  // Used to cache the read.
    return debugEnvVar;
  }

  static bool is_heuristics_enabled() {
    static bool heuristicsEnvVar = read_heuristics_env_var();  // Used to cache the read.
    return heuristicsEnvVar;
  }

  void log_debug(const std::string& key, const std::string& value) const {
    if (is_debug_enabled() || metrics_collection_mode) { debug_info[key] = value; }
  }

  void log_debug(const std::string& key, double value) const {
    if (is_debug_enabled() || metrics_collection_mode) { debug_info[key] = std::to_string(value); }
  }

  // Enable/disable metrics collection mode
  void set_metrics_collection_mode(bool enabled) const { metrics_collection_mode = enabled; }

  void clear_debug() const { debug_info.clear(); }

  void print_debug_info() const {
    std::cout << "=== Hardware Debug Info ===\n";
    for (const auto& [key, val] : debug_info) { std::cout << key << ": " << val << "\n"; }
    std::cout << "===========================\n";
  }

  // Extract analytical metrics to CSV format
  void extract_analytical_metrics_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing\n";
      return;
    }

    // Write CSV header
    file << "metric_name,metric_value\n";

    // Write all debug info as CSV rows
    for (const auto& [key, val] : debug_info) { file << key << "," << val << "\n"; }

    file.close();
    std::cout << "Analytical metrics exported to CSV: " << filename << "\n";
  }

  // Get all analytical metrics as a map (for programmatic access)
  std::unordered_map<std::string, std::string> get_analytical_metrics() const { return debug_info; }

  // Log debug info regardless of environment variable setting (for metrics extraction)
  void log_debug_force(const std::string& key, const std::string& value) const {
    debug_info[key] = value;
  }

  void log_debug_force(const std::string& key, double value) const {
    debug_info[key] = std::to_string(value);
  }

  size_t get_mi_latency(size_t MI_M, size_t MI_N, size_t MI_K, data_type_t mi_input_type) const {
    const auto& instruction_map = INSTRUCTION_MAP.at(arch);
    auto key                    = matrix_instruction(MI_M, MI_N, MI_K, mi_input_type);

    auto it = instruction_map.find(key);
    if (it != instruction_map.end()) {
      return it->second / parallel_mi_cu;
    } else {
      std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                << ", MI_K=" << MI_K << ", mi_input_type=" << to_string(mi_input_type)
                << ". Returning latency value of 32 (really slow).\n";
      return 32 / parallel_mi_cu;  // Default latency if instruction is not found
    }
  }

 private:
  static std::string get_before_first_colon(const std::string& input) {
    size_t pos = input.find(':');
    if (pos != std::string::npos) { return input.substr(0, pos); }
    return input;  // Return the whole string if ':' is not found
  }

  // Helper function to read the debug environment variable
  static bool read_debug_env_var() {
    const char* env = std::getenv("ANALYTICAL_GEMM_DEBUG");
    return env && std::string(env) == "1";
  }

  // Helper function to read the heuristics environment variable
  static bool read_heuristics_env_var() {
    const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS");
    return !(env && std::string(env) == "0");
  }
};
}  // namespace origami
