// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace origami {

// Static member definition
// clang-format off
const std::unordered_map<hardware_t::architecture_t,
                         std::unordered_map<matrix_instruction, size_t>>
    hardware_t::INSTRUCTION_MAP = {
        {hardware_t::architecture_t::gfx90a,
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
        {hardware_t::architecture_t::gfx942,
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
        {hardware_t::architecture_t::gfx950,
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
        {hardware_t::architecture_t::gfx1201,
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
        {hardware_t::architecture_t::gfx1100,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32}, // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16

             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32}, // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16

             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32}, // v_wmma_i32_16x16x16_iu8

             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16}, // v_wmma_i32_16x16x16_iu4
         }},
        {hardware_t::architecture_t::gfx1151,
         {
             // F16
             {matrix_instruction(16, 16, 16, data_type_t::Half), 32}, // v_wmma_f32_16x16x16_f16/v_wmma_f16_16x16x16_f16

             // BF16
             {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 32}, // v_wmma_f32_16x16x16_bf16/v_wmma_bf16_16x16x16_bf16

             // I8
             {matrix_instruction(16, 16, 16, data_type_t::Int8), 32}, // v_wmma_i32_16x16x16_iu8

             // I4
             {matrix_instruction(16, 16, 16, data_type_t::Int4), 16}, // v_wmma_i32_16x16x16_iu4
         }}};
// clang-format on

hardware_t::hardware_t(architecture_t arch,
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

hardware_t::hardware_t(hipDeviceProp_t properties)
    : hardware_t(get_hardware_for_properties(properties)) {}

hardware_t::hardware_t(const hardware_t& other)
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

hardware_t hardware_t::get_hardware_for_properties(hipDeviceProp_t properties) {
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

hardware_t hardware_t::get_hardware_for_device(int deviceId) {
  hipDeviceProp_t prop;
  hipError_t e = hipGetDeviceProperties(&prop, deviceId);
  if (e) { throw std::runtime_error(hipGetErrorString(e)); }
  return get_hardware_for_properties(prop);
}

bool hardware_t::is_hardware_supported(hipDeviceProp_t properties) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  return arch_enum != architecture_t::Count;
}

void hardware_t::print() const {
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

    std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N << ", MI_K=" << key.MI_K
              << ", mi_input_type=" << datatype_to_string(key.mi_input_type) << " bytes\n"
              << "  -> Latency (L_MI): " << L_MI << "\n";
  }
  std::cout << "===========================================================\n";
}

size_t hardware_t::get_mi_latency(size_t MI_M,
                                  size_t MI_N,
                                  size_t MI_K,
                                  data_type_t mi_input_type) const {
  const auto& instruction_map = INSTRUCTION_MAP.at(arch);
  auto key                    = matrix_instruction(MI_M, MI_N, MI_K, mi_input_type);

  auto it = instruction_map.find(key);
  if (it != instruction_map.end()) {
    return it->second / parallel_mi_cu;
  } else {
    if (origami::runtime_options().get().debug_enabled)
      std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                << ", MI_K=" << MI_K << ", mi_input_type=" << datatype_to_string(mi_input_type)
                << ". Returning latency value of 32 (really slow).\n";
    return 32 / parallel_mi_cu;  // Default latency if instruction is not found
  }
}

std::string hardware_t::get_before_first_colon(const std::string& input) {
  size_t pos = input.find(':');
  if (pos != std::string::npos) { return input.substr(0, pos); }
  return input;  // Return the whole string if ':' is not found
}

}  // namespace origami
