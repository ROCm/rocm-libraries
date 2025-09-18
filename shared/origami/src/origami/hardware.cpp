// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"

namespace origami
{
        const std::unordered_map<hardware_t::architecture_t, hardware_t::architecture_constants>
            hardware_t::ARCH_CONSTANT_MAP
            = {{hardware_t::architecture_t::gfx90a,
                hardware_t::architecture_constants(
                    1, 5.5, 1.21875121875121875122 * 1.2, 1.2, 4, std::make_tuple(0, 0.03, 0), 1.5)},
               {hardware_t::architecture_t::gfx942,
                hardware_t::architecture_constants(
                    8, 17, 1.21875121875121875122 * 6, 4, 4, std::make_tuple(0, 0.015, 0), 1.5)},
               {hardware_t::architecture_t::gfx950,
                // hardware_t::architecture_constants(
                //     8, 17, 1.21875121875121875122 * 7, 6, 4, std::make_tuple(-0.000013, 0.007070, 0.027355), 1.5)}};
                hardware_t::architecture_constants(
                    8, 17, 1.21875121875121875122 * 7, 6, 4, std::make_tuple(0, 0.008, 0), 1.5)}};

        // Schema : (MI_M, MI_N, MI_K, data_type_t)
        const std::unordered_map<hardware_t::architecture_t,
                                 std::unordered_map<matrix_instruction, size_t>>
            hardware_t::INSTRUCTION_MAP
            = {{hardware_t::architecture_t::gfx90a,
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
                    {matrix_instruction( 1,  1,  64, data_type_t::Half), 16}, // V_DOT2_F32_F16
                    {matrix_instruction( 1,  1,  64, data_type_t::BFloat16), 16}, // V_DOT2_F32_BF16
                }}};
} // namespace origami
