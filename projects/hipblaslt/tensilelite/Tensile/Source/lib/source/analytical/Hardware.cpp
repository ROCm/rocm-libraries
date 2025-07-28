/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/analytical/Hardware.hpp>

namespace TensileLite
{
    namespace analytical
    {
        const std::unordered_map<Hardware::Architecture, Hardware::ArchitectureConstants>
            Hardware::ARCH_CONSTANT_MAP
            = {{Hardware::Architecture::gfx942,
                Hardware::ArchitectureConstants(
                    8, 17, 1.21875121875121875122 * 6, 4, 4, 1.5e-2, 1.5)},
               {Hardware::Architecture::gfx950,
                Hardware::ArchitectureConstants(
                    8, 17, 1.21875121875121875122 * 7, 6, 4, 0.008, 1.5)}};

        // Schema : (MI_M, MI_N, MI_K, DataType)
        const std::unordered_map<Hardware::Architecture,
                                 std::unordered_map<MatrixInstruction, size_t>>
            Hardware::INSTRUCTION_MAP
            = {{Hardware::Architecture::gfx942,
                {
                    // F32
                    {MatrixInstruction(32, 32, 2, DataType::Float), 64}, // v_mfma_f32_32x32x2_f32
                    {MatrixInstruction(32, 32, 1, DataType::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {MatrixInstruction(16, 16, 4, DataType::Float), 32}, // v_mfma_f32_16x16x4_f32
                    {MatrixInstruction(16, 16, 1, DataType::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {MatrixInstruction(4, 4, 1, DataType::Float), 8}, // v_mfma_f32_4x4x1_16b_f32
                    // F64
                    {MatrixInstruction(16, 16, 4, DataType::Double), 32}, // v_mfma_f64_16x16x4_f64
                    {MatrixInstruction(4, 4, 4, DataType::Double), 16}, // v_mfma_f64_4x4x4_4b_f64
                    // TODO ComplexFloat
                    // TODO ComplexDouble
                    // F16
                    {MatrixInstruction(32, 32, 4, DataType::Half), 64}, // v_mfma_f32_32x32x4_2b_f16
                    {MatrixInstruction(32, 32, 8, DataType::Half), 32}, // v_mfma_f32_32x32x8_f16
                    {MatrixInstruction(16, 16, 4, DataType::Half), 32}, // v_mfma_f32_16x16x4_4b_f16
                    {MatrixInstruction(16, 16, 16, DataType::Half), 16}, // v_mfma_f32_16x16x16_f16
                    {MatrixInstruction(4, 4, 4, DataType::Half), 8}, // v_mfma_f32_4x4x4_16b_f16
                    // BF16
                    {MatrixInstruction(32, 32, 4, DataType::BFloat16), 64}, // v_mfma_f32_32x32x4_2b_bf16
                    {MatrixInstruction(32, 32, 8, DataType::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
                    {MatrixInstruction(16, 16, 4, DataType::BFloat16), 32}, // v_mfma_f32_16x16x4_4b_bf16
                    {MatrixInstruction(16, 16, 16, DataType::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    {MatrixInstruction(4, 4, 4, DataType::BFloat16), 8}, // v_mfma_f32_4x4x4_16b_bf16
                    // F8
                    {MatrixInstruction(32, 32, 16, DataType::Float8_fnuz), 32}, // v_mfma_f32_32x32x16_f8
                    {MatrixInstruction(16, 16, 32, DataType::Float8_fnuz), 16}, // v_mfma_f32_16x16x32_f8
                    // BF8
                    {MatrixInstruction(32, 32, 16, DataType::BFloat8_fnuz), 32}, // v_mfma_f32_32x32x16_bf8
                    {MatrixInstruction(16, 16, 32, DataType::BFloat8_fnuz), 16}, // v_mfma_f32_16x16x32_bf8
                    // F8B8
                    {MatrixInstruction(32, 32, 16, DataType::Float8BFloat8_fnuz), 32}, // v_mfma_f32_32x32x16_f8_bf8
                    {MatrixInstruction(16, 16, 32, DataType::Float8BFloat8_fnuz), 16}, // v_mfma_f32_16x16x32_f8_bf8
                    // B8F8
                    {MatrixInstruction(32, 32, 16, DataType::BFloat8Float8_fnuz), 32}, // v_mfma_f32_32x32x16_bf8_f8
                    {MatrixInstruction(16, 16, 32, DataType::BFloat8Float8_fnuz), 16}, // v_mfma_f32_16x16x32_bf8_f8
                    // I8
                    {MatrixInstruction(32, 32, 16, DataType::Int8), 32}, // v_mfma_f32_32x32x16_f8
                    {MatrixInstruction(32, 32, 4, DataType::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
                    {MatrixInstruction(16, 16, 32, DataType::Int8), 16}, // v_mfma_f32_16x16x32_i8
                    {MatrixInstruction(16, 16, 4, DataType::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {MatrixInstruction(4, 4, 4, DataType::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    // XF32
                    {MatrixInstruction(32, 32, 4, DataType::XFloat32), 32}, // v_mfma_f32_32x32x4_xf32
                    {MatrixInstruction(16, 16, 32, DataType::XFloat32), 16}, // v_mfma_f32_16x16x8_xf32
                }},
               {Hardware::Architecture::gfx950,
                {
                    // F32
                    {MatrixInstruction(32, 32, 2, DataType::Float), 64}, // v_mfma_f32_32x32x2_f32
                    {MatrixInstruction(32, 32, 1, DataType::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {MatrixInstruction(16, 16, 4, DataType::Float), 32}, // v_mfma_f32_16x16x4_f32
                    {MatrixInstruction(16, 16, 1, DataType::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {MatrixInstruction(4, 4, 1, DataType::Float), 8}, // v_mfma_f32_4x4x1_16b_f32
                    // F64
                    {MatrixInstruction(16, 16, 4, DataType::Double), 64}, // v_mfma_f64_16x16x4_f64
                    {MatrixInstruction(4, 4, 4, DataType::Double), 16}, // v_mfma_f64_4x4x4_4b_f64
                    // TODO ComplexFloat
                    // TODO ComplexDouble
                    // F16
                    {MatrixInstruction(32, 32, 8, DataType::Half), 32}, // v_mfma_f32_32x32x8_f16
                    {MatrixInstruction(32, 32, 16, DataType::Half), 32}, // v_mfma_f32_32x32x16_f16
                    {MatrixInstruction(16, 16, 16, DataType::Half), 16}, // v_mfma_f32_16x16x16_f16
                    {MatrixInstruction(16, 16, 32, DataType::Half), 16}, // v_mfma_f32_16x16x32_f16
                    // BF16
                    {MatrixInstruction(32, 32, 8, DataType::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
                    {MatrixInstruction(32, 32, 16, DataType::BFloat16), 32}, // v_mfma_f32_32x32x16_bf16
                    {MatrixInstruction(16, 16, 16, DataType::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    {MatrixInstruction(16, 16, 32, DataType::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    // F8
                    {MatrixInstruction(32, 32, 64, DataType::Float8), 64}, // v_mfma_f32_32x32x64_f8
                    {MatrixInstruction(32, 32, 16, DataType::Float8), 32}, // v_mfma_f32_32x32x16_f8
                    {MatrixInstruction(16, 16, 128, DataType::Float8), 32}, // v_mfma_f32_16x16x128_f8
                    {MatrixInstruction(16, 16, 32, DataType::Float8), 16}, // v_mfma_f32_16x16x32_f8
                    // BF8
                    {MatrixInstruction(32, 32, 64, DataType::BFloat8), 64}, // v_mfma_f32_32x32x64_bf8
                    {MatrixInstruction(32, 32, 16, DataType::BFloat8), 32}, // v_mfma_f32_32x32x16_bf8
                    {MatrixInstruction(16, 16, 128, DataType::BFloat8), 32}, // v_mfma_f32_16x16x128_bf8
                    {MatrixInstruction(16, 16, 32, DataType::BFloat8), 16}, // v_mfma_f32_16x16x32_bf8
                    // F8B8
                    {MatrixInstruction(32, 32, 64, DataType::Float8BFloat8), 64}, // v_mfma_f32_32x32x64_f8_bf8
                    {MatrixInstruction(32, 32, 16, DataType::Float8BFloat8), 32}, // v_mfma_f32_32x32x16_f8_bf8
                    {MatrixInstruction(16, 16, 128, DataType::Float8BFloat8), 32}, // v_mfma_f32_16x16x128_f8_bf8
                    {MatrixInstruction(16, 16, 32, DataType::Float8BFloat8), 16}, // v_mfma_f32_16x16x32_f8_bf8
                    // B8F8
                    {MatrixInstruction(32, 32, 64, DataType::BFloat8Float8), 64}, // v_mfma_f32_32x32x64_bf8_f8
                    {MatrixInstruction(32, 32, 16, DataType::BFloat8Float8), 32}, // v_mfma_f32_32x32x16_bf8_f8
                    {MatrixInstruction(16, 16, 128, DataType::BFloat8Float8), 32}, // v_mfma_f32_16x16x128_bf8_f8
                    {MatrixInstruction(16, 16, 32, DataType::BFloat8Float8), 16}, // v_mfma_f32_16x16x32_bf8_f8
                    // I8
                    {MatrixInstruction(32, 32, 16, DataType::Int8), 32}, // v_mfma_f32_32x32x16_f8
                    {MatrixInstruction(32, 32, 4, DataType::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
                    {MatrixInstruction(16, 16, 32, DataType::Int8), 16}, // v_mfma_f32_16x16x32_i8
                    {MatrixInstruction(16, 16, 4, DataType::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {MatrixInstruction(4, 4, 4, DataType::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    // XF32
                    {MatrixInstruction(32, 32, 8, DataType::BFloat16), 96}, // v_mfma_f32_32x32x8_bf16 * 3
                    {MatrixInstruction(32, 32, 16, DataType::BFloat16), 96}, // v_mfma_f32_32x32x16_bf16 * 3
                    {MatrixInstruction(16, 16, 16, DataType::BFloat16), 48}, // v_mfma_f32_16x16x16_bf16 * 3
                    {MatrixInstruction(16, 16, 32, DataType::BFloat16), 48}, // v_mfma_f32_16x16x16_bf16 * 3
                    // F6
                    {MatrixInstruction(32, 32, 64, DataType::Float6), 32}, // v_mfma_f32_32x32x64_f6
                    {MatrixInstruction(16, 16, 128, DataType::Float6), 16}, // v_mfma_f32_16x16x128_f6
                    // F4
                    {MatrixInstruction(32, 32, 64, DataType::Float4), 32}, // v_mfma_f32_32x32x64_f4
                    {MatrixInstruction(16, 16, 128, DataType::Float4), 16}, // v_mfma_f32_16x16x128_f4
                    // DOT2
                    {MatrixInstruction( 1,  1,  64, DataType::Half), 16}, // V_DOT2_F32_F16
                    {MatrixInstruction( 1,  1,  64, DataType::BFloat16), 16}, // V_DOT2_F32_BF16
                }}};

    } // namespace analytical
} // namespace TensileLite
