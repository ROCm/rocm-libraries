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
        const std::unordered_map<Hardware::Architecture,
                                 std::unordered_map<MatrixInstruction, size_t>>
            Hardware::INSTRUCTION_MAP
            = {{Hardware::Architecture::gfx942,
                {
                    //Schema : (MI_M,MI_N,MI_K,Datatype_Size)
                    {MatrixInstruction(16, 16, 1, 32), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {MatrixInstruction(16, 16, 4, 32), 32}, // v_mfma_f32_16x16x4_f32
                    {MatrixInstruction(16, 16, 4, 64), 32}, // v_mfma_f64_16x16x4_f64
                    {MatrixInstruction(4, 4, 4, 16), 8}, // v_mfma_f32_4x4x4_16b_f16
                    {MatrixInstruction(32, 32, 2, 32), 64}, // v_mfma_f32_32x32x2_f32
                    {MatrixInstruction(16, 16, 4, 16), 32}, // v_mfma_f32_16x16x4_4b_f16
                    {MatrixInstruction(32, 32, 4, 8), 64}, // v_mfma_i32_32x32x4_2b_i8/f8/b8
                    {MatrixInstruction(32, 32, 16, 8), 32}, // v_mfma_i32_32x32x16_i8
                    {MatrixInstruction(4, 4, 4, 64), 16}, // v_mfma_f64_4x4x4_4b_f64
                    {MatrixInstruction(32, 32, 1, 32), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {MatrixInstruction(32, 32, 8, 16), 32}, // v_mfma_f32_32x32x8_f16
                    {MatrixInstruction(16, 16, 16, 16), 16}, // v_mfma_f32_16x16x16_f16/bf16
                    {MatrixInstruction(32, 32, 4, 16), 64}, // v_mfma_f32_32x32x4_2b_f16
                    {MatrixInstruction(4, 4, 1, 32), 8}, // v_mfma_f32_4x4x1_16b_f32
                    {MatrixInstruction(16, 16, 32, 8), 16}, // v_mfma_i32_16x16x32_i8
                    {MatrixInstruction(4, 4, 4, 8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    {MatrixInstruction(32, 32, 4, 32), 32}, // v_mfma_f32_32x32x4_xf32
                    {MatrixInstruction(16, 16, 4, 8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {MatrixInstruction(16, 16, 8, 32), 16}, // v_mfma_f32_16x16x8_xf32
                    {MatrixInstruction(16, 16, 32, 32), 16}, // v_mfma_f32_16x16x8_xf32
                }},
               {Hardware::Architecture::gfx950, //TODO: NEed to make sure these are the right MFMAs
                {
                    {MatrixInstruction(16, 16, 32, 8), 16}, // v_mfma_i32_16x16x32_i8
                    {MatrixInstruction(16, 16, 1, 32), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {MatrixInstruction(16, 16, 4, 32), 32}, // v_mfma_f32_16x16x4_f32
                    {MatrixInstruction(16, 16, 4, 64), 32}, // v_mfma_f64_16x16x4_f64
                    {MatrixInstruction(4, 4, 4, 16), 8}, // v_mfma_f32_4x4x4_16b_f16
                    {MatrixInstruction(32, 32, 2, 32), 64}, // v_mfma_f32_32x32x2_f32
                    {MatrixInstruction(16, 16, 4, 16), 32}, // v_mfma_f32_16x16x4_4b_f16
                    {MatrixInstruction(32, 32, 4, 8), 64}, // v_mfma_i32_32x32x4_2b_i8/f8/b8
                    {MatrixInstruction(32, 32, 16, 8), 32}, // v_mfma_i32_32x32x16_i8
                    {MatrixInstruction(4, 4, 4, 64), 16}, // v_mfma_f64_4x4x4_4b_f64
                    {MatrixInstruction(32, 32, 1, 32), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {MatrixInstruction(32, 32, 8, 16), 32}, // v_mfma_f32_32x32x8_f16
                    {MatrixInstruction(16, 16, 16, 16), 16}, // v_mfma_f32_16x16x16_f16/bf16
                    {MatrixInstruction(32, 32, 4, 16), 64}, // v_mfma_f32_32x32x4_2b_f16
                    {MatrixInstruction(4, 4, 1, 32), 8}, // v_mfma_f32_4x4x1_16b_f32
                    {MatrixInstruction(16, 16, 32, 8), 16}, // v_mfma_i32_16x16x32_i8
                    {MatrixInstruction(4, 4, 4, 8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    {MatrixInstruction(32, 32, 4, 32), 32}, // v_mfma_f32_32x32x4_xf32
                    {MatrixInstruction(16, 16, 4, 8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {MatrixInstruction(16, 16, 8, 32), 16}, // v_mfma_f32_16x16x8_xf32
                    //----------
                    {MatrixInstruction(32, 32, 64, 8), 64}, // v_mfma_i32_32x32x16_i8
                    {MatrixInstruction(16, 16, 32, 16), 16}, // v_mfma_i32_16x16x32_f16/bf16
                    //----------
                    //TODO these need to be checked.
                    {MatrixInstruction(16, 16, 128, 8), 32}, // V_MFMA_F32_16X16X128_F8
                    {MatrixInstruction(16, 16, 128, 6), 16}, // V_MFMA_F32_16X16X128_F6
                    {MatrixInstruction(16, 16, 128, 4), 16}, // V_MFMA_F32_16X16X128_F4

                }}};

    } // namespace analytical
} // namespace TensileLite
