/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string>

#include <Tensile/Macros.hpp>
#include <rocisa/include/enum.hpp>

TENSILE_HIDDEN_BEGIN

namespace TensileLite
{
    /**
     * \ingroup DataTypes
     * \brief MX scale-format combination validation.
     *
     * The gfx1250 ISA constrains the legal (matrix_a_fmt, matrix_a_scale_fmt,
     * matrix_b_fmt, matrix_b_scale_fmt) tuples accepted by
     * v_wmma_scale_f32_16x16x128_f8f6f4. The AMDGPU assembler currently does
     * not enforce these joint constraints (see ROCm/llvm-project#2634), so
     * Tensilelite and hipBLASLt must validate them in the host before
     * configuring kernels that would otherwise be silently miscompiled.
     *
     * Per the ISA:
     *   - matrix class FP8 / BF8 / FP6 / BF6 must pair with scale E8 (UE8M0).
     *   - matrix class FP4 may pair with E8 (UE8M0), E5M3, or E4M3 (FP8 OCP).
     *   - when both A and B are FP4, the two scales must match.
     *
     * In Tensilelite's enum vocabulary:
     *   - "E8" scale          -> rocisa::DataType::E8
     *   - "E5M3" scale        -> rocisa::DataType::E5M3
     *   - "E4M3" scale        -> rocisa::DataType::Float8 (OCP FP8 E4M3)
     *
     * Any other scale type used with an MX matrix class is invalid.
     */

    /// True if dt is an MX matrix class governed by the gfx1250 f8f6f4 rules
    /// (FP8 / BF8 / FP6 / BF6 / FP4). Returns false for non-MX dtypes (Float,
    /// Half, BFloat16, ...).
    inline bool isMXMatrixDataType(rocisa::DataType dt)
    {
        switch(dt)
        {
        case rocisa::DataType::Float8:
        case rocisa::DataType::BFloat8:
        case rocisa::DataType::Float6:
        case rocisa::DataType::BFloat6:
        case rocisa::DataType::Float4:
            return true;
        default:
            return false;
        }
    }

    /// True if dt is the FP4 matrix class (the only class that admits more
    /// than one legal scale format).
    inline bool isFP4MatrixDataType(rocisa::DataType dt)
    {
        return dt == rocisa::DataType::Float4;
    }

    /// True if scaleDt is a legal MX scale format for the given matrix dtype.
    /// Per gfx1250 ISA:
    ///   - FP8/BF8/FP6/BF6 require E8.
    ///   - FP4 accepts E8, E5M3, or E4M3 (= Float8 in tensilelite enum).
    ///   - For any non-MX matrix dtype this returns true (the rules don't
    ///     apply; the host stack uses other paths for scaling there).
    inline bool isValidMXScaleFormatForDataType(rocisa::DataType matrixDt,
                                                rocisa::DataType scaleDt)
    {
        if(!isMXMatrixDataType(matrixDt))
            return true; // Non-MX class - the scale-format constraint does not apply.

        if(isFP4MatrixDataType(matrixDt))
        {
            return scaleDt == rocisa::DataType::E8
                   || scaleDt == rocisa::DataType::E5M3
                   || scaleDt == rocisa::DataType::Float8; // E4M3 (OCP FP8)
        }

        // FP8 / BF8 / FP6 / BF6 only support E8.
        return scaleDt == rocisa::DataType::E8;
    }

    /// True if the joint (aType, scaleAType, bType, scaleBType) tuple is a
    /// legal v_wmma_scale_f32_16x16x128_f8f6f4 combination on gfx1250.
    ///
    /// In addition to the per-side rules in isValidMXScaleFormatForDataType,
    /// the FP4xFP4 case requires the two scale formats to match.
    inline bool isValidMXScaleFormatCombination(rocisa::DataType aType,
                                                rocisa::DataType scaleAType,
                                                rocisa::DataType bType,
                                                rocisa::DataType scaleBType)
    {
        if(!isValidMXScaleFormatForDataType(aType, scaleAType))
            return false;
        if(!isValidMXScaleFormatForDataType(bType, scaleBType))
            return false;

        // FP4 x FP4 -> scales must match. (FP6/FP8/BF6/BF8 each already
        // pin scale to E8, so a mixed-class problem cannot have mismatching
        // scales except via the FP4-only rule.)
        if(isFP4MatrixDataType(aType) && isFP4MatrixDataType(bType))
        {
            if(scaleAType != scaleBType)
                return false;
        }

        return true;
    }

    /// Format a (aType, scaleAType, bType, scaleBType) tuple as a string
    /// suitable for error messages and log output.
    std::string formatMXScaleFormatCombination(rocisa::DataType aType,
                                               rocisa::DataType scaleAType,
                                               rocisa::DataType bType,
                                               rocisa::DataType scaleBType);

    /// Build a diagnostic explaining why the tuple is invalid. Returns the
    /// empty string when the combination is valid.
    std::string mxScaleFormatCombinationError(rocisa::DataType aType,
                                              rocisa::DataType scaleAType,
                                              rocisa::DataType bType,
                                              rocisa::DataType scaleBType);
} // namespace TensileLite

TENSILE_HIDDEN_END
