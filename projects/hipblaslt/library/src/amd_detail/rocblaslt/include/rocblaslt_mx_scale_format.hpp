/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef ROCBLASLT_MX_SCALE_FORMAT_HPP
#define ROCBLASLT_MX_SCALE_FORMAT_HPP

// ----------------------------------------------------------------------------
// Lean, log-free MX scale-format combination helpers for the hipBLASLt host
// API. This header is intentionally kept narrow so it can be exercised by
// unit tests without dragging in the rocblaslt logging machinery defined in
// utility.hpp (get_logger_os, log_base, etc.) that links only inside the
// hipBLASLt library.
//
// rocblaslt_mat_utils.hpp wraps these helpers in a thin logger-aware shim
// that preserves the existing rocblaslt_status return type for production
// callers (see validateMXScaleFormatCombination in that header).
// ----------------------------------------------------------------------------

#include "rocblaslt-types.h"

#include <Tensile/MXScaleFormatValidation.hpp>
#include <rocisa/include/enum.hpp>

#include <optional>
#include <string>

// Returns the matrix-scale dtype implied by the ScalingFormat, or std::nullopt
// for non-block formats (None/Scalar/Vector) where the gfx1250 MX rules do not
// apply.
inline std::optional<rocisa::DataType>
    rocblasltScalingFormatToMXScaleDataType(RocblasltContractionProblem::ScalingFormat fmt)
{
    switch(fmt)
    {
    case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0:
    case RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0:
    case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT:
        return rocisa::DataType::E8;
    case RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3:
    case RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3:
        return rocisa::DataType::Float8;
    case RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3:
    case RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3:
        return rocisa::DataType::E5M3;
    case RocblasltContractionProblem::ScalingFormat::None:
    case RocblasltContractionProblem::ScalingFormat::Scalar:
    case RocblasltContractionProblem::ScalingFormat::Vector:
    default:
        return std::nullopt;
    }
}

// Maps the subset of hipDataType values relevant to gfx1250 MX matrix classes
// (FP8/BF8/FP6/BF6/FP4 and their fnuz variants) to rocisa::DataType. Other
// types collapse to rocisa::DataType::None - the MX rules don't apply to
// them, so the joint validator will accept any scale on that side.
inline rocisa::DataType rocblasltHipDataTypeToMXMatrixDataType(hipDataType type)
{
    switch(type)
    {
    case HIP_R_8F_E4M3:
        return rocisa::DataType::Float8;
    case HIP_R_8F_E5M2:
        return rocisa::DataType::BFloat8;
    case HIP_R_8F_E4M3_FNUZ:
        return rocisa::DataType::Float8_fnuz;
    case HIP_R_8F_E5M2_FNUZ:
        return rocisa::DataType::BFloat8_fnuz;
    case HIP_R_6F_E2M3:
        return rocisa::DataType::Float6;
    case HIP_R_6F_E3M2:
        return rocisa::DataType::BFloat6;
    case HIP_R_4F_E2M1:
        return rocisa::DataType::Float4;
    default:
        return rocisa::DataType::None;
    }
}

// Pure validator: returns std::nullopt when the joint (a_type, b_type,
// scaleAFmt, scaleBFmt) tuple is legal under the gfx1250
// v_wmma_scale_f32_16x16x128_f8f6f4 rules (or when the MX rules do not
// apply at all). Otherwise returns a human-readable diagnostic suitable
// for log_error.
//
// Sides whose ScalingFormat is not a block MX format (None/Scalar/Vector)
// are skipped (treated as if the matrix dtype on that side is non-MX), so
// this guard only fires for real MX problems. In particular, the FP4xFP4
// joint rule does not trigger across one MX side and one non-MX side.
inline std::optional<std::string> checkMXScaleFormatCombination(
    hipDataType                                a_type,
    hipDataType                                b_type,
    RocblasltContractionProblem::ScalingFormat scaleAFmt,
    RocblasltContractionProblem::ScalingFormat scaleBFmt)
{
    auto scaleADt = rocblasltScalingFormatToMXScaleDataType(scaleAFmt);
    auto scaleBDt = rocblasltScalingFormatToMXScaleDataType(scaleBFmt);

    // Neither side uses a block MX scale: the gfx1250 joint rules do not
    // apply at all; let other validators handle scalar/vector scaling.
    if(!scaleADt.has_value() && !scaleBDt.has_value())
        return std::nullopt;

    rocisa::DataType aMatrixDt
        = scaleADt.has_value() ? rocblasltHipDataTypeToMXMatrixDataType(a_type)
                               : rocisa::DataType::None;
    rocisa::DataType bMatrixDt
        = scaleBDt.has_value() ? rocblasltHipDataTypeToMXMatrixDataType(b_type)
                               : rocisa::DataType::None;
    rocisa::DataType aScaleDt = scaleADt.value_or(rocisa::DataType::None);
    rocisa::DataType bScaleDt = scaleBDt.value_or(rocisa::DataType::None);

    if(!TensileLite::isValidMXScaleFormatCombination(aMatrixDt, aScaleDt, bMatrixDt, bScaleDt))
    {
        return TensileLite::mxScaleFormatCombinationError(
            aMatrixDt, aScaleDt, bMatrixDt, bScaleDt);
    }

    return std::nullopt;
}

#endif // ROCBLASLT_MX_SCALE_FORMAT_HPP
