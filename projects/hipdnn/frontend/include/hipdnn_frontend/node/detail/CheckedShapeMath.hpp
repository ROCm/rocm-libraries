// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <hipdnn_frontend/Error.hpp>

namespace hipdnn_frontend::detail
{
// Overflow-checked int64_t arithmetic for shape formulas. Each primitive returns false when the
// exact result does not fit int64_t and then leaves out unchanged. The checks compare against
// the int64_t limits before the operation, so no intermediate wraps and no compiler builtin is
// needed.

inline bool checkedAdd(int64_t a, int64_t b, int64_t& out)
{
    if((b > 0 && a > std::numeric_limits<int64_t>::max() - b)
       || (b < 0 && a < std::numeric_limits<int64_t>::min() - b))
    {
        return false;
    }
    out = a + b;
    return true;
}

inline bool checkedSub(int64_t a, int64_t b, int64_t& out)
{
    if((b > 0 && a < std::numeric_limits<int64_t>::min() + b)
       || (b < 0 && a > std::numeric_limits<int64_t>::max() + b))
    {
        return false;
    }
    out = a - b;
    return true;
}

inline bool checkedMul(int64_t a, int64_t b, int64_t& out)
{
    if(a == 0 || b == 0)
    {
        out = 0;
        return true;
    }

    const int64_t maxValue = std::numeric_limits<int64_t>::max();
    const int64_t minValue = std::numeric_limits<int64_t>::min();

    // Each bound is a quotient of an int64_t limit by a nonzero operand. Integer division
    // truncates toward zero, which is the correct rounding for every sign combination, and none
    // of these divisions can itself overflow (the divisor is never -1 with a dividend of
    // minValue).
    bool fits = false;
    if(a > 0)
    {
        fits = (b > 0) ? (a <= maxValue / b) : (b >= minValue / a);
    }
    else
    {
        fits = (b > 0) ? (a >= minValue / b) : (b >= maxValue / a);
    }

    if(!fits)
    {
        return false;
    }
    out = a * b;
    return true;
}

// INVALID_VALUE error for a shape term whose exact value does not fit int64_t.
inline Error shapeTermOverflowError(const std::string& nodeName,
                                    const std::string& term,
                                    size_t spatialIdx,
                                    const std::string& operands)
{
    return {ErrorCode::INVALID_VALUE,
            nodeName + ": " + term + " overflows int64 at spatial index "
                + std::to_string(spatialIdx) + " (" + operands + ")"};
}

// dilation * (kernel - 1) + 1: the extent a dilated kernel covers.
inline Error checkedDilatedKernelSize(
    const std::string& nodeName, size_t spatialIdx, int64_t dilation, int64_t kernel, int64_t& out)
{
    int64_t kernelMinusOne = 0;
    int64_t scaled = 0;
    int64_t result = 0;
    if(!checkedSub(kernel, 1, kernelMinusOne) || !checkedMul(dilation, kernelMinusOne, scaled)
       || !checkedAdd(scaled, 1, result))
    {
        return shapeTermOverflowError(nodeName,
                                      "dilated kernel size",
                                      spatialIdx,
                                      "dilation " + std::to_string(dilation) + ", kernel size "
                                          + std::to_string(kernel));
    }
    out = result;
    return {};
}

// extent + prePad + postPad - span, where span is the dilated kernel size or the resample
// window.
inline Error checkedPaddedNumerator(const std::string& nodeName,
                                    size_t spatialIdx,
                                    int64_t extent,
                                    int64_t prePad,
                                    int64_t postPad,
                                    int64_t span,
                                    int64_t& out)
{
    int64_t prePadded = 0;
    int64_t padded = 0;
    int64_t result = 0;
    if(!checkedAdd(extent, prePad, prePadded) || !checkedAdd(prePadded, postPad, padded)
       || !checkedSub(padded, span, result))
    {
        return shapeTermOverflowError(
            nodeName,
            "padded extent minus span",
            spatialIdx,
            "extent " + std::to_string(extent) + ", pre_padding " + std::to_string(prePad)
                + ", post_padding " + std::to_string(postPad) + ", span " + std::to_string(span));
    }
    out = result;
    return {};
}

// numerator / stride + 1 with truncating division. stride must be nonzero; callers validate
// stride >= 1 first.
inline Error checkedStridedExtent(
    const std::string& nodeName, size_t spatialIdx, int64_t numerator, int64_t stride, int64_t& out)
{
    int64_t result = 0;
    if((stride == -1 && numerator == std::numeric_limits<int64_t>::min())
       || !checkedAdd(numerator / stride, 1, result))
    {
        return shapeTermOverflowError(nodeName,
                                      "output extent",
                                      spatialIdx,
                                      "numerator " + std::to_string(numerator) + ", stride "
                                          + std::to_string(stride));
    }
    out = result;
    return {};
}

// (dy - 1) * stride - prePad - postPad + window: the input extent a strided window over dy
// spans.
inline Error checkedTransposedExtent(const std::string& nodeName,
                                     size_t spatialIdx,
                                     int64_t dy,
                                     int64_t stride,
                                     int64_t prePad,
                                     int64_t postPad,
                                     int64_t window,
                                     int64_t& out)
{
    int64_t dyMinusOne = 0;
    int64_t strided = 0;
    int64_t lessPrePad = 0;
    int64_t unpadded = 0;
    int64_t result = 0;
    if(!checkedSub(dy, 1, dyMinusOne) || !checkedMul(dyMinusOne, stride, strided)
       || !checkedSub(strided, prePad, lessPrePad) || !checkedSub(lessPrePad, postPad, unpadded)
       || !checkedAdd(unpadded, window, result))
    {
        return shapeTermOverflowError(
            nodeName,
            "input extent",
            spatialIdx,
            "dy " + std::to_string(dy) + ", stride " + std::to_string(stride) + ", pre_padding "
                + std::to_string(prePad) + ", post_padding " + std::to_string(postPad) + ", window "
                + std::to_string(window));
    }
    out = result;
    return {};
}
} // namespace hipdnn_frontend::detail
