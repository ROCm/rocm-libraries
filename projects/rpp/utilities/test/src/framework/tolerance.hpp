/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_TOLERANCE_H
#define RPP_TEST_TOLERANCE_H

#include "framework/config_param.hpp"

namespace rpptest {

// A per-dtype absolute comparison bound. Plain data: an op names the numbers it needs and
// hands the selected one to a comparator, so nothing here has to know about comparators,
// configs or ops. Unlisted dtypes stay 0.0 (bit-exact), which is the right default for an
// op that does not run them.
struct Tolerance {
    // Aggregate order: integers first, then floats, each widening.
    double u8 = 0.0, i8 = 0.0, i16 = 0.0, f16 = 0.0, f32 = 0.0;

    constexpr double operator()(DType dt) const {
        switch (dt) {
            case DType::U8: return u8;
            case DType::I8: return i8;
            case DType::I16: return i16;
            case DType::F16: return f16;
            case DType::F32: return f32;
        }
        return 0.0;
    }

    // Copies with one field replaced, for the ops that differ from a named constant in a
    // single dtype (C++17 has no designated initializers).
    constexpr Tolerance with_u8(double v) const { return {v, i8, i16, f16, f32}; }
    constexpr Tolerance with_i8(double v) const { return {u8, v, i16, f16, f32}; }
    constexpr Tolerance with_i16(double v) const { return {u8, i8, v, f16, f32}; }
    constexpr Tolerance with_f16(double v) const { return {u8, i8, i16, v, f32}; }
    constexpr Tolerance with_f32(double v) const { return {u8, i8, i16, f16, v}; }
};

// The common shape: the integer dtypes share one bound, each float dtype has its own.
constexpr Tolerance tolerance(double integer, double f32, double f16) {
    return {integer, integer, integer, f16, f32};
}

// Bit-exact: the op is a copy, a permutation, or integer-only arithmetic, so any deviation
// at all is a defect.
constexpr Tolerance kExact{};

// An op that computes in float and quantizes back: one LSB for the integer dtypes
// (round-to-nearest), and the accumulated rounding of a few float operations for F32/F16.
// The bound most such ops land on; an op needing a different one names its own.
constexpr Tolerance kRoundingTolerance = tolerance(1.0, 2e-3, 5e-3);

}  // namespace rpptest

#endif  // RPP_TEST_TOLERANCE_H
