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

#ifndef RPP_TEST_ARITHMETIC_TENSOR_REF_H
#define RPP_TEST_ARITHMETIC_TENSOR_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for the ND tensor-vs-tensor arithmetic ops (rppt_tensor_add_tensor /
// _subtract_tensor / _multiply_tensor / _divide_tensor). Modelled from the operations'
// definition and the public API header, NOT from the kernel; computed once on the host and
// used as the reference for both the HOST and HIP backends.
//
// Semantics: "element-wise addition/subtraction/multiplication/division of two
// N-dimensional tensors", with NumPy-style broadcasting -- "for every axis, the two input
// tensors must either have the same length or one of them must be 1", so an operand axis of
// extent 1 is reused across the corresponding output axis.
//
// Two per-dtype behaviors the header does not spell out, taken from the operations'
// definition and RPP's convention everywhere else:
//   - Integer dtypes hold integers, so the exact arithmetic result is rounded to nearest
//     (std::nearbyint) and saturated to the dtype's storable range. Only division produces a
//     fractional intermediate, so rounding is only observable there; addition, subtraction
//     and multiplication only exercise the saturation.
//   - Floating-point dtypes are NOT clamped. These are generic ND tensors rather than image
//     intensities, so a subtraction that goes negative or a division that exceeds 1 is the
//     intended result; clamping to [0,1] here would make subtract and divide meaningless.
// Arithmetic is carried in double and quantized once, at the store.
//
// Division by zero is undefined and is not modelled: the divide test shapes its second
// operand so no divisor is ever zero.

enum class ArithmeticTensorOp { Add, Subtract, Multiply, Divide };

inline double arithmetic_tensor_scalar(double a, double b, ArithmeticTensorOp op) {
    switch (op) {
        case ArithmeticTensorOp::Add:      return a + b;
        case ArithmeticTensorOp::Subtract: return a - b;
        case ArithmeticTensorOp::Multiply: return a * b;
        case ArithmeticTensorOp::Divide:   return a / b;
    }
    return 0.0;
}

// Stores the exact result into T: round-to-nearest + saturate for the integer dtypes, a plain
// (unclamped) conversion for the floating-point ones. Rpp16f is a class type, so it takes the
// non-integral branch.
template <typename T, bool Integral = std::is_integral<T>::value>
struct ArithmeticTensorStore {
    static T apply(double v) { return from_double<T>(v); }
};

template <typename T>
struct ArithmeticTensorStore<T, true> {
    static T apply(double v) {
        const double lo = static_cast<double>(std::numeric_limits<T>::lowest());
        const double hi = static_cast<double>(std::numeric_limits<T>::max());
        return static_cast<T>(clampd(std::nearbyint(v), lo, hi));
    }
};

template <typename T>
void arithmetic_tensor_reference(const T* src1, const T* src2, T* dst, const RpptGenericDesc& out,
                                 const RpptGenericDesc& s1, const RpptGenericDesc& s2,
                                 ArithmeticTensorOp op) {
    for_each_nd_element(out, s1, s2,
                        [&](std::size_t outIdx, std::size_t idx1, std::size_t idx2,
                            const NdDims&) {
                            const double v = arithmetic_tensor_scalar(
                                to_double(src1[idx1]), to_double(src2[idx2]), op);
                            dst[outIdx] = ArithmeticTensorStore<T>::apply(v);
                        });
}

}  // namespace rpptest

#endif  // RPP_TEST_ARITHMETIC_TENSOR_REF_H
