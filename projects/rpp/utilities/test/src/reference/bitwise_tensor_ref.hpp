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

#ifndef RPP_TEST_BITWISE_TENSOR_REF_H
#define RPP_TEST_BITWISE_TENSOR_REF_H

#include <rpp/rpp.h>

#include <cstddef>
#include <type_traits>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for the ND tensor-vs-tensor bitwise ops (rppt_tensor_and_tensor /
// _or_tensor / _xor_tensor). Modelled from the operations' definition and the public API
// header, NOT from the kernel; computed once on the host and used as the reference for
// both the HOST and HIP backends.
//
// Semantics: an elementwise bitwise combination of the two operands' stored bit patterns,
// with NumPy-style broadcasting -- an operand axis of extent 1 is reused across the
// corresponding output axis (the header permits broadcasting "when, for each axis, the
// corresponding dimensions of the input tensors are either equal or one of them is 1").
// The operation is on bits, not intensities: for signed I8 the byte is combined as-is, so
// there is no rounding, clamping or intensity-space conversion anywhere in this model and
// results are bit-exact (callers compare with zero tolerance).

enum class BitwiseTensorOp { And, Or, Xor };

template <typename T>
inline T bitwise_tensor_scalar(T a, T b, BitwiseTensorOp op) {
    using Bits = typename std::make_unsigned<T>::type;
    const Bits x = static_cast<Bits>(a);
    const Bits y = static_cast<Bits>(b);
    Bits r = 0;
    switch (op) {
        case BitwiseTensorOp::And: r = static_cast<Bits>(x & y); break;
        case BitwiseTensorOp::Or:  r = static_cast<Bits>(x | y); break;
        case BitwiseTensorOp::Xor: r = static_cast<Bits>(x ^ y); break;
    }
    return static_cast<T>(r);
}

template <typename T>
void bitwise_tensor_reference(const T* src1, const T* src2, T* dst, const RpptGenericDesc& out,
                              const RpptGenericDesc& s1, const RpptGenericDesc& s2,
                              BitwiseTensorOp op) {
    for_each_nd_element(out, s1, s2,
                        [&](std::size_t outIdx, std::size_t idx1, std::size_t idx2,
                            const NdDims&) {
                            dst[outIdx] = bitwise_tensor_scalar<T>(src1[idx1], src2[idx2], op);
                        });
}

}  // namespace rpptest

#endif  // RPP_TEST_BITWISE_TENSOR_REF_H
