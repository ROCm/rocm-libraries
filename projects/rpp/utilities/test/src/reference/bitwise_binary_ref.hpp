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

#ifndef RPP_TEST_BITWISE_BINARY_REF_H
#define RPP_TEST_BITWISE_BINARY_REF_H

#include <rpp/rpp.h>

#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: bitwise binary

RPP op
  rppt_bitwise_and / rppt_bitwise_or / rppt_bitwise_xor
  (Image / Bitwise)

Description
  Element-wise bitwise combination of two co-located sources. Each output byte
  is formed from the two source bytes at the same position.

Expression
  and   dst = a & b
  or    dst = a | b
  xor   dst = a ^ b

Per-type form
  All three are U8-only; the ops reject any other type. Results are
  bit-exact, so the caller compares with zero tolerance.
*/

enum class BitwiseOp { And, Or, Xor };

inline double bitwise_binary_scalar(double a, double b, BitwiseOp op) {
    const Rpp8u x = static_cast<Rpp8u>(a);
    const Rpp8u y = static_cast<Rpp8u>(b);
    Rpp8u r = 0;
    switch (op) {
        case BitwiseOp::And:
            r = x & y;
            break;
        case BitwiseOp::Or:
            r = x | y;
            break;
        case BitwiseOp::Xor:
            r = x ^ y;
            break;
    }
    return static_cast<double>(r);
}

template <typename T>
void bitwise_binary_reference(const T* src1, const T* src2, const RpptDesc& sd, T* dst,
                              const RpptDesc& dd, const RpptROI* roi, RpptRoiType roiType,
                              BitwiseOp op) {
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(bitwise_binary_scalar(
                            to_double(src1[srcIdx]), to_double(src2[srcIdx]), op));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_BITWISE_BINARY_REF_H
