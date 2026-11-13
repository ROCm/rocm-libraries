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

#ifndef RPP_TEST_BITWISE_NOT_REF_H
#define RPP_TEST_BITWISE_NOT_REF_H

#include <rpp/rpp.h>

#include <type_traits>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: bitwise_not

RPP op
  rppt_bitwise_not   (Image / Bitwise)

Description
  Element-wise bitwise complement. Each byte is replaced by its complement,
  which for U8 is exactly 255 - v, so the op is a tone inversion.

Expression
  dst(x, y, c) = ~src(x, y, c) = 255 - src(x, y, c)

Per-type form
  U8-only; the op rejects any other type. The result is bit-exact, so the
  caller compares with zero tolerance.
*/

inline Rpp8u bitwise_not_scalar(Rpp8u v) {
    return static_cast<Rpp8u>(~v);
}

template <typename T>
void bitwise_not_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                           const RpptROI* roi, RpptRoiType roiType) {
    static_assert(std::is_same_v<T, Rpp8u>, "bitwise_not is U8-only");
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = bitwise_not_scalar(src[srcIdx]);
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_BITWISE_NOT_REF_H
