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

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_bitwise_not. Computed once on the host and used as the
// reference for both the HOST and HIP backends.
//
// bitwise_not is a U8-only op (rppt_bitwise_not rejects any other dtype): each byte is
// replaced by its bitwise complement, which for U8 is exactly 255 - v. The result is
// bit-exact, so the caller compares with zero tolerance.

inline double bitwise_not_scalar(double v) {
    return static_cast<double>(static_cast<Rpp8u>(~static_cast<Rpp8u>(v)));
}

// Writes the bitwise-not result into dst, reading the source at the ROI offset and writing
// packed at the destination origin (matching the region and placement the RPP op uses).
// dst outside the written region is left as the caller initialized it.
template <typename T>
void bitwise_not_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                           RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(bitwise_not_scalar(to_double(src[srcIdx])));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_BITWISE_NOT_REF_H
