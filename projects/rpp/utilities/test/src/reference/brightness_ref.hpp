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

#ifndef RPP_TEST_BRIGHTNESS_REF_H
#define RPP_TEST_BRIGHTNESS_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_brightness, derived from the op's definition
// (out = alpha * pixel + beta), NOT from the RPP kernel. Used as the reference for both
// backends so kernel bugs surface as diffs.
//
// beta is expressed in [0,255] pixel units. Integer types work in [0,255] intensity space
// and round to nearest; I8 pixels are the same intensities shifted by -128:
//   U8  : clamp[0,255]  ( round(alpha * v + beta) )
//   I8  : clamp[-128,127]( round(alpha * (v + 128) + beta) - 128 )
//   F32 : clamp[0,1]    ( alpha * v + beta / 255 )
//   F16 : same as F32, stored as half
inline double brightness_scalar(double v, DType dt, double alpha, double beta) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v * alpha + beta), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint((v + 128.0) * alpha + beta) - 128.0, -128.0, 127.0);
        case DType::F16:
        case DType::F32:
            return clampd(v * alpha + beta / 255.0, 0.0, 1.0);
        default:
            return v;
    }
}

// Writes the brightness result into dst, reading the source at the ROI offset and writing
// packed at the destination origin (matching the region and placement the RPP op uses).
// dst outside the written region is left as the caller initialized it.
template <typename T>
void brightness_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                          RpptRoiType roiType, double alpha, double beta) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(
                            brightness_scalar(to_double(src[srcIdx]), dt, alpha, beta));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_BRIGHTNESS_REF_H
