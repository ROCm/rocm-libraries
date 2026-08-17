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

#ifndef RPP_TEST_THRESHOLD_REF_H
#define RPP_TEST_THRESHOLD_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_threshold, derived from the op's definition (a
// black/white binary mask: a pixel is white iff every channel value falls within its
// per-channel [min,max] cutoff, else black), NOT from the RPP kernel. Used as the reference
// for both backends so kernel bugs surface as diffs.
//
// minTensor/maxTensor are per-image, per-channel cutoffs expressed in the same units as the
// stored pixels (U8 [0,255], I8 [-128,127], F16/F32 [0,1]); they are the exact values handed
// to the op. The output is a binary mask in the dtype's black/white extremes:
//   U8      : white 255, black 0
//   I8      : white 127, black -128   (same intensities as U8, shifted by -128)
//   F16/F32 : white 1.0, black 0.0
inline double threshold_white(DType dt) {
    switch (dt) {
        case DType::U8: return 255.0;
        case DType::I8: return 127.0;
        case DType::F16:
        case DType::F32: return 1.0;
        default: return 0.0;
    }
}

inline double threshold_black(DType dt) { return dt == DType::I8 ? -128.0 : 0.0; }

// Writes the threshold mask into dst, reading each source pixel's channels at the ROI offset
// and writing packed at the destination origin (matching the region and placement the RPP op
// uses). dst outside the written region is left as the caller initialized it.
template <typename T>
void threshold_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                         RpptRoiType roiType, const Rpp32f* minTensor, const Rpp32f* maxTensor) {
    const double white = threshold_white(dt);
    const double black = threshold_black(dt);
    for_each_roi_pixel(d, roi, roiType,
                       [&](Rpp32u n, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
        bool inRange = true;
        for (Rpp32u c = 0; c < d.c; ++c) {
            const double v = to_double(src[channel_index(d, srcPix, c)]);
            const double lo = minTensor[n * d.c + c];
            const double hi = maxTensor[n * d.c + c];
            if (v < lo || v > hi) {
                inRange = false;
                break;
            }
        }
        const double out = inRange ? white : black;
        for (Rpp32u c = 0; c < d.c; ++c)
            dst[channel_index(d, dstPix, c)] = from_double<T>(out);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_THRESHOLD_REF_H
