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

#ifndef RPP_TEST_NON_LINEAR_BLEND_REF_H
#define RPP_TEST_NON_LINEAR_BLEND_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_non_linear_blend (two-source, scalar stdDev per image),
// derived from the op's definition (a spatially-varying alpha blend whose weight is a 2D gaussian
// centered on the region, of the given standard deviation), NOT from the kernel. Used for both
// backends.
//
// For output pixel (row j, col i) of a roiW x roiH region, with the gaussian centered at
// (roiW/2, roiH/2) (integer halves):
//     multiplier    = -0.5 / stdDev^2
//     gaussianValue = exp( ((j - roiH/2)^2 + (i - roiW/2)^2) * multiplier )   in (0, 1]
//     out           = (src1 - src2) * gaussianValue + src2
// The gaussian peaks at 1 in the center (out == src1) and decays to 0 at the edges (out == src2),
// so out is a convex combination of the two sources and never leaves [min,max] -- clamping is only
// a safety net. Integer types round to nearest (the +128 I8 offsets cancel in (src1 - src2), so the
// interpolation is identical in signed space).
//
// NOTE (semantics assumption): the public header describes only "standard deviation based
// non-linear alpha-blending" via a gaussian; the peak-1 gaussian (no 1/(2*pi*sigma^2) prefactor),
// the region center, and the round-to-nearest quantization above are the principled interpretation
// the golden holds to. A kernel using a different convention shows up as a diff -- a finding, not a
// reference bug.
inline double non_linear_blend_scalar(double s1, double s2, DType dt, double gaussian) {
    const double v = (s1 - s2) * gaussian + s2;
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(v), -128.0, 127.0);
        case DType::F16:
        case DType::F32:
            return clampd(v, 0.0, 1.0);
        default:
            return v;
    }
}

template <typename T>
void non_linear_blend_reference(const T* src1, const T* src2, T* dst, const RpptDesc& d, DType dt,
                                const RpptROI* roi, RpptRoiType roiType, double stdDev) {
    const double multiplier = -0.5 / (stdDev * stdDev);
    // Per-image gaussian center = (roiW/2, roiH/2), integer halves as in the reference definition.
    std::vector<double> halfW(d.n), halfH(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        halfW[n] = static_cast<double>(static_cast<int>(b.w) / 2.0);
        halfH[n] = static_cast<double>(static_cast<int>(b.h) / 2.0);
    }
    for_each_roi_io(
        d, roi, roiType,
        [&](Rpp32u n, Rpp32u, Rpp32u j, Rpp32u i, std::size_t srcIdx, std::size_t dstIdx) {
            const double iLoc = static_cast<double>(j) - halfH[n];
            const double jLoc = static_cast<double>(i) - halfW[n];
            const double gaussian = std::exp((iLoc * iLoc + jLoc * jLoc) * multiplier);
            dst[dstIdx] = from_double<T>(non_linear_blend_scalar(
                to_double(src1[srcIdx]), to_double(src2[srcIdx]), dt, gaussian));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_NON_LINEAR_BLEND_REF_H
