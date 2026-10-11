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
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: non_linear_blend

RPP op
  rppt_non_linear_blend   (Image / Effects augmentation)

Description
  Spatially-varying cross-fade between two sources, weighted by a 2D gaussian
  centred on the region with the given standard deviation. The gaussian peaks
  at 1 in the centre (out == src1) and decays to 0 at the edges (out == src2),
  so the output is a convex combination of the two sources and never leaves
  [min,max] -- clamping is only a safety net.

Expression
  For output pixel (row j, col i) of a roiW x roiH region, gaussian centred at
  (roiW/2, roiH/2) with integer halves:

  multiplier = -0.5 / stdDev^2
  gaussian   = exp( ((j - roiH/2)^2 + (i - roiW/2)^2) * multiplier )   in (0,1]
  dst        = (src1 - src2) * gaussian + src2

Per-type form
  Integer types round to nearest. The +128 I8 offsets cancel in
  (src1 - src2), so the interpolation is identical in signed space.

Notes
  The public header describes only "standard deviation based non-linear
  alpha-blending" via a gaussian. The peak-1 gaussian (no 1/(2*pi*sigma^2)
  prefactor), the region centre and the round-to-nearest quantization are the
  principled interpretation the golden holds to. A kernel using a different
  convention shows up as a diff -- a finding, not a reference bug.
*/
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
void non_linear_blend_reference(const T* src1, const T* src2, const RpptDesc& sd, T* dst,
                                const RpptDesc& dd, DType dt, const RpptROI* roi,
                                RpptRoiType roiType, double stdDev) {
    const double multiplier = -0.5 / (stdDev * stdDev);
    // Per-image gaussian center = (roiW/2, roiH/2), integer halves as in the reference definition:
    // an odd extent centres on the lower pixel rather than between two.
    std::vector<double> halfW(sd.n), halfH(sd.n);
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        const Rpp32u hw = b.w / 2, hh = b.h / 2;
        halfW[n] = static_cast<double>(hw);
        halfH[n] = static_cast<double>(hh);
    }
    for_each_roi_io(
        sd, dd, roi, roiType,
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
