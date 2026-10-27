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

#ifndef RPP_TEST_VIGNETTE_REF_H
#define RPP_TEST_VIGNETTE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: vignette

RPP op
  rppt_vignette   (Image / Effects augmentation)

Description
  Darkens the frame towards its edges with an isotropic Gaussian falloff about
  the centre, scaled by a single positive intensity. A larger intensity
  narrows the Gaussian and darkens the frame faster; the centre pixel is
  always left unchanged (g = 1).

Expression
  For pixel (i, j) of the processed region, with the half-extents truncated to
  whole pixels (hw = floor(w/2), hh = floor(h/2)) and radius = max(hw, hh):

  r2  = (i - hw)^2 + (j - hh)^2
  g   = exp(-0.25 * intensity * r2 / radius^2)   sigma = radius*sqrt(2/intensity)
  dst = src * g

Per-type form
  The multiply happens in normalized [0,1] intensity space via to_unit() /
  from_unit(), so U8 darkens towards 0 and I8 towards -128 rather than towards
  the raw stored zero, and integers round to nearest.

Notes
  The header documents `intensity` only as "> 0", so the sigma scale
  (2*sigma^2 = max(w,h)^2 / intensity) follows the op on both backends.

  The falloff is centred on and sized by the ROI, not the full image.

  The header notes HOST uses fastexpavx() and that up to 5% pixel mismatch is
  expected; the model uses exact exp() and the test tolerance covers it.
*/

// Gaussian falloff at (x, y) within a w x h region.
inline double vignette_weight(double x, double y, double w, double h, double intensity) {
    const double halfW = std::floor(w * 0.5), halfH = std::floor(h * 0.5);
    const double radius = halfW > halfH ? halfW : halfH;
    const double dx = x - halfW, dy = y - halfH;
    return std::exp(-0.25 * intensity * (dx * dx + dy * dy) / (radius * radius));
}

inline double vignette_scalar(double v, double weight, DType dt) {
    return from_unit(to_unit(v, dt) * weight, dt);
}

template <typename T>
void vignette_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                        const RpptROI* roi, RpptRoiType roiType, const Rpp32f* intensityTensor) {
    std::vector<RoiBounds> bounds(sd.n);
    for (Rpp32u n = 0; n < sd.n; ++n) bounds[n] = roi_bounds(roi[n], roiType);

    for_each_roi_io(
        sd, dd, roi, roiType,
        [&](Rpp32u n, Rpp32u, Rpp32u j, Rpp32u i, std::size_t srcIdx, std::size_t dstIdx) {
            const RoiBounds& b = bounds[n];
            const double g = vignette_weight(i, j, b.w, b.h, intensityTensor[n]);
            dst[dstIdx] = from_double<T>(vignette_scalar(to_double(src[srcIdx]), g, dt));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_VIGNETTE_REF_H
