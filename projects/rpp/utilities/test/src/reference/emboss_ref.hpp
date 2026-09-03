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

#ifndef RPP_TEST_EMBOSS_REF_H
#define RPP_TEST_EMBOSS_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: emboss   (kernel-derived REGRESSION golden)

RPP op
  rppt_emboss   (Image / Filter augmentation)

Description
  Directional relief filter: an anti-symmetric KxK tap table that lights one
  edge orientation and darkens the opposite, giving the raised-surface look.
  strength scales every tap.

  Both tables sum to 1, so the filter is DC-preserving at strength 1 and needs
  no mid-grey bias (the classic +128 emboss offset is absent here). Since
  strength scales every tap, the DC gain is the strength itself, and the
  kernel clamps strength at 2.0 from above (also undocumented, also
  transcribed).

  The taps are applied as a CORRELATION (no kernel flip), matching every other
  filter in the suite. This matters here and nowhere else: box/gaussian/median
  are symmetric, so their tests cannot tell correlation from convolution,
  whereas the emboss table is deliberately anti-symmetric.

Expression
  dst(j, i) = sum{ strength * tap[k] * src(j+dy, i+dx) }

Per-type form
  I8 stores intensities shifted by -128, so the window is lifted into [0,255]
  intensity space before the dot product and dropped back afterwards. For the
  suite's other linear filters this shift cancels (their weights sum to 1, so
  it contributes exactly +128 and the store removes it), which is why
  convolve_reference() ignores it. Here the weights sum to `strength`, so the
  shift contributes 128*strength and does NOT cancel: at strength 2 an I8
  result computed in stored space would be off by 128.

Notes
  This is a REGRESSION golden, not an independently-derived one. emboss's
  per-element semantics are documented nowhere: the public API header
  describes only the parameters and shows sample images, and there is no spec.
  It is also not recoverable from first principles -- several "standard" 3x3
  emboss kernels exist, the strength and bias conventions are unstated, and
  there is no canonical NxN emboss kernel at all for the larger sizes. The
  weight tables below are therefore transcribed from the RPP kernel with the
  user's explicit authorization, so this model LOCKS current behaviour rather
  than encoding intent: it will not catch a wrong-by-design kernel, only a
  change in it.

  Everything OUTSIDE the weight table is still the suite's own machinery: the
  KxK window, the clamp-to-ROI REPLICATE border (the only border type the op
  supports), the placement and the per-type quantization all come from
  filter_reference(). A border or placement defect therefore still surfaces as
  a diff; only the coefficients are locked.
*/

// The base emboss taps, row-major, top-left to bottom-right, in the same dy/dx order
// gather_roi_window() produces. Only the sizes the test grids are provided.
inline std::vector<double> emboss_kernel(Rpp32u kernelSize, double strength) {
    static const double k3[9] = {2, 1, 0, 1, 1, -1, 0, -1, -2};
    static const double k5[25] = {3,  3,  2, 1, 0,  3,  2,  1, 0,  -1, 2,  1, 1,
                                  -1, -2, 1, 0, -1, -2, -3, 0, -1, -2, -3, -3};
    const double* base = (kernelSize == 3) ? k3 : k5;
    const std::size_t count = static_cast<std::size_t>(kernelSize) * kernelSize;
    const double scale = (strength > 2.0) ? 2.0 : strength;  // clamped from above only
    std::vector<double> kernel(count);
    for (std::size_t i = 0; i < count; ++i) kernel[i] = base[i] * scale;
    return kernel;
}

template <typename T>
void emboss_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                      const RpptROI* roi, RpptRoiType type, double strength, Rpp32u kernelSize) {
    const std::vector<double> kernel = emboss_kernel(kernelSize, strength);
    const double shift = (dt == DType::I8) ? 128.0 : 0.0;
    filter_reference<T>(src, sd, dst, dd, roi, type, kernelSize, [&](const double* w, int kk) {
        double acc = 0.0;
        for (int k = 0; k < kk; ++k) acc += kernel[k] * (w[k] + shift);
        return quantize_stored(acc - shift, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_EMBOSS_REF_H
