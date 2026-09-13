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

#ifndef RPP_TEST_SOBEL_FILTER_REF_H
#define RPP_TEST_SOBEL_FILTER_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: sobel_filter

RPP op
  rppt_sobel_filter   (Image / Filter augmentation)

Description
  Canonical 3x3 Sobel edge operator. The two gradient kernels give the
  horizontal and vertical derivative, and sobelType selects which is written:
  0 -> gx, 1 -> gy, 2 -> the gradient magnitude.

  The window comes from filter_reference (reference/filter_common.hpp), which
  owns the REPLICATE border and the placement; sobel supplies only the
  reduction, since for the magnitude case it needs the raw gx/gy before
  quantization rather than a single convolution.

Expression
  Kernels, row-major (dy = -1..1 outer, dx = -1..1 inner):
    Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    Gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

  gx  = sum Gx[k] * w[k]
  gy  = sum Gy[k] * w[k]
  dst = sobelType == 0 ? gx : sobelType == 1 ? gy : sqrt(gx^2 + gy^2)

Per-type form
  The result is quantized via quantize_stored: U8 round + clamp[0,255], I8
  round + clamp[-128,127], F16/F32 clamp[0,1]. Gradients can be negative or
  out of range, so clamping is the intended "same depth as src" behaviour; any
  resulting diff is a finding, not a reference bug.

Notes
  Scope is PLN1 only, kernelSize = 3 only. sobel_filter's dstDesc is always
  single-channel greyscale (c=1, NCHW), so a 3-channel input would require an
  undocumented RGB->greyscale conversion that is not independently derivable,
  and the extended k=5/7 kernels have varying coefficient conventions. Both
  are deferred; this model covers greyscale-in/greyscale-out with the
  universally-defined 3x3 operator.
*/
template <typename T>
void sobel_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                            RpptRoiType type, Rpp32u sobelType, Rpp32u kernelSize) {
    static const double Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const double Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    filter_reference<T>(src, d, dst, d, roi, type, kernelSize, [&](const double* w, int kk) {
        double gx = 0.0, gy = 0.0;
        for (int k = 0; k < kk; ++k) {
            gx += Gx[k] * w[k];
            gy += Gy[k] * w[k];
        }
        const double result = sobelType == 0   ? gx
                              : sobelType == 1 ? gy
                                               : std::sqrt(gx * gx + gy * gy);
        return quantize_stored(result, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SOBEL_FILTER_REF_H
