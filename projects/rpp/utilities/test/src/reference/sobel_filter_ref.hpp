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
  owns the replicate border; sobel supplies the reduction because the
  magnitude needs the raw gx/gy.

Expression
  Kernels, row-major (dy = -1..1 outer, dx = -1..1 inner):
    Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    Gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

  gx  = sum Gx[k] * w[k]
  gy  = sum Gy[k] * w[k]
  dst = sobelType == 0 ? gx : sobelType == 1 ? gy : sqrt(gx^2 + gy^2)

Per-type form
  The gradient is computed in the dtype's intensity space: I8 is lifted by +128
  and shifted back before the store, as in emboss_ref. Gx and Gy sum to zero,
  so the lift only affects the store. The result is quantized via
  quantize_stored (U8 round + clamp[0,255], I8 round + clamp[-128,127],
  F16/F32 clamp[0,1]), clamping negative or out-of-range gradients.

Notes
  PLN1 and kernelSize 3 only. The output is always single-channel, so a
  3-channel input would need an unspecified RGB->greyscale conversion, and the
  5x5/7x7 kernels have no standard coefficients.
*/
template <typename T>
void sobel_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                            RpptRoiType type, Rpp32u sobelType, Rpp32u kernelSize) {
    static const double Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const double Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const double shift = (dt == DType::I8) ? 128.0 : 0.0;
    filter_reference<T>(src, d, dst, d, roi, type, kernelSize, [&](const double* w, int kk) {
        double gx = 0.0, gy = 0.0;
        for (int k = 0; k < kk; ++k) {
            gx += Gx[k] * (w[k] + shift);
            gy += Gy[k] * (w[k] + shift);
        }
        const double result = sobelType == 0   ? gx
                              : sobelType == 1 ? gy
                                               : std::sqrt(gx * gx + gy * gy);
        return quantize_stored(result - shift, dt);
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SOBEL_FILTER_REF_H
