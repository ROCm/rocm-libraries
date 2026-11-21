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
#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_to_greyscale_ref.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: sobel_filter

RPP op
  rppt_sobel_filter   (Image / Filter augmentation)

Description
  Sobel edge operator of size kernelSize = 3, 5 or 7. The two gradient kernels
  give the horizontal and vertical derivative, and sobelType selects which is
  written: 0 -> gx, 1 -> gy, 2 -> the gradient magnitude.

  The window comes from filter_reference (reference/filter_common.hpp), which
  owns the replicate border; sobel supplies the reduction because the
  magnitude needs the raw gx/gy.

  A 3-channel source (PKD3/PLN3) is first collapsed to one grey plane by
  color_to_greyscale_reference over the whole image (rppt_sobel_filter runs
  rppt_color_to_greyscale with RGB channel order and no ROI), and the Sobel
  operator is applied to that plane. The output is always single-channel.

Expression
  Kernels, row-major (dy = -r..r outer, dx = -r..r inner), the separable Sobel
  form: Gx = s (x) d and Gy = Gx^T, with s the binomial smoothing row of
  length k (1 2 1 / 1 4 6 4 1 / 1 6 15 20 15 6 1) and d the central difference
  (-1 0 1) convolved with the binomial row of length k - 2. For k = 3:
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
  F16/F32 clamp[0,1]); the clamp applies to the stored value only, never to gx
  or gy before the magnitude is formed.
*/

// Binomial row of length n: 1, 1 1, 1 2 1, 1 3 3 1, ...
inline std::vector<double> sobel_binomial_row(Rpp32u n) {
    std::vector<double> row(1, 1.0);
    for (Rpp32u len = 1; len < n; ++len) {
        std::vector<double> next(len + 1, 0.0);
        for (Rpp32u i = 0; i < len; ++i) {
            next[i] += row[i];
            next[i + 1] += row[i];
        }
        row = next;
    }
    return row;
}

// Gx (transpose = false) or Gy (transpose = true) for kernelSize k, row-major.
inline std::vector<double> sobel_kernel(Rpp32u k, bool transpose) {
    const std::vector<double> smooth = sobel_binomial_row(k);
    const std::vector<double> base = sobel_binomial_row(k - 2);
    std::vector<double> deriv(k, 0.0);  // (-1 0 1) convolved with base
    for (Rpp32u i = 0; i < base.size(); ++i) {
        deriv[i] -= base[i];
        deriv[i + 2] += base[i];
    }
    std::vector<double> g(k * k);
    for (Rpp32u y = 0; y < k; ++y)
        for (Rpp32u x = 0; x < k; ++x)
            g[y * k + x] = transpose ? smooth[x] * deriv[y] : smooth[y] * deriv[x];
    return g;
}

// src is addressed through srcDesc (c = 1, or c = 3 for RGB input) and dst through the
// single-channel dstDesc.
template <typename T>
void sobel_filter_reference(const T* src, const RpptDesc& srcDesc, T* dst, const RpptDesc& dstDesc,
                            DType dt, const RpptROI* roi, RpptRoiType type, Rpp32u sobelType,
                            Rpp32u kernelSize) {
    const std::vector<double> Gx = sobel_kernel(kernelSize, false);
    const std::vector<double> Gy = sobel_kernel(kernelSize, true);
    const double shift = (dt == DType::I8) ? 128.0 : 0.0;

    const T* plane = src;
    const RpptDesc* planeDesc = &srcDesc;
    std::vector<T> grey;
    if (srcDesc.c == 3) {
        grey.resize(element_count(dstDesc));
        const std::vector<RpptROI> whole = make_roi(dstDesc, Roi::Full);
        color_to_greyscale_reference<T>(src, srcDesc, grey.data(), dstDesc, dt, whole.data(), XYWH,
                                        RGBtype);
        plane = grey.data();
        planeDesc = &dstDesc;
    }

    filter_reference<T>(plane, *planeDesc, dst, dstDesc, roi, type, kernelSize,
                        [&](const double* w, int kk) {
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
