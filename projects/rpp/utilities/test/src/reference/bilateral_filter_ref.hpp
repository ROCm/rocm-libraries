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

#ifndef RPP_TEST_BILATERAL_FILTER_REF_H
#define RPP_TEST_BILATERAL_FILTER_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: bilateral_filter

RPP op
  rppt_bilateral_filter   (Image / Filter augmentation)

Description
  Edge-preserving smoothing: each output pixel is a weighted mean of its KxK
  neighbourhood, where a neighbour's weight is the product of a spatial
  Gaussian on its offset and a range Gaussian on how far its value is from the
  centre pixel's. Neighbours across an edge differ strongly in value, get a
  near-zero range weight, and so do not blur into the centre.

  Follows the rest of RPP's KxK filters (see filter_common.hpp): a square
  window, every channel filtered independently (the range distance is the
  channel's own |v - v0|, not a joint colour distance), REPLICATE border
  clamped to the ROI, source read at the ROI offset and output written at the
  destination origin.

  stdDevSpaceTensor is in pixels. stdDevRangeTensor is in the dtype's stored units -- the
  same units as the pixels it is compared against (U8 [0,255], I8 [-128,127],
  F16/F32 [0,1]), so the range term is |v - v0| with no rescaling. The I8 shift
  cancels in a difference, so I8 and U8 take the same stdDevRangeTensor value.

  The centre tap always has weight exp(0) * exp(0) = 1, so the denominator is
  >= 1 and the mean is always defined. Both standard deviations are per image.

Expression
  ws(dy, dx) = exp( -(dy^2 + dx^2) / (2 sigmaS^2) )                for dy, dx in [-r, r]
  wr(v)      = exp( -(v - v0)^2     / (2 sigmaR^2) )               v0 = src(j, i)
  dst(j, i)  = sum{ ws * wr(v) * v } / sum{ ws * wr(v) }           v  = src(j+dy, i+dx)

Limiting cases (each is a test parameter)
  sigmaR -> inf : wr == 1 everywhere, reduces to gaussian_filter with sigmaS.
  sigmaR -> 0   : only neighbours equal to v0 keep weight, so dst == src.
*/
template <typename T>
void bilateral_filter_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                DType dt, const RpptROI* roi, RpptRoiType type, Rpp32u kernelSize,
                                const Rpp32f* stdDevSpaceTensor, const Rpp32f* stdDevRangeTensor) {
    const int r = static_cast<int>(kernelSize / 2);
    const int kk = static_cast<int>(kernelSize * kernelSize);
    std::vector<double> window(kk), spatial(kk);
    for_each_roi_plane(
        sd, dd, roi, type,
        [&](Rpp32u n, const RoiBounds& b, Rpp32u, std::size_t srcBase, std::size_t dstBase) {
            // Spatial weights in gather_roi_window's row-major order (dy outer, dx inner).
            const double sigmaS = stdDevSpaceTensor[n];
            const double sigmaR = stdDevRangeTensor[n];
            int k = 0;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    spatial[k++] =
                        std::exp(-static_cast<double>(dy * dy + dx * dx) / (2.0 * sigmaS * sigmaS));
            const double twoSigmaRSq = 2.0 * sigmaR * sigmaR;

            for (int j = 0; j < static_cast<int>(b.h); ++j)
                for (int i = 0; i < static_cast<int>(b.w); ++i) {
                    gather_roi_window(src, sd, b, srcBase, j, i, r, window.data());
                    const double centre = window[kk / 2];
                    double num = 0.0, den = 0.0;
                    for (int t = 0; t < kk; ++t) {
                        const double d = window[t] - centre;
                        const double w = spatial[t] * std::exp(-(d * d) / twoSigmaRSq);
                        num += w * window[t];
                        den += w;
                    }
                    dst[plane_index(dd, dstBase, j, i)] =
                        from_double<T>(quantize_stored(num / den, dt));
                }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_BILATERAL_FILTER_REF_H
