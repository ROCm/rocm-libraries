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

#ifndef RPP_TEST_HISTOGRAM_EQUALIZE_REF_H
#define RPP_TEST_HISTOGRAM_EQUALIZE_REF_H

#include <rpp/rpp.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/intensity.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_ycbcr.hpp"

namespace rpptest {

/*
Reference model: histogram_equalize

RPP op
  rppt_histogram_equalize   (Image / Color augmentation)

Description
  Per-image histogram equalization over the ROI, spreading the intensity
  distribution to use the full range. Takes no parameters and is U8 only.

  This is a two-pass, per-image (non-pointwise) op: the output at each pixel
  depends on the whole ROI's intensity histogram, so it cannot use the
  pointwise scalar template.

  The BT.601 full-range (JPEG) transform itself lives in
  reference/color_ycbcr.hpp.

Expression
  1. Build the 256-bin histogram of the equalization channel over the ROI,
     with N = roiW * roiH pixels:
       1 channel   the pixel value itself
       3 channel   the luminance Y, in BT.601 full-range (JPEG) YCbCr

  2. LUT[v] = round( (cdf[v] - cdfMin) / (N - cdfMin) * 255 )
     the standard cdf-min-normalized mapping, cdfMin = first non-zero CDF
     value.

  3. Apply:
       1 channel   dst = LUT[src]
       3 channel   equalize only Y (Y' = LUT[Y]), keep the original Cb/Cr,
                   then convert YCbCr -> RGB

Per-type form
  U8 only. Y/Cb/Cr are quantized to U8 (round + clamp) as in a standard
  integer YCbCr pipeline, and the RGB round-trip clamps to [0,255].
*/

// Builds the cdf-min-normalized equalization LUT from a 256-bin histogram of N samples.
inline std::array<int, 256> he_build_lut(const std::array<long, 256>& hist, long N) {
    std::array<int, 256> lut{};
    long cdf = 0, cdfMin = 0;
    for (int v = 0; v < 256; ++v)
        if (hist[v] != 0) {
            cdfMin = hist[v];
            break;
        }
    const double denom = static_cast<double>(N - cdfMin);
    for (int v = 0; v < 256; ++v) {
        cdf += hist[v];
        const double mapped =
            denom > 0.0 ? std::nearbyint((cdf - cdfMin) / denom * 255.0) : static_cast<double>(v);
        lut[v] = static_cast<int>(clampd(mapped, 0.0, 255.0));
    }
    return lut;
}

template <typename T>
void histogram_equalize_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                  const RpptROI* roi, RpptRoiType type) {
    const bool rgb = sd.c == 3;
    const std::vector<std::size_t> N = roi_pixel_counts(sd, roi, type);

    // Pass 1: per-image histogram of the equalization channel (Y for RGB). Source only, so this
    // pass is addressed through sd alone.
    std::vector<std::array<long, 256>> hist(sd.n);
    for (auto& h : hist) h.fill(0);
    for_each_roi_pixel(sd, roi, type, [&](Rpp32u n, Rpp32u, Rpp32u, std::size_t s, std::size_t) {
        int y;
        if (rgb) {
            const double r = to_double(src[s]), g = to_double(src[channel_index(sd, s, 1)]),
                         b = to_double(src[channel_index(sd, s, 2)]);
            y = static_cast<int>(clampd(std::nearbyint(ycbcr_y(r, g, b)), 0.0, 255.0));
        } else {
            y = static_cast<int>(to_double(src[s]));
        }
        hist[n][y]++;
    });

    std::vector<std::array<int, 256>> lut(sd.n);
    for (Rpp32u n = 0; n < sd.n; ++n) lut[n] = he_build_lut(hist[n], static_cast<long>(N[n]));

    // Pass 2: apply the per-image LUT (equalizing only Y for RGB, preserving Cb/Cr).
    for_each_roi_pixel(
        sd, dd, roi, type, [&](Rpp32u n, Rpp32u, Rpp32u, std::size_t s, std::size_t o) {
            if (rgb) {
                const double r = to_double(src[s]), g = to_double(src[channel_index(sd, s, 1)]),
                             b = to_double(src[channel_index(sd, s, 2)]);
                const int y =
                    static_cast<int>(clampd(std::nearbyint(ycbcr_y(r, g, b)), 0.0, 255.0));
                const double cb = clampd(std::nearbyint(ycbcr_cb(r, g, b)), 0.0, 255.0);
                const double cr = clampd(std::nearbyint(ycbcr_cr(r, g, b)), 0.0, 255.0);
                const double yp = static_cast<double>(lut[n][y]);
                double rr, gg, bb;
                ycbcr_to_rgb(yp, cb, cr, rr, gg, bb);
                dst[o] = from_double<T>(clampd(std::nearbyint(rr), 0.0, 255.0));
                dst[channel_index(dd, o, 1)] =
                    from_double<T>(clampd(std::nearbyint(gg), 0.0, 255.0));
                dst[channel_index(dd, o, 2)] =
                    from_double<T>(clampd(std::nearbyint(bb), 0.0, 255.0));
            } else {
                dst[o] = from_double<T>(
                    static_cast<double>(lut[n][static_cast<int>(to_double(src[s]))]));
            }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_HISTOGRAM_EQUALIZE_REF_H
