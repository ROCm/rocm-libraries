#ifndef RPP_TEST_HISTOGRAM_EQUALIZE_REF_H
#define RPP_TEST_HISTOGRAM_EQUALIZE_REF_H

#include <rpp/rpp.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/reduction.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_histogram_equalize (single source, no params, U8 only),
// derived from the op's definition (per-image histogram equalization over the ROI), NOT from the
// kernel. The same reference serves both HOST and HIP backends.
//
// This is a two-pass, per-image (non-pointwise) op: the output at each pixel depends on the whole
// ROI's intensity histogram, so it cannot use the pointwise scalar template.
//
//   1. Build the 256-bin histogram of the equalization channel over the ROI (N = roiW*roiH pixels):
//        - 1 channel : the pixel value itself.
//        - 3 channel : the luminance Y, in BT.601 full-range (JPEG) YCbCr.
//   2. LUT[v] = round( (cdf[v] - cdfMin) / (N - cdfMin) * 255 ), the standard cdf-min-normalized
//      equalization mapping (cdfMin = first non-zero CDF value).
//   3. Apply: 1-channel out = LUT[in]; 3-channel equalizes only Y (Y' = LUT[Y]), keeps the original
//      Cb/Cr, and converts YCbCr->RGB. Y/Cb/Cr are quantized to U8 (round+clamp) as in a standard
//      integer YCbCr pipeline; the RGB round-trip clamps to [0,255].
//
// BT.601 full-range (JPEG):
//   Y  =  0.299 R + 0.587 G + 0.114 B
//   Cb = -0.168736 R - 0.331264 G + 0.5 B + 128
//   Cr =  0.5 R - 0.418688 G - 0.081312 B + 128
//   R  = Y + 1.402 (Cr-128)
//   G  = Y - 0.344136 (Cb-128) - 0.714136 (Cr-128)
//   B  = Y + 1.772 (Cb-128)

inline double he_luma_y(double r, double g, double b) { return 0.299 * r + 0.587 * g + 0.114 * b; }
inline double he_cb(double r, double g, double b) {
    return -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
}
inline double he_cr(double r, double g, double b) {
    return 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
}

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
void histogram_equalize_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                                  RpptRoiType type) {
    const bool rgb = d.c == 3;
    const std::vector<std::size_t> N = roi_pixel_counts(d, roi, type);

    // Pass 1: per-image histogram of the equalization channel (Y for RGB).
    std::vector<std::array<long, 256>> hist(d.n);
    for (auto& h : hist) h.fill(0);
    for_each_roi_pixel(d, roi, type, [&](Rpp32u n, Rpp32u, Rpp32u, std::size_t s, std::size_t) {
        int y;
        if (rgb) {
            const double r = to_double(src[s]), g = to_double(src[channel_index(d, s, 1)]),
                         b = to_double(src[channel_index(d, s, 2)]);
            y = static_cast<int>(clampd(std::nearbyint(he_luma_y(r, g, b)), 0.0, 255.0));
        } else {
            y = static_cast<int>(to_double(src[s]));
        }
        hist[n][y]++;
    });

    std::vector<std::array<int, 256>> lut(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) lut[n] = he_build_lut(hist[n], static_cast<long>(N[n]));

    // Pass 2: apply the per-image LUT (equalizing only Y for RGB, preserving Cb/Cr).
    for_each_roi_pixel(d, roi, type, [&](Rpp32u n, Rpp32u, Rpp32u, std::size_t s, std::size_t o) {
        if (rgb) {
            const double r = to_double(src[s]), g = to_double(src[channel_index(d, s, 1)]),
                         b = to_double(src[channel_index(d, s, 2)]);
            const int y = static_cast<int>(clampd(std::nearbyint(he_luma_y(r, g, b)), 0.0, 255.0));
            const double cb = clampd(std::nearbyint(he_cb(r, g, b)), 0.0, 255.0);
            const double cr = clampd(std::nearbyint(he_cr(r, g, b)), 0.0, 255.0);
            const double yp = static_cast<double>(lut[n][y]);
            const double rr = yp + 1.402 * (cr - 128.0);
            const double gg = yp - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0);
            const double bb = yp + 1.772 * (cb - 128.0);
            dst[o] = from_double<T>(clampd(std::nearbyint(rr), 0.0, 255.0));
            dst[channel_index(d, o, 1)] = from_double<T>(clampd(std::nearbyint(gg), 0.0, 255.0));
            dst[channel_index(d, o, 2)] = from_double<T>(clampd(std::nearbyint(bb), 0.0, 255.0));
        } else {
            dst[o] = from_double<T>(static_cast<double>(lut[n][static_cast<int>(to_double(src[s]))]));
        }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_HISTOGRAM_EQUALIZE_REF_H
