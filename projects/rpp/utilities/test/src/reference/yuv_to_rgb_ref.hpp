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

#ifndef RPP_TEST_YUV_TO_RGB_REF_H
#define RPP_TEST_YUV_TO_RGB_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model shared by rppt_yuv_to_rgb, rppt_yuv_to_rgb_linear_v and
// rppt_yuv_to_rgb_cubic_v. Modelled from the ops' documented definition (NV12 semi-planar 8-bit in,
// packed RGB24 out, byte pitches, RpptColorStandard / RpptColorRange) plus the published
// non-constant-luminance YCbCr -> RGB derivation, NOT from the RPP kernel. The three ops differ
// only in vertical chroma upsampling, so they share one parameterized reference and any kernel
// disagreement surfaces as a diff.

// Which vertical chroma upsampler the op under test documents. Horizontal upsampling is
// nearest-neighbour in all three (chroma column = luma column / 2).
enum class YuvChromaUpsample {
    Nearest,  // rppt_yuv_to_rgb:          chroma row = luma row / 2
    LinearV,  // rppt_yuv_to_rgb_linear_v: odd rows identity, even rows average two chroma rows
    CubicV    // rppt_yuv_to_rgb_cubic_v:  odd rows identity, even rows a symmetric 4-tap Mitchell
};

// ---- colour matrix ---------------------------------------------------------

struct YuvLumaWeights {
    double wr, wb;  // wg = 1 - wr - wb
};

// The luma coefficients each standard defines. The header documents unknown values as BT.709, so
// that is the default. RpptColorStandard_BT2020_CL is deliberately not special-cased: constant
// luminance is a genuinely different transfer and is out of scope for this golden (and not gridded).
inline YuvLumaWeights yuv_luma_weights(RpptColorStandard standard) {
    switch (standard) {
        case RpptColorStandard_FCC:         return {0.30, 0.11};
        case RpptColorStandard_BT470BG:
        case RpptColorStandard_BT601:       return {0.299, 0.114};
        case RpptColorStandard_SMPTE240M:   return {0.212, 0.087};
        case RpptColorStandard_BT2020_NCL:  return {0.2627, 0.0593};
        default:                            return {0.2126, 0.0722};  // BT.709
    }
}

// Y bias/scale and chroma scale for the requested range. The header documents values other than
// FULL as behaving like studio, so FULL is the tested-for case.
struct YuvRangeScale {
    double yBias, yScale, cScale;
};

inline YuvRangeScale yuv_range_scale(RpptColorRange range) {
    if (range == RpptColorRange_FULL) return {0.0, 1.0, 1.0};
    return {16.0, 255.0 / 219.0, 255.0 / 224.0};  // studio: luma 16-235, chroma 16-240
}

// Converts one (Y, U, V) triple of stored 8-bit codes into R, G, B stored codes:
//   Yn = (Y - yBias) * yScale,  Un = (U - 128) * cScale,  Vn = (V - 128) * cScale
//   R  = Yn + 2(1 - wr) Vn
//   B  = Yn + 2(1 - wb) Un
//   G  = Yn - (2(1 - wr) wr / wg) Vn - (2(1 - wb) wb / wg) Un
// then round-to-nearest and clamp to [0, 255].
inline void yuv_to_rgb_pixel(double y, double u, double v, RpptColorStandard standard,
                             RpptColorRange range, double rgb[3]) {
    const YuvLumaWeights w = yuv_luma_weights(standard);
    const YuvRangeScale s = yuv_range_scale(range);
    const double wg = 1.0 - w.wr - w.wb;

    const double yn = (y - s.yBias) * s.yScale;
    const double un = (u - 128.0) * s.cScale;
    const double vn = (v - 128.0) * s.cScale;

    const double rv = 2.0 * (1.0 - w.wr);
    const double bu = 2.0 * (1.0 - w.wb);

    rgb[0] = quantize_stored(yn + rv * vn, DType::U8);
    rgb[1] = quantize_stored(yn - (rv * w.wr / wg) * vn - (bu * w.wb / wg) * un, DType::U8);
    rgb[2] = quantize_stored(yn + bu * un, DType::U8);
}

// ---- vertical chroma upsampling -------------------------------------------

// Mitchell-Netravali reconstruction kernel, the standard piecewise cubic:
//   |x| < 1 : ((12 - 9B - 6C)|x|^3 + (-18 + 12B + 6C)|x|^2 + (6 - 2B)) / 6
//   |x| < 2 : ((-B - 6C)|x|^3 + (6B + 30C)|x|^2 + (-12B - 48C)|x| + (8B + 24C)) / 6
// The cubic op documents B = 0, C = 0.6. Written out rather than hard-coded as tap constants so
// the derivation is auditable.
inline double mitchell_netravali(double x, double b, double c) {
    x = std::fabs(x);
    if (x < 1.0)
        return ((12.0 - 9.0 * b - 6.0 * c) * x * x * x + (-18.0 + 12.0 * b + 6.0 * c) * x * x +
                (6.0 - 2.0 * b)) /
               6.0;
    if (x < 2.0)
        return ((-b - 6.0 * c) * x * x * x + (6.0 * b + 30.0 * c) * x * x +
                (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)) /
               6.0;
    return 0.0;
}

// The chroma rows contributing to one luma row, with their weights (already edge-clamped).
struct ChromaTaps {
    int row[4];
    double weight[4];
    int count;
};

// SEMANTICS ASSUMPTION (chroma phase). The headers say odd luma rows pass chroma through unchanged
// and even luma rows interpolate at frac = 0.5, but not *which* chroma rows are "nearest". Identity
// on odd rows pins the siting: chroma sample cr must be co-sited with luma row 2*cr + 1, so luma
// row y samples the chroma axis at the continuous position p = (y - 1) / 2. For odd y that is the
// integer cr = (y - 1) / 2 (identity, as documented); for even y it is p = y/2 - 0.5, i.e. exactly
// half way between chroma rows y/2 - 1 and y/2 (frac = 0.5, as documented). This is the only phase
// consistent with both statements -- the "forward" alternative (p = y/2, taps y/2 and y/2 + 1)
// would make *even* rows the identity ones. A kernel using that other convention will show up as a
// row-parity diff, which is a finding about the kernel, not a bug in this reference.
//
// Out-of-range chroma rows are clamped to [0, chromaHeight - 1] (edge replication); clamping before
// weighting means a duplicated row simply accumulates both weights.
inline ChromaTaps chroma_taps_v(YuvChromaUpsample mode, Rpp32u lumaRow, Rpp32u chromaHeight) {
    const int last = static_cast<int>(chromaHeight) - 1;
    const int cr = static_cast<int>(lumaRow / 2);
    auto clamp_row = [last](int r) { return r < 0 ? 0 : (r > last ? last : r); };

    ChromaTaps taps{};
    // Nearest samples cr in every row; for LinearV/CubicV an odd row is the documented identity
    // pass-through, whose co-sited chroma row (lumaRow - 1) / 2 is the same cr.
    if (mode == YuvChromaUpsample::Nearest || (lumaRow & 1u)) {
        taps.count = 1;
        taps.row[0] = clamp_row(cr);
        taps.weight[0] = 1.0;
        return taps;
    }

    if (mode == YuvChromaUpsample::LinearV) {
        taps.count = 2;
        taps.row[0] = clamp_row(cr - 1);
        taps.row[1] = clamp_row(cr);
        taps.weight[0] = 0.5;
        taps.weight[1] = 0.5;
        return taps;
    }

    // CubicV: four taps straddling p = cr - 0.5, at chroma rows cr-2 .. cr+1, so the tap distances
    // are 1.5, 0.5, 0.5, 1.5 -- the symmetric 4-tap filter the header describes.
    taps.count = 4;
    const double p = static_cast<double>(cr) - 0.5;
    double sum = 0.0;
    for (int t = 0; t < 4; ++t) {
        const int row = cr - 2 + t;
        taps.row[t] = clamp_row(row);
        taps.weight[t] = mitchell_netravali(static_cast<double>(row) - p, 0.0, 0.6);
        sum += taps.weight[t];
    }
    // Mitchell is a partition of unity so sum is already 1; normalising anyway keeps the tap
    // derivation self-checking rather than relying on that property.
    for (int t = 0; t < 4; ++t) taps.weight[t] /= sum;
    return taps;
}

// ---- reference -------------------------------------------------------------

// Writes packed RGB24 into dst for the whole width x height frame.
//   Y plane  : row r at srcY + r * srcYPitch, one byte per luma sample.
//   UV plane : half resolution both ways; chroma row cr at srcUV + cr * srcUVPitch, with chroma
//              sample cc occupying bytes [2*cc] = U and [2*cc + 1] = V.
//   dst      : pixel (x, y) at dst + y * dstPitch + 3*x, in R, G, B order.
// Rows are addressed only through the byte pitches (never a tight row width), so pitch slack is
// neither read nor written. width and height must be even (NV12).
inline void yuv_to_rgb_reference(const Rpp8u* srcY, const Rpp8u* srcUV, Rpp8u* dst,
                                 Rpp32u srcYPitch, Rpp32u srcUVPitch, Rpp32u dstPitch,
                                 Rpp32u width, Rpp32u height, RpptColorStandard standard,
                                 RpptColorRange range, YuvChromaUpsample upsample) {
    const Rpp32u chromaHeight = height / 2;

    for (Rpp32u y = 0; y < height; ++y) {
        const Rpp8u* lumaRow = srcY + static_cast<std::size_t>(y) * srcYPitch;
        Rpp8u* dstRow = dst + static_cast<std::size_t>(y) * dstPitch;
        const ChromaTaps taps = chroma_taps_v(upsample, y, chromaHeight);

        for (Rpp32u x = 0; x < width; ++x) {
            const Rpp32u cc = x / 2;  // horizontal upsampling is nearest in all three ops
            double u = 0.0, v = 0.0;
            for (int t = 0; t < taps.count; ++t) {
                const Rpp8u* chromaRow =
                    srcUV + static_cast<std::size_t>(taps.row[t]) * srcUVPitch;
                u += taps.weight[t] * static_cast<double>(chromaRow[2 * cc]);
                v += taps.weight[t] * static_cast<double>(chromaRow[2 * cc + 1]);
            }

            // Chroma is interpolated in continuous code space and fed straight to the matrix; only
            // the final RGB is quantized, so no intermediate rounding is invented here.
            double rgb[3];
            yuv_to_rgb_pixel(static_cast<double>(lumaRow[x]), u, v, standard, range, rgb);
            for (int c = 0; c < 3; ++c) dstRow[3 * x + c] = static_cast<Rpp8u>(rgb[c]);
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_YUV_TO_RGB_REF_H
