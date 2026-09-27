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

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"

namespace rpptest {

/*
Reference model: yuv_to_rgb

RPP op
  rppt_yuv_to_rgb / rppt_yuv_to_rgb_linear_v / rppt_yuv_to_rgb_cubic_v
  (Image / Data exchange)

Description
  Converts NV12 semi-planar 8-bit YUV to packed RGB24, with byte pitches and a
  caller-selected RpptColorStandard / RpptColorRange. The three ops differ only
  in vertical chroma upsampling, so they share one parameterized reference.

  Horizontal upsampling is nearest-neighbour in all three (chroma column =
  luma column / 2). Vertically:

    Nearest    rppt_yuv_to_rgb:          chroma row = luma row / 2
    LinearV    rppt_yuv_to_rgb_linear_v: odd rows identity, even rows average
                                         two chroma rows
    CubicV     rppt_yuv_to_rgb_cubic_v:  NV12 siting, two alternating 4-tap
                                         phases -- see the CubicV note below

Expression
  The published non-constant-luminance derivation, with wg = 1 - wr - wb:

  Yn = (Y - yBias) * yScale
  Un = (U - 128) * cScale
  Vn = (V - 128) * cScale

  R  = Yn + 2(1 - wr) Vn
  B  = Yn + 2(1 - wb) Un
  G  = Yn - (2(1 - wr) wr / wg) Vn - (2(1 - wb) wb / wg) Un

  then round-to-nearest and clamp to [0, 255]. This covers Nearest and
  LinearV; CubicV uses the fixed-point form below instead.

CubicV   (kernel-derived REGRESSION golden)
  The doxygen for rppt_yuv_to_rgb_cubic_v -- "odd luma rows pass through
  chroma unchanged (identity); even luma rows use a symmetric 4-tap filter" --
  does not describe the op. The kernel implements FFmpeg swscale's bicubic
  path: standard NV12 siting at p = (y - 0.5)/2 on BOTH parities via two
  alternating asymmetric phases, integer taps applied in a 15-bit domain, and
  FFmpeg's integer table colour math rather than the float matrix. See
  docs/doc-defects/yuv-to-rgb-cubic-v.md.

  The spec the kernel cites (FFmpeg9_YUV_to_RGB_spec.md) is not in the tree,
  so this branch is transcribed from the kernel with the user's explicit
  authorization: it LOCKS current behaviour rather than encoding intent.
  Everything outside the tap tables and the coefficient tables -- the row
  clamping, the pitch addressing, the frame walk -- is still the suite's own.

Notes
  The header documents unknown colour standards as behaving like BT.709 and
  unknown ranges as behaving like studio, so those are the defaults for the
  float path. The CubicV coefficient table follows the kernel, which falls
  back to BT.601 instead; that divergence is in the doc-defect note above.
  RpptColorStandard_BT2020_CL is deliberately not special-cased: constant
  luminance is a genuinely different transfer, out of scope for this golden
  and not gridded.
*/

// Which vertical chroma upsampler the op under test documents.
enum class YuvChromaUpsample { Nearest, LinearV, CubicV };

// ---- colour matrix ---------------------------------------------------------

struct YuvLumaWeights {
    double wr, wb;  // wg = 1 - wr - wb
};

// The luma coefficients each standard defines.
inline YuvLumaWeights yuv_luma_weights(RpptColorStandard standard) {
    switch (standard) {
        case RpptColorStandard_FCC:
            return {0.30, 0.11};
        case RpptColorStandard_BT470BG:
        case RpptColorStandard_BT601:
            return {0.299, 0.114};
        case RpptColorStandard_SMPTE240M:
            return {0.212, 0.087};
        case RpptColorStandard_BT2020_NCL:
            return {0.2627, 0.0593};
        default:
            return {0.2126, 0.0722};  // BT.709
    }
}

// Y bias/scale and chroma scale for the requested range.
struct YuvRangeScale {
    double yBias, yScale, cScale;
};

inline YuvRangeScale yuv_range_scale(RpptColorRange range) {
    if (range == RpptColorRange_FULL) return {0.0, 1.0, 1.0};
    return {16.0, 255.0 / 219.0, 255.0 / 224.0};  // studio: luma 16-235, chroma 16-240
}

// Converts one (Y, U, V) triple of stored 8-bit codes into R, G, B stored codes.
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

// ---- CubicV fixed-point colour math ---------------------------------------

// FFmpeg's ff_yuv2rgb_coeffs, scaled as ff_yuv2rgb_c_init_tables does for RGB24. cy is the luma
// gain in <<16 fixed point and C folds the black level, table offset and rounding.
struct YuvFixedCoeffs {
    int cy, c, crv, cbu, cgu, cgv;
};

inline YuvFixedCoeffs yuv_fixed_coeffs(RpptColorStandard standard, RpptColorRange range) {
    long long crv, cbu, cguAbs, cgvAbs;
    switch (standard) {
        case RpptColorStandard_BT709:
            crv = 117489, cbu = 138438, cguAbs = 13975, cgvAbs = 34925;
            break;
        case RpptColorStandard_FCC:
            crv = 104448, cbu = 132798, cguAbs = 24759, cgvAbs = 53109;
            break;
        case RpptColorStandard_SMPTE240M:
            crv = 117579, cbu = 136230, cguAbs = 16907, cgvAbs = 35559;
            break;
        case RpptColorStandard_BT2020_NCL:
        case RpptColorStandard_BT2020_CL:
            crv = 110013, cbu = 140363, cguAbs = 12277, cgvAbs = 42626;
            break;
        default:  // BT.601 / BT.470BG, and FFmpeg's fallback for anything else
            crv = 104597, cbu = 132201, cguAbs = 25675, cgvAbs = 53279;
            break;
    }
    long long cgu = -cguAbs, cgv = -cgvAbs;

    const bool full = (range == RpptColorRange_FULL);
    long long cy, oy;
    if (full) {
        cy = 1 << 16;
        oy = 0;
        crv = (crv * 224) / 255;
        cbu = (cbu * 224) / 255;
        cgu = (cgu * 224) / 255;
        cgv = (cgv * 224) / 255;
    } else {
        cy = ((long long)(1 << 16) * 255) / 219;
        oy = 16LL << 16;
    }

    YuvFixedCoeffs k{};
    k.cy = static_cast<int>(cy);
    k.c = static_cast<int>((long long)(full ? 384 : 326) * cy - (384LL << 16) - oy + 0x8000);
    k.crv = static_cast<int>((crv * (1LL << 16) + 0x8000) / cy);
    k.cbu = static_cast<int>((cbu * (1LL << 16) + 0x8000) / cy);
    k.cgu = static_cast<int>((cgu * (1LL << 16) + 0x8000) / cy);
    k.cgv = static_cast<int>((cgv * (1LL << 16) + 0x8000) / cy);
    return k;
}

// Chroma is truncated to an integer luma-table index offset before the luma gain; all shifts are
// arithmetic, matching FFmpeg's floor behaviour.
inline void yuv_to_rgb_pixel_fixed(int y, int u, int v, const YuvFixedCoeffs& k, double rgb[3]) {
    const int kr = y + ((v * k.crv) >> 16) - (k.crv >> 9);
    const int kb = y + ((u * k.cbu) >> 16) - (k.cbu >> 9);
    const int kg = y + ((u * k.cgu) >> 16) - (k.cgu >> 9) + ((v * k.cgv) >> 16) - (k.cgv >> 9);
    rgb[0] = clampd(static_cast<double>((kr * k.cy + k.c) >> 16), 0.0, 255.0);
    rgb[1] = clampd(static_cast<double>((kg * k.cy + k.c) >> 16), 0.0, 255.0);
    rgb[2] = clampd(static_cast<double>((kb * k.cy + k.c) >> 16), 0.0, 255.0);
}

// ---- vertical chroma upsampling -------------------------------------------

// CubicV's integer taps (sum 4096) and their first chroma row, before clamping. Rows 0 and 2 get
// FFmpeg initFilter's folded+renormalized boundary taps, which are not what clamping the interior
// taps would give; the bottom edge needs no special case.
struct CubicVPhase {
    int base, tap[4];
};

inline CubicVPhase cubic_v_phase(Rpp32u lumaRow) {
    const int y = static_cast<int>(lumaRow);
    if (y == 0) return {0, {4432, -336, 0, 0}};
    if (y == 2) return {0, {959, 3473, -336, 0}};
    if (y & 1) return {(y >> 1) - 1, {-346, 3572, 985, -115}};
    return {(y >> 1) - 2, {-115, 985, 3572, -346}};
}

// The chroma rows contributing to one luma row, with their weights (already edge-clamped).
struct ChromaTaps {
    int row[4];
    double weight[4];
    int count;
};

// Nearest and LinearV only. The LinearV siting is pinned by its documented identity on odd rows:
// chroma sample cr is co-sited with luma row 2*cr + 1, so luma row y samples at p = (y - 1) / 2.
// Out-of-range chroma rows are clamped to [0, chromaHeight - 1] (edge replication).
inline ChromaTaps chroma_taps_v(YuvChromaUpsample mode, Rpp32u lumaRow, Rpp32u chromaHeight) {
    const int last = static_cast<int>(chromaHeight) - 1;
    const int cr = static_cast<int>(lumaRow / 2);
    auto clamp_row = [last](int r) { return r < 0 ? 0 : (r > last ? last : r); };

    ChromaTaps taps{};
    if (mode == YuvChromaUpsample::Nearest || (lumaRow & 1u)) {
        taps.count = 1;
        taps.row[0] = clamp_row(cr);
        taps.weight[0] = 1.0;
        return taps;
    }

    taps.count = 2;
    taps.row[0] = clamp_row(cr - 1);
    taps.row[1] = clamp_row(cr);
    taps.weight[0] = 0.5;
    taps.weight[1] = 0.5;
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
                                 Rpp32u srcYPitch, Rpp32u srcUVPitch, Rpp32u dstPitch, Rpp32u width,
                                 Rpp32u height, RpptColorStandard standard, RpptColorRange range,
                                 YuvChromaUpsample upsample) {
    const Rpp32u chromaHeight = height / 2;
    const int last = static_cast<int>(chromaHeight) - 1;
    const bool cubic = (upsample == YuvChromaUpsample::CubicV);
    const YuvFixedCoeffs fixedCoeffs = yuv_fixed_coeffs(standard, range);

    for (Rpp32u y = 0; y < height; ++y) {
        const Rpp8u* lumaRow = srcY + static_cast<std::size_t>(y) * srcYPitch;
        Rpp8u* dstRow = dst + static_cast<std::size_t>(y) * dstPitch;

        if (cubic) {
            const CubicVPhase phase = cubic_v_phase(y);
            const Rpp8u* chromaRow[4];
            for (int t = 0; t < 4; ++t) {
                const int r = phase.base + t;
                chromaRow[t] = srcUV + static_cast<std::size_t>(r < 0 ? 0 : (r > last ? last : r)) *
                                           srcUVPitch;
            }
            for (Rpp32u x = 0; x < width; ++x) {
                const Rpp32u cc = x / 2;
                // 8-bit chroma lifted to the 15-bit domain, accumulated with the integer taps and
                // rounded back: (acc + (1 << 18)) >> 19.
                int uAcc = 1 << 18, vAcc = 1 << 18;
                for (int t = 0; t < 4; ++t) {
                    uAcc += phase.tap[t] * (static_cast<int>(chromaRow[t][2 * cc]) << 7);
                    vAcc += phase.tap[t] * (static_cast<int>(chromaRow[t][2 * cc + 1]) << 7);
                }
                const int u = std::min(std::max(uAcc >> 19, 0), 255);
                const int v = std::min(std::max(vAcc >> 19, 0), 255);

                double rgb[3];
                yuv_to_rgb_pixel_fixed(static_cast<int>(lumaRow[x]), u, v, fixedCoeffs, rgb);
                for (int c = 0; c < 3; ++c) dstRow[3 * x + c] = static_cast<Rpp8u>(rgb[c]);
            }
            continue;
        }

        const ChromaTaps taps = chroma_taps_v(upsample, y, chromaHeight);
        for (Rpp32u x = 0; x < width; ++x) {
            const Rpp32u cc = x / 2;  // horizontal upsampling is nearest in all three ops
            double u = 0.0, v = 0.0;
            for (int t = 0; t < taps.count; ++t) {
                const Rpp8u* chromaRow = srcUV + static_cast<std::size_t>(taps.row[t]) * srcUVPitch;
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
