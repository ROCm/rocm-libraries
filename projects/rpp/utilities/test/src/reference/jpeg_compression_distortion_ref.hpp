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

#ifndef RPP_TEST_JPEG_COMPRESSION_DISTORTION_REF_H
#define RPP_TEST_JPEG_COMPRESSION_DISTORTION_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_ycbcr.hpp"

namespace rpptest {

// Independent host golden model for rppt_jpeg_compression_distortion, derived from the definition
// of baseline JPEG lossy compression (ITU-T T.81 + the IJG quality convention every encoder uses)
// and the op's header description ("converting the image to the frequency domain using the DCT,
// applying quantization, and then reconstructing the image using the IDCT"), NOT from the RPP
// kernel. The same reference serves both HOST and HIP backends.
//
// This is a block-structured (non-pointwise) op: each output pixel depends on its whole 8x8 block,
// so it cannot use the pointwise scalar template. The pipeline, per image, over the ROI:
//
//   1. Load into [0,255] intensity space (U8 v; I8 v+128; F16/F32 v*255).
//   2. 3 channels: RGB -> YCbCr, BT.601 full-range -- the JPEG color transform, shared with
//      histogram_equalize via reference/color_ycbcr.hpp. 1 channel: the value is Y.
//   3. Y is processed at full resolution; Cb/Cr are subsampled 4:2:0 (see below).
//   4. Tile each plane into 8x8 blocks from its top-left origin, level-shift by -128, forward
//      DCT-II (T.81 eq. 8-3), quantize q = round(F / Q), dequantize q * Q, inverse DCT (eq. 8-2),
//      level-shift by +128 and clamp to [0,255]. A block running past the right/bottom edge
//      replicates the last valid column/row into the padding (the encoder-side edge extension).
//   5. 3 channels: upsample Cb/Cr back to full resolution, then YCbCr -> RGB.
//   6. Store: U8 round+clamp[0,255], I8 round-128+clamp[-128,127], F16/F32 /255 clamp[0,1].
//
// Chroma follows the 4:2:0 sampling that is the default of every baseline JPEG encoder: each
// chroma sample is the 2x2 box average of the luma-resolution plane, and the image is padded out
// to whole 16x16 MCUs by edge replication before subsampling -- so a chroma plane is always a
// whole number of 8x8 blocks, ceil(dim/16)*8 samples per axis. Reconstruction replicates each
// chroma sample back over its 2x2 pixel group (box upsampling).
//
// Q is the T.81 Annex K table (K.1 luminance for Y / a 1-channel image, K.2 chrominance for
// Cb/Cr) scaled by the IJG quality mapping:
//   scale = quality < 50 ? 5000/quality : 200 - 2*quality;  Q = clamp((base*scale + 50)/100, 1, 255)
// The result is an 8-bit integer because baseline JPEG stores quantization values in 8 bits for
// 8-bit sample precision (T.81 B.2.4.1): no encoder can quantize by 3.4, nor by more than 255.
// RPP keeps the scaled table real-valued and unclamped, so it quantizes by neither an integer nor
// anything a decoder could read back, and that single difference is what the q10 and q90 halves of
// the grid disagree on -- with the table rounded and clamped this model reproduces both backends
// bit-exactly. It deliberately stays spec-correct.
//
// One degree of freedom the header does not pin down, chosen as a documented assumption: the
// intermediate YCbCr samples are kept continuous rather than re-quantized to 8 bits between the
// color transform and the DCT, so the model is dtype-independent -- the distortion comes from the
// coefficient quantization, and only the final store quantizes.

namespace jpeg_detail {

// ITU-T T.81 Annex K, Table K.1 -- luminance quantization table.
constexpr int kLumaTable[64] = {16, 11, 10, 16,  24,  40,  51,  61,  12,  12, 14, 19, 26,
                                58, 60, 55, 14,  13,  16,  24,  40,  57,  69, 56, 14, 17,
                                22, 29, 51, 87,  80,  62,  18,  22,  37,  56, 68, 109, 103,
                                77, 24, 35, 55,  64,  81,  104, 113, 92,  49, 64, 78, 87,
                                103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

// ITU-T T.81 Annex K, Table K.2 -- chrominance quantization table.
constexpr int kChromaTable[64] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99,
                                  99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66,
                                  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                                  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                                  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

// IJG quality -> table scaling percentage (libjpeg's jpeg_quality_scaling).
inline int quality_scaling(int quality) {
    quality = std::max(1, std::min(quality, 100));
    return quality < 50 ? 5000 / quality : 200 - quality * 2;
}

// The quality-scaled quantization table, rounded and clamped to a valid 8-bit quantizer.
inline std::array<double, 64> scaled_table(const int base[64], int quality) {
    const int scale = quality_scaling(quality);
    std::array<double, 64> q{};
    for (int i = 0; i < 64; ++i) {
        const int v = (base[i] * scale + 50) / 100;
        q[i] = std::max(1, std::min(v, 255));
    }
    return q;
}

// cos((2x+1) u pi / 16), the only transcendental the DCT pair needs.
inline const std::array<double, 64>& cos_table() {
    static const std::array<double, 64> t = [] {
        std::array<double, 64> c{};
        for (int x = 0; x < 8; ++x)
            for (int u = 0; u < 8; ++u)
                c[x * 8 + u] = std::cos((2.0 * x + 1.0) * u * M_PI / 16.0);
        return c;
    }();
    return t;
}

inline double dct_norm(int u) { return u == 0 ? 0.70710678118654752440 : 1.0; }  // 1/sqrt(2)

// Quantization round-trip of one level-shifted 8x8 block: forward DCT (T.81 eq. 8-3), quantize,
// dequantize, inverse DCT (eq. 8-2). in/out are 64 samples in row-major order.
inline void block_roundtrip(const double in[64], double out[64], const std::array<double, 64>& q) {
    const std::array<double, 64>& cs = cos_table();

    double coeff[64];
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            double sum = 0.0;
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x) sum += in[y * 8 + x] * cs[x * 8 + v] * cs[y * 8 + u];
            const double f = 0.25 * dct_norm(u) * dct_norm(v) * sum;
            coeff[u * 8 + v] = std::nearbyint(f / q[u * 8 + v]) * q[u * 8 + v];
        }

    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            double sum = 0.0;
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    sum += dct_norm(u) * dct_norm(v) * coeff[u * 8 + v] * cs[x * 8 + v] *
                           cs[y * 8 + u];
            out[y * 8 + x] = 0.25 * sum;
        }
}

// Runs the 8x8 block round-trip over a whole h x w plane held in [0,255] intensity space. Edge
// blocks replicate the last valid row/column into the padding.
inline void plane_roundtrip(const std::vector<double>& in, std::vector<double>& out, int h, int w,
                            const std::array<double, 64>& q) {
    double blk[64], res[64];
    for (int by = 0; by < h; by += 8)
        for (int bx = 0; bx < w; bx += 8) {
            for (int j = 0; j < 8; ++j) {
                const int y = std::min(by + j, h - 1);
                for (int i = 0; i < 8; ++i)
                    blk[j * 8 + i] = in[static_cast<std::size_t>(y) * w + std::min(bx + i, w - 1)] -
                                     128.0;
            }
            block_roundtrip(blk, res, q);
            for (int j = 0; j < 8 && by + j < h; ++j)
                for (int i = 0; i < 8 && bx + i < w; ++i)
                    out[static_cast<std::size_t>(by + j) * w + bx + i] =
                        clampd(res[j * 8 + i] + 128.0, 0.0, 255.0);
        }
}

// [0,255] intensity space is where the whole pipeline works, so a stored pixel is lifted into it
// on load and quantized back on store. Both compose the shared per-dtype rules rather than
// restating them: U8 v, I8 v+128, F16/F32 v*255.
inline double load_intensity(double stored, DType dt) { return to_unit(stored, dt) * 255.0; }
inline double store_intensity(double v, DType dt) { return from_unit(v / 255.0, dt); }

// Chroma plane extent per axis: the image padded out to whole 16x16 MCUs, halved. Always a
// multiple of 8, so the subsampled plane tiles into whole 8x8 blocks.
inline int chroma_extent(int dim) { return ((dim + 15) / 16) * 8; }

// The 4:2:0 counterpart of plane_roundtrip, same contract on a full-resolution plane: each chroma
// sample is the 2x2 box average of the edge-replicated plane, the subsampled plane makes the same
// round trip (whole 8x8 blocks by construction), and each sample is replicated back over its 2x2
// pixel group.
inline void chroma_roundtrip(const std::vector<double>& in, std::vector<double>& out, int h, int w,
                             const std::array<double, 64>& q) {
    const int hs = chroma_extent(h), ws = chroma_extent(w);
    std::vector<double> small(static_cast<std::size_t>(hs) * ws), smallOut(small.size());
    for (int y = 0; y < hs; ++y)
        for (int x = 0; x < ws; ++x) {
            double sum = 0.0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    sum += in[static_cast<std::size_t>(std::min(2 * y + dy, h - 1)) * w +
                              std::min(2 * x + dx, w - 1)];
            small[static_cast<std::size_t>(y) * ws + x] = sum * 0.25;
        }
    plane_roundtrip(small, smallOut, hs, ws, q);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            out[static_cast<std::size_t>(y) * w + x] =
                smallOut[static_cast<std::size_t>(y / 2) * ws + x / 2];
}

}  // namespace jpeg_detail

// Writes the JPEG-distorted result into dst, reading the source at the ROI offset and writing the
// roiH x roiW region packed at the destination origin (the placement every RPP image op uses).
template <typename T>
void jpeg_compression_distortion_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                                           const RpptROI* roi, RpptRoiType roiType, int quality) {
    const bool rgb = d.c == 3;
    const std::array<double, 64> lumaQ = jpeg_detail::scaled_table(jpeg_detail::kLumaTable, quality);
    const std::array<double, 64> chromaQ =
        jpeg_detail::scaled_table(jpeg_detail::kChromaTable, quality);

    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds bounds = roi_bounds(roi[n], roiType);
        const int h = static_cast<int>(bounds.h), w = static_cast<int>(bounds.w);
        const std::size_t plane = static_cast<std::size_t>(h) * w;

        // Gather the ROI into per-channel planes in [0,255] intensity space.
        std::vector<std::vector<double>> in(d.c, std::vector<double>(plane));
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = plane_base(d, n, c);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    in[c][static_cast<std::size_t>(y) * w + x] = jpeg_detail::load_intensity(
                        to_double(src[plane_index(d, base, bounds.y0 + y, bounds.x0 + x)]), dt);
        }

        if (rgb)
            for (std::size_t i = 0; i < plane; ++i)
                rgb_to_ycbcr(in[0][i], in[1][i], in[2][i], in[0][i], in[1][i], in[2][i]);

        std::vector<std::vector<double>> out(d.c, std::vector<double>(plane));
        for (Rpp32u c = 0; c < d.c; ++c) {
            if (rgb && c > 0)
                jpeg_detail::chroma_roundtrip(in[c], out[c], h, w, chromaQ);
            else
                jpeg_detail::plane_roundtrip(in[c], out[c], h, w, lumaQ);
        }

        if (rgb)
            for (std::size_t i = 0; i < plane; ++i)
                ycbcr_to_rgb(out[0][i], out[1][i], out[2][i], out[0][i], out[1][i], out[2][i]);

        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = plane_base(d, n, c);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    dst[plane_index(d, base, y, x)] = from_double<T>(jpeg_detail::store_intensity(
                        out[c][static_cast<std::size_t>(y) * w + x], dt));
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_JPEG_COMPRESSION_DISTORTION_REF_H
