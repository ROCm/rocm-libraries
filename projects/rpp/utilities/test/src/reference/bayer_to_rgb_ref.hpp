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

#ifndef RPP_TEST_BAYER_TO_RGB_REF_H
#define RPP_TEST_BAYER_TO_RGB_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/interpolation.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: bayer_to_rgb

RPP op
  rppt_bayer_to_rgb   (Image / Data exchange)

Description
  Bilinear demosaic of a single-channel Bayer mosaic (PLN1) into R, G, B
  (PKD3 or PLN3). The pattern is anchored at the ROI's top-left pixel and
  neighbours outside the ROI are mirrored without repeating the edge pixel
  (Reflect101). That mirror moves a coordinate by an even amount, so every
  neighbour keeps the colour its unmirrored position has.

Expression
  colour(y, x) = pattern[y % 2][x % 2], (y, x) ROI-local
  s(y, x)      = src(y0 + reflect101(y), x0 + reflect101(x))

  For output channel C at (j, i):
    C == colour(j, i)             s(j, i)
    C == G                        mean of the 4 edge neighbours
    site G, C == colour(j, i+1)   mean of s(j, i-1), s(j, i+1)
    site G, C == colour(j+1, i)   mean of s(j-1, i), s(j+1, i)
    otherwise                     mean of the 4 diagonal neighbours

Per-type form
  Every output is a mean of stored values, so the I8 offset cancels; quantized
  once via quantize_stored.
*/

// 'R', 'G' or 'B' at ROI-local (y, x); y and x may be -1 (parity is what matters).
inline char bayer_colour(RpptBayerPattern pattern, int y, int x) {
    static const char* kPatterns[4] = {"RGGB", "BGGR", "GRBG", "GBRG"};
    return kPatterns[pattern][((y & 1) << 1) | (x & 1)];
}

template <typename T>
void bayer_to_rgb_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                            const RpptROI* roi, RpptRoiType type, RpptBayerPattern pattern) {
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        const int h = static_cast<int>(b.h), w = static_cast<int>(b.w);
        const std::size_t srcBase = plane_base(sd, n, 0);
        auto s = [&](int y, int x) {
            const int sy = border_index(y, h, BorderMode::Reflect101);
            const int sx = border_index(x, w, BorderMode::Reflect101);
            return to_double(src[plane_index(sd, srcBase, b.y0 + sy, b.x0 + sx)]);
        };
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                const char site = bayer_colour(pattern, j, i);
                const double edges = (s(j - 1, i) + s(j + 1, i) + s(j, i - 1) + s(j, i + 1)) / 4.0;
                const double diagonals =
                    (s(j - 1, i - 1) + s(j - 1, i + 1) + s(j + 1, i - 1) + s(j + 1, i + 1)) / 4.0;
                const double rowPair = (s(j, i - 1) + s(j, i + 1)) / 2.0;
                const double colPair = (s(j - 1, i) + s(j + 1, i)) / 2.0;
                static const char kChannels[3] = {'R', 'G', 'B'};
                for (Rpp32u c = 0; c < 3; ++c) {
                    const char want = kChannels[c];
                    double v;
                    if (want == site)
                        v = s(j, i);
                    else if (want == 'G')
                        v = edges;
                    else if (site == 'G')
                        v = (bayer_colour(pattern, j, i + 1) == want) ? rowPair : colPair;
                    else
                        v = diagonals;
                    dst[plane_index(dd, plane_base(dd, n, c), j, i)] =
                        from_double<T>(quantize_stored(v, dt));
                }
            }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_BAYER_TO_RGB_REF_H
