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

#ifndef RPP_TEST_GLITCH_REF_H
#define RPP_TEST_GLITCH_REF_H

#include <rpp/rpp.h>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: glitch

RPP op
  rppt_glitch   (Image / Effects augmentation)

Description
  Per-channel RGB pixel shift -- the classic colour-fringe glitch. Each
  channel is read from the source displaced by its own (x, y) offset, given as
  one RpptChannelOffsets per image, so the three channels separate at edges.

  Greyscale is out of scope: the op takes three offsets and the legacy harness
  rejects PLN1.

Expression
  sx = x0 + i + offset[c].x
  sy = y0 + j + offset[c].y
  dst(i, j, c) = src(sx, sy, c)

Per-type form
  The op only moves whole pixels, so no interpolation, rounding or clamping is
  involved and every type is a verbatim copy.

Notes
  The header documents no behaviour for a shifted coordinate that leaves the
  ROI. The model passes the unshifted source pixel through, which keeps the
  frame intact and is what an RGB-shift glitch does; a kernel that blacks or
  clamps those pixels instead shows up as a diff -- a finding, not a reference
  bug.
*/

// The (x, y) offset for channel c of RpptChannelOffsets.
inline RpptPoint2D glitch_channel_offset(const RpptChannelOffsets& o, Rpp32u c) {
    return c == 0 ? o.r : (c == 1 ? o.g : o.b);
}

template <typename T>
void glitch_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                      RpptRoiType roiType, const RpptChannelOffsets* offsets) {
    for_each_roi_plane(
        d, roi, roiType, [&](Rpp32u n, const RoiBounds& b, Rpp32u c, std::size_t base) {
            const RpptPoint2D off = glitch_channel_offset(offsets[n], c);
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    const int sx = static_cast<int>(b.x0 + i) + off.x;
                    const int sy = static_cast<int>(b.y0 + j) + off.y;
                    const bool inside =
                        sx >= static_cast<int>(b.x0) && sy >= static_cast<int>(b.y0) &&
                        sx < static_cast<int>(b.x0 + b.w) && sy < static_cast<int>(b.y0 + b.h);
                    const std::size_t srcIdx =
                        inside ? plane_index(d, base, static_cast<std::size_t>(sy),
                                             static_cast<std::size_t>(sx))
                               : plane_index(d, base, b.y0 + j, b.x0 + i);
                    dst[plane_index(d, base, j, i)] = src[srcIdx];
                }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_GLITCH_REF_H
