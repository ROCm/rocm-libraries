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

#ifndef RPP_TEST_RANDOM_ERASE_REF_H
#define RPP_TEST_RANDOM_ERASE_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: random_erase

RPP op
  rppt_random_erase   (Image / Effects augmentation)

Description
  Fills one user-defined rectangular region per image with noise sampled
  spatially from a caller-supplied buffer, passing every other pixel through
  unchanged. Unlike erase and coarse_dropout it takes exactly one box per
  image (anchorBoxInfoTensor[n], no numBoxes/maxBoxesPerImage), and the fill is
  not a solid colour but a tiled lookup into a noiseBuffer of size
  255*255*channels.

  Despite the name there is no seed anywhere in this API: every "random" input
  (box placement, noise contents) is supplied by the caller, so the op is a
  deterministic tiling lookup and this golden is bit-exact.

  anchorBoxInfoTensor[n] gives the box as an RpptRoiLtrb in ABSOLUTE image
  coordinates, inclusive on both corners (same convention as erase_ref.hpp /
  roi_bounds: width rb-lt+1), covering columns [lt.x, rb.x] and rows
  [lt.y, rb.y].

Expression
  For a pixel at absolute source coordinate (sx, sy) inside the box:

  dst = noiseBuffer[ (((sy + n) % 255) * 255 + (sx % 255)) * channels + c ]

  and dst = src everywhere else.

Per-type form
  No arithmetic beyond the modulo and the lookup, so every type is bit-exact.

Notes
  The row phase is shifted by the image's own batch index n. Both backends add
  their per-image batch index to the row before the modulo, independently of
  one another, so this is the intended per-image tile phase -- each image in a
  batch samples a different vertical slice of the tile -- rather than a shared
  bug.
*/
template <typename T>
void random_erase_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                            RpptRoiType roiType, const RpptRoiLtrb* boxes, const T* noiseBuffer) {
    (void)dt;
    const Rpp32u channels = d.c;
    std::vector<RoiBounds> b(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) b[n] = roi_bounds(roi[n], roiType);
    for_each_roi_io(
        d, roi, roiType,
        [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t srcIdx, std::size_t dstIdx) {
            const int sx = static_cast<int>(b[n].x0 + i);
            const int sy = static_cast<int>(b[n].y0 + j);
            const RpptRoiLtrb& bx = boxes[n];
            if (sx >= bx.lt.x && sx <= bx.rb.x && sy >= bx.lt.y && sy <= bx.rb.y) {
                const int tx = sx % 255;
                const int ty = (sy + static_cast<int>(n)) % 255;
                dst[dstIdx] = noiseBuffer[static_cast<std::size_t>(ty * 255 + tx) * channels + c];
            } else {
                dst[dstIdx] = src[srcIdx];
            }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_RANDOM_ERASE_REF_H
