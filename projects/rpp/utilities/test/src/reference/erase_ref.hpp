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

#ifndef RPP_TEST_ERASE_REF_H
#define RPP_TEST_ERASE_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: erase

RPP op
  rppt_erase   (Image / Effects augmentation)

Description
  Overwrites one or more user-defined rectangular regions with caller-supplied
  solid colours, passing every other pixel through unchanged.

  anchorBoxInfoTensor gives each erase-region as an RpptRoiLtrb in ABSOLUTE
  image coordinates, inclusive on both corners (RpptRoiLtrb is inclusive
  throughout RPP -- see roi_bounds, which spans [lt, rb] with width
  rb-lt+1), so a box covers columns [lt.x, rb.x] and rows [lt.y, rb.y]. The
  box and colour tensors are packed with a per-image stride of
  maxBoxesPerImage: box k of image n is at [n*maxBoxesPerImage + k], and its
  colour for channel c at [(n*maxBoxesPerImage + k)*channels + c] (a per-box
  RGB triplet, one value per channel). numBoxes[n] is the count of active
  boxes for image n, and they are required not to overlap.

Expression
  dst(x, y, c) = inside box k ? colors[k][c] : src(x, y, c)

Per-type form
  No arithmetic is performed -- an output pixel is either a box colour or a
  source value, stored directly -- so every type is bit-exact.
*/
template <typename T>
void erase_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                     const RpptROI* roi, RpptRoiType roiType, const RpptRoiLtrb* boxes,
                     const Rpp32u* numBoxes, Rpp32u maxBoxesPerImage, const T* colors) {
    (void)dt;
    const Rpp32u channels = sd.c;
    std::vector<RoiBounds> b(sd.n);
    for (Rpp32u n = 0; n < sd.n; ++n) b[n] = roi_bounds(roi[n], roiType);
    for_each_roi_io(
        sd, dd, roi, roiType,
        [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t srcIdx, std::size_t dstIdx) {
            const int sx = static_cast<int>(b[n].x0 + i);
            const int sy = static_cast<int>(b[n].y0 + j);
            int hit = -1;
            for (Rpp32u k = 0; k < numBoxes[n]; ++k) {
                const RpptRoiLtrb& bx = boxes[n * maxBoxesPerImage + k];
                if (sx >= bx.lt.x && sx <= bx.rb.x && sy >= bx.lt.y && sy <= bx.rb.y) {
                    hit = static_cast<int>(k);
                    break;
                }
            }
            if (hit >= 0)
                dst[dstIdx] =
                    colors[(n * maxBoxesPerImage + static_cast<Rpp32u>(hit)) * channels + c];
            else
                dst[dstIdx] = src[srcIdx];
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_ERASE_REF_H
