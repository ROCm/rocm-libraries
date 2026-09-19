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

#ifndef RPP_TEST_FLIP_REF_H
#define RPP_TEST_FLIP_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: flip

RPP op
  rppt_flip   (Image / Geometric augmentation)

Description
  Mask-controlled mirror of the ROI region about its vertical and/or
  horizontal axis. The two flags are independent, so setting both rotates the
  region by 180 degrees.

Expression
  dst(j, i) = src( y0 + (vertical   ? h-1-j : j),
                   x0 + (horizontal ? w-1-i : i) )

Per-type form
  A pure permutation of source elements. No arithmetic is performed, so there
  is no rounding or clamping and every type is bit-exact.
*/
template <typename T>
void flip_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                    const RpptROI* roi, RpptRoiType roiType, Rpp32u horizontal, Rpp32u vertical) {
    for_each_roi_plane(
        sd, dd, roi, roiType,
        [&](Rpp32u, const RoiBounds& b, Rpp32u, std::size_t srcBase, std::size_t dstBase) {
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    const Rpp32u srcRow = b.y0 + (vertical ? (b.h - 1 - j) : j);
                    const Rpp32u srcCol = b.x0 + (horizontal ? (b.w - 1 - i) : i);
                    dst[plane_index(dd, dstBase, j, i)] =
                        src[plane_index(sd, srcBase, srcRow, srcCol)];
                }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_FLIP_REF_H
