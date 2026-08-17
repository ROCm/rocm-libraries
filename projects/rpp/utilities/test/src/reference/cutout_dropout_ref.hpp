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

#ifndef RPP_TEST_CUTOUT_DROPOUT_REF_H
#define RPP_TEST_CUTOUT_DROPOUT_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_cutout_dropout, derived from the op's definition
// (overwrite one or more rectangular boxes of an image with caller-supplied solid colors) and
// its public API doc, NOT from the RPP kernel. Used as the reference for both backends so
// kernel bugs surface as diffs.
//
// Boxes are given in absolute image coordinates, LTRB inclusive (rb is the last covered pixel,
// so a box spans columns [lt.x, rb.x] and rows [lt.y, rb.y]). anchorBoxInfoTensor and colors are
// laid out with a per-image stride of maxBoxesPerImage: box k of image n is at
// [n * maxBoxesPerImage + k], and its color channel c at [(n * maxBoxesPerImage + k) * c... ].
// numBoxes[n] gives the count of active boxes for image n. Any output pixel whose absolute source
// coordinate falls inside an active box is a direct, bit-exact store of that box's color for the
// pixel's channel; all other pixels pass the source through unchanged. No arithmetic, so every
// dtype is bit-exact.
template <typename T>
void cutout_dropout_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                              const RpptROI* roi, RpptRoiType roiType,
                              const RpptRoiLtrb* boxes, const Rpp32u* numBoxes,
                              Rpp32u maxBoxesPerImage, const T* colors) {
    (void)dt;
    const Rpp32u channels = d.c;
    std::vector<RoiBounds> b(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) b[n] = roi_bounds(roi[n], roiType);
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t srcIdx,
                        std::size_t dstIdx) {
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
                                colors[(n * maxBoxesPerImage + static_cast<Rpp32u>(hit)) * channels +
                                       c];
                        else
                            dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CUTOUT_DROPOUT_REF_H
