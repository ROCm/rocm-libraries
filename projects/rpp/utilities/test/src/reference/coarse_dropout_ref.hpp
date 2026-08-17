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

#ifndef RPP_TEST_COARSE_DROPOUT_REF_H
#define RPP_TEST_COARSE_DROPOUT_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_coarse_dropout, derived from the op's definition
// (erase user-selected rectangular regions of an image to black) and its public API doc, NOT
// from the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// anchorBoxInfoTensor holds erase-region boxes as RpptRoiLtrb, laid out with a per-image stride
// of maxBoxesPerImage: box k of image n is at [n * maxBoxesPerImage + k]. numBoxesTensor[n] is
// the count of active boxes for image n (only k < numBoxesTensor[n] are read). Boxes are given
// in ABSOLUTE image coordinates, LTRB inclusive (box width = rb.x - lt.x + 1). Per the API doc
// the user-supplied boxes must not overlap. Every pixel inside any active box is "erased", i.e.
// set to black -- 0 intensity in the suite's shared intensity model:
//   U8  : 0        I8  : -128 (0 intensity shifted by -128)
//   F16 : 0.0      F32 : 0.0
// Everything else is a bit-exact passthrough of the source.
inline double coarse_dropout_scalar(double v, DType dt, bool erased) {
    return erased ? from_unit(0.0, dt) : v;
}

// Writes the coarse-dropout result into dst, reading the source at the ROI offset and writing
// packed at the destination origin (matching the region and placement the RPP op uses). Box
// membership is tested against the ABSOLUTE source coordinate (x0 + i, y0 + j). dst outside the
// written region is left as the caller initialized it.
template <typename T>
void coarse_dropout_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                              const RpptROI* roi, RpptRoiType roiType, const RpptRoiLtrb* boxes,
                              const Rpp32u* numBoxes, Rpp32u maxBoxesPerImage) {
    std::vector<RoiBounds> b(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) b[n] = roi_bounds(roi[n], roiType);
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u n, Rpp32u, Rpp32u j, Rpp32u i, std::size_t srcIdx,
                        std::size_t dstIdx) {
                        const int sx = static_cast<int>(b[n].x0 + i);
                        const int sy = static_cast<int>(b[n].y0 + j);
                        bool erased = false;
                        for (Rpp32u k = 0; k < numBoxes[n] && !erased; ++k) {
                            const RpptRoiLtrb& bx = boxes[n * maxBoxesPerImage + k];
                            if (sx >= bx.lt.x && sx <= bx.rb.x && sy >= bx.lt.y && sy <= bx.rb.y)
                                erased = true;
                        }
                        dst[dstIdx] = from_double<T>(
                            coarse_dropout_scalar(to_double(src[srcIdx]), dt, erased));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_COARSE_DROPOUT_REF_H
