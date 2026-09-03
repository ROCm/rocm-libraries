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

#ifndef RPP_TEST_GRID_DROPOUT_REF_H
#define RPP_TEST_GRID_DROPOUT_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: grid_dropout

RPP op
  rppt_grid_dropout   (Image / Effects augmentation)

Description
  Erases a regular grid of rectangular holes from an image, passing every
  other pixel through unchanged.

  The holes are provided directly as boxes (anchorBoxInfoTensor), laid out per
  image as [n*boxesInEachImage + k]. Each box is an RpptRoiLtrb in ABSOLUTE
  image coordinates, LTRB inclusive (covering columns [lt.x, rb.x] and rows
  [lt.y, rb.y]).

Expression
  dst(x, y, c) = inside any hole ? black : src(x, y, c)

Per-type form
  "Black" is 0 intensity in the suite's shared intensity model.

    U8  0        I8  -128 (0 intensity shifted by -128)
    F16 0.0      F32 0.0

Notes
  The op carries maxHoleW/maxHoleH as separate scalars; the golden ignores
  them and uses the boxes directly, so a kernel that misuses maxHoleW/H
  surfaces as a diff.
*/
inline double grid_dropout_scalar(double v, DType dt, bool erased) {
    return erased ? from_unit(0.0, dt) : v;
}

// Hole membership is tested against the ABSOLUTE source coordinate (x0 + i, y0 + j).
template <typename T>
void grid_dropout_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                            const RpptROI* roi, RpptRoiType roiType, const RpptRoiLtrb* boxes,
                            Rpp32u boxesInEachImage) {
    std::vector<RoiBounds> b(sd.n);
    for (Rpp32u n = 0; n < sd.n; ++n) b[n] = roi_bounds(roi[n], roiType);
    for_each_roi_io(
        sd, dd, roi, roiType,
        [&](Rpp32u n, Rpp32u, Rpp32u j, Rpp32u i, std::size_t srcIdx, std::size_t dstIdx) {
            const int sx = static_cast<int>(b[n].x0 + i);
            const int sy = static_cast<int>(b[n].y0 + j);
            bool erased = false;
            for (Rpp32u k = 0; k < boxesInEachImage && !erased; ++k) {
                const RpptRoiLtrb& bx = boxes[n * boxesInEachImage + k];
                if (sx >= bx.lt.x && sx <= bx.rb.x && sy >= bx.lt.y && sy <= bx.rb.y) erased = true;
            }
            dst[dstIdx] = from_double<T>(grid_dropout_scalar(to_double(src[srcIdx]), dt, erased));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_GRID_DROPOUT_REF_H
