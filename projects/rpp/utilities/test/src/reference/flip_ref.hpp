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

// Independent host golden model for rppt_flip, derived from the op's definition (a
// mask-controlled mirror of the ROI region about its vertical and/or horizontal axis), NOT from
// the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// Flip is a pure permutation of source elements -- no arithmetic, so no rounding or clamping and
// every dtype is bit-exact. The source is read at the ROI offset and the output written packed at
// the destination origin (the placement every RPP op uses), so output element (j, i) comes from
// source (y0 + [vertical ? h-1-j : j], x0 + [horizontal ? w-1-i : i]).
template <typename T>
void flip_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                    RpptRoiType roiType, Rpp32u horizontal, Rpp32u vertical) {
    for_each_roi_plane(d, roi, roiType, [&](Rpp32u, const RoiBounds& b, Rpp32u, std::size_t base) {
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i) {
                const Rpp32u srcRow = b.y0 + (vertical ? (b.h - 1 - j) : j);
                const Rpp32u srcCol = b.x0 + (horizontal ? (b.w - 1 - i) : i);
                dst[plane_index(d, base, j, i)] = src[plane_index(d, base, srcRow, srcCol)];
            }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_FLIP_REF_H
