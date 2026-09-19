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

#ifndef RPP_TEST_ERODE_REF_H
#define RPP_TEST_ERODE_REF_H

#include <rpp/rpp.h>

#include <algorithm>

#include "framework/config_param.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

/*
Reference model: erode

RPP op
  rppt_erode   (Image / Morphological)

Description
  Greyscale morphological erosion: the per-channel MIN over a flat KxK square
  window centred on each pixel, which shrinks bright regions and grows dark
  ones. The window and its REPLICATE border come from filter_reference
  (reference/filter_common.hpp); erosion supplies only the MIN reduction.

Expression
  dst(j, i) = min{ src(j+dy, i+dx) : dy, dx in [-r, r] }

Per-type form
  The min selects an existing source value and performs no arithmetic, so the
  result is bit-exact for every type.
*/
template <typename T>
void erode_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType /*dt*/,
                     const RpptROI* roi, RpptRoiType type, Rpp32u kernelSize) {
    filter_reference<T>(src, sd, dst, dd, roi, type, kernelSize,
                        [](const double* w, int kk) { return *std::min_element(w, w + kk); });
}

}  // namespace rpptest

#endif  // RPP_TEST_ERODE_REF_H
