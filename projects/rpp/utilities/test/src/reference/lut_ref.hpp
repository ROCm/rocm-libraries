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

#ifndef RPP_TEST_LUT_REF_H
#define RPP_TEST_LUT_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: lut

RPP op
  rppt_lut   (Image / Color augmentation)

Description
  Pointwise look-up table remap. Each output element is a direct table entry
  selected by the source pixel's intensity, so the table defines an arbitrary
  tone curve. The look-up is a pure copy of an entry, so the result is
  bit-exact with no arithmetic, rounding, or clamping.

Expression
  dst(x, y, c) = lut[ index(src(x, y, c)) ]

Per-type form
  The 256-entry table is indexed by the pixel's unsigned intensity in [0,255].

    U8    index = v
    I8    index = v + 128        (so -128 -> 0, 127 -> 255)

  Modelled for the integer types only, where the 256-entry index is
  unambiguous; the float types' index semantics are not defined by the
  public API.
*/
inline int lut_index(double v, DType dt) {
    return (dt == DType::I8) ? static_cast<int>(v) + 128 : static_cast<int>(v);
}

template <typename T>
void lut_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                   RpptRoiType roiType, const T* lut) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = lut[lut_index(to_double(src[srcIdx]), dt)];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_LUT_REF_H
