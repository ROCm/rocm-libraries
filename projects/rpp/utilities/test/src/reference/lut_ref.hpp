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
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_lut, derived from the op's definition (each output
// element is a direct look-up of the source pixel's intensity in a caller-supplied table:
// out = lut[index(src)]), NOT from the RPP kernel. Used as the reference for both backends so
// kernel bugs surface as diffs.
//
// The table is indexed by the pixel's unsigned intensity in [0,255]. For U8 the index is the
// pixel value; for I8 the intensity is the pixel shifted by +128 (so -128 -> 0, 127 -> 255).
// The look-up is a pure copy of a table entry, so there is no arithmetic, rounding, or
// clamping and the result is bit-exact. Restricted to the integer dtypes (U8/I8) where the
// 256-entry index is unambiguous; the float dtypes' index semantics are not defined by the
// public API and are not modeled here.
inline int lut_index(double v, DType dt) {
    return (dt == DType::I8) ? static_cast<int>(v) + 128 : static_cast<int>(v);
}

// Writes the lut result into dst, reading the source at the ROI offset and writing packed at
// the destination origin (matching the region and placement the RPP op uses). lut is the
// same table handed to the kernel. dst outside the written region is left as the caller
// initialized it.
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
