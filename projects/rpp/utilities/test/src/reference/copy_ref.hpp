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

#ifndef RPP_TEST_COPY_REF_H
#define RPP_TEST_COPY_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_copy, derived from the op's definition (dst = src, a
// bit-exact buffer copy of every element), NOT from the RPP kernel. Used as the reference for
// both backends so kernel bugs surface as diffs.
//
// copy takes no ROI and no scalar params: every dtype is copied verbatim with no arithmetic,
// rounding, or clamping, so the result is bit-exact for U8/I8/F16/F32 alike.
template <typename T>
void copy_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                    RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_COPY_REF_H
