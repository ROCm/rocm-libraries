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

#ifndef RPP_TEST_CHANNEL_DROPOUT_REF_H
#define RPP_TEST_CHANNEL_DROPOUT_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_channel_dropout, derived from the op's definition
// (erase user-selected channels from an image) and its public API doc, NOT from the RPP
// kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// dropoutTensor holds one 0/1 value per channel per image (size batchSize * channels),
// laid out as [image * channels + channel]: 1 = Keep (pass the channel through unchanged),
// 0 = Drop (erase the channel). "Erase" means set the channel to black, i.e. 0 intensity in
// the suite's shared intensity model:
//   U8  : 0        I8  : -128 (0 intensity shifted by -128)
//   F16 : 0.0      F32 : 0.0
inline double channel_dropout_scalar(double v, DType dt, bool keep) {
    return keep ? v : from_unit(0.0, dt);
}

// Writes the channel-dropout result into dst, reading the source at the ROI offset and
// writing packed at the destination origin (matching the region and placement the RPP op
// uses). dst outside the written region is left as the caller initialized it.
template <typename T>
void channel_dropout_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                               const RpptROI* roi, RpptRoiType roiType, const Rpp8u* dropout) {
    const Rpp32u channels = d.c;
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u n, Rpp32u c, Rpp32u, Rpp32u, std::size_t srcIdx,
                        std::size_t dstIdx) {
                        const bool keep = dropout[n * channels + c] != 0;
                        dst[dstIdx] = from_double<T>(
                            channel_dropout_scalar(to_double(src[srcIdx]), dt, keep));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CHANNEL_DROPOUT_REF_H
