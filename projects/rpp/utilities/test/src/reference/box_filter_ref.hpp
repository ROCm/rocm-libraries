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

#ifndef RPP_TEST_BOX_FILTER_REF_H
#define RPP_TEST_BOX_FILTER_REF_H

#include <rpp/rpp.h>

#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/filter_common.hpp"

namespace rpptest {

// Independent host golden model for rppt_box_filter, derived from the box-filter definition
// (per-channel arithmetic mean over a KxK window with a clamp-to-edge / REPLICATE border), NOT
// from the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
// The uniform kernel weights (each 1/(K*K)) are handed to convolve_reference, which owns the
// shared window/border/quantization math (see reference/filter_common.hpp).
template <typename T>
void box_filter_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                          RpptRoiType type, Rpp32u kernelSize) {
    const std::size_t kk = static_cast<std::size_t>(kernelSize) * kernelSize;
    const std::vector<double> weights(kk, 1.0 / static_cast<double>(kk));
    convolve_reference<T>(src, dst, d, dt, roi, type, kernelSize, weights);
}

}  // namespace rpptest

#endif  // RPP_TEST_BOX_FILTER_REF_H
