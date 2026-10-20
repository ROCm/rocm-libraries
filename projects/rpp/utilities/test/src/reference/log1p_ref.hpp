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

#ifndef RPP_TEST_LOG1P_REF_H
#define RPP_TEST_LOG1P_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: log1p

RPP op
  rppt_log1p   (ND / Arithmetic)

Description
  Element-wise log(1 + x) over the whole densely packed tensor (a single
  source, no broadcast, no op params). The header specifies that the absolute
  value of the input is used. Computed in double, no clamping.

Expression
  dst = log1p(|src|) = log(1 + |src|)

Notes
  std::log1p is used rather than std::log(1 + x) to keep precision for small
  |x|.
*/

inline double log1p_scalar(double v) {
    return std::log1p(std::fabs(v));
}

template <typename Tin, typename Tout>
void log1p_reference(const Tin* src, Tout* dst, const RpptGenericDesc& srcDesc,
                     const RpptGenericDesc& dstDesc) {
    // src and dst may differ in dtype and stride padding, so each uses its own descriptor.
    for_each_nd_coord(dstDesc, [&](const NdDims& coord) {
        dst[nd_offset(dstDesc, coord)] =
            from_double<Tout>(log1p_scalar(to_double(src[nd_offset(srcDesc, coord)])));
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_LOG1P_REF_H
