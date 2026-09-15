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

#include "framework/generic_tensor_setup.hpp"
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"

namespace rpptest {

/*
Reference model: log1p

RPP op
  rppt_log1p   (ND / Arithmetic)

Description
  Element-wise log(1 + x) over the whole densely packed tensor (a single
  source, no broadcast, no op params). The header specifies that the absolute
  value of the input is used, so negative inputs are folded before the log.
  Everything is computed in double; no clamping is applied, since the
  documented output type (f32) is floating point.

Expression
  dst = log1p(|src|) = log(1 + |src|)

Notes
  Taking the absolute value first makes the argument >= 0, so the result is
  defined for every representable input -- including zero, where log1p(0) = 0
  exactly. Unlike rppt_log (undefined at zero, which the header dodges with an
  unspecified nextafter) there is no ambiguous edge case here, so the standard
  input fill is used as-is, zeros included.

  std::log1p is used rather than std::log(1 + x) so the golden keeps full
  precision for small |x|, where 1 + x loses the low bits of x.
*/

inline double log1p_scalar(double v) {
    return std::log1p(std::fabs(v));
}

template <typename Tin, typename Tout>
void log1p_reference(const Tin* src, Tout* dst, const RpptGenericDesc& srcDesc,
                     const RpptGenericDesc& dstDesc) {
    // Same logical shape, differing dtype and possibly stride padding: address each through its
    // own descriptor rather than walking either buffer flat.
    for_each_nd_coord(dstDesc, [&](const NdDims& coord) {
        dst[nd_offset(dstDesc, coord)] =
            from_double<Tout>(log1p_scalar(to_double(src[nd_offset(srcDesc, coord)])));
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_LOG1P_REF_H
