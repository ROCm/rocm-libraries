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

#ifndef RPP_TEST_LOG_REF_H
#define RPP_TEST_LOG_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_log. Modelled from the operation's definition and the public
// API header, NOT from the kernel; computed once on the host and used as the reference for
// both the HOST and HIP backends.
//
// Semantics, per the header: "Computes Log to base e(natural log) of the input", which
// "Uses Absolute of input for log computation":
//
//     dst = log(|src|)
//
// applied element-wise over the whole densely packed tensor (a single source, no broadcast,
// no op params). Everything is computed in double; no clamping is applied, since a log
// result is signed and the documented output dtypes (f32, f16) are floating point.
//
// The header also states that a zero input is replaced via nextafter() "to avoid undefined
// result", but it pins neither the precision nor the direction of that nextafter:
// nextafterf(0.f, 1.f) = 1.4e-45 gives log = -103.28, while the double form 4.9e-324 gives
// -744.44. The expected value is therefore ambiguous, so the tests deliberately feed no zero
// inputs rather than encoding a guess, and this model does not special-case zero.

inline double log_scalar(double v) { return std::log(std::fabs(v)); }

// src and dst have the same logical shape; only the dtype (and possibly the stride padding)
// differs, so each is addressed through its own descriptor.
template <typename Tin, typename Tout>
void log_reference(const Tin* src, Tout* dst, const RpptGenericDesc& srcDesc,
                   const RpptGenericDesc& dstDesc) {
    for_each_nd_coord(dstDesc, [&](const NdDims& coord) {
        dst[nd_offset(dstDesc, coord)] =
            from_double<Tout>(log_scalar(to_double(src[nd_offset(srcDesc, coord)])));
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_LOG_REF_H
