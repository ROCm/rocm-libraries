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

template <typename Tin, typename Tout>
void log_reference(const Tin* src, Tout* dst, const RpptGenericDesc& srcDesc,
                   const RpptGenericDesc& dstDesc) {
    (void)dstDesc;  // src and dst are the same packed shape; only the dtype differs
    const std::size_t count = generic_element_count(srcDesc);
    for (std::size_t i = 0; i < count; ++i)
        dst[i] = from_double<Tout>(log_scalar(to_double(src[i])));
}

}  // namespace rpptest

#endif  // RPP_TEST_LOG_REF_H
