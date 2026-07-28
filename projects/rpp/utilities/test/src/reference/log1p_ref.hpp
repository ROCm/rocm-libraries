#ifndef RPP_TEST_LOG1P_REF_H
#define RPP_TEST_LOG1P_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_log1p. Modelled from the operation's definition and the public
// API header, NOT from the kernel; computed once on the host and used as the reference for
// both the HOST and HIP backends.
//
// Semantics, per the header: "Computes Log1p i.e (log(1 + x)) of the input", which "Uses
// Absolute of input for log1p computation to avoid undefined result":
//
//     dst = log1p(|src|) = log(1 + |src|)
//
// applied element-wise over the whole densely packed tensor (a single source, no broadcast,
// no op params). Everything is computed in double; no clamping is applied, since the
// documented output dtype (f32) is floating point.
//
// Taking the absolute value first makes the argument of log1p >= 0, so the result is defined
// for every representable input -- including zero, where log1p(0) = 0 exactly. Unlike
// rppt_log (undefined at zero, which the header dodges with an unspecified nextafter) there
// is no ambiguous edge case here, so the standard input fill is used as-is, zeros included.
//
// std::log1p is used rather than std::log(1 + x) so the golden keeps full precision for small
// |x|, where 1 + x loses the low bits of x.

inline double log1p_scalar(double v) { return std::log1p(std::fabs(v)); }

template <typename Tin, typename Tout>
void log1p_reference(const Tin* src, Tout* dst, const RpptGenericDesc& srcDesc,
                     const RpptGenericDesc& dstDesc) {
    // Same logical shape, differing dtype and possibly stride padding: address each through its
    // own descriptor rather than walking either buffer flat.
    for_each_nd_coord(dstDesc, [&](const std::vector<Rpp32u>& coord) {
        dst[nd_offset(dstDesc, coord)] =
            from_double<Tout>(log1p_scalar(to_double(src[nd_offset(srcDesc, coord)])));
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_LOG1P_REF_H
