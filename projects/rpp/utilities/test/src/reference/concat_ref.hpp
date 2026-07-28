#ifndef RPP_TEST_CONCAT_REF_H
#define RPP_TEST_CONCAT_REF_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for the ND concatenation op (rppt_concat). Modelled from the operation's
// definition and the public API header, NOT from the kernel; computed once on the host and used
// as the reference for both the HOST and HIP backends.
//
// Semantics: "concatenates two 2D, 3D or ND tensors along a specified axis ... all dimensions
// except the concatenation axis must match". With A = src1's extent along the axis:
//   - the output extent along the axis is src1's + src2's, every other extent is unchanged;
//   - dst[c] = src1[c]                       when c[axis] <  A,
//     dst[c] = src2[c with c[axis] -= A]     when c[axis] >= A.
// The axis is given 0-based over the per-sample axes (the batch axis is excluded), so it indexes
// the descriptors' dims/strides at axis + 1.
//
// Concat only relocates elements -- no arithmetic, no dtype conversion -- so the golden copies
// the stored value verbatim (no double round-trip, which would perturb F16) and the comparison
// is bit-exact for every dtype.

// Every tensor is addressed through its own strides, so operands and output may be dense or
// padded independently -- the descriptor, not the buffer layout, defines where a coordinate lives.
template <typename T>
void concat_reference(const T* src1, const T* src2, T* dst, const RpptGenericDesc& out,
                      const RpptGenericDesc& s1, const RpptGenericDesc& s2, Rpp32u axis) {
    const std::size_t rank = out.numDims;   // includes the batch axis
    const std::size_t descAxis = axis + 1;  // concat axis in descriptor coordinates
    const Rpp32u split = s1.dims[descAxis];  // where src1's slab ends along the axis

    for_each_nd_coord(out, [&](const std::vector<Rpp32u>& coord) {
        const bool fromFirst = coord[descAxis] < split;
        const RpptGenericDesc& src = fromFirst ? s1 : s2;
        std::size_t idx = 0;
        for (std::size_t a = 0; a < rank; ++a) {
            const Rpp32u c = (a == descAxis && !fromFirst) ? coord[a] - split : coord[a];
            idx += static_cast<std::size_t>(c) * src.strides[a];
        }
        dst[nd_offset(out, coord)] = fromFirst ? src1[idx] : src2[idx];
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CONCAT_REF_H
