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

// Every tensor is addressed by logical coordinate through its own strides (nd_offset), so operands
// and output may be dense or padded independently -- the descriptor, not the buffer layout, defines
// where a coordinate lives.
template <typename T>
void concat_reference(const T* src1, const T* src2, T* dst, const RpptGenericDesc& out,
                      const RpptGenericDesc& s1, const RpptGenericDesc& s2, Rpp32u axis) {
    const std::size_t descAxis = axis + 1;   // concat axis in descriptor coordinates
    const Rpp32u split = s1.dims[descAxis];  // where src1's slab ends along the axis

    NdDims srcCoord(out.numDims);
    for_each_nd_coord(out, [&](const NdDims& coord) {
        srcCoord = coord;
        const bool fromFirst = coord[descAxis] < split;
        if (!fromFirst) srcCoord[descAxis] -= split;
        dst[nd_offset(out, coord)] =
            fromFirst ? src1[nd_offset(s1, srcCoord)] : src2[nd_offset(s2, srcCoord)];
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_CONCAT_REF_H
