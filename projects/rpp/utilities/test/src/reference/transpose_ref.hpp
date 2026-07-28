#ifndef RPP_TEST_TRANSPOSE_REF_H
#define RPP_TEST_TRANSPOSE_REF_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_transpose. Modelled from the operation's definition and the
// public API header ("an input-permutation based transpose on a generic ND Tensor"), NOT
// from the kernel; computed once on the host and used as the reference for both the HOST
// and HIP backends.
//
// Semantics: permTensor is a permutation of the nDim per-sample axes -- the batch axis is
// neither permuted nor part of perm. Output axis k reads source axis perm[k]:
//
//     dst[b][c_0, ..., c_{n-1}] == src[b][s_0, ..., s_{n-1}]   where s[perm[k]] = c[k]
//
// so the output extent along axis k is the source ROI extent along axis perm[k].
//
// This is pure data movement: no arithmetic, no dtype conversion, no clamping. Every output
// element is a bit-exact copy of exactly one source element and the coordinate map is a
// bijection, so the reference reproduces the operation exactly at every dtype -- any
// difference from the kernel is a real defect, never a rounding artifact.

// The destination extents implied by a source shape and a permutation, batch axis first.
// Both tensors are densely packed, so this plus make_generic_descriptor fixes the layout.
inline NdDims transpose_dst_dims(const NdDims& srcDims, const std::vector<Rpp32u>& perm) {
    NdDims dstDims(srcDims.size());
    dstDims[0] = srcDims[0];
    for (std::size_t k = 0; k < perm.size(); ++k) dstDims[k + 1] = srcDims[perm[k] + 1];
    return dstDims;
}

// Both tensors are addressed through their own strides, so either may be dense or padded.
template <typename T>
void transpose_reference(const T* src, T* dst, const RpptGenericDesc& srcDesc,
                         const RpptGenericDesc& dstDesc, const Rpp32u* perm) {
    const std::size_t rank = dstDesc.numDims;  // counts the batch axis
    for_each_nd_coord(dstDesc, [&](const std::vector<Rpp32u>& coord) {
        // Destination axis a > 0 reads source axis perm[a-1] + 1; axis 0 (batch) maps to itself.
        std::size_t srcIdx = 0;
        for (std::size_t a = 0; a < rank; ++a) {
            const std::size_t srcAxis = (a == 0) ? 0 : static_cast<std::size_t>(perm[a - 1]) + 1;
            srcIdx += static_cast<std::size_t>(coord[a]) * srcDesc.strides[srcAxis];
        }
        dst[nd_offset(dstDesc, coord)] = src[srcIdx];
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_TRANSPOSE_REF_H
