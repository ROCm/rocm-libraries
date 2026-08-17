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
inline NdDims transpose_dst_dims(const NdDims& srcDims, const std::vector<Rpp32u>& perm) {
    NdDims dstDims(srcDims.size());
    dstDims[0] = srcDims[0];
    for (std::size_t k = 0; k < perm.size(); ++k) dstDims[k + 1] = srcDims[perm[k] + 1];
    return dstDims;
}

// Both tensors are addressed by logical coordinate through their own strides (nd_offset), so
// either may be dense or padded.
template <typename T>
void transpose_reference(const T* src, T* dst, const RpptGenericDesc& srcDesc,
                         const RpptGenericDesc& dstDesc, const Rpp32u* perm) {
    const std::size_t rank = dstDesc.numDims;  // counts the batch axis
    NdDims srcCoord(rank);
    for_each_nd_coord(dstDesc, [&](const NdDims& coord) {
        // Destination axis a > 0 reads source axis perm[a-1] + 1; axis 0 (batch) maps to itself.
        srcCoord[0] = coord[0];
        for (std::size_t a = 1; a < rank; ++a) srcCoord[perm[a - 1] + 1] = coord[a];
        dst[nd_offset(dstDesc, coord)] = src[nd_offset(srcDesc, srcCoord)];
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_TRANSPOSE_REF_H
