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

#ifndef RPP_TEST_SLICE_REF_H
#define RPP_TEST_SLICE_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: slice

RPP op
  rppt_slice   (ND / Geometric augmentation)

Description
  Selects a region of interest from the source tensor and copies it to the
  destination. anchorTensor holds the per-axis start index of the slice and
  shapeTensor its per-axis length (both batchSize * nDim, batch axis
  excluded); roiTensor (batchSize * nDim * 2 -- per-axis starts followed by
  per-axis lengths) delimits the valid source region.

  The slice is written at the destination origin. Elements whose source
  coordinate lies outside the ROI on any axis take fillValue.

  Both tensors are addressed by logical coordinate through their own strides
  (nd_offset), so either may be dense or padded.

Expression
  s[a] = anchor[a] + c[a]
  dst[c] = in-ROI ? src[s] : fillValue
  in-ROI = roiStart[a] <= s[a] < roiStart[a] + roiLength[a]  for every axis a

Per-type form
  Slice moves values and never computes on them, so the result is bit-exact
  for every type.

Notes
  - The out-of-ROI result with enablePadding false is unspecified, so callers
    must keep the slice inside the ROI in that case; enablePadding is not a
    parameter here.
  - fillValue's type is unspecified (void pointer). Callers pass 0, which is
    the same under every interpretation.
*/
template <typename T>
void slice_reference(const T* src, T* dst, const RpptGenericDesc& srcDesc,
                     const RpptGenericDesc& dstDesc, const Rpp32s* anchorTensor,
                     const Rpp32s* shapeTensor, const Rpp32u* roiTensor, double fillValue) {
    const Rpp32u batch = srcDesc.dims[0];
    const Rpp32u nDim = static_cast<Rpp32u>(srcDesc.numDims) - 1;
    NdDims srcCoord(nDim + 1), dstCoord(nDim + 1);

    for (Rpp32u n = 0; n < batch; ++n) {
        const Rpp32s* anchor = anchorTensor + static_cast<std::size_t>(n) * nDim;
        const Rpp32s* shape = shapeTensor + static_cast<std::size_t>(n) * nDim;
        const Rpp32u* roi = roiTensor + static_cast<std::size_t>(n) * 2 * nDim;
        srcCoord[0] = dstCoord[0] = n;

        const NdDims sliceExtents(shape, shape + nDim);
        for_each_coord(sliceExtents, [&](const NdDims& coord) {
            // Out-of-ROI coordinates can be negative, so they are floored at 0; srcCoord is only
            // read when the element is in the ROI.
            bool inRoi = true;
            for (Rpp32u a = 0; a < nDim; ++a) {
                dstCoord[a + 1] = coord[a];
                const Rpp32s s = anchor[a] + static_cast<Rpp32s>(coord[a]);
                const Rpp32s lo = static_cast<Rpp32s>(roi[a]);
                if (s < lo || s >= lo + static_cast<Rpp32s>(roi[nDim + a])) inRoi = false;
                srcCoord[a + 1] = static_cast<Rpp32u>(s < 0 ? 0 : s);
            }
            dst[nd_offset(dstDesc, dstCoord)] =
                inRoi ? src[nd_offset(srcDesc, srcCoord)] : from_double<T>(fillValue);
        });
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_SLICE_REF_H
