#ifndef RPP_TEST_SLICE_REF_H
#define RPP_TEST_SLICE_REF_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for the ND slice op (rppt_slice). Modelled from the operation's definition
// and the public API header, NOT from the kernel; computed once on the host and used as the
// reference for both the HOST and HIP backends.
//
// Semantics: "selecting a region of interest from the source tensor and copying it to the
// destination tensor". anchorTensor holds the per-axis start index of the slice and shapeTensor
// its per-axis length (both batchSize * nDim, batch axis excluded); roiTensor (batchSize * nDim
// * 2 -- per-axis starts followed by per-axis lengths) delimits the valid source region.
//
// The slice is written densely at the destination origin: destination coordinate c takes source
// coordinate s[a] = anchor[a] + c[a]. When every s[a] lies inside the ROI
// (roiStart[a] <= s[a] < roiStart[a] + roiLength[a]) the element is copied verbatim -- slice
// moves values and never computes on them, so the result is bit-exact for every dtype. When some
// s[a] falls outside, the element has no source and takes fillValue.
//
// Two things the header leaves open, and how they are handled rather than guessed:
//   - The out-of-ROI result with enablePadding false is unspecified, so the caller is required to
//     keep the slice inside the ROI in that case; the fill branch is then unreachable rather than
//     an encoded guess. enablePadding is consequently not a parameter here.
//   - fillValue's type is unspecified (the API takes a void pointer). The caller passes 0, which
//     is the same value under every interpretation, so the golden only needs the scalar.
//
// The walk covers shape[] per sample, which is exactly the destination extents the caller builds
// (dstDims = {batch, shape...}), so every destination element is written.
template <typename T>
void slice_reference(const T* src, T* dst, const RpptGenericDesc& srcDesc,
                     const RpptGenericDesc& dstDesc, const Rpp32s* anchorTensor,
                     const Rpp32s* shapeTensor, const Rpp32u* roiTensor, double fillValue) {
    const Rpp32u batch = srcDesc.dims[0];
    const Rpp32u nDim = static_cast<Rpp32u>(srcDesc.numDims) - 1;
    std::vector<Rpp32u> coord(nDim, 0);

    for (Rpp32u n = 0; n < batch; ++n) {
        const Rpp32s* anchor = anchorTensor + static_cast<std::size_t>(n) * nDim;
        const Rpp32s* shape = shapeTensor + static_cast<std::size_t>(n) * nDim;
        const Rpp32u* roi = roiTensor + static_cast<std::size_t>(n) * 2 * nDim;

        std::size_t sliceCount = 1;
        for (Rpp32u a = 0; a < nDim; ++a) sliceCount *= static_cast<std::size_t>(shape[a]);

        for (std::size_t linear = 0; linear < sliceCount; ++linear) {
            // The slice's own row-major coordinates (trailing axis fastest).
            std::size_t rem = linear;
            for (Rpp32u a = nDim; a-- > 0;) {
                coord[a] = static_cast<Rpp32u>(rem % static_cast<std::size_t>(shape[a]));
                rem /= static_cast<std::size_t>(shape[a]);
            }

            std::size_t dstIdx = static_cast<std::size_t>(n) * dstDesc.strides[0];
            for (Rpp32u a = 0; a < nDim; ++a)
                dstIdx += static_cast<std::size_t>(coord[a]) * dstDesc.strides[a + 1];

            // The source index is only formed once every axis is known to be inside the ROI:
            // an out-of-ROI coordinate can be negative and has no address to speak of.
            bool inRoi = true;
            std::size_t srcIdx = static_cast<std::size_t>(n) * srcDesc.strides[0];
            for (Rpp32u a = 0; a < nDim; ++a) {
                const Rpp32s s = anchor[a] + static_cast<Rpp32s>(coord[a]);
                const Rpp32s lo = static_cast<Rpp32s>(roi[a]);
                const Rpp32s hi = lo + static_cast<Rpp32s>(roi[nDim + a]);
                if (s < lo || s >= hi) {
                    inRoi = false;
                    break;
                }
                srcIdx += static_cast<std::size_t>(s) * srcDesc.strides[a + 1];
            }

            dst[dstIdx] = inRoi ? src[srcIdx] : from_double<T>(fillValue);
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_SLICE_REF_H
