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

#ifndef RPP_TEST_PIXELATE_REF_H
#define RPP_TEST_PIXELATE_REF_H

#include <rpp/rpp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/resize_ref.hpp"

namespace rpptest {

// Host golden model for rppt_pixelate. Used as the reference for both the HOST and HIP backends.
//
// The public API header documents only that pixelationPercentage "controls how much pixelation is
// applied" (0 to 100) -- it pins neither the block size nor the aggregation. The op's definition is
// instead stated at the dispatch level: pixelation is a two-step resize, the image scaled DOWN with
// bilinear interpolation and then back UP with nearest-neighbour, which is what turns the lost
// detail into blocks. That definition is what this model encodes:
//
//   interW = roiW * (100 - p) / 100      interH = roiH * (100 - p) / 100     (truncated)
//   step 1: bilinear resize of the source ROI down to interW x interH, packed at the origin
//   step 2: nearest-neighbour resize of that intermediate back up to roiW x roiH
//
// Only the composition above is taken from the operator; the resize itself is the suite's own
// resize_reference (drift-free pixel-CENTER inverse map, edge-clamped, per-dtype round-to-nearest
// quantization -- see reference/resize_ref.hpp), NOT the kernel's. A resize defect therefore
// surfaces here as a diff rather than being reproduced, which is the point: pixelate inherits
// whatever the resize paths get wrong.
//
// Step 2's source is the intermediate image, which step 1 wrote packed at the destination ORIGIN
// (the placement rule every resize in the suite follows). Its ROI origin is therefore (0,0)
// regardless of where the source ROI sat -- the intermediate is not offset.
//
// The intermediate is held in the source dtype at the source descriptor's strides, matching the
// scratch buffer the op requires (n * strides.nStride elements).

// The downsampled extent of one ROI edge. Truncating, matching the Rpp32u image-patch field the
// extent is carried in. Floored at 1: a zero extent has no pixels to sample and the second resize
// would divide by it (no test configuration reaches this -- it needs p at or near 100).
inline Rpp32u pixelate_intermediate_extent(Rpp32u extent, double pixelationPercentage) {
    const double scaled =
        (static_cast<double>(extent) * (100.0 - pixelationPercentage)) / 100.0;
    return std::max(1u, static_cast<Rpp32u>(scaled));
}

template <typename T>
void pixelate_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                        RpptRoiType roiType, double pixelationPercentage) {
    std::vector<RpptImagePatch> interSizes(d.n), roiSizes(d.n);
    std::vector<RpptROI> interRoi(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        interSizes[n] = {pixelate_intermediate_extent(b.w, pixelationPercentage),
                         pixelate_intermediate_extent(b.h, pixelationPercentage)};
        roiSizes[n] = {b.w, b.h};
        interRoi[n].xywhROI = {{0, 0}, static_cast<int>(interSizes[n].width),
                               static_cast<int>(interSizes[n].height)};
    }

    std::vector<T> inter(element_count(d), from_double<T>(dtype_black(dt)));
    resize_reference<T>(src, d, inter.data(), d, dt, roi, roiType, interSizes.data(), BILINEAR);
    resize_reference<T>(inter.data(), d, dst, d, dt, interRoi.data(), XYWH, roiSizes.data(),
                        NEAREST_NEIGHBOR);
}

}  // namespace rpptest

#endif  // RPP_TEST_PIXELATE_REF_H
