#ifndef RPP_TEST_CROP_AND_PATCH_REF_H
#define RPP_TEST_CROP_AND_PATCH_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_crop_and_patch, derived from the op's definition (copy
// the 2nd image, then overlay the rectangular crop taken from the 1st image at the patch
// co-ordinates), NOT from the RPP kernel. Used as the reference for both backends so kernel bugs
// surface as diffs.
//
// crop_and_patch does no arithmetic, rounding, or clamping: every output element is copied verbatim
// from one of the two sources, so the result is bit-exact for U8/I8/F16/F32 alike. Crop size ==
// patch size (no resize), the documented unambiguous case. Coordinates are absolute (image origin
// (0,0)); pixels whose source or destination fall outside [0,d.h) x [0,d.w) are skipped.
template <typename T>
void crop_and_patch_reference(const T* src1, const T* src2, T* dst, const RpptDesc& d,
                              const RpptROI* /*dstRoi*/, const RpptROI* cropRoi,
                              const RpptROI* patchRoi, RpptRoiType roiType) {
    // (1) Output is a copy of the 2nd image over the whole frame.
    for (Rpp32u n = 0; n < d.n; ++n)
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (Rpp32u y = 0; y < d.h; ++y)
                for (Rpp32u x = 0; x < d.w; ++x) {
                    const std::size_t idx = base + y * d.strides.hStride + x * d.strides.wStride;
                    dst[idx] = src2[idx];
                }
        }

    // (2) Overlay the src1 crop at the patch location, in-bounds pixels only.
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds cb = roi_bounds(cropRoi[n], roiType);
        const RoiBounds pb = roi_bounds(patchRoi[n], roiType);
        for (Rpp32u r = 0; r < pb.h; ++r)
            for (Rpp32u col = 0; col < pb.w; ++col) {
                const Rpp32u sy = cb.y0 + r, sx = cb.x0 + col;
                const Rpp32u dy = pb.y0 + r, dx = pb.x0 + col;
                if (sy >= d.h || sx >= d.w || dy >= d.h || dx >= d.w) continue;
                for (Rpp32u c = 0; c < d.c; ++c) {
                    const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                             static_cast<std::size_t>(c) * d.strides.cStride;
                    const std::size_t srcIdx = base + sy * d.strides.hStride + sx * d.strides.wStride;
                    const std::size_t dstIdx = base + dy * d.strides.hStride + dx * d.strides.wStride;
                    dst[dstIdx] = src1[srcIdx];
                }
            }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_CROP_AND_PATCH_REF_H
