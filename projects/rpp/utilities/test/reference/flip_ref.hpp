#ifndef RPP_TEST_FLIP_REF_H
#define RPP_TEST_FLIP_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_flip.
template <typename T>
void flip_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                    RpptRoiType roiType, Rpp32u horizontal, Rpp32u vertical) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                     static_cast<std::size_t>(c) * d.strides.cStride;
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    const Rpp32u srcRow = b.y0 + (vertical ? (b.h - 1 - j) : j);
                    const Rpp32u srcCol = b.x0 + (horizontal ? (b.w - 1 - i) : i);
                    const std::size_t srcIdx =
                        base + srcRow * d.strides.hStride + srcCol * d.strides.wStride;
                    const std::size_t dstIdx = base + j * d.strides.hStride + i * d.strides.wStride;
                    dst[dstIdx] = src[srcIdx];
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_FLIP_REF_H
