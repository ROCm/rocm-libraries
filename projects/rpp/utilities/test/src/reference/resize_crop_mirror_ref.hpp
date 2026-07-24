#ifndef RPP_TEST_RESIZE_CROP_MIRROR_REF_H
#define RPP_TEST_RESIZE_CROP_MIRROR_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/interpolation.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_resize_crop_mirror, derived from the op's definition (resize
// the source crop -- the ROI -- to fill a per-image destination size, then optionally mirror it
// horizontally), NOT from the RPP kernel. Used as the reference for both backends.
//
// The resize is the same drift-free, edge-clamped inverse map as the resize golden (pixel-CENTER
// convention; the source ROI [x0,x0+roiW) x [y0,y0+roiH) maps onto the whole destination so every
// output pixel samples INSIDE the crop -- no synthetic border). The optional horizontal mirror flips
// the destination left-to-right, which is equivalent to sampling output column (dstW-1-i):
//     scaleX = roiWidth / dstW ,  scaleY = roiHeight / dstH
//     ii   = mirror ? (dstW-1-i) : i
//     srcX = x0 + (ii + 0.5) * scaleX - 0.5   (edge-clamped into the ROI)
//     srcY = y0 + (j  + 0.5) * scaleY - 0.5   (edge-clamped into the ROI)
//
// NOTE (semantics assumption): the public header documents neither the scale/offset convention nor
// the boundary handling; the pixel-center map + edge-clamp above are the principled resize (matching
// the resize golden), and the mirror is a plain horizontal flip. A kernel using a different
// convention shows up as a diff -- a finding, not a reference bug.
template <typename T>
void resize_crop_mirror_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                  DType dt, const RpptROI* roi, RpptRoiType roiType,
                                  const RpptImagePatch* dstSizes, const Rpp32u* mirror,
                                  RpptInterpolationType interp) {
    const double border = dtype_black(dt);  // clamped away below; present only for sample()'s API
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const Rpp32u dstW = dstSizes[n].width, dstH = dstSizes[n].height;
        const double scaleX = static_cast<double>(b.w) / dstW;
        const double scaleY = static_cast<double>(b.h) / dstH;
        const bool mir = mirror[n] != 0;
        for (Rpp32u c = 0; c < sd.c; ++c) {
            const std::size_t srcBase = static_cast<std::size_t>(n) * sd.strides.nStride +
                                        static_cast<std::size_t>(c) * sd.strides.cStride;
            const std::size_t dstBase = static_cast<std::size_t>(n) * dd.strides.nStride +
                                        static_cast<std::size_t>(c) * dd.strides.cStride;
            for (Rpp32u j = 0; j < dstH; ++j)
                for (Rpp32u i = 0; i < dstW; ++i) {
                    const Rpp32u ii = mir ? (dstW - 1 - i) : i;
                    double sx = rx0 + (ii + 0.5) * scaleX - 0.5;
                    double sy = ry0 + (j + 0.5) * scaleY - 0.5;
                    // Edge-clamp so the boundary replicates the crop edge (resize has no border).
                    sx = clampd(sx, rx0, rx1 - 1);
                    sy = clampd(sy, ry0, ry1 - 1);
                    const double v =
                        sample(src, sd, srcBase, sx, sy, rx0, ry0, rx1, ry1, interp, border);
                    const std::size_t dstIdx = dstBase +
                                               static_cast<std::size_t>(j) * dd.strides.hStride +
                                               static_cast<std::size_t>(i) * dd.strides.wStride;
                    dst[dstIdx] = from_double<T>(quantize_stored(v, dt));
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_RESIZE_CROP_MIRROR_REF_H
