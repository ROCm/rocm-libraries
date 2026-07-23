#ifndef RPP_TEST_RESIZE_REF_H
#define RPP_TEST_RESIZE_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/interpolation.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_resize, derived from the op's definition (scale the source
// ROI to fill a per-image destination size), NOT from the RPP kernel. Used as the reference for both
// backends so kernel bugs surface as diffs.
//
// resize is an inverse map: for output pixel (i,j) of a dstW x dstH image, the source coordinate is
// found by scaling with the pixel-CENTER convention (the resize that has no half-texel drift, so an
// exact-integer scale is a verbatim copy and scale 1 is the identity):
//     scaleX = roiWidth / dstW ,  scaleY = roiHeight / dstH
//     srcX = x0 + (i + 0.5) * scaleX - 0.5
//     srcY = y0 + (j + 0.5) * scaleY - 0.5
// The source ROI [x0,x0+roiW) x [y0,y0+roiH) maps onto the whole destination, so every output pixel
// samples INSIDE the source region: the boundary is edge-clamped (replicate), not black-filled -- a
// resize does not introduce border pixels (this is the key difference from the warp/rotate golden,
// which black-fills off-image samples). Sampling (nearest / bilinear) and per-dtype round-to-nearest
// quantization stay independent of the kernel.
//
// Unlike the same-size warps this does not reuse geometric_reference(): resize genuinely has two
// distinct descriptors (source and destination differ in size and stride), so the walk is written
// here while the sampler (interpolation.hpp) stays shared.
//
// NOTE (semantics assumption): the public header documents neither the scale/offset convention nor
// the boundary handling. The pixel-center map and edge-clamped boundary above are the mathematically
// principled resize (drift-free, no synthetic border) and are what the reference holds to; a kernel
// that uses a different convention shows up as a diff, which is a finding, not a reference bug.
template <typename T>
void resize_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                      const RpptROI* roi, RpptRoiType roiType, const RpptImagePatch* dstSizes,
                      RpptInterpolationType interp) {
    const double border = dtype_black(dt);  // clamped away below; present only for sample()'s API
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const Rpp32u dstW = dstSizes[n].width, dstH = dstSizes[n].height;
        const double scaleX = static_cast<double>(b.w) / dstW;
        const double scaleY = static_cast<double>(b.h) / dstH;
        for (Rpp32u c = 0; c < sd.c; ++c) {
            const std::size_t srcBase = static_cast<std::size_t>(n) * sd.strides.nStride +
                                        static_cast<std::size_t>(c) * sd.strides.cStride;
            const std::size_t dstBase = static_cast<std::size_t>(n) * dd.strides.nStride +
                                        static_cast<std::size_t>(c) * dd.strides.cStride;
            for (Rpp32u j = 0; j < dstH; ++j)
                for (Rpp32u i = 0; i < dstW; ++i) {
                    double sx = rx0 + (i + 0.5) * scaleX - 0.5;
                    double sy = ry0 + (j + 0.5) * scaleY - 0.5;
                    // Edge-clamp so the boundary replicates the source edge (resize has no border).
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

#endif  // RPP_TEST_RESIZE_REF_H
