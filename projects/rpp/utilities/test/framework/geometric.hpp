#ifndef RPP_TEST_GEOMETRIC_H
#define RPP_TEST_GEOMETRIC_H

#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/interpolation.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Generic inverse-map driver shared by every geometric ("move pixels around") golden model. The
// op supplies only its output->source coordinate map; the sampling, interpolation, border fill,
// and per-dtype quantization live here, so each op reference is just its transform + this call.

// Destination region size (packed at the destination origin) per image. For same-size ops
// (flip/rotate/warp/remap) this is the source ROI size; resize overrides it with dstImgSizes.
struct OutSize {
    Rpp32u w, h;
};

// Output size == source ROI size, the common case.
inline std::vector<OutSize> roi_out_sizes(const RpptDesc& d, const RpptROI* roi, RpptRoiType type) {
    std::vector<OutSize> out(d.n);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        out[n] = {b.w, b.h};
    }
    return out;
}

// For each output pixel (i,j) of image n's destination region, invMap yields the source coordinate
// (srcX,srcY) in the full-image absolute frame (texel (0,0) = image origin); the source is sampled
// there with `interp`, samples outside the ROI rectangle filled with the dtype's black, and the
// quantized result written packed at the destination origin. dst outside the written region is left
// as the caller initialized it.
//
// The source is sampled in the ABSOLUTE image frame (not ROI-local): RPP's warp maps the origin-
// based output index straight to a source coordinate and ignores the ROI offset (the ROI only sets
// the output size and the valid-source rectangle [x0,x0+w) x [y0,y0+h), outside which the sample is
// black). Ops whose source region is placed differently add the offset inside their own invMap;
// this keeps the driver's sampling model uniform.
//
// invMap signature: void(Rpp32u n, double outX, double outY, double& srcX, double& srcY)
//   outX/outY are origin-based output pixel indices; the op owns any pixel-center (+0.5) convention.
template <typename T, typename InvMap>
void geometric_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                         RpptRoiType roiType, const std::vector<OutSize>& outSize,
                         RpptInterpolationType interp, InvMap invMap) {
    const double border = dtype_black(dt);
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const OutSize os = outSize[n];
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t imgBase = static_cast<std::size_t>(n) * d.strides.nStride +
                                        static_cast<std::size_t>(c) * d.strides.cStride;
            for (Rpp32u j = 0; j < os.h; ++j)
                for (Rpp32u i = 0; i < os.w; ++i) {
                    double sx, sy;
                    invMap(n, static_cast<double>(i), static_cast<double>(j), sx, sy);
                    const double v =
                        sample(src, d, imgBase, sx, sy, rx0, ry0, rx1, ry1, interp, border);
                    const std::size_t dstIdx = imgBase +
                                               static_cast<std::size_t>(j) * d.strides.hStride +
                                               static_cast<std::size_t>(i) * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(quantize_stored(v, dt));
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_GEOMETRIC_H
