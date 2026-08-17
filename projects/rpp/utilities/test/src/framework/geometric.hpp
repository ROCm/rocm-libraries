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

// Scaling driver shared by the ops that map the source ROI onto a per-image destination size
// (resize, resize_crop_mirror, resize_mirror_normalize). It owns the one definition of that resize:
// the drift-free pixel-CENTER inverse map
//     scaleX = roiWidth / dstW ,  srcX = x0 + (i + 0.5) * scaleX - 0.5
// edge-clamped into the ROI so the boundary replicates the source edge -- a resize introduces no
// border, which is what separates it from the warp driver above. An exact-integer scale is then a
// verbatim copy and scale 1 is the identity.
//
// mirror (may be null) flips the destination left-to-right, equivalent to sampling output column
// (dstW-1-i). store(v, n, c) turns the sampled value into the stored element: resize and
// resize_crop_mirror use quantizing_store() below, while resize_mirror_normalize folds its
// normalize in so the pipeline quantizes exactly once (and can keep its own float-range rule).
template <typename T, typename Store>
void resize_driver(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                   const RpptROI* roi, RpptRoiType roiType, const RpptImagePatch* dstSizes,
                   const Rpp32u* mirror, RpptInterpolationType interp, Store store) {
    const double border = dtype_black(dt);  // clamped away below; present only for sample()'s API
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const Rpp32u dstW = dstSizes[n].width, dstH = dstSizes[n].height;
        const double scaleX = static_cast<double>(b.w) / dstW;
        const double scaleY = static_cast<double>(b.h) / dstH;
        const bool mir = mirror != nullptr && mirror[n] != 0;
        for (Rpp32u c = 0; c < sd.c; ++c) {
            const std::size_t srcBase = plane_base(sd, n, c);
            const std::size_t dstBase = plane_base(dd, n, c);
            for (Rpp32u j = 0; j < dstH; ++j)
                for (Rpp32u i = 0; i < dstW; ++i) {
                    const Rpp32u ii = mir ? (dstW - 1 - i) : i;
                    const double sx = clampd(rx0 + (ii + 0.5) * scaleX - 0.5, rx0, rx1 - 1);
                    const double sy = clampd(ry0 + (j + 0.5) * scaleY - 0.5, ry0, ry1 - 1);
                    const double v =
                        sample(src, sd, srcBase, sx, sy, rx0, ry0, rx1, ry1, interp, border);
                    dst[plane_index(dd, dstBase, j, i)] = store(v, n, c);
                }
        }
    }
}

// The plain store: round/clamp the sampled value into the dtype and convert. What an op that only
// moves pixels around (resize, resize_crop_mirror) needs.
template <typename T>
inline auto quantizing_store(DType dt) {
    return [dt](double v, Rpp32u, Rpp32u) { return from_double<T>(quantize_stored(v, dt)); };
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
    for_each_roi_plane(d, roi, roiType, [&](Rpp32u n, const RoiBounds& b, Rpp32u,
                                            std::size_t base) {
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const OutSize os = outSize[n];
        for (Rpp32u j = 0; j < os.h; ++j)
            for (Rpp32u i = 0; i < os.w; ++i) {
                double sx, sy;
                invMap(n, static_cast<double>(i), static_cast<double>(j), sx, sy);
                const double v = sample(src, d, base, sx, sy, rx0, ry0, rx1, ry1, interp, border);
                dst[plane_index(d, base, j, i)] = from_double<T>(quantize_stored(v, dt));
            }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_GEOMETRIC_H
