#ifndef RPP_TEST_SATURATION_REF_H
#define RPP_TEST_SATURATION_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_hsv.hpp"

namespace rpptest {

// Independent host golden model for rppt_saturation, derived from the op's definition (a
// saturation scale in HSV space: RGB -> HSV, S = clamp(S * factor, 0, 1), HSV -> RGB), NOT
// from the RPP kernel. Used as the reference for both backends so kernel bugs surface as diffs.
//
// saturation is a 3-channel (RGB, ch0=R/ch1=G/ch2=B) op. The scale is computed on normalized
// [0,1] RGB; per dtype the pixel is normalized into [0,1], scaled, then written back:
//   U8  : v/255       -> scale -> clamp[0,255]  ( round(v'*255) )
//   I8  : (v+128)/255 -> scale -> clamp[-128,127]( round(v'*255) - 128 )
//   F32 : v           -> scale -> clamp[0,1]
//   F16 : same as F32, stored as half
// Grey pixels (R==G==B) have S==0 and are invariant to the scale.

// Writes the saturation result into dst, reading each source pixel's three channels at the ROI
// offset and writing packed at the destination origin (matching the region and placement the
// RPP op uses). dst outside the written region is left as the caller initialized it.
template <typename T>
void saturation_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                          RpptRoiType roiType, double factor) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds bnd = roi_bounds(roi[n], roiType);
        for (Rpp32u j = 0; j < bnd.h; ++j)
            for (Rpp32u i = 0; i < bnd.w; ++i) {
                const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride;
                const std::size_t srcPix =
                    base + (bnd.y0 + j) * d.strides.hStride + (bnd.x0 + i) * d.strides.wStride;
                const std::size_t dstPix = base + j * d.strides.hStride + i * d.strides.wStride;

                double rgb[3];
                for (int c = 0; c < 3; ++c) {
                    const double v = to_double(src[srcPix + c * d.strides.cStride]);
                    rgb[c] = (dt == DType::U8) ? v / 255.0
                             : (dt == DType::I8) ? (v + 128.0) / 255.0
                                                 : v;  // F16/F32 already [0,1]
                }
                saturation_scale_rgb(rgb[0], rgb[1], rgb[2], factor);
                for (int c = 0; c < 3; ++c) {
                    double out;
                    switch (dt) {
                        case DType::U8:
                            out = clampd(std::nearbyint(rgb[c] * 255.0), 0.0, 255.0);
                            break;
                        case DType::I8:
                            out = clampd(std::nearbyint(rgb[c] * 255.0) - 128.0, -128.0, 127.0);
                            break;
                        default:
                            out = clampd(rgb[c], 0.0, 1.0);
                            break;
                    }
                    dst[dstPix + c * d.strides.cStride] = from_double<T>(out);
                }
            }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_SATURATION_REF_H
