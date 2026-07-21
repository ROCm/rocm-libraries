#ifndef RPP_TEST_COLOR_TWIST_REF_H
#define RPP_TEST_COLOR_TWIST_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/color_hsv.hpp"

namespace rpptest {

// Independent host golden model for rppt_color_twist, derived from the op's intended fused
// definition (hue rotate -> saturation scale -> brightness/contrast affine, all in normalized
// [0,1] continuous space with a single quantization at the end), NOT from the RPP kernel. Used
// as the reference for both backends so kernel bugs surface as diffs.
//
// Per dtype each channel is normalized into [0,1]: U8 v/255, I8 (v+128)/255, F16/F32 as-is.
// The three stages are carried through as doubles (no clamp/quantize between them):
//   Stage 1 (3-channel):    RGB->HSV, H = (H + hueDeg) mod 360, HSV->RGB
//   Stage 2 (3-channel):    RGB->HSV, S = clamp(S * satFactor, 0, 1), HSV->RGB
//   Stage 3 (per channel):  x = brightness * x + contrast/255
// A single quantization closes the pipeline: U8 clamp[0,255](round(x*255)),
// I8 clamp[-128,127](round(x*255)-128), F32/F16 clamp[0,1](x). For a 1-channel (PLN1) image
// hue/saturation are no-ops (S==0, no hue), so only the brightness/contrast affine is applied.

inline double ct_normalize(double v, DType dt) {
    return (dt == DType::U8) ? v / 255.0 : (dt == DType::I8) ? (v + 128.0) / 255.0 : v;
}

inline double ct_quantize(double x, DType dt) {
    switch (dt) {
        case DType::U8: return clampd(std::nearbyint(x * 255.0), 0.0, 255.0);
        case DType::I8: return clampd(std::nearbyint(x * 255.0) - 128.0, -128.0, 127.0);
        default:        return clampd(x, 0.0, 1.0);  // F16/F32
    }
}

// Writes the color_twist result into dst, reading each source pixel at the ROI offset and
// writing packed at the destination origin (matching the region and placement the RPP op uses).
// dst outside the written region is left as the caller initialized it.
template <typename T>
void color_twist_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                           RpptRoiType roiType, double brightness, double contrast, double hueDeg,
                           double satFactor) {
    const double beta = contrast / 255.0;
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds bnd = roi_bounds(roi[n], roiType);
        for (Rpp32u j = 0; j < bnd.h; ++j)
            for (Rpp32u i = 0; i < bnd.w; ++i) {
                const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride;
                const std::size_t srcPix =
                    base + (bnd.y0 + j) * d.strides.hStride + (bnd.x0 + i) * d.strides.wStride;
                const std::size_t dstPix = base + j * d.strides.hStride + i * d.strides.wStride;

                if (d.c == 3) {
                    double rgb[3];
                    for (int c = 0; c < 3; ++c)
                        rgb[c] = ct_normalize(to_double(src[srcPix + c * d.strides.cStride]), dt);
                    hue_rotate_rgb(rgb[0], rgb[1], rgb[2], hueDeg);                    // Stage 1
                    saturation_scale_rgb(rgb[0], rgb[1], rgb[2], satFactor);           // Stage 2
                    for (int c = 0; c < 3; ++c) rgb[c] = brightness * rgb[c] + beta;  // Stage 3
                    for (int c = 0; c < 3; ++c)
                        dst[dstPix + c * d.strides.cStride] = from_double<T>(ct_quantize(rgb[c], dt));
                } else {  // 1-channel: only the Stage 1 affine (hue/saturation are no-ops)
                    const double x = brightness * ct_normalize(to_double(src[srcPix]), dt) + beta;
                    dst[dstPix] = from_double<T>(ct_quantize(x, dt));
                }
            }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_TWIST_REF_H
