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

#ifndef RPP_TEST_CROP_MIRROR_NORMALIZE_REF_H
#define RPP_TEST_CROP_MIRROR_NORMALIZE_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: crop_mirror_normalize

RPP op
  rppt_crop_mirror_normalize   (Image / Geometric augmentation)

Description
  Crops each image to its ROI, optionally mirrors it left-to-right, then
  applies a per-channel affine normalize. The affine form is what the API's
  ranges (offset <= 0, multiplier > 0) imply and what the legacy harness
  encodes as offset = -mean/stdDev, multiplier = 1/stdDev.

  offset and multiplier are per image AND per channel (index n*c + c), which
  is how the legacy harness fills them; the header says size batchSize.

Expression
  dst(j, i) = f( src(y0 + j, x0 + (mirror ? w-1-i : i)) )
  f(v)      = v * multiplier + offset

Per-type form
  offset is in [0,255] intensity units (the brightness / contrast /
  resize_mirror_normalize convention); multiplier is a ratio. I8 is the same
  intensity shifted -128.

    U8      clamp[0,255]   ( round( v * multiplier + offset ) )
    I8      clamp[-128,127]( round( (v + 128) * multiplier + offset ) - 128 )
    F16/F32                ( v * multiplier + offset/255 )

  Floats are deliberately NOT clamped to [0,1] -- a normalized result is
  signed by construction -- matching the resize_mirror_normalize and ND
  normalize goldens.

Notes
  The header states neither the intensity space nor the clamping. The test's
  offset-0 sets are invariant to both, so a diff on the Normalize set is a
  finding, not a reference bug.
*/

// Normalizes one cropped value, taking and returning STORED units.
inline double crop_mirror_normalize_scalar(double v, DType dt, double offset, double multiplier) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v * multiplier + offset), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint((v + 128.0) * multiplier + offset) - 128.0, -128.0, 127.0);
        default:
            return v * multiplier + offset / 255.0;  // F16/F32 -- legitimately signed, not clamped
    }
}

// offsetTensor / multiplierTensor hold one value per channel per image; mirrorTensor one flag per
// image.
template <typename T>
void crop_mirror_normalize_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                     DType dt, const RpptROI* roi, RpptRoiType roiType,
                                     const Rpp32f* offsetTensor, const Rpp32f* multiplierTensor,
                                     const Rpp32u* mirrorTensor) {
    for_each_roi_plane(
        sd, dd, roi, roiType,
        [&](Rpp32u n, const RoiBounds& b, Rpp32u c, std::size_t srcBase, std::size_t dstBase) {
            const std::size_t k = static_cast<std::size_t>(n) * sd.c + c;
            const double offset = offsetTensor[k];
            const double multiplier = multiplierTensor[k];
            const bool mirror = mirrorTensor[n] != 0;
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    const Rpp32u srcCol = b.x0 + (mirror ? (b.w - 1 - i) : i);
                    const double v = to_double(src[plane_index(sd, srcBase, b.y0 + j, srcCol)]);
                    dst[plane_index(dd, dstBase, j, i)] =
                        from_double<T>(crop_mirror_normalize_scalar(v, dt, offset, multiplier));
                }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_CROP_MIRROR_NORMALIZE_REF_H
