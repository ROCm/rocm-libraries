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

#ifndef RPP_TEST_RESIZE_MIRROR_NORMALIZE_REF_H
#define RPP_TEST_RESIZE_MIRROR_NORMALIZE_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"
#include "framework/intensity.hpp"

namespace rpptest {

/*
Reference model: resize_mirror_normalize

RPP op
  rppt_resize_mirror_normalize   (Image / Geometric augmentation)

Description
  Resizes the source ROI to the per-image destination size, optionally mirrors
  it left-to-right, then normalizes per channel. Resize and mirror use
  resize_driver() (framework/geometric.hpp); the normalize runs before the
  single per-type quantization.

  mean and stdDev are per image and per channel, so channel c of image n reads
  index n*channels + c.

Expression
  dst = (pixel - mean) / stdDev

Per-type form
  mean is in [0,255] intensity units for every type; stdDev is a ratio and is
  not rescaled. I8 is the same intensity shifted -128.

    U8      clamp[0,255]   ( round( (v - mean) / stdDev ) )
    I8      clamp[-128,127]( round( ((v + 128) - mean) / stdDev ) - 128 )
    F16/F32                ( (v - mean/255) / stdDev )

  Integer types round to nearest and saturate. Float types are not clamped,
  since a normalized result is signed; this is why the store below is not
  quantizing_store().

Notes
  The header states neither the intensity space of mean/stdDev nor the
  clamping. The NormalizeIdentity set (mean 0, stdDev 1) is the identity under
  any reading. stdDev == 0 is undefined and is not exercised.
*/

// Stores a normalized value: integers round to nearest and saturate, floats pass through unclamped.
inline double resize_mirror_normalize_store(double v, DType dt) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(v), -128.0, 127.0);
        default:
            return v;  // F16/F32
    }
}

// Normalizes one sampled value, taking and returning STORED units.
inline double resize_mirror_normalize_scalar(double v, DType dt, double mean, double stdDev) {
    switch (dt) {
        case DType::U8:
            return (v - mean) / stdDev;
        case DType::I8:
            return ((v + 128.0) - mean) / stdDev - 128.0;
        case DType::F16:
        case DType::F32:
            return (v - mean / 255.0) / stdDev;
        default:
            return v;
    }
}

// meanTensor / stdDevTensor hold one value per channel per image; mirrorTensor one flag per image.
template <typename T>
void resize_mirror_normalize_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                       DType dt, const RpptROI* roi, RpptRoiType roiType,
                                       const RpptImagePatch* dstSizes, const Rpp32f* meanTensor,
                                       const Rpp32f* stdDevTensor, const Rpp32u* mirrorTensor,
                                       RpptInterpolationType interp) {
    const Rpp32u channels = sd.c;
    resize_driver<T>(src, sd, dst, dd, dt, roi, roiType, dstSizes, mirrorTensor, interp,
                     [&](double v, Rpp32u n, Rpp32u c) {
                         const std::size_t k = static_cast<std::size_t>(n) * channels + c;
                         const double norm =
                             resize_mirror_normalize_scalar(v, dt, meanTensor[k], stdDevTensor[k]);
                         return from_double<T>(resize_mirror_normalize_store(norm, dt));
                     });
}

}  // namespace rpptest

#endif  // RPP_TEST_RESIZE_MIRROR_NORMALIZE_REF_H
