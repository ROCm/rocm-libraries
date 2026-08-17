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
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_resize_mirror_normalize, derived from the op's definition
// ("an image resize, an optional mirror and/or normalize"), NOT from the RPP kernel. Used as the
// reference for both backends so kernel bugs surface as diffs.
//
// Three stages, in the order the name gives them:
//   1. resize the source ROI onto the per-image destination size,
//   2. optionally mirror the destination left-to-right,
//   3. normalize:  out = (pixel - mean) / stdDev,  per channel.
// Mirror is a coordinate permutation and normalize is pointwise, so 2 and 3 commute; only the
// resize has to come first. Stages 1-2 are resize_driver() (framework/geometric.hpp) -- the same
// drift-free pixel-CENTER, edge-clamped map resize and resize_crop_mirror use, so this golden
// cannot disagree with those two about what a resize is -- and the normalize is the driver's
// post-transform, which runs BEFORE the single per-dtype quantization. Quantizing once (rather than
// after the resize and again after the normalize) is what the fused op's name implies.
//
// mean/stdDev are per image AND per channel (batchSize * 3 for RGB, batchSize for greyscale), so
// channel c of image n reads index n*c_count + c.
//
// Per-dtype behavior. mean is a pixel-intensity offset expressed in [0,255] units -- the same
// convention this suite already models for brightness's beta and contrast's center, and the one the
// legacy harness corroborates by passing mean = {60, 80, 100} unchanged for U8, I8, F16 and F32
// alike. stdDev is a ratio and is not rescaled. I8 pixels are the same intensities shifted by -128:
//   U8      : clamp[0,255]  ( round( (v - mean) / stdDev ) )
//   I8      : clamp[-128,127]( round( ((v + 128) - mean) / stdDev ) - 128 )
//   F16/F32 : clamp[0,1]    ( (v - mean/255) / stdDev )
// The integer dtypes round to nearest and saturate, because that is all their storage can hold. The
// FLOAT dtypes are deliberately NOT clamped to [0,1]: a normalized result is signed by construction
// (that is the point of subtracting a mean), and clamping it away would make the op useless for the
// preprocessing it exists to do. This matches the ND rppt_normalize golden in this suite, which
// documents the same reasoning, and it is why the store below is not the shared quantizing_store().
//
// NOTE (semantics assumption): the header states neither the intensity space of mean/stdDev nor the
// clamping. The NormalizeIdentity parameter set below (mean 0, stdDev 1) is invariant to that
// choice -- it is the identity normalize under every reading -- so it validates the resize, mirror,
// placement and quantization independently of the assumption. A diff on the other sets is a
// finding, not a reference bug. stdDev == 0 is undefined and is not exercised.

// Stores a normalized value: integers round to nearest and saturate, floats pass through unclamped.
inline double resize_mirror_normalize_store(double v, DType dt) {
    switch (dt) {
        case DType::U8: return clampd(std::nearbyint(v), 0.0, 255.0);
        case DType::I8: return clampd(std::nearbyint(v), -128.0, 127.0);
        default:        return v;  // F16/F32 -- a normalized result is legitimately signed
    }
}

// Normalizes one sampled value, taking and returning STORED units.
inline double resize_mirror_normalize_scalar(double v, DType dt, double mean, double stdDev) {
    switch (dt) {
        case DType::U8: return (v - mean) / stdDev;
        case DType::I8: return ((v + 128.0) - mean) / stdDev - 128.0;
        case DType::F16:
        case DType::F32: return (v - mean / 255.0) / stdDev;
        default: return v;
    }
}

// meanTensor / stdDevTensor hold one value per channel per image; mirrorTensor one flag per image.
template <typename T>
void resize_mirror_normalize_reference(const T* src, const RpptDesc& sd, T* dst,
                                       const RpptDesc& dd, DType dt, const RpptROI* roi,
                                       RpptRoiType roiType, const RpptImagePatch* dstSizes,
                                       const Rpp32f* meanTensor, const Rpp32f* stdDevTensor,
                                       const Rpp32u* mirrorTensor, RpptInterpolationType interp) {
    const Rpp32u channels = sd.c;
    resize_driver<T>(src, sd, dst, dd, dt, roi, roiType, dstSizes, mirrorTensor, interp,
                     [&](double v, Rpp32u n, Rpp32u c) {
                         const std::size_t k = static_cast<std::size_t>(n) * channels + c;
                         const double norm = resize_mirror_normalize_scalar(
                             v, dt, meanTensor[k], stdDevTensor[k]);
                         return from_double<T>(resize_mirror_normalize_store(norm, dt));
                     });
}

}  // namespace rpptest

#endif  // RPP_TEST_RESIZE_MIRROR_NORMALIZE_REF_H
