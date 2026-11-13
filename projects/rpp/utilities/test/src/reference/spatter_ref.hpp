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

#ifndef RPP_TEST_SPATTER_REF_H
#define RPP_TEST_SPATTER_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: spatter   (fixed point only)

RPP op
  rppt_spatter   (Image / Effects augmentation)

Description
  Per-texel alpha composite of the source over a user-defined spatter colour,
  in normalized [0,1] intensity space, with `mask` sampled from the op's
  1920x1080 spatter texture. The channel mapping follows the header's "RGB
  values to use for the spatter augmentation": channel 0 takes R, 1 takes G, 2
  takes B. A 1-channel image has no channel to map onto, so the colour
  collapses to its mean intensity.

Expression
  dst = src * (1 - mask) + color * mask

Per-type form
  U8/I8 normalize to [0,1] on load and quantize back with round-to-nearest on
  store (to_unit / from_unit); F16/F32 are already unit intensities and only
  clamp to [0,1].

Scope
  The texture is not modelled: it lives in the non-installed header
  src/include/tensor/spatter_mask.hpp, and both backends pick the sampling
  window per image from an mt19937 seeded by std::random_device. Only the
  channel mapping and the fixed point where the source already equals the
  spatter colour are modelled, since those hold for any texel.
*/

// Documented spatter intensity for channel `c` of a `channels`-channel image, in [0,1].
inline double spatter_color_unit(RpptRGB color, Rpp32u channels, Rpp32u c) {
    if (channels == 1)
        return (static_cast<double>(color.R) + static_cast<double>(color.G) +
                static_cast<double>(color.B)) /
               3.0 / 255.0;
    const Rpp8u component = (c == 0) ? color.R : (c == 1) ? color.G : color.B;
    return static_cast<double>(component) / 255.0;
}

// The stored-dtype pixel a fully opaque (mask == 1) spatter texel produces.
inline double spatter_color_stored(RpptRGB color, Rpp32u channels, Rpp32u c, DType dt) {
    return from_unit(spatter_color_unit(color, channels, c), dt);
}

template <typename T>
void spatter_identity_reference(T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                                RpptRoiType roiType, RpptRGB color) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u c, Rpp32u, Rpp32u, std::size_t, std::size_t dstIdx) {
                        dst[dstIdx] = from_double<T>(spatter_color_stored(color, d.c, c, dt));
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_SPATTER_REF_H
