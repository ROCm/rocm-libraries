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
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_spatter, used for both HOST and HIP.
//
// The blend itself is modeled from the op's public API doc comment: a per-texel alpha composite of
// the source over a user-defined spatter color, in normalized [0,1] intensity space,
//   out = src * (1 - mask) + color * mask
// with `mask` sampled from the op's 1920x1080 spatter texture. The channel mapping follows the
// header's "RGB values to use for the spatter augmentation": channel 0 takes R, 1 takes G, 2 takes
// B. A 1-channel image has no channel to map onto, so the color collapses to its mean intensity.
//
// The texture itself is not modeled here, and cannot be: it is a baked constant pair in the private
// header src/include/tensor/spatter_mask.hpp, which is not installed with the library, and the
// window the kernel samples from is drawn per image from an unseedable std::random_device-seeded
// mt19937 on both backends. There is therefore no pointwise golden for this op. What is modeled
// instead are the parts of the blend that hold whatever texel the RNG lands on -- the documented
// channel mapping, and the fixed point where the source already equals the spatter colour.
//
// Per dtype: U8/I8 normalize to [0,1] on load and quantize back with round-to-nearest on store
// (to_unit / from_unit); F16/F32 are already unit intensities and only clamp to [0,1].

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

// mask + maskInv == 1 everywhere, so a source already equal to the spatter color is a fixed point
// of the blend whatever window the RNG lands on. This is the only exact golden that holds at an
// arbitrary image size.
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
