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

#ifndef RPP_TEST_COLOR_TO_GREYSCALE_REF_H
#define RPP_TEST_COLOR_TO_GREYSCALE_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: color_to_greyscale

RPP op
  rppt_color_to_greyscale   (Image / Data exchange)

Description
  Collapses a 3-channel colour image to a single luma plane. Each output
  element is the weighted sum of that pixel's colour channels, so the op is
  not pointwise in the usual sense: it reads a whole pixel and writes one
  element, and the source and destination descriptors differ (srcDesc c = 3
  in PKD3/PLN3, dstDesc c = 1 planar).

  srcSubpixelLayout names the source channel order: RGBtype => (R,G,B),
  BGRtype => (B,G,R), i.e. R and B swap and G is unchanged.

Expression
  The public header does not name the coefficients, so the standard BT.601
  luma weights are used -- the same ones histogram_equalize applies.

  grey(x, y) = 0.299 R + 0.587 G + 0.114 B

Per-type form
  The weights sum to exactly 1, so the luma is a convex combination and
  commutes with the I8 intensity offset:
  0.299(R+128) + 0.587(G+128) + 0.114(B+128) = luma + 128. The dot product is
  therefore taken directly on stored values for every type and quantized
  once, with no detour through [0,1] intensity space.
*/
inline double color_to_greyscale_scalar(double c0, double c1, double c2, DType dt,
                                        RpptSubpixelLayout subpixel) {
    // srcSubpixelLayout names the source channel order: RGBtype => (R,G,B), BGRtype => (B,G,R),
    // i.e. R and B swap and G is unchanged.
    const double r = (subpixel == BGRtype) ? c2 : c0;
    const double g = c1;
    const double b = (subpixel == BGRtype) ? c0 : c2;
    return quantize_stored(0.299 * r + 0.587 * g + 0.114 * b, dt);
}

// The source walk supplies the channel-0 pixel offset and the channels are reached with
// channel_index(); the destination offset is computed against dstDesc.
template <typename T>
void color_to_greyscale_reference(const T* src, const RpptDesc& srcDesc, T* dst,
                                  const RpptDesc& dstDesc, DType dt, const RpptROI* roi,
                                  RpptRoiType roiType, RpptSubpixelLayout subpixel) {
    for_each_roi_pixel(
        srcDesc, roi, roiType, [&](Rpp32u n, Rpp32u j, Rpp32u i, std::size_t srcPix, std::size_t) {
            const double c0 = to_double(src[channel_index(srcDesc, srcPix, 0)]);
            const double c1 = to_double(src[channel_index(srcDesc, srcPix, 1)]);
            const double c2 = to_double(src[channel_index(srcDesc, srcPix, 2)]);
            const std::size_t dstIdx = plane_index(dstDesc, plane_base(dstDesc, n, 0), j, i);
            dst[dstIdx] = from_double<T>(color_to_greyscale_scalar(c0, c1, c2, dt, subpixel));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_TO_GREYSCALE_REF_H
