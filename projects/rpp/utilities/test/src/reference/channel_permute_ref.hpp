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

#ifndef RPP_TEST_CHANNEL_PERMUTE_REF_H
#define RPP_TEST_CHANNEL_PERMUTE_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: channel_permute

RPP op
  rppt_channel_permute  (Image / Data exchange)

Description
  Per-image reordering of the 3 channels. The permutation tensor holds n
  contiguous triples (perm[n*3 + 0..2], each in 0..2). A rotation like {2,0,1}
  distinguishes this convention from its inverse, which would be
  source channel c -> output perm[c].

Expression
  dst(x, y, i) = src(x, y, perm[i])

Per-type form
  A pure data exchange: only whole channel values move, with no arithmetic,
  rounding, or clamping, so the result is bit-exact for U8, I8, F16 and F32
  alike.
*/
template <typename T>
void channel_permute_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                               const Rpp32u* perm, const RpptROI* roi, RpptRoiType roiType) {
    for_each_roi_pixel(sd, dd, roi, roiType,
                       [&](Rpp32u n, Rpp32u, Rpp32u, std::size_t srcPix, std::size_t dstPix) {
                           for (Rpp32u c = 0; c < sd.c; ++c) {
                               const Rpp32u srcC = perm[n * 3 + c];
                               dst[channel_index(dd, dstPix, c)] =
                                   src[channel_index(sd, srcPix, srcC)];
                           }
                       });
}

}  // namespace rpptest

#endif  // RPP_TEST_CHANNEL_PERMUTE_REF_H
