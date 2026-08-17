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

#ifndef RPP_TEST_RESIZE_REF_H
#define RPP_TEST_RESIZE_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_resize, derived from the op's definition (scale the source
// ROI to fill a per-image destination size), NOT from the RPP kernel. Used as the reference for both
// backends so kernel bugs surface as diffs.
//
// resize is an inverse map: for output pixel (i,j) of a dstW x dstH image, the source coordinate is
// found by scaling with the pixel-CENTER convention (the resize that has no half-texel drift, so an
// exact-integer scale is a verbatim copy and scale 1 is the identity):
//     scaleX = roiWidth / dstW ,  scaleY = roiHeight / dstH
//     srcX = x0 + (i + 0.5) * scaleX - 0.5
//     srcY = y0 + (j + 0.5) * scaleY - 0.5
// The source ROI [x0,x0+roiW) x [y0,y0+roiH) maps onto the whole destination, so every output pixel
// samples INSIDE the source region: the boundary is edge-clamped (replicate), not black-filled -- a
// resize does not introduce border pixels (this is the key difference from the warp/rotate golden,
// which black-fills off-image samples). Sampling (nearest / bilinear) and per-dtype round-to-nearest
// quantization stay independent of the kernel.
//
// The walk itself is resize_driver() (framework/geometric.hpp), shared with resize_crop_mirror and
// resize_mirror_normalize, which owns the pixel-center map, the edge clamp and the quantization;
// resize adds neither a mirror nor a post-transform.
//
// NOTE (semantics assumption): the public header documents neither the scale/offset convention nor
// the boundary handling. The pixel-center map and edge-clamped boundary above are the mathematically
// principled resize (drift-free, no synthetic border) and are what the reference holds to; a kernel
// that uses a different convention shows up as a diff, which is a finding, not a reference bug.
template <typename T>
void resize_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                      const RpptROI* roi, RpptRoiType roiType, const RpptImagePatch* dstSizes,
                      RpptInterpolationType interp) {
    resize_driver<T>(src, sd, dst, dd, dt, roi, roiType, dstSizes, /*mirror=*/nullptr, interp,
                     quantizing_store<T>(dt));
}

}  // namespace rpptest

#endif  // RPP_TEST_RESIZE_REF_H
