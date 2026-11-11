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

#ifndef RPP_TEST_RESIZE_CROP_MIRROR_REF_H
#define RPP_TEST_RESIZE_CROP_MIRROR_REF_H

#include <rpp/rpp.h>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"

namespace rpptest {

/*
Reference model: resize_crop_mirror

RPP op
  rppt_resize_crop_mirror   (Image / Geometric augmentation)

Description
  Resizes the source ROI to fill a per-image destination size, then optionally
  mirrors it horizontally. The walk is resize_driver() (framework/geometric.hpp),
  the same pixel-centre, edge-clamped map as resize.

Expression
  scaleX = roiWidth / dstW,  scaleY = roiHeight / dstH
  ii     = mirror ? (dstW-1-i) : i
  srcX   = x0 + (ii + 0.5) * scaleX - 0.5   (edge-clamped into the ROI)
  srcY   = y0 + (j  + 0.5) * scaleY - 0.5   (edge-clamped into the ROI)

Notes
  The public header documents neither the scale/offset convention nor the
  boundary handling; the model uses the pixel-centre map and edge clamp above.
*/
template <typename T>
void resize_crop_mirror_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                                  DType dt, const RpptROI* roi, RpptRoiType roiType,
                                  const RpptImagePatch* dstSizes, const Rpp32u* mirror,
                                  RpptInterpolationType interp) {
    resize_driver<T>(src, sd, dst, dd, dt, roi, roiType, dstSizes, mirror, interp,
                     quantizing_store<T>(dt));
}

}  // namespace rpptest

#endif  // RPP_TEST_RESIZE_CROP_MIRROR_REF_H
