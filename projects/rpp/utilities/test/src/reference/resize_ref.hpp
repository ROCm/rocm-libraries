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

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"

namespace rpptest {

/*
Reference model: resize

RPP op
  rppt_resize   (Image / Geometric augmentation)

Description
  Scales the source ROI to fill a per-image destination size. Each output pixel
  inverse-maps to the source with the pixel-centre convention, so an integer
  scale is a verbatim copy and scale 1 is the identity. The boundary is
  edge-clamped (replicate), so no border pixels are introduced.

  The walk is resize_driver() (framework/geometric.hpp), shared with
  resize_crop_mirror and resize_mirror_normalize.

Expression
  scaleX = roiWidth / dstW,  scaleY = roiHeight / dstH
  srcX   = x0 + (i + 0.5) * scaleX - 0.5
  srcY   = y0 + (j + 0.5) * scaleY - 0.5

Notes
  The public header documents neither the scale/offset convention nor the
  boundary handling; the model uses the pixel-centre map and edge clamp above.
*/
template <typename T>
void resize_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                      const RpptROI* roi, RpptRoiType roiType, const RpptImagePatch* dstSizes,
                      RpptInterpolationType interp) {
    resize_driver<T>(src, sd, dst, dd, dt, roi, roiType, dstSizes, /*mirror=*/nullptr, interp,
                     quantizing_store<T>(dt));
}

}  // namespace rpptest

#endif  // RPP_TEST_RESIZE_REF_H
