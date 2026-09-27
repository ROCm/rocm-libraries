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

#ifndef RPP_TEST_CROP_AND_PATCH_REF_H
#define RPP_TEST_CROP_AND_PATCH_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: crop_and_patch

RPP op
  rppt_crop_and_patch   (Image / Geometric augmentation)

Description
  Copies the 2nd image, then overlays the rectangular crop taken from the 1st
  image at the patch co-ordinates. Crop size == patch size (no resize), the
  documented unambiguous case.

  dstRoi sizes the output, which is written packed at the destination origin
  while src2 is read at the dstRoi offset. crop is absolute in src1 and gives
  both the origin and the extent of the overlaid rectangle; patch gives its
  origin only, in the packed output frame -- neither backend reads the patch
  extent.

Expression
  For packed output coordinate (r, c), 0 <= r < dstRoi.h, 0 <= c < dstRoi.w:

    inPatch      = patch.y0 <= r < patch.y0 + crop.h
                && patch.x0 <= c < patch.x0 + crop.w
    dst(r, c)    = inPatch ? src1(crop.y0 + r - patch.y0, crop.x0 + c - patch.x0)
                           : src2(dstRoi.y0 + r,          dstRoi.x0 + c)

  Reads that fall outside [0,h) x [0,w) are skipped rather than clamped.

Per-type form
  No arithmetic, rounding, or clamping is performed -- every output element is
  copied verbatim from one of the two sources -- so the result is bit-exact
  for U8, I8, F16 and F32 alike.
*/
template <typename T>
void crop_and_patch_reference(const T* src1, const T* src2, T* dst, const RpptDesc& d,
                              const RpptROI* dstRoi, const RpptROI* cropRoi,
                              const RpptROI* patchRoi, RpptRoiType roiType) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds db = roi_bounds(dstRoi[n], roiType);
        const RoiBounds cb = roi_bounds(cropRoi[n], roiType);
        const RoiBounds pb = roi_bounds(patchRoi[n], roiType);
        for (Rpp32u r = 0; r < db.h; ++r)
            for (Rpp32u col = 0; col < db.w; ++col) {
                const bool inPatch =
                    (r >= pb.y0 && r < pb.y0 + cb.h) && (col >= pb.x0 && col < pb.x0 + cb.w);
                const T* src = inPatch ? src1 : src2;
                const Rpp32u sy = inPatch ? cb.y0 + (r - pb.y0) : db.y0 + r;
                const Rpp32u sx = inPatch ? cb.x0 + (col - pb.x0) : db.x0 + col;
                if (sy >= d.h || sx >= d.w) continue;
                for (Rpp32u c = 0; c < d.c; ++c) {
                    const std::size_t base = plane_base(d, n, c);
                    dst[plane_index(d, base, r, col)] = src[plane_index(d, base, sy, sx)];
                }
            }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_CROP_AND_PATCH_REF_H
