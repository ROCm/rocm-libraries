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

#ifndef RPP_TEST_JITTER_REF_H
#define RPP_TEST_JITTER_REF_H

#include <rpp/rpp.h>

#include "framework/tensor_setup.hpp"

namespace rpptest {

/*
Reference model: jitter   (RNG-free corner only)

RPP op
  rppt_jitter   (Image / Effects augmentation)

Description
  Standard pixel-jitter augmentation (DALI/Albumentations-style). For every
  output pixel at absolute source-space coordinate (sx, sy) inside an image's
  ROI, a per-PIXEL random integer offset (dx, dy) is drawn, each independently
  uniform over [-r, r] with r = kernelSize / 2 (integer division; kernelSize
  is odd, so kernelSize = 3 gives r = 1, a 3x3 neighbourhood).

  The SAME offset is used for every channel of that pixel, so a multi-channel
  pixel is displaced as a whole rather than splitting its channels apart.
  `seed` seeds the RNG for the whole call: same seed -> bit-identical output,
  different seed -> a different output somewhere.

Expression
  dst(sx, sy) = src( clamp(sx + dx), clamp(sy + dy) )

  clamped independently on each axis into the ROI bounds
  [roi.x0, roi.x0+roi.w-1] x [roi.y0, roi.y0+roi.h-1] if the offset would
  leave the ROI.

Scope
  This header encodes only the RNG-FREE corner: kernelSize = 1 gives r = 0, so
  the offset window collapses to {0} on both axes regardless of any RNG draw
  and the op is forced to identity for every element, independent of seed.
  That is the only case a static golden can match bit-for-bit without
  reimplementing the kernel's PRNG, which would make the golden depend on the
  kernel and defeat its purpose.

  The general kernelSize > 1 case is checked at runtime against the real
  kernel output via structural invariants -- membership of the output pixel in
  its legal candidate window, same-seed repeatability, different-seed
  divergence -- in jitter_test.cpp, not here.
*/

// Clamps an integer coordinate into [lo, hi]. Shared by the identity case (trivially, since the
// window is {0}) and the runtime membership-invariant check in the test file.
inline int clamp_coord(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Writes the kernelSize = 1 identity result into dst.
template <typename T>
void jitter_identity_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd,
                               const RpptROI* roi, RpptRoiType roiType) {
    for_each_roi_io(sd, dd, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_JITTER_REF_H
