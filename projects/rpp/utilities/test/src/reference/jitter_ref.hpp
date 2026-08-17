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

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_jitter, derived from the op's public API doc and the
// standard "pixel jitter" augmentation definition (DALI/Albumentations-style), NOT from the RPP
// kernel:
//
// For every output pixel at absolute source-space coordinate (sx, sy) inside an image's ROI, a
// per-PIXEL random integer offset (dx, dy) is drawn, each independently uniform over [-r, r] with
// r = kernelSize / 2 (integer division; kernelSize is odd, e.g. kernelSize=3 -> r=1, a 3x3
// neighbourhood). The SAME offset is used for every channel of that pixel, so a multi-channel
// pixel is displaced as a whole rather than splitting its channels apart. The output pixel is the
// source pixel at (sx+dx, sy+dy), independently clamped on each axis into the ROI bounds
// ([roi.x0, roi.x0+roi.w-1], [roi.y0, roi.y0+roi.h-1]) if the offset would leave the ROI. `seed`
// seeds the RNG for the whole call: same seed -> bit-identical output, different seed -> a
// different output somewhere.
//
// This header only encodes the RNG-FREE corner of that definition: kernelSize = 1 gives r = 0, so
// the offset window collapses to {0} for both axes regardless of any RNG draw -- the op is forced
// to identity (output pixel == input pixel) for every element, independent of seed. That is the
// only case a static golden buffer can match bit-for-bit without reimplementing the kernel's PRNG
// (which would make the golden dependent on the kernel, defeating its purpose). The general
// kernelSize > 1 case is instead checked at runtime against the real kernel output via structural
// invariants (membership of the output pixel in its legal candidate window; same-seed repeatability;
// different-seed divergence) -- see jitter_test.cpp, not this file.

// Clamps an integer coordinate into [lo, hi]. Shared by the identity case (trivially, since the
// window is {0}) and the runtime membership-invariant check in the test file.
inline int clamp_coord(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Writes the kernelSize=1 identity result into dst, reading the source at the ROI offset and
// writing packed at the destination origin (matching the region and placement the RPP op uses).
// dst outside the written region is left as the caller initialized it.
template <typename T>
void jitter_identity_reference(const T* src, T* dst, const RpptDesc& d, const RpptROI* roi,
                               RpptRoiType roiType) {
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t srcIdx, std::size_t dstIdx) {
                        dst[dstIdx] = src[srcIdx];
                    });
}

}  // namespace rpptest

#endif  // RPP_TEST_JITTER_REF_H
