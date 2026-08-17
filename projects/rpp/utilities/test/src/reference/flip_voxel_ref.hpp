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

#ifndef RPP_TEST_FLIP_VOXEL_REF_H
#define RPP_TEST_FLIP_VOXEL_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/voxel_tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_flip_voxel, derived from the op's definition (a
// mask-controlled mirror of the ROI box about its horizontal/vertical/depth axis) and the public
// API header, NOT from the RPP kernel. Used as the reference for both backends so kernel bugs
// surface as diffs.
//
// Flip is a pure permutation of source elements -- no arithmetic, so no rounding or clamping and
// every dtype is bit-exact (the op is documented f32 -> f32 and u8 -> u8 only). Following the
// voxel-domain pointwise convention (for_each_voxel_roi_io), the source is read at the ROI offset
// and the output written packed at the destination origin, so output voxel (z, y, x) comes from
// source (z0 + [depth ? d-1-z : z], y0 + [vertical ? h-1-y : y], x0 + [horizontal ? w-1-x : x]).
template <typename T>
void flip_voxel_reference(const T* src, T* dst, const RpptGenericDesc& desc, const RpptROI3D* roi,
                          Roi3D roiType, const Rpp32u* horizontalTensor,
                          const Rpp32u* verticalTensor, const Rpp32u* depthTensor) {
    for_each_voxel_roi_plane(
        desc, roi, roiType, [&](Rpp32u n, Rpp32u, const VoxelBox& b, std::size_t base) {
            const Rpp32u horizontal = horizontalTensor[n];
            const Rpp32u vertical = verticalTensor[n];
            const Rpp32u depth = depthTensor[n];
            for (Rpp32u z = 0; z < b.d; ++z) {
                const Rpp32u srcZ = b.z0 + (depth ? (b.d - 1 - z) : z);
                for (Rpp32u y = 0; y < b.h; ++y) {
                    const Rpp32u srcY = b.y0 + (vertical ? (b.h - 1 - y) : y);
                    for (Rpp32u x = 0; x < b.w; ++x) {
                        const Rpp32u srcX = b.x0 + (horizontal ? (b.w - 1 - x) : x);
                        dst[voxel_plane_index(desc, base, z, y, x)] =
                            src[voxel_plane_index(desc, base, srcZ, srcY, srcX)];
                    }
                }
            }
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_FLIP_VOXEL_REF_H
