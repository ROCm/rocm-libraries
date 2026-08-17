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

#ifndef RPP_TEST_VOXEL_TENSOR_SETUP_H
#define RPP_TEST_VOXEL_TENSOR_SETUP_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Construction, traversal and comparison for the Voxel domain: a 5D RpptGenericDesc (NCDHW or
// NDHWC) plus a per-sample RpptROI3D box. Descriptors are built with generic_tensor_setup.hpp's
// GenericDescriptor (dense strides, as the legacy voxel harness uses), so only the 3D layout
// mapping, the ROI3D box and an in-box traversal are added here.

inline Rpp32u voxel_channels(VoxelLayout l) { return l == VoxelLayout::NCDHW1 ? 1 : 3; }

inline bool voxel_is_packed(VoxelLayout l) { return l == VoxelLayout::NDHWC3; }

inline bool voxel_is_packed(const RpptGenericDesc& d) { return d.layout == RpptLayout::NDHWC; }

inline RpptLayout to_rpp_layout_3d(VoxelLayout l) {
    return voxel_is_packed(l) ? RpptLayout::NDHWC : RpptLayout::NCDHW;
}

// Which descriptor axis each logical axis occupies: {n, d, h, w, c} packed, {n, c, d, h, w}
// planar. The single statement of that order -- extents (voxel_dims) and addressing
// (voxel_plane_base / voxel_plane_index) are both derived from it, so they cannot disagree.
struct VoxelAxes {
    std::size_t c, z, y, x;
};

inline VoxelAxes voxel_axes(bool packed) {
    return packed ? VoxelAxes{4, 1, 2, 3} : VoxelAxes{1, 2, 3, 4};
}

inline NdDims voxel_dims(const VoxelSize& s, VoxelLayout l) {
    const VoxelAxes a = voxel_axes(voxel_is_packed(l));
    NdDims dims(5, s.n);  // dims[0] is the batch axis
    dims[a.c] = voxel_channels(l);
    dims[a.z] = s.d;
    dims[a.y] = s.h;
    dims[a.x] = s.w;
    return dims;
}

inline Rpp32u voxel_channels(const RpptGenericDesc& d) {
    return d.dims[voxel_axes(voxel_is_packed(d)).c];
}

// ---- element addressing ----------------------------------------------------
//
// Addressed through the descriptor's own strides, never by walking the buffer flat, so the layout
// is the descriptor's business and a padded convention would need no change here. The image
// domain's plane_base / plane_index pair, one dimension up.

// Origin of sample n's channel-c volume.
inline std::size_t voxel_plane_base(const RpptGenericDesc& d, Rpp32u n, Rpp32u c) {
    return static_cast<std::size_t>(n) * d.strides[0] +
           static_cast<std::size_t>(c) * d.strides[voxel_axes(voxel_is_packed(d)).c];
}

// Voxel (z, y, x) of the volume whose origin is `base`.
inline std::size_t voxel_plane_index(const RpptGenericDesc& d, std::size_t base, Rpp32u z, Rpp32u y,
                                     Rpp32u x) {
    const VoxelAxes a = voxel_axes(voxel_is_packed(d));
    return base + static_cast<std::size_t>(z) * d.strides[a.z] +
           static_cast<std::size_t>(y) * d.strides[a.y] +
           static_cast<std::size_t>(x) * d.strides[a.x];
}

// ---- ROI3D -----------------------------------------------------------------

// x is the width axis, y the height axis, z the depth axis (RpptRoiXyzwhd's own naming).
struct VoxelBox {
    Rpp32u x0, y0, z0, w, h, d;
};

// LTFRBB is read inclusive of both corners, matching the image domain's LTRB convention
// (roi_bounds() in tensor_setup.hpp).
inline VoxelBox voxel_box(const RpptROI3D& r, Roi3D type) {
    if (type == Roi3D::XYZWHD)
        return {static_cast<Rpp32u>(r.xyzwhdROI.xyz.x), static_cast<Rpp32u>(r.xyzwhdROI.xyz.y),
                static_cast<Rpp32u>(r.xyzwhdROI.xyz.z), static_cast<Rpp32u>(r.xyzwhdROI.roiWidth),
                static_cast<Rpp32u>(r.xyzwhdROI.roiHeight),
                static_cast<Rpp32u>(r.xyzwhdROI.roiDepth)};
    return {static_cast<Rpp32u>(r.ltfrbbROI.ltf.x),
            static_cast<Rpp32u>(r.ltfrbbROI.ltf.y),
            static_cast<Rpp32u>(r.ltfrbbROI.ltf.z),
            static_cast<Rpp32u>(r.ltfrbbROI.rbb.x - r.ltfrbbROI.ltf.x + 1),
            static_cast<Rpp32u>(r.ltfrbbROI.rbb.y - r.ltfrbbROI.ltf.y + 1),
            static_cast<Rpp32u>(r.ltfrbbROI.rbb.z - r.ltfrbbROI.ltf.z + 1)};
}

inline RpptRoi3DType to_rpp_roi3d_type(Roi3D t) {
    return t == Roi3D::XYZWHD ? RpptRoi3DType::XYZWHD : RpptRoi3DType::LTFRBB;
}

// Per-sample ROI3D: the whole volume, or a centered half-extent box in every axis.
inline std::vector<RpptROI3D> make_voxel_roi(const VoxelSize& s, Roi mode, Roi3D type) {
    // One axis of the box; a half-extent is kept at least 1, so a thin depth axis still yields a
    // usable box.
    auto span = [mode](Rpp32u extent) {
        if (mode == Roi::Full) return std::pair<Rpp32u, Rpp32u>{0, extent};
        return std::pair<Rpp32u, Rpp32u>{extent / 4, extent / 2 ? extent / 2 : 1};
    };
    const auto sx = span(s.w), sy = span(s.h), sz = span(s.d);

    std::vector<RpptROI3D> roi(s.n);
    for (Rpp32u i = 0; i < s.n; ++i) {
        RpptROI3D r{};
        if (type == Roi3D::XYZWHD) {
            r.xyzwhdROI.xyz.x = static_cast<int>(sx.first);
            r.xyzwhdROI.xyz.y = static_cast<int>(sy.first);
            r.xyzwhdROI.xyz.z = static_cast<int>(sz.first);
            r.xyzwhdROI.roiWidth = static_cast<int>(sx.second);
            r.xyzwhdROI.roiHeight = static_cast<int>(sy.second);
            r.xyzwhdROI.roiDepth = static_cast<int>(sz.second);
        } else {
            r.ltfrbbROI.ltf.x = static_cast<int>(sx.first);
            r.ltfrbbROI.ltf.y = static_cast<int>(sy.first);
            r.ltfrbbROI.ltf.z = static_cast<int>(sz.first);
            r.ltfrbbROI.rbb.x = static_cast<int>(sx.first + sx.second - 1);
            r.ltfrbbROI.rbb.y = static_cast<int>(sy.first + sy.second - 1);
            r.ltfrbbROI.rbb.z = static_cast<int>(sz.first + sz.second - 1);
        }
        roi[i] = r;
    }
    return roi;
}

// ---- traversal --------------------------------------------------------------

// Invokes fn(n, c, box, base) once per sample-channel volume, box being that sample's ROI and base
// the volume's origin. The outer walk, for goldens that address a neighbourhood or mirror
// coordinates rather than a single voxel (the image domain's for_each_roi_plane, one dimension up).
template <typename Fn>
void for_each_voxel_roi_plane(const RpptGenericDesc& desc, const RpptROI3D* roi, Roi3D type,
                              Fn fn) {
    const Rpp32u channels = voxel_channels(desc);
    for (Rpp32u n = 0; n < desc.dims[0]; ++n) {
        const VoxelBox box = voxel_box(roi[n], type);
        for (Rpp32u c = 0; c < channels; ++c) fn(n, c, box, voxel_plane_base(desc, n, c));
    }
}

// Invokes fn(n, c, z, y, x, srcIdx, dstIdx) for every voxel of each sample's ROI box, (z, y, x)
// being the box-relative coordinate.
//
// Like the image-domain pointwise ops (for_each_roi_io in tensor_setup.hpp), the voxel ops read the
// source at the ROI offset and write the output packed at the destination origin: box voxel
// (z, y, x) comes from source (z0 + z, y0 + y, x0 + x) and lands at destination (z, y, x). This is
// the one definition of that mapping -- the goldens and the comparator both drive it, so they
// cannot disagree.
template <typename Fn>
void for_each_voxel_roi_io(const RpptGenericDesc& desc, const RpptROI3D* roi, Roi3D type, Fn fn) {
    for_each_voxel_roi_plane(
        desc, roi, type, [&](Rpp32u n, Rpp32u c, const VoxelBox& b, std::size_t base) {
            for (Rpp32u z = 0; z < b.d; ++z)
                for (Rpp32u y = 0; y < b.h; ++y)
                    for (Rpp32u x = 0; x < b.w; ++x)
                        fn(n, c, z, y, x,
                           voxel_plane_index(desc, base, b.z0 + z, b.y0 + y, b.x0 + x),
                           voxel_plane_index(desc, base, z, y, x));
        });
}

// ---- comparison -------------------------------------------------------------

// Compares the ROI box's output region only -- the destination-origin block the op fills. What
// these ops leave outside it is not documented, so the suite does not assert it. The bound is
// absTolerance + relTolerance * |reference|.
template <typename T>
::testing::AssertionResult compare_voxel_roi(const T* actual, const T* reference,
                                             const RpptGenericDesc& desc, const RpptROI3D* roi,
                                             Roi3D type, double absTolerance,
                                             double relTolerance = 0.0) {
    bool failed = false;
    std::string coords;
    double got = 0.0, want = 0.0, diff = 0.0, tolerance = 0.0;
    for_each_voxel_roi_io(
        desc, roi, type,
        [&](Rpp32u n, Rpp32u c, Rpp32u z, Rpp32u y, Rpp32u x, std::size_t, std::size_t idx) {
            if (failed) return;  // report the first mismatch only
            const double a = to_double(actual[idx]);
            const double r = to_double(reference[idx]);
            const double delta = std::fabs(a - r);
            const double bound = absTolerance + relTolerance * std::fabs(r);
            if (delta <= bound) return;
            failed = true;
            got = a, want = r, diff = delta, tolerance = bound;
            coords = std::to_string(n) + "," + std::to_string(c) + "," + std::to_string(z) + "," +
                     std::to_string(y) + "," + std::to_string(x);
        });
    if (failed)
        return ::testing::AssertionFailure()
               << "mismatch at [n,c,z,y,x = " << coords << "]: actual=" << got
               << " reference=" << want << " diff=" << diff << " tolerance=" << tolerance;
    return ::testing::AssertionSuccess();
}

}  // namespace rpptest

#endif  // RPP_TEST_VOXEL_TENSOR_SETUP_H
