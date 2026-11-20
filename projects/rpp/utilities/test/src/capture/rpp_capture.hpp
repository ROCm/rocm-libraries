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

#ifndef RPP_TEST_CAPTURE_RPP_CAPTURE_H
#define RPP_TEST_CAPTURE_RPP_CAPTURE_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "capture/bundle.hpp"
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "framework/voxel_config_param.hpp"

// Maps RPP descriptors onto capture bundles. Each array holds the whole logical tensor in a
// canonical channels-last order, so the viewer never sees layouts, strides, or padding. Elements
// are marked with the comparators' own walks and tolerance rule, so the status mask is exactly
// what the verdict was computed over.

namespace rpptest::capture {

template <>
struct NpyDescr<Rpp16f> {
    static std::string get() {
        return "<f2";
    }
};

namespace detail {

inline void mark(Comparison& cmp, std::size_t i, double actual, double reference, Bound bound) {
    const double diff = actual - reference;
    cmp.mark(i, diff, std::fabs(diff) <= bound(reference));
}

}  // namespace detail

// Image tensor -> (n, h, w, c).
template <typename T>
void record_roi(const T* actual, const T* reference, const RpptDesc& d, const RpptROI* roi,
                RpptRoiType roiType, Bound bound) {
    if (!enabled()) return;
    auto dense = [&](std::size_t n, std::size_t c, std::size_t y, std::size_t x) {
        return ((n * d.h + y) * d.w + x) * d.c + c;
    };
    Comparison cmp = Comparison::of<T, T>("image", {"n", "h", "w", "c"}, {d.n, d.h, d.w, d.c});
    for (Rpp32u n = 0; n < d.n; ++n)
        for (Rpp32u c = 0; c < d.c; ++c)
            for (Rpp32u y = 0; y < d.h; ++y)
                for (Rpp32u x = 0; x < d.w; ++x) {
                    const std::size_t src = plane_index(d, plane_base(d, n, c), y, x);
                    cmp.reference.put(dense(n, c, y, x), reference[src]);
                    cmp.actual.put(dense(n, c, y, x), actual[src]);
                }
    for_each_roi_io(d, roi, roiType,
                    [&](Rpp32u n, Rpp32u c, Rpp32u j, Rpp32u i, std::size_t, std::size_t idx) {
                        detail::mark(cmp, dense(n, c, j, i), to_double(actual[idx]),
                                     to_double(reference[idx]), bound);
                    });
    submit(std::move(cmp));
}

// Generic tensor -> its logical dims, batch axis first.
template <typename T>
void record_nd(const T* actual, const T* reference, const RpptGenericDesc& d, Bound bound) {
    if (!enabled()) return;
    std::vector<std::string> axes{"n"};
    std::vector<std::size_t> shape;
    for (std::size_t a = 0; a < d.numDims; ++a) {
        if (a) axes.push_back("d" + std::to_string(a));
        shape.push_back(d.dims[a]);
    }
    Comparison cmp = Comparison::of<T, T>("nd", axes, shape);
    std::size_t dense = 0;
    for_each_nd_coord(d, [&](const NdDims& coord) {
        const std::size_t i = nd_offset(d, coord);
        cmp.reference.put(dense, reference[i]);
        cmp.actual.put(dense, actual[i]);
        detail::mark(cmp, dense, to_double(actual[i]), to_double(reference[i]), bound);
        ++dense;
    });
    submit(std::move(cmp));
}

// Voxel tensor -> (n, d, h, w, c).
template <typename T>
void record_voxel_roi(const T* actual, const T* reference, const RpptGenericDesc& desc,
                      const RpptROI3D* roi, Roi3D type, Bound bound) {
    if (!enabled()) return;
    const VoxelAxes ax = voxel_axes(voxel_is_packed(desc));
    const std::size_t N = desc.dims[0], C = desc.dims[ax.c], D = desc.dims[ax.z],
                      H = desc.dims[ax.y], W = desc.dims[ax.x];
    auto dense = [&](std::size_t n, std::size_t c, std::size_t z, std::size_t y, std::size_t x) {
        return (((n * D + z) * H + y) * W + x) * C + c;
    };
    Comparison cmp = Comparison::of<T, T>("voxel", {"n", "d", "h", "w", "c"}, {N, D, H, W, C});
    for (Rpp32u n = 0; n < N; ++n)
        for (Rpp32u c = 0; c < C; ++c) {
            const std::size_t base = voxel_plane_base(desc, n, c);
            for (Rpp32u z = 0; z < D; ++z)
                for (Rpp32u y = 0; y < H; ++y)
                    for (Rpp32u x = 0; x < W; ++x) {
                        const std::size_t src = voxel_plane_index(desc, base, z, y, x);
                        cmp.reference.put(dense(n, c, z, y, x), reference[src]);
                        cmp.actual.put(dense(n, c, z, y, x), actual[src]);
                    }
        }
    for_each_voxel_roi_io(
        desc, roi, type,
        [&](Rpp32u n, Rpp32u c, Rpp32u z, Rpp32u y, Rpp32u x, std::size_t, std::size_t idx) {
            detail::mark(cmp, dense(n, c, z, y, x), to_double(actual[idx]),
                         to_double(reference[idx]), bound);
        });
    submit(std::move(cmp));
}

// Reduction results -> (i). The golden is double-precision, so the reference array is float64.
template <typename TOut>
void record_reduction(const TOut* actual, const std::vector<double>& golden, Bound bound) {
    if (!enabled()) return;
    Comparison cmp = Comparison::of<double, TOut>("reduction", {"i"}, {golden.size()});
    for (std::size_t i = 0; i < golden.size(); ++i) {
        cmp.reference.put(i, golden[i]);
        cmp.actual.put(i, actual[i]);
        detail::mark(cmp, i, to_double(actual[i]), golden[i], bound);
    }
    submit(std::move(cmp));
}

}  // namespace rpptest::capture

#endif  // RPP_TEST_CAPTURE_RPP_CAPTURE_H
