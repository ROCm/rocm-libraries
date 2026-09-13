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

#ifndef RPP_TEST_VOXEL_CONFIG_PARAM_H
#define RPP_TEST_VOXEL_CONFIG_PARAM_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <ostream>
#include <string>
#include <vector>

#include "framework/backend_param.hpp"
#include "framework/config_param.hpp"

// Grid axes for the Voxel (3D) ops.
// The shared axes (DType, Roi, the WithParams machinery) live in config_param.hpp.

namespace rpptest {

// The voxel ops take a 5D RpptGenericDesc (NCDHW / NDHWC) plus an RpptROI3D box instead of the
// image domain's RpptDesc + XYWH ROI. The layout slot carries the 3D layout together with its
// channel count, and the ROI slot gains a second token for the ROI3D encoding, so the label keeps
// the same structural slots: "<Backend>_<DTypeConv>_<Layout>_<Roi>_<Roi3DType>_<Shape>".
// Descriptor construction, traversal and comparison live in framework/voxel_tensor_setup.hpp.

enum class VoxelLayout { NCDHW1, NCDHW3, NDHWC3 };
enum class Roi3D { XYZWHD, LTFRBB };

// Spatial extent of the test volume; the channel count comes from VoxelLayout.
struct VoxelSize {
    Rpp32u n, d, h, w;
};

inline std::string voxel_layout_name(VoxelLayout l) {
    switch (l) {
        case VoxelLayout::NCDHW1:
            return "NCDHW1";
        case VoxelLayout::NCDHW3:
            return "NCDHW3";
        case VoxelLayout::NDHWC3:
            return "NDHWC3";
    }
    return "UNK";
}

inline std::string roi3d_name(Roi3D t) {
    return t == Roi3D::XYZWHD ? "XYZWHD" : "LTFRBB";
}

inline std::string voxel_size_name(VoxelSize s) {
    return std::to_string(s.n) + "x" + std::to_string(s.d) + "x" + std::to_string(s.h) + "x" +
           std::to_string(s.w);
}

struct VoxelConfig {
    RppBackend backend;
    DType dtype;
    VoxelLayout layout;
    Roi roi;
    Roi3D roiType;
    VoxelSize size;
};

inline std::string voxel_config_name(const VoxelConfig& c) {
    return backend_name(c.backend) + "_" + dtype_name(c.dtype) + "to" + dtype_name(c.dtype) + "_" +
           voxel_layout_name(c.layout) + "_" + roi_name(c.roi) + "_" + roi3d_name(c.roiType) + "_" +
           voxel_size_name(c.size);
}

namespace presets {
// The default test volume: every extent is a multiple of 4, so the vector loops see no tail.
inline constexpr VoxelSize kDefaultVolume{2, 4, 12, 16};

// Odd width and odd depth. Voxel tensors are dense -- unlike an image row, nothing pads the end of
// a row -- so this is the shape that puts a tail element at the end of the vector span and live
// data immediately after the last store.
inline constexpr VoxelSize kTailVolume{2, 3, 10, 19};
}  // namespace presets

inline std::vector<VoxelConfig> make_voxel_configs(
    const std::vector<DType>& dtypes, const std::vector<VoxelLayout>& layouts,
    const std::vector<Roi>& rois, const std::vector<Roi3D>& roiTypes,
    const std::vector<VoxelSize>& sizes = {{2, 4, 12, 16}}) {
    std::vector<VoxelConfig> configs;
    for (RppBackend backend : available_backends())
        for (DType dtype : dtypes)
            for (VoxelLayout layout : layouts)
                for (Roi roi : rois)
                    for (Roi3D roiType : roiTypes)
                        for (VoxelSize size : sizes)
                            configs.push_back({backend, dtype, layout, roi, roiType, size});
    return configs;
}

// GTest name generator for voxel ops with no scalar parameters (the voxel counterpart of
// config_param_name).
inline std::string voxel_config_param_name(const ::testing::TestParamInfo<VoxelConfig>& info) {
    return voxel_config_name(info.param);
}

// The voxel counterpart of WithParams. P must provide std::string name() const.
template <typename P>
struct VoxelWithParams {
    VoxelConfig cfg;
    P op;
};

template <typename P>
inline std::vector<VoxelWithParams<P>> voxel_with_params(const std::vector<VoxelConfig>& base,
                                                         const std::vector<P>& params) {
    std::vector<VoxelWithParams<P>> out;
    out.reserve(base.size() * params.size());
    for (const auto& c : base)
        for (const auto& p : params) out.push_back({c, p});
    return out;
}

template <typename P>
inline std::string voxel_op_config_name(const ::testing::TestParamInfo<VoxelWithParams<P>>& info) {
    const std::string suffix = info.param.op.name();
    return voxel_config_name(info.param.cfg) + (suffix.empty() ? "" : "_" + suffix);
}

inline void PrintTo(const VoxelConfig&, std::ostream*) {}

template <typename P>
void PrintTo(const VoxelWithParams<P>&, std::ostream*) {}

}  // namespace rpptest

#endif  // RPP_TEST_VOXEL_CONFIG_PARAM_H
