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

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/normalize_ref.hpp"

using namespace rpptest;

namespace {

struct NormalizeParams {
    Rpp32u axisMask;     // bit i set => axis i is reduced
    Rpp8u computeMode;   // bit 0 => compute mean, bit 1 => compute stddev, 0 => caller-supplied
    std::string name() const {
        return "axis" + std::to_string(axisMask) + "_cms" + std::to_string(computeMode);
    }
};

// Non-trivial scale/shift so a kernel that drops either is caught (the legacy harness runs
// scale=1, shift=0, which cannot distinguish them).
constexpr Rpp32f kScale = 1.5f;
constexpr Rpp32f kShift = 0.25f;

// Caller-supplied mean/stddev for the modes that do not compute them. Deterministic, and
// the stddev is always non-zero so the reference never divides by zero.
Rpp32f supplied_mean(std::size_t i) { return 0.25f * static_cast<Rpp32f>(i % 7); }
Rpp32f supplied_stddev(std::size_t i) { return 1.0f + 0.5f * static_cast<Rpp32f>(i % 3); }

// Tolerances reflect legitimate floating-point error only: the golden accumulates in double
// while the kernel reduces in float (and stores half for F16), so the bound is dominated by
// the relative term.
double abs_tolerance(DType out) { return out == DType::F16 ? 2e-2 : 1e-4; }
double rel_tolerance(DType out) { return out == DType::F16 ? 4e-3 : 1e-5; }

template <typename Tin, typename Tout>
void run_normalize(const NdConfig& cfg, const NormalizeParams& p) {
    const NdDims dims = nd_extents(cfg.nDim);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor srcDesc(cfg.backend, dims, cfg.dtypeIn);
    GenericDescriptor dstDesc(cfg.backend, dims, cfg.dtypeOut);

    const std::size_t count = generic_element_count(*srcDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeOut);

    const auto paramDims = normalize_param_dims(dims, p.axisMask);
    const Rpp32u paramSize = normalize_param_size(paramDims);
    const std::size_t paramCount = static_cast<std::size_t>(paramSize) * dims[0];

    // Mean/stddev and the roiTensor live in host-accessible (pinned for HIP) memory.
    PinnedArray<Rpp32f> mean(cfg.backend, paramCount);
    PinnedArray<Rpp32f> stdDev(cfg.backend, paramCount);
    for (std::size_t i = 0; i < paramCount; ++i) {
        mean[i] = supplied_mean(i);
        stdDev[i] = supplied_stddev(i);
    }
    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(dims);
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    // (1) Host golden model.
    std::vector<Tin> input(count);
    std::vector<Tout> golden(count), actual(count);
    fill_input_nd<Tin>(input.data(), *srcDesc, cfg.dtypeIn);
    normalize_reference<Tin, Tout>(input.data(), golden.data(), *srcDesc, *dstDesc, p.axisMask,
                                   mean.data(), stdDev.data(), p.computeMode, kScale, kShift);

    // (2) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);

    RppHandle handle(cfg.backend, dims[0]);
    ASSERT_EQ(rppt_normalize(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), p.axisMask,
                             mean.data(), stdDev.data(), p.computeMode, kScale, kShift, roi.data(),
                             handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (3) Compare the whole output tensor.
    EXPECT_TRUE(compare_nd<Tout>(actual.data(), golden.data(), *dstDesc, abs_tolerance(cfg.dtypeOut),
                                 rel_tolerance(cfg.dtypeOut)));
}

// Representative reduction masks per rank: innermost axis only, every axis, and one mixed
// mask. These are the three distinct param-tensor shapes (per-element, scalar, partial).
std::vector<Rpp32u> masks_for(Rpp32u nDim) {
    switch (nDim) {
        case 2:  return {2, 3, 1};
        case 3:  return {4, 7, 6};
        case 4:  return {8, 15, 12};
        default: return {1};
    }
}

// axisMask validity depends on the rank, so the grid is built rank by rank rather than as a
// flat cross product.
std::vector<NdWithParams<NormalizeParams>> normalize_grid() {
    const std::vector<DTypeConv> convs = {{DType::U8, DType::F32},
                                          {DType::I8, DType::F32},
                                          {DType::F16, DType::F16},
                                          {DType::F32, DType::F32}};
    std::vector<NdWithParams<NormalizeParams>> grid;
    for (Rpp32u nDim : {2u, 3u, 4u})
        for (const NdConfig& cfg : make_nd_configs(convs, {nDim}))
            for (Rpp32u axisMask : masks_for(nDim))
                for (Rpp8u mode : {0, 1, 2, 3})
                    grid.push_back({cfg, NormalizeParams{axisMask, mode}});
    return grid;
}

}  // namespace

// Full name:
// Misc_Statistical/NormalizeTest.Correctness/<Backend>_<DTypeConv>_<Rank>_axis<M>_cms<C>_<Shape>
class NormalizeTest : public SkipListTest<NdWithParams<NormalizeParams>> {};

TEST_P(NormalizeTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const NormalizeParams p = GetParam().op;
    if (cfg.dtypeIn == DType::U8 && cfg.dtypeOut == DType::F32)
        run_normalize<Rpp8u, Rpp32f>(cfg, p);
    else if (cfg.dtypeIn == DType::I8 && cfg.dtypeOut == DType::F32)
        run_normalize<Rpp8s, Rpp32f>(cfg, p);
    else if (cfg.dtypeIn == DType::F16 && cfg.dtypeOut == DType::F16)
        run_normalize<Rpp16f, Rpp16f>(cfg, p);
    else if (cfg.dtypeIn == DType::F32 && cfg.dtypeOut == DType::F32)
        run_normalize<Rpp32f, Rpp32f>(cfg, p);
    else
        FAIL() << "unsupported dtype conversion for normalize";
}

// The dtype conversions are the set the header documents ("Supports u8->f32, i8->f32,
// f16->f16 and f32->f32"). Note the legacy harness instead exercises same-dtype
// (u8->u8, i8->i8); the two disagree, and the header is taken as the spec here.
//
// 223 of these 288 cases are red against four documented kernel defects, all deterministic
// (identical failure set across repeated runs). The goldens and tolerances are deliberately
// left correct:
//   - 144: U8toF32 / I8toF32 rejected with RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE, though the
//     header documents them.
//   -  36: cms0 (both statistics supplied) applies sample 0's mean/stddev to every sample,
//     both backends.
//   -  18: cms1 on HOST multiplies by the supplied stddev instead of by scale/stddev; HIP is
//     correct.
//   -  25: cms2/cms3 on HOST at rank >= 3 (all zeros at 3D), plus F16 partial masks at 2D; HIP
//     matches the golden at every rank.
INSTANTIATE_TEST_SUITE_P(Misc_Statistical, NormalizeTest, ::testing::ValuesIn(normalize_grid()),
                         nd_op_config_name<NormalizeParams>);
