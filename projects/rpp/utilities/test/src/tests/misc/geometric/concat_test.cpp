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
#include "framework/tensor_setup.hpp"
#include "reference/concat_ref.hpp"

using namespace rpptest;

namespace {

// The concat axis is rank-dependent, so it is parameterized by kind rather than by a literal
// index. At rank 2 Middle and Last coincide -- harmless duplicate coverage with distinct labels.
enum class AxisKind { First, Middle, Last };

struct ConcatParams {
    AxisKind kind;
    std::string name() const {
        switch (kind) {
            case AxisKind::First:  return "AxisFirst";
            case AxisKind::Middle: return "AxisMiddle";
            case AxisKind::Last:   return "AxisLast";
        }
        return "UNK";
    }
};

// 0-based over the per-sample axes (the batch axis is excluded), which is what the op's axisMask
// carries: the legacy misc driver sizes the output as
// dstRoi[nDim + axisMask] = roi1[nDim + axisMask] + roi2[nDim + axisMask].
Rpp32u concat_axis(AxisKind kind, Rpp32u nDim) {
    switch (kind) {
        case AxisKind::First:  return 0;
        case AxisKind::Middle: return nDim / 2;
        case AxisKind::Last:   return nDim - 1;
    }
    return 0;
}

// src2 matches src1 on every axis except the concat axis, where it deliberately gets a *different*
// extent so that a swapped operand or an off-by-one axis cannot pass by coincidence. Halving is
// a no-op at extent 2 (the rank-4 extents include one), hence the second branch.
NdDims concat_src2_dims(Rpp32u nDim, Rpp32u axis) {
    NdDims dims = nd_extents(nDim);
    const Rpp32u e1 = dims[axis + 1];
    dims[axis + 1] = (e1 > 2) ? e1 / 2 + 1 : e1 + 1;
    return dims;
}

// Output shape: the operands' shared shape with the concat axis summed.
NdDims concat_dst_dims(const NdDims& dims1, const NdDims& dims2, Rpp32u axis) {
    NdDims dims = dims1;
    dims[axis + 1] = dims1[axis + 1] + dims2[axis + 1];
    return dims;
}

template <typename T>
void run_concat(const NdConfig& cfg, AxisKind kind) {
    const Rpp32u axis = concat_axis(kind, cfg.nDim);
    const NdDims dims1 = nd_extents(cfg.nDim);
    const NdDims dims2 = concat_src2_dims(cfg.nDim, axis);
    const NdDims outDims = concat_dst_dims(dims1, dims2, axis);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor desc1(cfg.backend, dims1, cfg.dtypeIn);
    GenericDescriptor desc2(cfg.backend, dims2, cfg.dtypeIn);
    GenericDescriptor descOut(cfg.backend, outDims, cfg.dtypeIn);

    const std::size_t count1 = generic_element_count(*desc1);
    const std::size_t count2 = generic_element_count(*desc2);
    const std::size_t countOut = generic_element_count(*descOut);

    // (1) Host golden model. Two distinct fills (different salts) so the two halves of the output
    // are distinguishable. The op writes every output element, so golden needs no pre-seeding.
    std::vector<T> input1(count1), input2(count2), golden(countOut), actual(countOut);
    fill_input_nd<T>(input1.data(), *desc1, cfg.dtypeIn, 0);
    fill_input_nd<T>(input2.data(), *desc2, cfg.dtypeIn, 1);
    concat_reference<T>(input1.data(), input2.data(), golden.data(), *descOut, *desc1, *desc2,
                        axis);

    // (2) roiTensors live in host-accessible (pinned for HIP) memory. Each operand is exercised
    // whole, so its roiTensor is just its own extents.
    const std::vector<Rpp32u> roiVec1 = make_nd_roi_tensor(dims1);
    const std::vector<Rpp32u> roiVec2 = make_nd_roi_tensor(dims2);
    PinnedArray<Rpp32u> roi1(cfg.backend, roiVec1.size());
    PinnedArray<Rpp32u> roi2(cfg.backend, roiVec2.size());
    for (std::size_t i = 0; i < roiVec1.size(); ++i) roi1[i] = roiVec1[i];
    for (std::size_t i = 0; i < roiVec2.size(); ++i) roi2[i] = roiVec2[i];

    // (3) Run RPP on the configured backend.
    const std::size_t bytes1 = generic_byte_size(*desc1, cfg.dtypeIn);
    const std::size_t bytes2 = generic_byte_size(*desc2, cfg.dtypeIn);
    const std::size_t bytesOut = generic_byte_size(*descOut, cfg.dtypeIn);
    DeviceTensor src1(cfg.backend, bytes1), src2(cfg.backend, bytes2), dst(cfg.backend, bytesOut);
    src1.write(input1.data(), bytes1);
    src2.write(input2.data(), bytes2);

    RppHandle handle(cfg.backend, outDims[0]);
    ASSERT_EQ(rppt_concat(src1.ptr(), src2.ptr(), desc1.get(), desc2.get(), dst.ptr(),
                          descOut.get(), axis, roi1.data(), roi2.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytesOut);

    // (4) Compare the whole output tensor. Concat only copies elements -- it performs no
    // arithmetic and no dtype conversion -- so every dtype is bit-exact and any diff is a defect.
    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *descOut, 0.0, 0.0));
}

}  // namespace

// Full name: Misc_Geometric/ConcatTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Axis>_<Shape>
class ConcatTest : public ::testing::TestWithParam<NdWithParams<ConcatParams>> {};

TEST_P(ConcatTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const AxisKind kind = GetParam().op.kind;
    switch (cfg.dtypeIn) {
        case DType::U8:
            run_concat<Rpp8u>(cfg, kind);
            break;
        case DType::I8:
            run_concat<Rpp8s>(cfg, kind);
            break;
        case DType::F16:
            run_concat<Rpp16f>(cfg, kind);
            break;
        case DType::F32:
            run_concat<Rpp32f>(cfg, kind);
            break;
        default:
            FAIL() << "unsupported dtype for concat";
    }
}

// 72 cases: U8/F16/F32/I8 (what the op accepts) x ranks 2/3/4 x 3 axis kinds x HOST/HIP.
//
// 32 green, 40 red against one documented kernel defect, identically on both backends and for every
// dtype: concat returns RPP_ERROR_INVALID_DIM_LENGTHS (-25) when the two operands' extents differ
// along a non-final concat axis -- which is the axis concat exists to differ on
// (issues/concat-rejects-unequal-extents-on-non-final-axis.md). Every AxisLast case is bit-exact,
// as is rank-2 AxisMiddle (where "middle" is the last axis). The axis parameter itself is handled
// correctly: axisMask is an axis index, confirmed by sweeping it against per-axis goldens.
//
// The shape token in the label is src1's shape; src2 differs from it along the concat axis and the
// output is their sum there (see concat_src2_dims), so the label identifies the case, not the
// output extent.
//
// Note the HIP descriptors must be device-addressable (GenericDescriptor handles this): at
// rank >= 4 the ND kernels read dims/strides on the device. Undocumented and rank-dependent.
INSTANTIATE_TEST_SUITE_P(Misc_Geometric, ConcatTest,
                         ::testing::ValuesIn(nd_with_params<ConcatParams>(
                             make_nd_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                             {2, 3, 4}),
                             {{AxisKind::First}, {AxisKind::Middle}, {AxisKind::Last}})),
                         nd_op_config_name<ConcatParams>);
