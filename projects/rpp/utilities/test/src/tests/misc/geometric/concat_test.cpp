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
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/skip_list.hpp"
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
            case AxisKind::First:
                return "AxisFirst";
            case AxisKind::Middle:
                return "AxisMiddle";
            case AxisKind::Last:
                return "AxisLast";
        }
        return "UNK";
    }
};

// axisMask is a 0-based index over the per-sample axes (the batch axis is excluded).
Rpp32u concat_axis(AxisKind kind, Rpp32u nDim) {
    switch (kind) {
        case AxisKind::First:
            return 0;
        case AxisKind::Middle:
            return nDim / 2;
        case AxisKind::Last:
            return nDim - 1;
    }
    return 0;
}

// src2 matches src1 except along the concat axis, where its extent differs so a swapped operand or
// wrong axis cannot pass (halving leaves an extent of 2 unchanged, hence the second branch). Copied
// from dims1 rather than rebuilt from the rank so the Tail shape's innermost extent agrees.
NdDims concat_src2_dims(const NdDims& dims1, Rpp32u axis) {
    NdDims dims = dims1;
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
    const NdDims dims1 = nd_extents(cfg);
    const NdDims dims2 = concat_src2_dims(dims1, axis);
    const NdDims outDims = concat_dst_dims(dims1, dims2, axis);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor desc1(cfg.backend, dims1, cfg.dtypeIn);
    GenericDescriptor desc2(cfg.backend, dims2, cfg.dtypeIn);
    GenericDescriptor descOut(cfg.backend, outDims, cfg.dtypeIn);

    const std::size_t count1 = generic_element_count(*desc1);
    const std::size_t count2 = generic_element_count(*desc2);
    const std::size_t countOut = generic_element_count(*descOut);

    // Distinct salts so the two halves of the output are distinguishable.
    std::vector<T> input1(count1), input2(count2), golden(countOut), actual(countOut);
    fill_input_nd<T>(input1.data(), *desc1, cfg.dtypeIn, 0);
    fill_input_nd<T>(input2.data(), *desc2, cfg.dtypeIn, 1);
    concat_reference<T>(input1.data(), input2.data(), golden.data(), *descOut, *desc1, *desc2,
                        axis);

    const std::vector<Rpp32u> roiVec1 = make_nd_roi_tensor(dims1);
    const std::vector<Rpp32u> roiVec2 = make_nd_roi_tensor(dims2);
    PinnedArray<Rpp32u> roi1(cfg.backend, roiVec1.size());
    PinnedArray<Rpp32u> roi2(cfg.backend, roiVec2.size());
    for (std::size_t i = 0; i < roiVec1.size(); ++i) roi1[i] = roiVec1[i];
    for (std::size_t i = 0; i < roiVec2.size(); ++i) roi2[i] = roiVec2[i];

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

    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *descOut, 0.0));
}

}  // namespace

// Full name: Misc_Geometric/ConcatTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Axis>_<Shape>
class ConcatTest : public SkipListTest<NdWithParams<ConcatParams>> {};

TEST_P(ConcatTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const AxisKind kind = GetParam().op.kind;
    dispatch_dtype<DType::U8, DType::I8, DType::F16, DType::F32>(
        cfg.dtypeIn, [&](auto tag) { run_concat<Element<decltype(tag)>>(cfg, kind); });
}

// Known kernel defect (skip-listed): concat returns RPP_ERROR_INVALID_DIM_LENGTHS when the
// operands' extents differ along a non-final concat axis.
//
// The shape token in the label is src1's shape, not the output's.
INSTANTIATE_TEST_SUITE_P(Misc_Geometric, ConcatTest,
                         ::testing::ValuesIn(nd_with_params<ConcatParams>(
                             make_nd_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                             {2, 3, 4}, {NdShape::VectorAligned, NdShape::Tail}),
                             {{AxisKind::First}, {AxisKind::Middle}, {AxisKind::Last}})),
                         nd_op_config_name<ConcatParams>);
