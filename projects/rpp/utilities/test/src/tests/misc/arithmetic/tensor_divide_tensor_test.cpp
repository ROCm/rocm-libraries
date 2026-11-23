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

#include <cstddef>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/compare.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/arithmetic_tensor_ref.hpp"

using namespace rpptest;

namespace {

// U8/I8 compare bit-exactly against the golden's round-to-nearest quotient, so a kernel that
// truncates fails.
constexpr Tolerance kAbsTolerance = tolerance(0.0, 1e-5, 2e-3);

constexpr Tolerance kRelTolerance = tolerance(0.0, 1e-6, 2e-3);

// Division by zero is undefined, so zero divisors are replaced by the smallest magnitude the
// dtype's fill uses.
template <typename T>
void fill_divisor(T* buf, const RpptGenericDesc& d, DType dt, unsigned salt) {
    fill_input_nd<T>(buf, d, dt, salt);
    const double replacement = (dt == DType::U8 || dt == DType::I8) ? 1.0 : 1.0 / 255.0;
    for_each_nd_coord(d, [&](const NdDims& coord) {
        T& v = buf[nd_offset(d, coord)];
        if (to_double(v) == 0.0) v = from_double<T>(replacement);
    });
}

template <typename T>
void run_tensor_divide_tensor(const NdConfig& cfg, Broadcast broadcast) {
    const NdDims dims1 = nd_operand_dims(cfg, broadcast, 1);
    const NdDims dims2 = nd_operand_dims(cfg, broadcast, 2);
    const NdDims outDims = nd_broadcast_dims(dims1, dims2);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor desc1(cfg.backend, dims1, cfg.dtypeIn);
    GenericDescriptor desc2(cfg.backend, dims2, cfg.dtypeIn);
    GenericDescriptor descOut(cfg.backend, outDims, cfg.dtypeIn);

    const std::size_t count1 = generic_element_count(*desc1);
    const std::size_t count2 = generic_element_count(*desc2);
    const std::size_t countOut = generic_element_count(*descOut);

    std::vector<T> input1(count1), input2(count2), golden(countOut), actual(countOut);
    fill_input_nd<T>(input1.data(), *desc1, cfg.dtypeIn, 0);
    fill_divisor<T>(input2.data(), *desc2, cfg.dtypeIn, 1);
    arithmetic_tensor_reference<T>(input1.data(), input2.data(), golden.data(), *descOut, *desc1,
                                   *desc2, ArithmeticTensorOp::Divide);

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
    ASSERT_EQ(rppt_tensor_divide_tensor(src1.ptr(), src2.ptr(), desc1.get(), desc2.get(), dst.ptr(),
                                        descOut.get(), to_rpp_broadcast(broadcast), roi1.data(),
                                        roi2.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytesOut);

    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *descOut,
                              Bound{kAbsTolerance(cfg.dtypeIn), kRelTolerance(cfg.dtypeIn)}));
}

}  // namespace

// Full name:
// Misc_Arithmetic/TensorDivideTensorTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Broadcast>_<Shape>
class TensorDivideTensorTest : public SkipListTest<NdWithParams<BroadcastParams>> {};

TEST_P(TensorDivideTensorTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const Broadcast broadcast = GetParam().op.mode;
    dispatch_dtype<DType::U8, DType::I8, DType::F16, DType::F32>(cfg.dtypeIn, [&](auto tag) {
        run_tensor_divide_tensor<Element<decltype(tag)>>(cfg, broadcast);
    });
}

// Scoped to U8/I8/F16/F32: the op also accepts U16/I16/U32/I32, which the DType axis does not
// model.
//
// Known kernel defect (skip-listed): U8, I8 and F16 division is unimplemented; the op returns
// RPP_SUCCESS without writing the output.
INSTANTIATE_TEST_SUITE_P(Misc_Arithmetic, TensorDivideTensorTest,
                         ::testing::ValuesIn(nd_with_params<BroadcastParams>(
                             make_nd_configs({DType::U8, DType::I8, DType::F16, DType::F32},
                                             {2, 3, 4}, {NdShape::VectorAligned, NdShape::Tail}),
                             {{Broadcast::None}, {Broadcast::Src1}, {Broadcast::Src2}})),
                         nd_op_config_name<BroadcastParams>);
