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
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/tensor_setup.hpp"
#include "framework/tolerance.hpp"
#include "reference/arithmetic_tensor_ref.hpp"

using namespace rpptest;

namespace {

// Tolerances model legitimate floating-point error only. Division is the one arithmetic op here
// with a fractional intermediate, and the golden rounds it to nearest before storing, so U8/I8 are
// still compared bit-exactly -- a truncating kernel must show up as a failure, not be absorbed.
constexpr Tolerance kAbsTolerance = tolerance(0.0, 1e-5, 2e-3);

constexpr Tolerance kRelTolerance = tolerance(0.0, 1e-6, 2e-3);

// The golden does not model division by zero (undefined), so every divisor is forced away from
// zero: the integer fills contain 0 and the [0,1] float fills contain exactly 0.0. The replacement
// is the smallest magnitude the dtype represents in the fill's own units (1 for U8/I8, 1/255 for
// F16/F32), so the divisor distribution is otherwise untouched.
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
    const NdDims dims1 = nd_operand_dims(cfg.nDim, broadcast, 1);
    const NdDims dims2 = nd_operand_dims(cfg.nDim, broadcast, 2);
    const NdDims outDims = nd_broadcast_dims(dims1, dims2);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor desc1(cfg.backend, dims1, cfg.dtypeIn);
    GenericDescriptor desc2(cfg.backend, dims2, cfg.dtypeIn);
    GenericDescriptor descOut(cfg.backend, outDims, cfg.dtypeIn);

    const std::size_t count1 = generic_element_count(*desc1);
    const std::size_t count2 = generic_element_count(*desc2);
    const std::size_t countOut = generic_element_count(*descOut);

    // (1) Host golden model. Two distinct fills (different salts) so the quotient is non-trivial;
    // the second operand additionally goes through fill_divisor so it is never zero. The op writes
    // every output element, so golden needs no pre-seeding. Operand order matters here: src1 / src2,
    // in the order the API declares them -- src2 is the divisor.
    std::vector<T> input1(count1), input2(count2), golden(countOut), actual(countOut);
    fill_input_nd<T>(input1.data(), *desc1, cfg.dtypeIn, 0);
    fill_divisor<T>(input2.data(), *desc2, cfg.dtypeIn, 1);
    arithmetic_tensor_reference<T>(input1.data(), input2.data(), golden.data(), *descOut, *desc1,
                                   *desc2, ArithmeticTensorOp::Divide);

    // (2) roiTensors live in host-accessible (pinned for HIP) memory.
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
    DeviceTensor src1(cfg.backend, bytes1), src2(cfg.backend, bytes2),
        dst(cfg.backend, bytesOut);
    src1.write(input1.data(), bytes1);
    src2.write(input2.data(), bytes2);

    RppHandle handle(cfg.backend, outDims[0]);
    ASSERT_EQ(rppt_tensor_divide_tensor(src1.ptr(), src2.ptr(), desc1.get(), desc2.get(), dst.ptr(),
                                        descOut.get(), to_rpp_broadcast(broadcast), roi1.data(),
                                        roi2.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytesOut);

    // (4) Compare the whole output tensor.
    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *descOut, kAbsTolerance(cfg.dtypeIn),
                              kRelTolerance(cfg.dtypeIn)));
}

}  // namespace

// Full name:
// Misc_Arithmetic/TensorDivideTensorTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Broadcast>_<Shape>
class TensorDivideTensorTest : public ::testing::TestWithParam<NdWithParams<BroadcastParams>> {};

TEST_P(TensorDivideTensorTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const Broadcast broadcast = GetParam().op.mode;
    dispatch_dtype<DType::U8, DType::I8, DType::F16, DType::F32>(cfg.dtypeIn, [&](auto tag) {
        run_tensor_divide_tensor<Element<decltype(tag)>>(cfg, broadcast);
    });
}

// Dtypes are scoped to U8/I8/F16/F32: the op also accepts U16/I16/U32/I32, which the framework's
// DType axis does not model for this op.
//
// Division is the one arithmetic op whose exact result is fractional, which drives two choices:
//   - Every divisor is non-zero by construction (fill_divisor above), because division by zero is
//     undefined and the golden does not model it.
//   - The U8/I8 tolerance is 0, holding the kernel to the golden's round-to-nearest store
//     (documented in reference/arithmetic_tensor_ref.hpp). A kernel that truncates the quotient
//     instead is exactly the defect class this catches (RPP truncates I8 in several other ops),
//     so the tolerance must not be widened to hide it.
//
// 54 of these 72 cases are red, deterministically, against a single kernel defect: U8, I8 and F16
// division is unimplemented, yet the op returns RPP_SUCCESS and leaves the destination buffer
// entirely untouched. Until that is fixed the integer rounding tripwire above cannot actually
// observe anything. F32 is green on both backends across every rank and broadcast mode.
//
// Note the ND descriptors must be device-addressable on HIP (GenericDescriptor handles this): at
// rank >= 4 the non-broadcast kernel reads dims/strides on the device. Undocumented and
// rank-dependent.
INSTANTIATE_TEST_SUITE_P(Misc_Arithmetic, TensorDivideTensorTest,
                         ::testing::ValuesIn(nd_with_params<BroadcastParams>(
                             make_nd_configs({DType::U8, DType::I8, DType::F16, DType::F32},
                                             {2, 3, 4}),
                             {{Broadcast::None}, {Broadcast::Src1}, {Broadcast::Src2}})),
                         nd_op_config_name<BroadcastParams>);
