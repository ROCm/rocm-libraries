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

#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/bitwise_tensor_ref.hpp"

using namespace rpptest;

namespace {

template <typename T>
void run_tensor_xor_tensor(const NdConfig& cfg, Broadcast broadcast) {
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

    // (1) Host golden model. Two distinct fills (different salts) so the XOR is exercised on
    // differing bit patterns. The op writes every output element, so golden needs no pre-seeding.
    std::vector<T> input1(count1), input2(count2), golden(countOut), actual(countOut);
    fill_input_nd<T>(input1.data(), *desc1, cfg.dtypeIn, 0);
    fill_input_nd<T>(input2.data(), *desc2, cfg.dtypeIn, 1);
    bitwise_tensor_reference<T>(input1.data(), input2.data(), golden.data(), *descOut, *desc1, *desc2,
                                BitwiseTensorOp::Xor);

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
    ASSERT_EQ(rppt_tensor_xor_tensor(src1.ptr(), src2.ptr(), desc1.get(), desc2.get(), dst.ptr(), descOut.get(),
                                     to_rpp_broadcast(broadcast), roi1.data(), roi2.data(),
                                     handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytesOut);

    // (4) Compare the whole output tensor. A bitwise XOR is bit-exact for every dtype, so the
    // tolerance is zero -- do not loosen it.
    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *descOut, 0.0));
}

}  // namespace

// Full name: Misc_Bitwise/TensorXorTensorTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Broadcast>_<Shape>
class TensorXorTensorTest : public SkipListTest<NdWithParams<BroadcastParams>> {};

TEST_P(TensorXorTensorTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const Broadcast broadcast = GetParam().op.mode;
    dispatch_dtype<DType::U8, DType::I8>(cfg.dtypeIn, [&](auto tag) {
        run_tensor_xor_tensor<Element<decltype(tag)>>(cfg, broadcast);
    });
}

// Restricted to the integer dtypes (U8/I8): the op also accepts the 16/32-bit integer types, but
// the framework's DType enum does not carry them yet.
//
// Full grid passes on both backends. Note the HIP descriptors must be device-addressable
// (GenericDescriptor handles this): at rank >= 4 the non-broadcast kernel reads dims/strides on
// the device. Undocumented and rank-dependent.
INSTANTIATE_TEST_SUITE_P(Misc_Bitwise, TensorXorTensorTest,
                         ::testing::ValuesIn(nd_with_params<BroadcastParams>(
                             make_nd_configs({DType::U8, DType::I8}, {2, 3, 4}),
                             {{Broadcast::None}, {Broadcast::Src1}, {Broadcast::Src2}})),
                         nd_op_config_name<BroadcastParams>);
