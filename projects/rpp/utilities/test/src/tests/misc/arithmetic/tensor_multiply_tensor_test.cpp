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
#include "framework/generic_tensor_setup.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/arithmetic_tensor_ref.hpp"

using namespace rpptest;

namespace {

// Multiplying two full-range integer operands saturates almost every element, which would test
// the clamp and nothing else. So the integer fills are folded down to a small magnitude: most
// products then land inside the dtype range while a few still saturate, keeping the clamp
// covered. C++ '%' keeps the sign of the dividend, so the I8 fold stays two-sided.
//   U8 -> [0,18]   (products 0..324; ~2-3% of elements saturate at 255)
//   I8 -> [-14,14] (products -196..196; a few percent saturate at each of -128 and 127)
// F16/F32 fills are already in [0,1], so their products stay in range and are left alone.
template <typename T>
void fill_multiply_operand(T* buf, const RpptGenericDesc& d, DType dt, unsigned salt) {
    fill_input_nd<T>(buf, d, dt, salt);
    const int modulus = dt == DType::U8 ? 19 : dt == DType::I8 ? 15 : 0;
    if (modulus == 0) return;  // F16/F32 fills are already in range
    for_each_nd_coord(d, [&](const NdDims& coord) {
        T& v = buf[nd_offset(d, coord)];
        v = static_cast<T>(static_cast<int>(v) % modulus);
    });
}

// Integer multiplication is exact (only saturation is involved), so U8/I8 are bit-exact. The
// float tolerances model legitimate fp rounding of a single multiply only.
struct Tolerance {
    double abs, rel;
};

Tolerance tolerance_for(DType dt) {
    switch (dt) {
        case DType::F32: return {1e-5, 1e-6};
        case DType::F16: return {2e-3, 2e-3};
        default:         return {0.0, 0.0};
    }
}

template <typename T>
void run_tensor_multiply_tensor(const NdConfig& cfg, Broadcast broadcast) {
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

    // (1) Host golden model. Two distinct fills (different salts) so the two operands never
    // carry the same value at the same coordinate. The op writes every output element, so
    // golden needs no pre-seeding.
    std::vector<T> input1(count1), input2(count2), golden(countOut), actual(countOut);
    fill_multiply_operand<T>(input1.data(), *desc1, cfg.dtypeIn, 0);
    fill_multiply_operand<T>(input2.data(), *desc2, cfg.dtypeIn, 1);
    arithmetic_tensor_reference<T>(input1.data(), input2.data(), golden.data(), *descOut, *desc1,
                                   *desc2, ArithmeticTensorOp::Multiply);

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
    ASSERT_EQ(rppt_tensor_multiply_tensor(src1.ptr(), src2.ptr(), desc1.get(), desc2.get(),
                                          dst.ptr(), descOut.get(), to_rpp_broadcast(broadcast),
                                          roi1.data(), roi2.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), bytesOut);

    // (4) Compare the whole output tensor.
    const Tolerance tol = tolerance_for(cfg.dtypeIn);
    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *descOut, tol.abs, tol.rel));
}

}  // namespace

// Full name: Misc_Arithmetic/TensorMultiplyTensorTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Broadcast>_<Shape>
class TensorMultiplyTensorTest : public ::testing::TestWithParam<NdWithParams<BroadcastParams>> {};

TEST_P(TensorMultiplyTensorTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const Broadcast broadcast = GetParam().op.mode;
    switch (cfg.dtypeIn) {
        case DType::U8:
            run_tensor_multiply_tensor<Rpp8u>(cfg, broadcast);
            break;
        case DType::I8:
            run_tensor_multiply_tensor<Rpp8s>(cfg, broadcast);
            break;
        case DType::F16:
            run_tensor_multiply_tensor<Rpp16f>(cfg, broadcast);
            break;
        case DType::F32:
            run_tensor_multiply_tensor<Rpp32f>(cfg, broadcast);
            break;
        default:
            FAIL() << "unsupported dtype for tensor_multiply_tensor";
    }
}

// The integer operands are folded to a small magnitude before the op runs (U8 -> [0,18],
// I8 -> [-14,14]): at full range nearly every product would saturate, so the test would only
// exercise the clamp. At these ranges most products are representable and a few percent still
// saturate, in every rank/broadcast combination.
//
// Restricted to U8/I8/F16/F32: the op also accepts the 16/32-bit integer types, but the
// framework's DType enum does not carry them for this op.
//
// 54 of these 72 cases are red against three documented kernel defects, deterministically. The
// golden and tolerances are deliberately left correct:
//   - 27: U8 (both backends) and I8 (HIP) overflow wraps modulo 256 instead of saturating
//     (18 * 15 -> 14, not 255).
//   -  9: HOST I8 returns 127 for every negative product, in range or not (-5 * 6 -> 127).
//   - 18: F16 is unimplemented -- the op returns RPP_SUCCESS on both backends and never writes
//     the output.
// F32 is green on both backends across every rank and broadcast mode.
//
// Note the HIP descriptors must be device-addressable (GenericDescriptor handles this): at
// rank >= 4 the non-broadcast kernel reads dims/strides on the device. Undocumented and
// rank-dependent.
INSTANTIATE_TEST_SUITE_P(Misc_Arithmetic, TensorMultiplyTensorTest,
                         ::testing::ValuesIn(nd_with_params<BroadcastParams>(
                             make_nd_configs({DType::U8, DType::I8, DType::F16, DType::F32},
                                             {2, 3, 4}),
                             {{Broadcast::None}, {Broadcast::Src1}, {Broadcast::Src2}})),
                         nd_op_config_name<BroadcastParams>);
