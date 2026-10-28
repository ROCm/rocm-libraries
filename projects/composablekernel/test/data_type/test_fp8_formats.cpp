// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstdint>

#include "gtest/gtest.h"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/library/utility/device_memory.hpp"

namespace {

template <typename T>
__global__ void convert_fp8_formats(const float* input, uint8_t* encoded, float* decoded)
{
    const auto i = threadIdx.x;
    encoded[i]   = ck::bit_cast<uint8_t>(ck::type_convert<T>(input[i]));
    decoded[i]   = ck::type_convert<float>(ck::bit_cast<T>(static_cast<uint8_t>(i)));
}

template <typename T>
class FP8Formats : public ::testing::Test
{
};

using Formats = ::testing::Types<ck::f8_fnuz_t, ck::bf8_fnuz_t, ck::f8_ocp_t, ck::bf8_ocp_t>;
TYPED_TEST_SUITE(FP8Formats, Formats);

TYPED_TEST(FP8Formats, HostAndDeviceAgreeForEveryEncoding)
{
    // Exercise all exponent/mantissa combinations, including subnormals, signed
    // zero and NaNs. Re-encode finite values, which are exactly representable and
    // therefore independent of the configured rounding mode.
    std::array<float, 256> input{}, reference{}, decoded{};
    std::array<uint8_t, 256> encoded{};
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        reference[i] = ck::type_convert<float>(ck::bit_cast<TypeParam>(static_cast<uint8_t>(i)));
        input[i]     = std::isfinite(reference[i]) ? reference[i] : 1.0f;
    }

    ck::DeviceMem input_device(sizeof(input));
    ck::DeviceMem encoded_device(sizeof(encoded));
    ck::DeviceMem decoded_device(sizeof(decoded));
    input_device.ToDevice(input.data());
    hipLaunchKernelGGL(convert_fp8_formats<TypeParam>,
                       dim3(1),
                       dim3(256),
                       0,
                       nullptr,
                       static_cast<const float*>(input_device.GetDeviceBuffer()),
                       static_cast<uint8_t*>(encoded_device.GetDeviceBuffer()),
                       static_cast<float*>(decoded_device.GetDeviceBuffer()));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    encoded_device.FromDevice(encoded.data());
    decoded_device.FromDevice(decoded.data());

    for(std::size_t i = 0; i < input.size(); ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(encoded[i], ck::bit_cast<uint8_t>(ck::type_convert<TypeParam>(input[i])));
        if(std::isnan(reference[i]))
        {
            EXPECT_TRUE(std::isnan(decoded[i]));
        }
        else
        {
            EXPECT_EQ(decoded[i], reference[i]);
            EXPECT_EQ(std::signbit(decoded[i]), std::signbit(reference[i]));
        }
    }
}

// Host-encoded inputs distinguish native decoding from a format mismatch. Use
// exact values and normalize by K so every output is -2 for either MFMA shape.
template <typename A, typename B, int Size>
__global__ void multiply_fp8_formats(A a, B b, float* output)
{
#if defined(__gfx942__) || defined(__gfx950__)
    constexpr auto instruction = ck::MfmaSelector<A, Size, Size, B>::selected_mfma;
    constexpr int KPack        = instruction.k_per_blk;
    constexpr int Registers    = instruction.num_regs_per_blk;
    ck::vector_type<A, KPack> av;
    ck::vector_type<B, KPack> bv;
    ck::vector_type<float, Registers> cv;
    ck::static_for<0, KPack, 1>{}([&](auto i) {
        av.template AsType<A>()(i) = a;
        bv.template AsType<B>()(i) = b;
    });
    ck::static_for<0, Registers, 1>{}([&](auto i) { cv.template AsType<float>()(i) = 0.0f; });
    instruction.template run<Size, Size>(
        av.template AsType<typename ck::vector_type<A, KPack>::type>()[ck::Number<0>{}],
        bv.template AsType<typename ck::vector_type<B, KPack>::type>()[ck::Number<0>{}],
        cv);
    ck::static_for<0, Registers, 1>{}([&](auto i) {
        output[threadIdx.x * Registers + i] =
            cv.template AsType<float>()[i] / (KPack * instruction.num_input_blks);
    });
#else
    (void)a;
    (void)b;
    (void)output;
#endif
}

template <typename A, typename B, int Size>
void check_fp8_multiply()
{
    std::array<float, Size * Size> output{};
    ck::DeviceMem result(sizeof(output));
    hipLaunchKernelGGL((multiply_fp8_formats<A, B, Size>),
                       dim3(1),
                       dim3(64),
                       0,
                       nullptr,
                       ck::type_convert<A>(1.0f),
                       ck::type_convert<B>(-2.0f),
                       static_cast<float*>(result.GetDeviceBuffer()));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    result.FromDevice(output.data());
    for(std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_EQ(output[i], -2.0f) << "element " << i;
    }
}

TYPED_TEST(FP8Formats, MatrixInstructionsPreserveHostEncoding)
{
    const auto device = ck::get_device_name();
    if(device != "gfx942" && device != "gfx950")
        GTEST_SKIP() << "Requires gfx942 or gfx950 MFMA";

    check_fp8_multiply<TypeParam, TypeParam, 16>();
    check_fp8_multiply<TypeParam, TypeParam, 32>();
}

TEST(FP8MixedFormats, MatrixInstructionsPreserveBothOperandEncodings)
{
    const auto device = ck::get_device_name();
    if(device != "gfx942" && device != "gfx950")
        GTEST_SKIP() << "Requires gfx942 or gfx950 MFMA";

    // In each case one operand is nonnative. This also checks that either
    // operand can select the single-rate path with its FP32 MFMA fallback.
    check_fp8_multiply<ck::f8_ocp_t, ck::bf8_fnuz_t, 16>();
    check_fp8_multiply<ck::bf8_fnuz_t, ck::f8_ocp_t, 16>();
    check_fp8_multiply<ck::f8_ocp_t, ck::bf8_fnuz_t, 32>();
    check_fp8_multiply<ck::bf8_fnuz_t, ck::f8_ocp_t, 32>();
}

} // namespace
