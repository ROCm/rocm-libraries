// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/KernelArguments.hpp>

using namespace TensileLite;

TEST(KernelArguments, HalfAlphaBetaUsePackedHalfAbiAndPreserveValues)
{
    KernelArguments args;
    ConstantVariant alpha = Half(2.0f);
    ConstantVariant beta  = Half(-2.0f);

    args.append("alpha", alpha, rocisa::DataType::Half);
    args.append("beta", beta, rocisa::DataType::Half);

    EXPECT_EQ(args.size(), 2 * sizeof(uint32_t));
    const auto alphaBits
        = *reinterpret_cast<const uint32_t*>(KernelArguments::const_iterator(args, "alpha")->first);
    const auto betaBits
        = *reinterpret_cast<const uint32_t*>(KernelArguments::const_iterator(args, "beta")->first);
    EXPECT_EQ(alphaBits & 0xffff, alphaBits >> 16);
    EXPECT_EQ(betaBits & 0xffff, betaBits >> 16);
}

TEST(KernelArguments, ExactPackedHalfBitEncodings)
{
    // Test exact packed-half scalar bit encodings for 0.0, 1.0, 0.5, 2.0, and -2.0
    KernelArguments args;
    args.append("zero", ConstantVariant(Half(0.0f)), rocisa::DataType::Half);
    args.append("one", ConstantVariant(Half(1.0f)), rocisa::DataType::Half);
    args.append("half", ConstantVariant(Half(0.5f)), rocisa::DataType::Half);
    args.append("alpha", ConstantVariant(Half(2.0f)), rocisa::DataType::Half);
    args.append("beta", ConstantVariant(Half(-2.0f)), rocisa::DataType::Half);

    const auto bits_zero
        = *reinterpret_cast<const uint16_t*>(KernelArguments::const_iterator(args, "zero")->first);
    const auto bits_one
        = *reinterpret_cast<const uint16_t*>(KernelArguments::const_iterator(args, "one")->first);
    const auto bits_half
        = *reinterpret_cast<const uint16_t*>(KernelArguments::const_iterator(args, "half")->first);
    const auto bits_alpha
        = *reinterpret_cast<const uint32_t*>(KernelArguments::const_iterator(args, "alpha")->first);
    const auto bits_beta
        = *reinterpret_cast<const uint32_t*>(KernelArguments::const_iterator(args, "beta")->first);

    EXPECT_EQ(bits_zero, 0x0000u);
    EXPECT_EQ(bits_one, 0x3c00u);
    EXPECT_EQ(bits_half, 0x3800u);
    EXPECT_EQ(bits_alpha, 0x40004000u);
    EXPECT_EQ(bits_beta, 0xc000c000u);

    // Verify duplication logic on alpha and beta fields
    EXPECT_EQ(bits_alpha & 0xffff, bits_alpha >> 16);
    EXPECT_EQ(bits_beta & 0xffff, bits_beta >> 16);
}

TEST(KernelArguments, CounterMatchesPackedHalfAlphaBetaAbiFieldByField)
{
    KernelArguments        args;
    KernelArgumentsCounter counter;
    ConstantVariant        alpha = Half(2.0f);
    ConstantVariant        beta  = Half(1.0f);

    args.append("alpha", alpha, rocisa::DataType::Half);
    args.append("beta", beta, rocisa::DataType::Half);

    counter.append("alpha", alpha, rocisa::DataType::Half);
    counter.append("beta", beta, rocisa::DataType::Half);

    EXPECT_EQ(args.size(), counter.size());
    EXPECT_EQ(args.size(), 8u);
}

TEST(KernelArguments, Solution23ProductionDeviceUserArgumentsLayout104Bytes)
{
    KernelArguments        args;
    KernelArgumentsCounter counter;

    // Production direct kernel arguments for Solution 23 (MT64x8x32) - 104 bytes exact direct ABI layout
    args.append("m", 64u);
    args.append("n", 64u);
    args.append("k", 32u);
    args.append("batchCount", 1u);

    void* d = (void*)0x1000;
    void* c = (void*)0x2000;
    void* a = (void*)0x3000;
    void* b = (void*)0x4000;

    args.append("d", d);
    args.append("c", c);
    args.append("a", a);
    args.append("b", b);

    args.append("offsetD", 0u);
    args.append("offsetC", 0u);
    args.append("offsetA", 0u);
    args.append("offsetB", 0u);

    args.append("strideD1", 64u);
    args.append("strideD2", 64u * 64u);
    args.append("strideC1", 64u);
    args.append("strideC2", 64u * 64u);
    args.append("strideA1", 64u);
    args.append("strideA2", 64u * 32u);
    args.append("strideB1", 32u);
    args.append("strideB2", 32u * 64u);

    args.append("alpha", ConstantVariant(Half(2.0f)), rocisa::DataType::Half);
    args.append("beta", ConstantVariant(Half(0.5f)), rocisa::DataType::Half);

    // Counter matching
    counter.append("m", 64u);
    counter.append("n", 64u);
    counter.append("k", 32u);
    counter.append("batchCount", 1u);

    counter.append("d", d);
    counter.append("c", c);
    counter.append("a", a);
    counter.append("b", b);

    counter.append("offsetD", 0u);
    counter.append("offsetC", 0u);
    counter.append("offsetA", 0u);
    counter.append("offsetB", 0u);

    counter.append("strideD1", 64u);
    counter.append("strideD2", 64u * 64u);
    counter.append("strideC1", 64u);
    counter.append("strideC2", 64u * 64u);
    counter.append("strideA1", 64u);
    counter.append("strideA2", 64u * 32u);
    counter.append("strideB1", 32u);
    counter.append("strideB2", 32u * 64u);

    counter.append("alpha", ConstantVariant(Half(2.0f)), rocisa::DataType::Half);
    counter.append("beta", ConstantVariant(Half(0.5f)), rocisa::DataType::Half);

    // Assert exact 104-byte size and counter agreement
    EXPECT_EQ(args.size(), 104u);
    EXPECT_EQ(counter.size(), args.size());

    // Assert every field size matches field-by-field
    EXPECT_EQ(KernelArguments::const_iterator(args, "m")->second, 4u);
    EXPECT_EQ(KernelArguments::const_iterator(args, "d")->second, 8u);
    EXPECT_EQ(KernelArguments::const_iterator(args, "strideD1")->second, 4u);
    EXPECT_EQ(KernelArguments::const_iterator(args, "alpha")->second, 4u);
    EXPECT_EQ(KernelArguments::const_iterator(args, "beta")->second, 4u);
}

TEST(KernelArguments, ProductionDeviceUserArgumentsPackedHalfLayout)
{
    // Verify production DeviceUserArguments<float> packed struct alignment, offsets, and 196-byte size
    EXPECT_EQ(sizeof(DeviceUserArguments<float>), 196u);

    EXPECT_EQ(offsetof(DeviceUserArguments<float>, m), 0u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, n), 4u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, batch), 8u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, k), 12u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, d), 16u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, c), 24u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, a), 32u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, b), 40u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideD1), 48u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideD2), 52u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideC1), 56u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideC2), 60u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideA1), 64u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideA2), 68u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideB1), 72u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideB2), 76u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, alpha), 80u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, beta), 96u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, scaleA), 112u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, scaleB), 120u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, scaleC), 128u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, scaleD), 136u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, scaleAlphaVec), 144u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, bias), 152u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, biasType), 160u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, reserved), 164u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, e), 168u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideE1), 176u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, strideE2), 180u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, act0), 184u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, act1), 188u);
    EXPECT_EQ(offsetof(DeviceUserArguments<float>, activationType), 192u);

    // Verify API sets arguments equivalent to direct struct manipulation
    std::vector<ContractionSolution::Problem> problems;
    problems.push_back(ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            rocisa::DataType::Half,
                                                            rocisa::DataType::Half,
                                                            rocisa::DataType::Half,
                                                            rocisa::DataType::Half,
                                                            64,
                                                            64,
                                                            32,
                                                            1,
                                                            64,
                                                            64 * 32, // lda, aStride
                                                            32,
                                                            32 * 64, // ldb, bStride
                                                            64,
                                                            64 * 64, // ldc, cStride
                                                            64,
                                                            64 * 64, // ldd, dStride
                                                            0.5 // beta
                                                            ));

    problems[0].setAlphaType(rocisa::DataType::Half);
    problems[0].setBetaType(rocisa::DataType::Half);

    ContractionGroupedInputs inputs;
    inputs.grouped.resize(1);

    inputs.grouped[0].d = (void*)0x1000;
    inputs.grouped[0].c = (void*)0x2000;
    inputs.grouped[0].a = (void*)0x3000;
    inputs.grouped[0].b = (void*)0x4000;

    inputs.grouped[0].alpha = ConstantVariant(Half(2.0f));
    inputs.grouped[0].beta  = ConstantVariant(Half(0.5f));

    DeviceUserArguments<float> userArgs{};
    setDeviceUserArgs(problems, inputs, &userArgs);

    EXPECT_EQ(userArgs.m, 64u);
    EXPECT_EQ(userArgs.n, 64u);
    EXPECT_EQ(userArgs.k, 32u);
    EXPECT_EQ(userArgs.batch, 1u);
    EXPECT_EQ(userArgs.d, (void*)0x1000);
    EXPECT_EQ(userArgs.c, (void*)0x2000);
    EXPECT_EQ(userArgs.a, (void*)0x3000);
    EXPECT_EQ(userArgs.b, (void*)0x4000);
    EXPECT_EQ(userArgs.strideD1, 64u);
    EXPECT_EQ(userArgs.strideD2, 64 * 64u);
    EXPECT_EQ(userArgs.strideC1, 64u);
    EXPECT_EQ(userArgs.strideC2, 64 * 64u);
    EXPECT_EQ(userArgs.strideA1, 64u);
    EXPECT_EQ(userArgs.strideA2, 64 * 32u);
    EXPECT_EQ(userArgs.strideB1, 32u);
    EXPECT_EQ(userArgs.strideB2, 32 * 64u);
    EXPECT_EQ(*reinterpret_cast<const uint32_t*>(userArgs.alpha), 0x40004000u);
    EXPECT_EQ(*reinterpret_cast<const uint32_t*>(userArgs.beta), 0x38003800u);
}
