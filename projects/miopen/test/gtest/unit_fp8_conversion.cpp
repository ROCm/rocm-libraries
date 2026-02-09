// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include <miopen/handle.hpp>

#include <array>
#include <string>
#include <vector>

#include "get_handle.hpp"

namespace {

constexpr std::size_t value_count = 5;

struct Fp8Format
{
    std::string name;
    int ieee_exponent_bias;
    std::array<unsigned char, value_count> fp8;
    std::array<unsigned char, value_count> bfp8;
};

std::string GetKernelSource()
{
    return R"(
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif
#include "fp8_dev.hpp"

extern "C" __global__ void fp8_conversion_test(const float* input,
                                                 unsigned char* fp8,
                                                 unsigned char* bfp8,
                                                 float* fp8_output,
                                                 float* bfp8_output)
{
    for(unsigned int i = 0; i < 5; ++i)
    {
        fp8[i]        = float_to_fp8(input[i]);
        bfp8[i]       = float_to_bfp8(input[i]);
        fp8_output[i] = fp8_to_float(fp8[i]);
        bfp8_output[i] = bfp8_to_float(bfp8[i]);
    }
}
)";
}

enum class Fp8Type
{
    Fp8,
    Bfp8
};

void TestConversions(const Fp8Format& format, bool force_software_fallback, Fp8Type fp8_type)
{
    const std::vector<float> input = {0.0f, 0.5f, 1.0f, 2.0f, -1.0f};
    const std::vector<unsigned char> empty_bytes(value_count);
    const std::vector<float> empty_floats(value_count);

    auto& handle      = get_handle();
    auto input_dev    = handle.Write(input);
    auto fp8_dev      = handle.Write(empty_bytes);
    auto bfp8_dev     = handle.Write(empty_bytes);
    auto fp8_out_dev  = handle.Write(empty_floats);
    auto bfp8_out_dev = handle.Write(empty_floats);

    auto options = std::string{" -DMIOPEN_USE_FP8=1 -DMIOPEN_USE_BFP8=1"} +
                   " -DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" + std::to_string(format.ieee_exponent_bias) +
                   " -DMIOPEN_FP8_CLIPPING=1";
    if(force_software_fallback)
        options += " -DMIOPEN_DISABLE_NATIVE_FP8_CONVERSION=1";

    const auto network_config = format.name + (force_software_fallback ? "_software" : "_native");
    handle.AddKernel("Fp8ConversionTest",
                     network_config,
                     "fp8_conversion_test.cpp",
                     "fp8_conversion_test",
                     {1, 1, 1},
                     {1, 1, 1},
                     options,
                     0,
                     GetKernelSource())(
        input_dev.get(), fp8_dev.get(), bfp8_dev.get(), fp8_out_dev.get(), bfp8_out_dev.get());

    if(fp8_type == Fp8Type::Fp8)
    {
        EXPECT_EQ(handle.Read<unsigned char>(fp8_dev, value_count),
                  std::vector<unsigned char>(format.fp8.begin(), format.fp8.end()));
        EXPECT_EQ(handle.Read<float>(fp8_out_dev, value_count), input);
    }
    else
    {
        EXPECT_EQ(handle.Read<unsigned char>(bfp8_dev, value_count),
                  std::vector<unsigned char>(format.bfp8.begin(), format.bfp8.end()));
        EXPECT_EQ(handle.Read<float>(bfp8_out_dev, value_count), input);
    }
}

void TestFormats(Fp8Type fp8_type)
{
    const std::array<Fp8Format, 2> formats = {
        Fp8Format{"fnuz", 0, {0x00, 0x38, 0x40, 0x48, 0xC0}, {0x00, 0x3C, 0x40, 0x44, 0xC0}},
        Fp8Format{"ieee", 1, {0x00, 0x30, 0x38, 0x40, 0xB8}, {0x00, 0x38, 0x3C, 0x40, 0xBC}}};

    for(const auto& format : formats)
    {
        SCOPED_TRACE(format.name + " default path");
        TestConversions(format, false, fp8_type);
    }

    for(const auto& format : formats)
    {
        SCOPED_TRACE(format.name + " software fallback");
        TestConversions(format, true, fp8_type);
    }
}

} // namespace

// NOLINTNEXTLINE(google-readability-avoid-underscore-in-googletest-name)
TEST(GPU_Fp8Conversion_FP8, DefaultAndSoftwareFallback) { TestFormats(Fp8Type::Fp8); }

// NOLINTNEXTLINE(google-readability-avoid-underscore-in-googletest-name)
TEST(GPU_Bfp8Conversion_BFP8, DefaultAndSoftwareFallback) { TestFormats(Fp8Type::Bfp8); }

#else

// NOLINTNEXTLINE(google-readability-avoid-underscore-in-googletest-name)
TEST(GPU_Fp8Conversion_FP8, SkippedForNonHipBackend)
{
    GTEST_SKIP() << "HIP backend not available";
}

// NOLINTNEXTLINE(google-readability-avoid-underscore-in-googletest-name)
TEST(GPU_Bfp8Conversion_BFP8, SkippedForNonHipBackend)
{
    GTEST_SKIP() << "HIP backend not available";
}

#endif
