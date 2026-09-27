// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include <miopen/handle.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "get_handle.hpp"

namespace {

constexpr std::size_t value_count = 7;

struct ConversionData
{
    std::array<float, value_count> input;
    std::array<unsigned char, value_count> encoded;
    std::array<float, value_count> decoded;
};

struct Fp8Format
{
    std::string name;
    int ieee_exponent_bias;
    ConversionData fp8;
    ConversionData bfp8;
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
    for(unsigned int i = 0; i < 7; ++i)
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
    const auto& data = fp8_type == Fp8Type::Fp8 ? format.fp8 : format.bfp8;
    const std::vector<float> input(data.input.begin(), data.input.end());
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
                  std::vector<unsigned char>(data.encoded.begin(), data.encoded.end()));
    }
    else
    {
        EXPECT_EQ(handle.Read<unsigned char>(bfp8_dev, value_count),
                  std::vector<unsigned char>(data.encoded.begin(), data.encoded.end()));
    }

    const auto output = fp8_type == Fp8Type::Fp8 ? handle.Read<float>(fp8_out_dev, value_count)
                                                 : handle.Read<float>(bfp8_out_dev, value_count);
    for(std::size_t i = 0; i < value_count; ++i)
    {
        SCOPED_TRACE("value index " + std::to_string(i));
        if(std::isnan(data.decoded[i]))
            EXPECT_TRUE(std::isnan(output[i]));
        else
            EXPECT_EQ(output[i], data.decoded[i]);

        if(data.decoded[i] == 0.0f)
            EXPECT_EQ(std::signbit(output[i]), std::signbit(data.decoded[i]));
    }
}

void TestFormats(Fp8Type fp8_type)
{
    constexpr auto inf                     = std::numeric_limits<float>::infinity();
    constexpr auto nan                     = std::numeric_limits<float>::quiet_NaN();
    constexpr auto max                     = std::numeric_limits<float>::max();
    const std::array<Fp8Format, 2> formats = {
        Fp8Format{"fnuz",
                  0,
                  {{1.1f, -1.1f, 0x1p-10f, max, inf, nan, -0.0f},
                   {0x41, 0xC1, 0x01, 0x7F, 0x80, 0x80, 0x00},
                   {1.125f, -1.125f, 0x1p-10f, 240.0f, nan, nan, 0.0f}},
                  {{1.1f, -1.1f, 0x1p-17f, max, inf, nan, -0.0f},
                   {0x40, 0xC0, 0x01, 0x7F, 0x80, 0x80, 0x00},
                   {1.0f, -1.0f, 0x1p-17f, 57344.0f, nan, nan, 0.0f}}},
        Fp8Format{"ieee",
                  1,
                  {{1.1f, -1.1f, 0x1p-9f, max, inf, nan, -0.0f},
                   {0x39, 0xB9, 0x01, 0x7E, 0x7F, 0x7F, 0x80},
                   {1.125f, -1.125f, 0x1p-9f, 448.0f, nan, nan, -0.0f}},
                  {{1.1f, -1.1f, 0x1p-16f, max, inf, nan, -0.0f},
                   {0x3C, 0xBC, 0x01, 0x7B, 0x7C, 0x7D, 0x80},
                   {1.0f, -1.0f, 0x1p-16f, 57344.0f, inf, nan, -0.0f}}}};

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
