// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include <half/half.hpp>
#include <miopen/bfloat16.hpp>
#include <miopen/handle.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "get_handle.hpp"

namespace {

// Private kernel ABI: keep in sync with MIOpenCheckNumerics.cpp/check_numerics.cpp.
struct NumericsResult
{
    float sum    = 0.0f;
    float absSum = 0.0f;
    float min    = std::numeric_limits<float>::infinity();
    float max    = -std::numeric_limits<float>::infinity();
    int hasZero  = 0;
    int hasNan   = 0;
    int hasInf   = 0;
};

template <class T>
const char* KernelName()
{
    if constexpr(std::is_same_v<T, float>)
        return "check_numerics_fp32";
    else if constexpr(std::is_same_v<T, half_float::half>)
        return "check_numerics_fp16";
    else
        return "check_numerics_bf16";
}

template <class T>
NumericsResult RunKernel(const std::vector<T>& input, bool compute_stats)
{
    auto& handle   = get_handle();
    auto input_dev = handle.Write(input);
    NumericsResult result;
    auto result_dev = handle.Create(sizeof(result));
    handle.WriteTo(&result, result_dev, sizeof(result));

    const std::string options =
        " -DMIOPEN_FP8_CLIPPING=" + std::to_string(MIOPEN_FP8_CLIPPING) +
        " -DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" + std::to_string(MIOPEN_FP8_IEEE_EXPONENT_BIAS);
    // Two blocks: a partly occupied block and a completely idle block. Both must
    // contribute neutral extrema, not zero or a finite numeric_limits sentinel.
    handle.AddKernel("CheckNumericsRegression",
                     KernelName<T>(),
                     "MIOpenCheckNumerics.cpp",
                     KernelName<T>(),
                     {256, 1, 1},
                     {512, 1, 1},
                     options)(input_dev.get(), input.size(), result_dev.get(), compute_stats);
    handle.ReadTo(&result, result_dev, sizeof(result));
    return result;
}

template <class T>
struct NumericsCase
{
    std::string name;
    std::vector<T> input;
    float min;
    float max;
    int has_zero;
    int has_nan;
    int has_inf;
    bool compute_stats = true;
};

template <class T>
std::vector<T> MakeInput(std::initializer_list<float> values)
{
    std::vector<T> input;
    input.reserve(values.size());
    for(float value : values)
        input.emplace_back(value);
    return input;
}

template <class T>
std::vector<NumericsCase<T>> GetCases()
{
    constexpr float inf                = std::numeric_limits<float>::infinity();
    constexpr float nan                = std::numeric_limits<float>::quiet_NaN();
    std::vector<NumericsCase<T>> cases = {
        {"PositiveOnly", MakeInput<T>({4.0f, 1.0f, 2.0f}), 1.0f, 4.0f, 0, 0, 0},
        {"NegativeOnly", MakeInput<T>({-4.0f, -1.0f, -2.0f}), -4.0f, -1.0f, 0, 0, 0},
        {"Zeros", MakeInput<T>({0.0f, -0.0f}), 0.0f, 0.0f, 1, 0, 0},
        {"PositiveInfinities", MakeInput<T>({inf, inf}), inf, inf, 0, 0, 1},
        {"NegativeInfinities", MakeInput<T>({-inf, -inf}), -inf, -inf, 0, 0, 1},
        {"NonfiniteWithStats", MakeInput<T>({0.0f, nan, inf, -inf}), -inf, inf, 1, 1, 1},
        {"NonfiniteWithoutStats", MakeInput<T>({0.0f, nan, inf, -inf}), -inf, inf, 1, 1, 1, false}};

    if constexpr(std::is_same_v<T, bfloat16>)
    {
        // Exact raw storage avoids rounding or flushing subnormals in input generation.
        constexpr float smallest = 0x1p-133f;
        cases.push_back({"PositiveSubnormals",
                         {bfloat16::generate(0x0001), bfloat16::generate(0x0002)},
                         smallest,
                         2.0f * smallest,
                         0,
                         0,
                         0});
        cases.push_back({"NegativeSubnormals",
                         {bfloat16::generate(0x8001), bfloat16::generate(0x8002)},
                         -2.0f * smallest,
                         -smallest,
                         0,
                         0,
                         0});
    }
    return cases;
}

template <class T>
class CheckNumerics : public ::testing::TestWithParam<NumericsCase<T>>
{
public:
    void RunTest()
    {
        const auto& test  = this->GetParam();
        const auto result = RunKernel(test.input, test.compute_stats);
        if(test.compute_stats)
        {
            // Exact equality matters: ULP-based float comparisons can accept zero
            // instead of a BF16 subnormal, hiding conversion/extrema regressions.
            EXPECT_EQ(result.min, test.min);
            EXPECT_EQ(result.max, test.max);
        }
        EXPECT_EQ(result.hasZero, test.has_zero);
        EXPECT_EQ(result.hasNan, test.has_nan);
        EXPECT_EQ(result.hasInf, test.has_inf);
    }
};

struct CaseName
{
    template <class T>
    std::string operator()(const ::testing::TestParamInfo<NumericsCase<T>>& info) const
    {
        return info.param.name;
    }
};

} // namespace

using GPU_CheckNumerics_FP32  = CheckNumerics<float>;
using GPU_CheckNumerics_FP16  = CheckNumerics<half_float::half>;
using GPU_CheckNumerics_BFP16 = CheckNumerics<bfloat16>;

TEST_P(GPU_CheckNumerics_FP32, ExtremaAndFlags) { this->RunTest(); }
TEST_P(GPU_CheckNumerics_FP16, ExtremaAndFlags) { this->RunTest(); }
TEST_P(GPU_CheckNumerics_BFP16, ExtremaAndFlags) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_FP32,
                         ::testing::ValuesIn(GetCases<float>()),
                         CaseName{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_FP16,
                         ::testing::ValuesIn(GetCases<half_float::half>()),
                         CaseName{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_BFP16,
                         ::testing::ValuesIn(GetCases<bfloat16>()),
                         CaseName{});

#endif
