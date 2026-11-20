// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include "gtest_common.hpp"
#include <hip_float8.hpp>
#include <miopen/bfloat16.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "get_handle.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_CHECK_NUMERICS)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_LOGGING_QUIETING_DISABLE)

namespace {

using float8_fnuz  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8_fnuz = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

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
    else if constexpr(std::is_same_v<T, bfloat16>)
        return "check_numerics_bf16";
    else if constexpr(std::is_same_v<T, float8_fnuz>)
        return "check_numerics_fp8";
    else
        return "check_numerics_bf8";
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
        " -DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" +
        std::to_string(miopen::UsesIeeeExponentBias(handle.GetDeviceName()) ? 1 : 0);
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
    bool compute_stats        = true;
    std::vector<T> ieee_input = {};
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
std::vector<T> MakeRawInput(std::initializer_list<uint8_t> values)
{
    std::vector<T> input;
    input.reserve(values.size());
    for(uint8_t value : values)
        input.emplace_back(value);
    return input;
}

template <class T>
std::vector<NumericsCase<T>> GetFp8Cases()
{
    constexpr bool fp8        = std::is_same_v<T, float8_fnuz>;
    constexpr float largest   = fp8 ? 240.0f : 57344.0f;
    constexpr uint8_t one     = fp8 ? 0x38 : 0x3c;
    constexpr uint8_t maximum = fp8 ? 0x77 : 0x7b;
    constexpr uint8_t nan     = fp8 ? 0x7c : 0x7e;
    // Raw FNUZ and IEEE-bias encodings represent the same independently known values.
    // FNUZ has no infinity: 0x80 is NaN, not negative zero or infinity.
    return {{"PositiveOnly",
             MakeRawInput<T>({0x40, 0x7f}),
             1.0f,
             largest,
             0,
             0,
             0,
             true,
             MakeRawInput<T>({one, maximum})},
            {"NegativeOnly",
             MakeRawInput<T>({0xc0, 0xff}),
             -largest,
             -1.0f,
             0,
             0,
             0,
             true,
             MakeRawInput<T>({uint8_t(one | 0x80), uint8_t(maximum | 0x80)})},
            {"NanAndZeroWithStats",
             MakeRawInput<T>({0x00, 0x80, 0x40, 0xc0}),
             -1.0f,
             1.0f,
             1,
             1,
             0,
             true,
             MakeRawInput<T>({0x00, nan, one, uint8_t(one | 0x80)})},
            {"NanAndZeroWithoutStats",
             MakeRawInput<T>({0x00, 0x80, 0x40, 0xc0}),
             -1.0f,
             1.0f,
             1,
             1,
             0,
             false,
             MakeRawInput<T>({0x00, nan, one, uint8_t(one | 0x80)})}};
}

template <class T>
class CheckNumerics : public ::testing::TestWithParam<NumericsCase<T>>
{
public:
    void RunTest()
    {
        const auto& test = this->GetParam();
        const bool ieee =
            !test.ieee_input.empty() && miopen::UsesIeeeExponentBias(get_handle().GetDeviceName());
        const auto& input = ieee ? test.ieee_input : test.input;
        const auto result = RunKernel(input, test.compute_stats);
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

// Like the logging tests, redirect cerr with RAII so exceptions and fatal assertions
// cannot leave the following test's log output redirected.
struct NumericsLogCapture
{
    std::ostringstream stream;
    std::streambuf* previous = std::cerr.rdbuf(stream.rdbuf());
    ~NumericsLogCapture() { std::cerr.rdbuf(previous); }
};

void ExpectLogValue(const std::string& log,
                    const std::string& key,
                    float expected,
                    float tolerance = 0.0f)
{
    const auto marker = " " + key + "=";
    const auto offset = log.find(marker);
    ASSERT_NE(offset, std::string::npos) << "Missing numerical field " << key << " in " << log;
    const float actual = std::stof(log.substr(offset + marker.size()));
    if(std::isnan(expected))
        EXPECT_TRUE(std::isnan(actual)) << key;
    else if(tolerance == 0.0f)
        EXPECT_EQ(actual, expected) << key;
    else
        EXPECT_NEAR(actual, expected, tolerance) << key;
}

enum class NumericsInputKind
{
    FiniteGridStride,
    Nonfinite,
    Empty
};

class CheckNumericsInput : public ::testing::TestWithParam<NumericsInputKind>
{
protected:
    // Explicit severity works for both debug and release logging defaults. Override
    // quieting through a scoped environment, not the suite-global LoggingQuiet flag.
    ScopedEnvironment<int> numerics{
        MIOPEN_CHECK_NUMERICS, miopen::CheckNumerics::Info | miopen::CheckNumerics::ComputeStats};
    ScopedEnvironment<std::string> log_level{MIOPEN_LOG_LEVEL, "6"};
    ScopedEnvironment<std::string> quieting{MIOPEN_DEBUG_LOGGING_QUIETING_DISABLE, "1"};
};

} // namespace

using GPU_CheckNumerics_FP32  = CheckNumerics<float>;
using GPU_CheckNumerics_FP16  = CheckNumerics<half_float::half>;
using GPU_CheckNumerics_BFP16 = CheckNumerics<bfloat16>;

using GPU_CheckNumerics_FP8       = CheckNumerics<float8_fnuz>;
using GPU_CheckNumerics_BFP8      = CheckNumerics<bfloat8_fnuz>;
using GPU_CheckNumericsInput_FP32 = CheckNumericsInput;

TEST_P(GPU_CheckNumerics_FP32, ExtremaAndFlags) { this->RunTest(); }
TEST_P(GPU_CheckNumerics_FP16, ExtremaAndFlags) { this->RunTest(); }
TEST_P(GPU_CheckNumerics_BFP16, ExtremaAndFlags) { this->RunTest(); }
TEST_P(GPU_CheckNumerics_FP8, ExtremaAndFlags) { this->RunTest(); }
TEST_P(GPU_CheckNumerics_BFP8, ExtremaAndFlags) { this->RunTest(); }

TEST_P(GPU_CheckNumericsInput_FP32, StatsAndFlags)
{
    auto& handle = get_handle();
    if(GetParam() == NumericsInputKind::Empty)
    {
        // Internal JSON imports can represent empty tensors even though constructors reject them.
        const nlohmann::json metadata = {
            {"lengths", {0}}, {"strides", {1}}, {"packed", true}, {"type", miopenFloat}};
        const auto desc = metadata.get<miopen::TensorDescriptor>();
        ASSERT_EQ(desc.GetElementSize(), 0);
        EXPECT_FALSE(miopen::checkNumericsInput(handle, desc, nullptr));
        return;
    }
    const bool abnormal = GetParam() == NumericsInputKind::Nonfinite;
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> input;
    if(abnormal)
    {
        input = {2.0f, 0.0f, nan, inf, -inf, -4.0f};
    }
    else
    {
        // More than the capped host grid can cover in one pass, across multiple blocks.
        // This validates every element, not the exact geometry: an uncapped grid is valid too.
        const size_t count = size_t{256} * 8 * handle.GetMaxHardwareComputeUnits() + 512;
        input.resize(count);
        for(size_t i = 0; i < count; i += 4)
        {
            input[i]     = -4.0f;
            input[i + 1] = 2.0f;
            input[i + 2] = 0.0f;
            input[i + 3] = 8.0f;
        }
        // Put unique extrema beyond the first grid-stride pass.
        input[count - 4] = -16.0f;
        input[count - 1] = 20.0f;
    }
    auto input_dev = handle.Write(input);
    const miopen::TensorDescriptor desc(miopenFloat, {input.size()});

    NumericsLogCapture capture;
    EXPECT_EQ(miopen::checkNumericsInput(handle, desc, input_dev.get()), abnormal);
    const auto log = capture.stream.str();
    // Logs use six significant digits; allow their rounding, not reduction error.
    ExpectLogValue(log, "mean", abnormal ? nan : 1.5f, 5e-6f);
    ExpectLogValue(log, "absmean", abnormal ? nan : 3.5f + 24.0f / input.size(), 5e-6f);
    ExpectLogValue(log, "min", abnormal ? -inf : -16.0f);
    ExpectLogValue(log, "max", abnormal ? inf : 20.0f);
    ExpectLogValue(log, "zeros", 1);
    ExpectLogValue(log, "nans", abnormal ? 1 : 0);
    ExpectLogValue(log, "infs", abnormal ? 1 : 0);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumericsInput_FP32,
                         ::testing::Values(NumericsInputKind::FiniteGridStride,
                                           NumericsInputKind::Nonfinite,
                                           NumericsInputKind::Empty),
                         [](const ::testing::TestParamInfo<NumericsInputKind>& info) {
                             if(info.param == NumericsInputKind::Empty)
                                 return "Empty";
                             return info.param == NumericsInputKind::Nonfinite ? "Nonfinite"
                                                                               : "FiniteGridStride";
                         });

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
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_FP8,
                         ::testing::ValuesIn(GetFp8Cases<float8_fnuz>()),
                         CaseName{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_BFP8,
                         ::testing::ValuesIn(GetFp8Cases<bfloat8_fnuz>()),
                         CaseName{});

#endif
