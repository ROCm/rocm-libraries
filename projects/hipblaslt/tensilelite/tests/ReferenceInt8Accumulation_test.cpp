// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    struct DotProductCase
    {
        const char* name;
        size_t      prefixCount;
        int8_t      prefixB;
        size_t      extraTerms;
        bool        floatAlpha;
        float       alpha;
        float       beta;
        int32_t     c;
        int32_t     expected;
    };

    void PrintTo(const DotProductCase& test, std::ostream* out)
    {
        *out << test.name;
    }

    // Regression: https://github.com/ROCm/rocm-libraries/pull/15266#discussion_r4554390239
    // 1024 products of (-128)*(-128) reach 2^24 exactly. At that magnitude,
    // float accumulation loses each trailing 1*1 term, whereas integer
    // accumulation preserves both terms before the epilogue conversion.
    // One trailing term rounds away even when the completed sum is cast once.
    // The negative case reaches -2^24 with 2048 products of (-128)*64.
    constexpr DotProductCase cases[] = {
        {"AtFloatExactLimit", 1024, -128, 0, true, 0.5f, 0.0f, 0, 8388608},
        {"RoundsCompletedSumOnce", 1024, -128, 1, true, 0.5f, 0.0f, 0, 8388608},
        {"PreservesTrailingTerms", 1024, -128, 2, true, 0.5f, 0.0f, 0, 8388609},
        {"FractionalBeta", 1024, -128, 2, true, 0.5f, 0.25f, 4, 8388610},
        {"NegativeDotProduct", 2048, 64, 2, true, 0.5f, 0.0f, 0, -8388609},
        {"IntegerAlphaControl", 1024, -128, 2, false, 1.0f, 0.0f, 0, 16777218},
    };

    ContractionProblemGemm makeProblem(size_t           k,
                                       bool             floatAlpha,
                                       rocisa::DataType outputType = rocisa::DataType::Int32)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            rocisa::DataType::Int8,
                                                            rocisa::DataType::Int8,
                                                            outputType,
                                                            outputType,
                                                            1,
                                                            1,
                                                            k,
                                                            1,
                                                            1,
                                                            k,
                                                            k,
                                                            k,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            0.0);
        problem.setComputeInputTypeA(rocisa::DataType::Int8);
        problem.setComputeInputTypeB(rocisa::DataType::Int8);
        const auto alphaType = floatAlpha ? rocisa::DataType::Float : rocisa::DataType::Int32;
        problem.setAlphaType(alphaType);
        problem.setBetaType(alphaType);
        return problem;
    }
} // namespace

class ReferenceInt8Accumulation : public ::testing::TestWithParam<std::tuple<DotProductCase, bool>>
{
};

TEST_P(ReferenceInt8Accumulation, IntegerDotProductBeforeEpilogue)
{
    const auto& [test, tryFastPath] = GetParam();
    const size_t        k           = test.prefixCount + test.extraTerms;
    std::vector<int8_t> a(k, -128);
    std::vector<int8_t> b(k, test.prefixB);
    const int8_t        tailB = test.prefixB > 0 ? -1 : 1;

    for(size_t i = test.prefixCount; i < k; ++i)
    {
        a[i] = 1;
        b[i] = tailB;
    }

    int64_t integerDot = 0;
    for(size_t i = 0; i < k; ++i)
    {
        integerDot += int64_t(a[i]) * int64_t(b[i]);
        // Every prefix fits Int32, excluding overflow as a source of mismatch.
        ASSERT_GE(integerDot, std::numeric_limits<int32_t>::min());
        ASSERT_LE(integerDot, std::numeric_limits<int32_t>::max());
    }

    auto    problem = makeProblem(k, test.floatAlpha);
    int32_t c       = test.c;
    int32_t d       = -1;

    if(test.floatAlpha)
    {
        ContractionInputs inputs(a.data(), b.data(), &c, &d, test.alpha, test.beta);
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, tryFastPath);
    }
    else
    {
        ContractionInputs inputs(a.data(), b.data(), &c, &d, int32_t(1), int32_t(0));
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, tryFastPath);
    }

    EXPECT_EQ(d, test.expected) << "K=" << k << ", exact integer dot product=" << integerDot;
}

INSTANTIATE_TEST_SUITE_P(
    CPUReference,
    ReferenceInt8Accumulation,
    ::testing::Combine(::testing::ValuesIn(cases), ::testing::Bool()),
    [](const ::testing::TestParamInfo<ReferenceInt8Accumulation::ParamType>& info) {
        const auto& test        = std::get<0>(info.param);
        const bool  tryFastPath = std::get<1>(info.param);
        return std::string(test.name) + (tryFastPath ? "TryFastPath" : "SlowPath");
    });

template <typename Tuple>
class ReferenceInt8Output : public ::testing::Test
{
};

// Tuple wrappers allow gtest to print type information for Half (_Float16).
using Int8OutputTypes = ::testing::
    Types<std::tuple<int32_t>, std::tuple<int8_t>, std::tuple<Half>, std::tuple<BFloat16>>;
TYPED_TEST_SUITE(ReferenceInt8Output, Int8OutputTypes);

TYPED_TEST(ReferenceInt8Output, AmaxPreservesFloatEpilogueBeforeOutputCast)
{
    using OutputT         = typename std::tuple_element<0, TypeParam>::type;
    const size_t        k = 1026;
    std::vector<int8_t> a(k, -128);
    std::vector<int8_t> b(k, -128);
    a[k - 2] = a[k - 1] = b[k - 2] = b[k - 1] = 1;

    auto problem = makeProblem(k, true, TypeInfo<OutputT>::Enum);
    problem.setOutputAmaxD(true);
    problem.setAmaxD(rocisa::DataType::Float, /*isOutput=*/true);

    OutputT           c     = static_cast<OutputT>(0);
    OutputT           d     = static_cast<OutputT>(-1);
    float             amaxD = -1.0f;
    ContractionInputs inputs(a.data(), b.data(), &c, &d, 0x1p-19f, 0.0f);
    inputs.amaxD = &amaxD;
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);

    // (2^24 + 2) * 2^-19 = 32 + 2^-18, exactly one float ULP above 32.
    // Every output type rounds/truncates D to 32, but float AMAX-D retains it.
    EXPECT_EQ(static_cast<float>(d), 32.0f);
    EXPECT_EQ(amaxD, 0x1.000002p5f);
}

TEST(ReferenceInt8Wrapping, CompletedSumIsSignedInt32BeforeFloatEpilogue)
{
    // 2^17 products of 2^14 reach 2^31: the 32-bit MAC result is INT32_MIN.
    // The fractional epilogue then produces an in-range Int32 output.
    const size_t        k = 131072;
    std::vector<int8_t> a(k, -128);
    std::vector<int8_t> b(k, -128);
    auto                problem = makeProblem(k, true);
    int32_t             c       = 0;

    for(bool tryFastPath : {false, true})
    {
        SCOPED_TRACE(tryFastPath);
        int32_t           d = -1;
        ContractionInputs inputs(a.data(), b.data(), &c, &d, 0.5f, 0.0f);
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, tryFastPath);
        EXPECT_EQ(d, -1073741824);
    }
}
