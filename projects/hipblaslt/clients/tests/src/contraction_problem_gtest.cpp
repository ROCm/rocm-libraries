// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The TensileLite problem factories give the compute input types, which are part of the
// solution-cache key, a defined value.

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem_Detail.hpp>

#include <functional>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm halfGemm()
    {
        return ContractionProblemGemm::GEMM_Strides(false,
                                                    false,
                                                    rocisa::DataType::Half,
                                                    rocisa::DataType::Half,
                                                    rocisa::DataType::Float,
                                                    rocisa::DataType::Float,
                                                    128,
                                                    256,
                                                    64,
                                                    1,
                                                    128,
                                                    128 * 64,
                                                    64,
                                                    64 * 256,
                                                    128,
                                                    128 * 256,
                                                    128,
                                                    128 * 256,
                                                    1.0);
    }

    TEST(ContractionProblemFactory, smoke_ComputeInputTypesFollowInputTypes)
    {
        auto half = halfGemm();
        EXPECT_EQ(half.computeInputTypeA(), rocisa::DataType::Half);
        EXPECT_EQ(half.computeInputTypeB(), rocisa::DataType::Half);

        auto sgemm
            = ContractionProblemGemm::GEMM(false, false, 128, 256, 64, 128, 64, 128, 1.0, false, 1);
        EXPECT_EQ(sgemm.computeInputTypeA(), rocisa::DataType::Float);
        EXPECT_EQ(sgemm.computeInputTypeB(), rocisa::DataType::Float);

        ContractionProblemGemm empty;
        EXPECT_EQ(empty.computeInputTypeA(), empty.a().dataType());
        EXPECT_EQ(empty.computeInputTypeB(), empty.b().dataType());
    }

    TEST(ContractionProblemFactory, smoke_IdenticalProblemsShareCacheKey)
    {
        auto first  = halfGemm();
        auto second = halfGemm();
        EXPECT_TRUE(first == second);
        EXPECT_EQ(std::hash<ContractionProblemGemm>{}(first),
                  std::hash<ContractionProblemGemm>{}(second));
    }
} // namespace
