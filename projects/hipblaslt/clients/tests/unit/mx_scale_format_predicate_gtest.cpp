// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the TensileLite MXScaleFormatEquals predicate, which keeps
// solutions for a host scale layout that selection must tell apart (format 3 and
// later) away from every other MX solution.

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>

#include <gtest/gtest.h>

#include <functional>

namespace
{
    TensileLite::ContractionProblemGemm problemWithScaleFormat(int format)
    {
        auto problem = TensileLite::ContractionProblemGemm::GEMM_Strides(true,
                                                                         false,
                                                                         rocisa::DataType::Float6,
                                                                         rocisa::DataType::Float6,
                                                                         rocisa::DataType::BFloat16,
                                                                         rocisa::DataType::BFloat16,
                                                                         256,
                                                                         256,
                                                                         128,
                                                                         1,
                                                                         128,
                                                                         128 * 256,
                                                                         128,
                                                                         128 * 256,
                                                                         256,
                                                                         256 * 256,
                                                                         256,
                                                                         256 * 256,
                                                                         0.0);
        problem.setMXScaleFormat(format);
        return problem;
    }
}

// Formats 0..2 cannot be told apart from the host scale mode, so they match each
// other; format 3 matches only format 3.
TEST(HostUnitMXScaleFormatPredicate, smoke_OnlyTheExactFormatsArePinned)
{
    using TensileLite::Predicates::Contraction::MXScaleFormatEquals;
    for(int sol = 0; sol <= 3; ++sol)
    {
        for(int prob = 0; prob <= 3; ++prob)
        {
            bool const expected = (sol < 3 && prob < 3) || sol == prob;
            EXPECT_EQ(MXScaleFormatEquals(sol)(problemWithScaleFormat(prob)), expected)
                << "solution format " << sol << ", problem format " << prob;
        }
    }
}

TEST(HostUnitMXScaleFormatPredicate, smoke_ProblemsStartAtFormatZero)
{
    auto problem = TensileLite::ContractionProblemGemm::GEMM(
        true, false, 256, 256, 128, 128, 128, 256, 1.0, false, 1);
    EXPECT_EQ(problem.mxScaleFormat(), 0);
    problem.setMXScaleFormat(3);
    EXPECT_EQ(problem.mxScaleFormat(), 3);
}

// The solution cache keys on problem equality, so the format has to take part in it.
TEST(HostUnitMXScaleFormatPredicate, smoke_FormatDistinguishesCachedProblems)
{
    auto const plain = problemWithScaleFormat(0);
    auto const tiled = problemWithScaleFormat(3);
    EXPECT_TRUE(plain == problemWithScaleFormat(0));
    EXPECT_FALSE(plain == tiled);
    EXPECT_NE(std::hash<TensileLite::ContractionProblemGemm>{}(plain),
              std::hash<TensileLite::ContractionProblemGemm>{}(tiled));
}

// A solution with AssertAlphaValue must not be reused from the cache for another alpha.
TEST(HostUnitMXScaleFormatPredicate, smoke_AlphaRestrictionDistinguishesCachedProblems)
{
    using TensileLite::ScalarValue;
    using TensileLite::Predicates::Contraction::AlphaValue;
    auto one = problemWithScaleFormat(3);
    auto any = problemWithScaleFormat(3);
    one.setAlphaRestriction(ScalarValue::One);
    any.setAlphaRestriction(ScalarValue::Any);
    EXPECT_TRUE(AlphaValue(ScalarValue::One)(one));
    EXPECT_FALSE(AlphaValue(ScalarValue::One)(any));
    EXPECT_FALSE(one == any);
    EXPECT_NE(std::hash<TensileLite::ContractionProblemGemm>{}(one),
              std::hash<TensileLite::ContractionProblemGemm>{}(any));
}
