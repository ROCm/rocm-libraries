/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>

TEST(Predicates, ArithmeticIntensity)
{
    using namespace TensileLite;

    ContractionProblemGemm a = ContractionProblemGemm::GEMM(
        false, true, 1000, 1500, 500, 2000, 2000, 2000, 3.0, false, 1); // 88.4
    ContractionProblemGemm b = ContractionProblemGemm::GEMM(
        false, true, 500, 1000, 1000, 2000, 2000, 2000, 0.0, false, 5); // 125
    ContractionProblemGemm c = ContractionProblemGemm::GEMM(
        false, true, 2000, 100, 2000, 2000, 2000, 2000, 1.0, false, 10); // 43.5
    ContractionProblemGemm d = ContractionProblemGemm::GEMM(
        false, true, 2000, 2000, 450, 2000, 2000, 2000, 2.0, false, 1); // 92.04

    auto pg1 = std::make_shared<Predicates::Contraction::AIGreaterThanEqual>(100);
    auto pg2 = std::make_shared<Predicates::Contraction::AIGreaterThanEqual>(75);
    auto pl1 = std::make_shared<Predicates::Contraction::AILessThanEqual>(100);
    auto pl2 = std::make_shared<Predicates::Contraction::AILessThanEqual>(75);

    EXPECT_EQ(false, (*pg1)(a));
    EXPECT_EQ(true, (*pg2)(a));
    EXPECT_EQ(true, (*pl1)(a));
    EXPECT_EQ(false, (*pl2)(a));

    EXPECT_EQ(true, (*pg1)(b));
    EXPECT_EQ(true, (*pg2)(b));
    EXPECT_EQ(false, (*pl1)(b));
    EXPECT_EQ(false, (*pl2)(b));

    EXPECT_EQ(false, (*pg1)(c));
    EXPECT_EQ(false, (*pg2)(c));
    EXPECT_EQ(true, (*pl1)(c));
    EXPECT_EQ(true, (*pl2)(c));

    EXPECT_EQ(false, (*pg1)(d));
    EXPECT_EQ(true, (*pg2)(d));
    EXPECT_EQ(true, (*pl1)(d));
    EXPECT_EQ(false, (*pl2)(d));
}

// ----------------------------------------------------------------------------
// WorkgroupMappingXCCCheck: CPU-only tests with injected cuCount (ROCM-2963).
// These use the test-only constructor so we don't need a GPU. See
// docs/solution-selection-unit-test-pattern.md.
// ----------------------------------------------------------------------------

TEST(Predicates, WorkgroupMappingXCCCheck_38CU_XCC4_Fails)
{
    using namespace TensileLite;
    // 38 % 4 != 0 -> predicate must reject (would have caught ROCM-2963).
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{4, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_FALSE((*pred)(problem)) << "38 CUs with XCC=4 should fail (38 % 4 != 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_38CU_XCC1_Passes)
{
    using namespace TensileLite;
    // 38 % 1 == 0 -> predicate must accept (fix for ROCM-2963).
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{1, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "38 CUs with XCC=1 should pass (38 % 1 == 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_80CU_XCC4_Passes)
{
    using namespace TensileLite;
    // 80 % 4 == 0 -> predicate must accept.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{4, -1}, 80u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "80 CUs with XCC=4 should pass (80 % 4 == 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_XCCMinus1_AlwaysPasses)
{
    using namespace TensileLite;
    // value[0] == -1 means no check.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{-1, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "XCC=-1 should always pass";
}

TEST(Predicates, WorkgroupMappingXCCCheck_FallbackTreatsXCCAs1)
{
    using namespace TensileLite;
    // When problem is cu-fallback, effective XCC is 1 so 38 % 1 == 0 -> pass.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{4, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    problem.setParams().setFallbackStatus(true);
    EXPECT_TRUE((*pred)(problem)) << "With fallback status, effective XCC=1 so 38 % 1 == 0";
}

// ----------------------------------------------------------------------------
// WorkgroupMappingXCCCheck on non-standard-CU devices.
// A non-standard-CU device (e.g. 228 CU) must accept a generic XCC=8 entry on
// the first heuristic lookup, before the runtime sets fallbackStatus(). The
// predicate treats a non-standard-CU device as an implicit fallback. The
// device classification is the public isStandardCUDevice member, set directly
// here so the branch is exercised on CPU without a GPU.
// ----------------------------------------------------------------------------

TEST(Predicates, WorkgroupMappingXCCCheck_NonStandardCUDevice_XCC8_Passes)
{
    using namespace TensileLite;
    // 228 CU, generic XCC=8, no fallbackStatus. 228 % 8 != 0, but a non-standard
    // CU device is treated as an implicit fallback -> effective XCC=1 -> pass.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{8, -1}, 228u);
    pred->isStandardCUDevice = false;
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem))
        << "Non-standard-CU (228) must accept generic XCC=8 on first lookup";
}

TEST(Predicates, WorkgroupMappingXCCCheck_StandardCUDevice_228_XCC8_Fails)
{
    using namespace TensileLite;
    // Same shape (228 CU, XCC=8, no fallbackStatus) but a standard-CU device: the
    // implicit-fallback branch is not taken, so 228 % 8 != 0 rejects. Isolates the
    // device classification as the single deciding factor.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{8, -1}, 228u);
    pred->isStandardCUDevice = true;
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_FALSE((*pred)(problem))
        << "Standard-CU device does not coerce XCC=8 to 1 (228 % 8 != 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_StandardCUDevice_304_XCC8_Passes)
{
    using namespace TensileLite;
    // 304 CU, standard-CU, generic XCC=8, no fallbackStatus: a no-op for the
    // non-standard-CU path (304 % 8 == 0 regardless), confirming standard-CU
    // devices are unaffected.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{8, -1}, 304u);
    pred->isStandardCUDevice = true;
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "Standard-CU (304) keeps passing XCC=8 (304 % 8 == 0)";
}

// ----------------------------------------------------------------------------
// Device-tuned CU libraries (gfx942_80cu, gfx942_152cu, ...) are unaffected.
//
// Solutions in a CU-variant logic directory are guaranteed by the build-time
// ValidWorkGroupMappingXCC check (Tensile/TensileLogic/ValidWorkGroupMappingXCC.py)
// to carry a WorkGroupMappingXCC that is -1 or a power of two dividing the
// directory's CU count. Such a solution therefore passes the strict check
// already, and coercing XCC to 1 cannot change the verdict: the implicit
// fallback is a no-op for every device-tuned lib.
// ----------------------------------------------------------------------------

TEST(Predicates, WorkgroupMappingXCCCheck_TunedCULib_UnaffectedByImplicitFallback)
{
    using namespace TensileLite;
    // (cuCount, WorkGroupMappingXCC) pairs taken from the shipped CU-variant
    // logic dirs: 80cu uses XCC=4/8/16, 152cu uses XCC=4, 228cu uses XCC=1.
    struct Case
    {
        unsigned cuCount;
        int      xcc;
    };
    const Case cases[] = {{80u, 4}, {80u, 8}, {80u, 16}, {152u, 4}, {228u, 1}, {64u, 8}};

    auto problem = ContractionProblemGemm::GEMM(
        false, false, 1024, 1024, 1024, 1024, 1024, 1024, 1.0, false, 1);

    for(auto const& c : cases)
    {
        auto strictPred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
            std::array<int, 2>{c.xcc, -1}, c.cuCount);
        strictPred->isStandardCUDevice = true;

        auto relaxedPred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
            std::array<int, 2>{c.xcc, -1}, c.cuCount);
        relaxedPred->isStandardCUDevice = false;

        EXPECT_EQ((*strictPred)(problem), (*relaxedPred)(problem))
            << "Implicit fallback must be a no-op for a tuned lib: cuCount=" << c.cuCount
            << " WGMXCC=" << c.xcc;
        EXPECT_TRUE((*relaxedPred)(problem))
            << "Tuned-lib solution must be accepted: cuCount=" << c.cuCount
            << " WGMXCC=" << c.xcc;
    }
}
