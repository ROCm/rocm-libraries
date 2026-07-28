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
// ClusterReductionIterCheck: StreamK cluster-reduction split-barrier safety.
// The C cluster peers split a tile's itersPerTile = ceil(K /
// DepthU) K-iterations; the split barrier over-signals unless itersPerTile % C
// == 0. When itersPerTile % C != 0 the predicate is a HARD REJECT of the
// cluster-reduction solution (deliberate, user-visible via debugEval -- NOT a
// silent fallback): the "must be a multiple of the cluster size" limitation.
// K-tail (K % DepthU != 0) is safe on its own. value = {DepthU, C}.
// Problems set K via ContractionProblemGemm::GEMM(..., k, ...).
// ----------------------------------------------------------------------------

namespace
{
    TensileLite::ContractionProblemGemm gemmWithK(size_t k)
    {
        using namespace TensileLite;
        // TN like the MX-FP8 cluster-reduction configs; only K matters here.
        return ContractionProblemGemm::GEMM(true, false, 256, 256, k, 256, 256, 256, 1.0, false, 1);
    }
}

TEST(Predicates, ClusterReductionIterCheck_EvenSplit_NoTail_Passes)
{
    using namespace TensileLite;
    // DepthU=256, C=4, K=2048 -> itersPerTile=8, 8 % 4 == 0 -> safe.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 4});
    EXPECT_TRUE((*pred)(gemmWithK(2048))) << "itersPerTile=8, 8 % 4 == 0 should pass";
}

TEST(Predicates, ClusterReductionIterCheck_EvenSplit_WithKTail_Passes)
{
    using namespace TensileLite;
    // DepthU=256, C=4, K=1920 -> itersPerTile=ceil(1920/256)=8, 8 % 4 == 0.
    // K % 256 = 128 (K-tail) but that alone is SAFE.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 4});
    EXPECT_TRUE((*pred)(gemmWithK(1920)))
        << "itersPerTile=8, 8 % 4 == 0 must pass even with a K-tail";
}

TEST(Predicates, ClusterReductionIterCheck_CGreaterThanItersPerTile_Rejects)
{
    using namespace TensileLite;
    // DepthU=256, C=8, K=1024 -> itersPerTile=4, 4 % 8 == 4 != 0 -> unsafe
    // (C > itersPerTile). Would over-signal the split barrier -> HARD REJECT.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 8});
    EXPECT_FALSE((*pred)(gemmWithK(1024)))
        << "C=8 > itersPerTile=4 (4 % 8 != 0) must be rejected (not a fallback)";
}

TEST(Predicates, ClusterReductionIterCheck_UnevenSplit_NoTail_Rejects)
{
    using namespace TensileLite;
    // DepthU=256, C=4, K=1536 (=6*256, no K-tail) -> itersPerTile=6, 6 % 4 == 2
    // != 0 -> unsafe even without a K-tail.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 4});
    EXPECT_FALSE((*pred)(gemmWithK(1536)))
        << "itersPerTile=6, 6 % 4 != 0 must be rejected (no K-tail involved)";
}

TEST(Predicates, ClusterReductionIterCheck_UnevenSplit_WithKTail_Rejects)
{
    using namespace TensileLite;
    // DepthU=256, C=2, K=1056 -> itersPerTile=ceil(1056/256)=5, 5 % 2 == 1 != 0.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 2});
    EXPECT_FALSE((*pred)(gemmWithK(1056))) << "itersPerTile=5, 5 % 2 != 0 must be rejected";
}

// Focused hard-reject test: the rejection is user-visible and its diagnostic
// message names the "multiple of the cluster size" limitation (debugEval output
// is what the solution selector surfaces on the DID_NOT_SATISFY_ASSERTS path).
TEST(Predicates, ClusterReductionIterCheck_RejectMessageNamesLimitation)
{
    using namespace TensileLite;
    // DepthU=256, C=4, K=1024 -> itersPerTile=4, 4 % 4 == 0 -> the safe case
    // (no message emitted for a passing predicate in non-verbose mode) is not
    // what we assert here; instead take a non-conforming case:
    // DepthU=256, C=4, K=1280 -> itersPerTile=5, 5 % 4 != 0 -> reject.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 4});
    auto               problem = gemmWithK(1280);
    std::ostringstream oss;
    bool               rv = pred->debugEval(problem, oss);
    EXPECT_FALSE(rv) << "itersPerTile=5, 5 % 4 != 0 must be rejected";
    const std::string msg = oss.str();
    EXPECT_NE(msg.find("iterations-per-tile ceil(K/DepthU)"), std::string::npos)
        << "reject diagnostic must name iterations-per-tile; got: " << msg;
    EXPECT_NE(msg.find("multiple of ClusterDim[1]"), std::string::npos)
        << "reject diagnostic must name the multiple-of-cluster-size limitation; got: " << msg;
    EXPECT_NE(msg.find("[!!]"), std::string::npos)
        << "reject diagnostic must mark the predicate as failing; got: " << msg;
}

TEST(Predicates, ClusterReductionIterCheck_NoCluster_AlwaysPasses)
{
    using namespace TensileLite;
    // C <= 1 -> predicate inert (non-cluster / multicast never emits it, but
    // guard defensively): any K passes.
    auto pred = std::make_shared<Predicates::Contraction::ClusterReductionIterCheck>(
        std::array<int, 2>{256, 1});
    EXPECT_TRUE((*pred)(gemmWithK(1056))) << "C=1 must always pass (guard inert)";
    EXPECT_TRUE((*pred)(gemmWithK(1024))) << "C=1 must always pass (guard inert)";
}
