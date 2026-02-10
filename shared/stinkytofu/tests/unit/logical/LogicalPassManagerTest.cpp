/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include <gtest/gtest.h>

#include "stinkytofu/core/stinkytofu.hpp"

using namespace stinkytofu;

// Simplified PassManager tests focusing only on pass flow and analysis
// invalidation/caching behavior. No instruction semantics are exercised.

// Dummy Analysis
class DummyAnalysis : public AnalysisPass
{
public:
    static Pass::ID ID;

    int runCount = 0;

    Pass::ID getPassID() const override
    {
        return ID;
    }

    const char* getName() const override
    {
        return "DummyAnalysis";
    }

    void run(Function&, PassContext&) override
    {
        runCount++;
    }
};
Pass::ID DummyAnalysis::ID = &DummyAnalysis::ID;

// No-Op Transform Pass
class NoOpPass : public Pass
{
public:
    static Pass::ID ID;

    Pass::ID getPassID() const override
    {
        return ID;
    }

    const char* getName() const override
    {
        return "NoOpPass";
    }

    void run(Function&, PassContext&) override {}
};
Pass::ID NoOpPass::ID = &NoOpPass::ID;

// Pass that invalidates DummyAnalysis
class InvalidateAnalysisPass : public Pass
{
public:
    static Pass::ID ID;

    Pass::ID getPassID() const override
    {
        return ID;
    }

    const char* getName() const override
    {
        return "InvalidateAnalysisPass";
    }

    void run(Function&, PassContext& ctx) override
    {
        ctx.getAnalysisManager().invalidate(DummyAnalysis::ID);
    }
};
Pass::ID InvalidateAnalysisPass::ID = &InvalidateAnalysisPass::ID;

class PassManagerFlowTest : public ::testing::Test, public stinkytofu::PassManager
{
protected:
    PassManagerFlowTest()           = default;
    ~PassManagerFlowTest() override = default;

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(PassManagerFlowTest, AnalysisLazilyComputedAndCached)
{
    registerAnalysisPass(std::make_unique<DummyAnalysis>());
    run();

    auto& mgr = passCtx.getAnalysisManager();
    auto& a1  = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a1.runCount, 1);
    auto& a2 = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a2.runCount, 1);
}

TEST_F(PassManagerFlowTest, ManualInvalidationRecomputesOnDemand)
{
    registerAnalysisPass(std::make_unique<DummyAnalysis>());
    run();

    auto& mgr = passCtx.getAnalysisManager();
    auto& a1  = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    ASSERT_EQ(a1.runCount, 1);
    mgr.invalidate(DummyAnalysis::ID);
    auto& a2 = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a2.runCount, 2);
}

TEST_F(PassManagerFlowTest, NoOpPassDoesNotInvalidateAnalysis)
{
    registerAnalysisPass(std::make_unique<DummyAnalysis>());
    addPass(std::make_unique<NoOpPass>());
    run();

    auto& mgr = passCtx.getAnalysisManager();
    auto& a1  = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    ASSERT_EQ(a1.runCount, 1);
    addPass(std::make_unique<NoOpPass>());

    run();
    auto& a2 = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a2.runCount, 1);
}

TEST_F(PassManagerFlowTest, InvalidatingPassTriggersRecomputeOnNextQuery)
{
    registerAnalysisPass(std::make_unique<DummyAnalysis>());
    addPass(std::make_unique<InvalidateAnalysisPass>());
    run();
    auto& mgr = passCtx.getAnalysisManager();
    auto& a1  = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a1.runCount, 1);
    addPass(std::make_unique<InvalidateAnalysisPass>());
    run();
    auto& a2 = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a2.runCount, 2);
}

TEST_F(PassManagerFlowTest, MixedPassSequenceMinimizesRecomputations)
{
    registerAnalysisPass(std::make_unique<DummyAnalysis>());
    addPass(std::make_unique<NoOpPass>());
    run();
    auto& mgr = passCtx.getAnalysisManager();
    auto& a1  = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a1.runCount, 1);
    addPass(std::make_unique<NoOpPass>());
    run();
    auto& a2 = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a2.runCount, 1);
    addPass(std::make_unique<InvalidateAnalysisPass>());
    run();
    auto& a3 = mgr.getResult<DummyAnalysis>(passCtx.getFunction(), passCtx);
    EXPECT_EQ(a3.runCount, 2);
}
