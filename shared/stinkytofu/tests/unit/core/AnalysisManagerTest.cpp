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

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/PassManager.hpp"

using namespace stinkytofu;

// -----------------------------------------------------------------------
// Mock analyses for testing
// -----------------------------------------------------------------------

// A mock analysis with a run counter to verify caching behavior.
struct CountingAnalysis {
    static inline AnalysisKey Key;
    static AnalysisKey* ID() {
        return &Key;
    }
    static const char* name() {
        return "CountingAnalysis";
    }
    struct Result {
        int value = 42;
    };
    static inline int runCount = 0;
    static Result run(Function& /*F*/, AnalysisManager& /*AM*/) {
        ++runCount;
        return Result{};
    }
};

// A second independent mock analysis.
struct SecondAnalysis {
    static inline AnalysisKey Key;
    static AnalysisKey* ID() {
        return &Key;
    }
    static const char* name() {
        return "SecondAnalysis";
    }
    struct Result {
        int value = 99;
    };
    static inline int runCount = 0;
    static Result run(Function& /*F*/, AnalysisManager& /*AM*/) {
        ++runCount;
        return Result{};
    }
};

// A mock analysis that depends on CountingAnalysis.
struct DependentAnalysis {
    static inline AnalysisKey Key;
    static AnalysisKey* ID() {
        return &Key;
    }
    static const char* name() {
        return "DependentAnalysis";
    }
    struct Result {
        int derivedValue;
    };
    static inline int runCount = 0;
    static Result run(Function& F, AnalysisManager& AM) {
        ++runCount;
        auto& base = AM.getResult<CountingAnalysis>(F);
        return Result{base.value * 2};
    }
};

// -----------------------------------------------------------------------
// Test fixture
// -----------------------------------------------------------------------

class AnalysisManagerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        CountingAnalysis::runCount = 0;
        SecondAnalysis::runCount = 0;
        DependentAnalysis::runCount = 0;
    }

    Function func{"test"};
};

// -----------------------------------------------------------------------
// PreservedAnalyses data structure
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, DefaultIsEmpty) {
    PreservedAnalyses PA;
    EXPECT_FALSE(PA.areAllPreserved());
    EXPECT_FALSE(PA.isPreserved(CountingAnalysis::ID()));
}

TEST_F(AnalysisManagerTest, All) {
    auto PA = PreservedAnalyses::all();
    EXPECT_TRUE(PA.areAllPreserved());
    EXPECT_TRUE(PA.isPreserved(CountingAnalysis::ID()));
    EXPECT_TRUE(PA.isPreserved(SecondAnalysis::ID()));
}

TEST_F(AnalysisManagerTest, None) {
    auto PA = PreservedAnalyses::none();
    EXPECT_FALSE(PA.areAllPreserved());
    EXPECT_FALSE(PA.isPreserved(CountingAnalysis::ID()));
}

TEST_F(AnalysisManagerTest, PreserveSpecific) {
    PreservedAnalyses PA;
    PA.preserve<CountingAnalysis>();
    EXPECT_TRUE(PA.isPreserved<CountingAnalysis>());
    EXPECT_FALSE(PA.isPreserved<SecondAnalysis>());
}

TEST_F(AnalysisManagerTest, PreserveMultiple) {
    PreservedAnalyses PA;
    PA.preserve<CountingAnalysis>();
    PA.preserve<SecondAnalysis>();
    EXPECT_TRUE(PA.isPreserved<CountingAnalysis>());
    EXPECT_TRUE(PA.isPreserved<SecondAnalysis>());
}

TEST_F(AnalysisManagerTest, PreserveCFGGroup) {
    auto PA = preserveCFGAnalyses();
    EXPECT_TRUE(PA.isPreserved<BBIndexAnalysis>());
    EXPECT_TRUE(PA.isPreserved<DominanceAnalysis>());
    EXPECT_TRUE(PA.isPreserved<LoopAnalysis>());
}

TEST_F(AnalysisManagerTest, Intersect) {
    PreservedAnalyses PA1;
    PA1.preserve<CountingAnalysis>();
    PA1.preserve<SecondAnalysis>();

    PreservedAnalyses PA2;
    PA2.preserve<CountingAnalysis>();

    PA1.intersect(PA2);
    EXPECT_TRUE(PA1.isPreserved<CountingAnalysis>());
    EXPECT_FALSE(PA1.isPreserved<SecondAnalysis>());
}

TEST_F(AnalysisManagerTest, IntersectWithAll) {
    PreservedAnalyses PA;
    PA.preserve<CountingAnalysis>();
    PA.preserve<SecondAnalysis>();

    PA.intersect(PreservedAnalyses::all());
    EXPECT_TRUE(PA.isPreserved<CountingAnalysis>());
    EXPECT_TRUE(PA.isPreserved<SecondAnalysis>());
}

TEST_F(AnalysisManagerTest, IntersectWithNone) {
    PreservedAnalyses PA;
    PA.preserve<CountingAnalysis>();
    PA.preserve<SecondAnalysis>();

    PA.intersect(PreservedAnalyses::none());
    EXPECT_FALSE(PA.isPreserved<CountingAnalysis>());
    EXPECT_FALSE(PA.isPreserved<SecondAnalysis>());
}

// -----------------------------------------------------------------------
// AnalysisManager cache + lazy evaluation
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, LazyComputation) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    EXPECT_EQ(CountingAnalysis::runCount, 0);
    auto& result = AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(result.value, 42);
}

TEST_F(AnalysisManagerTest, CacheHit) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
}

TEST_F(AnalysisManagerTest, ClearEvictsAll) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);

    AM.clear();
    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);
}

TEST_F(AnalysisManagerTest, GetCachedResultNull) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    EXPECT_EQ(AM.getCachedResult<CountingAnalysis>(), nullptr);
}

TEST_F(AnalysisManagerTest, GetCachedResultHit) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    auto* cached = AM.getCachedResult<CountingAnalysis>();
    ASSERT_NE(cached, nullptr);
    EXPECT_EQ(cached->value, 42);
}

// -----------------------------------------------------------------------
// Invalidation
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, InvalidateEvicts) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);

    AM.invalidate(func, PreservedAnalyses::none());
    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);
}

TEST_F(AnalysisManagerTest, PreservationKeeps) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);

    PreservedAnalyses PA;
    PA.preserve<CountingAnalysis>();
    AM.invalidate(func, PA);

    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
}

TEST_F(AnalysisManagerTest, AllPreservedKeepsEverything) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<SecondAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    AM.getResult<SecondAnalysis>(func);

    AM.invalidate(func, PreservedAnalyses::all());

    AM.getResult<CountingAnalysis>(func);
    AM.getResult<SecondAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(SecondAnalysis::runCount, 1);
}

TEST_F(AnalysisManagerTest, SelectiveInvalidation) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<SecondAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    AM.getResult<SecondAnalysis>(func);

    PreservedAnalyses PA;
    PA.preserve<CountingAnalysis>();
    AM.invalidate(func, PA);

    AM.getResult<CountingAnalysis>(func);
    AM.getResult<SecondAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(SecondAnalysis::runCount, 2);
}

TEST_F(AnalysisManagerTest, MultipleInvalidations) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);

    AM.invalidate(func, PreservedAnalyses::none());
    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);

    AM.invalidate(func, PreservedAnalyses::none());
    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 3);
}

// -----------------------------------------------------------------------
// Analysis depends on analysis
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, DependencyTriggersComputation) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<DependentAnalysis>();

    auto& result = AM.getResult<DependentAnalysis>(func);
    EXPECT_EQ(DependentAnalysis::runCount, 1);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(result.derivedValue, 84);  // 42 * 2
}

TEST_F(AnalysisManagerTest, DependencyCacheShared) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<DependentAnalysis>();

    AM.getResult<DependentAnalysis>(func);
    auto& base = AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(base.value, 42);
}

TEST_F(AnalysisManagerTest, InvalidatingDependency) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<DependentAnalysis>();

    AM.getResult<DependentAnalysis>(func);
    EXPECT_EQ(DependentAnalysis::runCount, 1);
    EXPECT_EQ(CountingAnalysis::runCount, 1);

    // Invalidate the dependency (CountingAnalysis), keep DependentAnalysis
    PreservedAnalyses PA;
    PA.preserve<DependentAnalysis>();
    AM.invalidate(func, PA);

    // CountingAnalysis was evicted, request it again
    AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);
}

// -----------------------------------------------------------------------
// PassManager + AM integration (end-to-end)
// -----------------------------------------------------------------------

// Helper: a pass that requests an analysis and preserves everything
class AnalysisUserPass : public Pass {
   public:
    static char ID;
    std::function<PreservedAnalyses(AnalysisManager&, Function&)> action;

    AnalysisUserPass(std::function<PreservedAnalyses(AnalysisManager&, Function&)> action)
        : action(std::move(action)) {}

    PassID getPassID() const override {
        return &ID;
    }
    const char* getName() const override {
        return "AnalysisUserPass";
    }

    PreservedAnalyses run(Function& func, PassContext& /*ctx*/, AnalysisManager& AM) override {
        return action(AM, func);
    }
};
char AnalysisUserPass::ID = 0;

TEST_F(AnalysisManagerTest, PipelineCacheReuse) {
    CountingAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();

    // PassA: requests CountingAnalysis, preserves it
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        PreservedAnalyses PA;
        PA.preserve<CountingAnalysis>();
        return PA;
    }));

    // PassB: requests CountingAnalysis (should be cached)
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
}

TEST_F(AnalysisManagerTest, PipelineRecomputation) {
    CountingAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();

    // PassA: requests CountingAnalysis, preserves nothing
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        return PreservedAnalyses::none();
    }));

    // PassB: requests CountingAnalysis (must recompute)
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);
}

TEST_F(AnalysisManagerTest, PipelineAllPreserved) {
    CountingAnalysis::runCount = 0;
    SecondAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();
    PM.getAnalysisManager().registerPass<SecondAnalysis>();

    // PassA: requests both, returns all()
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        AM.getResult<SecondAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    // PassB: requests both
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        AM.getResult<SecondAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(SecondAnalysis::runCount, 1);
}

TEST_F(AnalysisManagerTest, PipelineNonePreserved) {
    CountingAnalysis::runCount = 0;
    SecondAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();
    PM.getAnalysisManager().registerPass<SecondAnalysis>();

    // PassA: requests both, returns none()
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        AM.getResult<SecondAnalysis>(F);
        return PreservedAnalyses::none();
    }));

    // PassB: requests both (must recompute)
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        AM.getResult<SecondAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);
    EXPECT_EQ(SecondAnalysis::runCount, 2);
}

TEST_F(AnalysisManagerTest, ScopeAdaptorEvictsOuterCache) {
    // Simulate: PassA caches analysis → ScopeAdaptor returns none() →
    // PassB must recompute. This mirrors the real pattern where
    // ScopeAdaptor extracts/splices instructions, invalidating everything.
    CountingAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();

    // PassA: computes and caches analysis
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    // Simulated ScopeAdaptor: returns none() (instructions were extracted/spliced)
    PM.addPass(std::make_unique<AnalysisUserPass>(
        [](AnalysisManager&, Function&) { return PreservedAnalyses::none(); }));

    // PassB: must recompute — cache was wiped by ScopeAdaptor
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);  // computed in PassA and PassB
}

// -----------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, EmptyFunction) {
    // func has no BBs — analyses should handle gracefully
    AnalysisManager AM;
    AM.registerPass<BBIndexAnalysis>();
    AM.registerPass<DominanceAnalysis>();
    AM.registerPass<LoopAnalysis>();

    auto& bbIndex = AM.getResult<BBIndexAnalysis>(func);
    EXPECT_TRUE(bbIndex.rpo.empty());

    auto& domInfo = AM.getResult<DominanceAnalysis>(func);
    EXPECT_TRUE(domInfo.rpo.empty());

    auto& loops = AM.getResult<LoopAnalysis>(func);
    EXPECT_TRUE(loops.empty());
}

TEST_F(AnalysisManagerTest, SingleBlockFunction) {
    func.createBasicBlock("entry");

    AnalysisManager AM;
    AM.registerPass<BBIndexAnalysis>();
    AM.registerPass<DominanceAnalysis>();
    AM.registerPass<LoopAnalysis>();

    auto& bbIndex = AM.getResult<BBIndexAnalysis>(func);
    EXPECT_EQ(bbIndex.rpo.size(), 1u);

    auto& domInfo = AM.getResult<DominanceAnalysis>(func);
    EXPECT_EQ(domInfo.rpo.size(), 1u);

    auto& loops = AM.getResult<LoopAnalysis>(func);
    EXPECT_TRUE(loops.empty());
}

TEST_F(AnalysisManagerTest, InvalidateEmptyCache) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    // invalidate on empty cache — should not crash
    AM.invalidate(func, PreservedAnalyses::none());
}

TEST_F(AnalysisManagerTest, ClearEmptyCache) {
    AnalysisManager AM;
    // clear on empty cache — should not crash
    AM.clear();
}

TEST_F(AnalysisManagerTest, RegisterPassTwice) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    // Second registration overwrites — should not crash
    AM.registerPass<CountingAnalysis>();

    auto& result = AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(result.value, 42);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
}

// -----------------------------------------------------------------------
// Real analyses with CFG
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, DiamondCFGAnalyses) {
    // Build diamond: entry -> then, entry -> else, then -> merge, else -> merge
    BasicBlock* entry = func.createBasicBlock("entry");
    BasicBlock* thenBB = func.createBasicBlock("then");
    BasicBlock* elseBB = func.createBasicBlock("else");
    BasicBlock* mergeBB = func.createBasicBlock("merge");

    func.addEdge(entry, thenBB);
    func.addEdge(entry, elseBB);
    func.addEdge(thenBB, mergeBB);
    func.addEdge(elseBB, mergeBB);

    AnalysisManager AM;
    AM.registerPass<BBIndexAnalysis>();
    AM.registerPass<DominanceAnalysis>();
    AM.registerPass<LoopAnalysis>();

    auto& bbIndex = AM.getResult<BBIndexAnalysis>(func);
    EXPECT_EQ(bbIndex.rpo.size(), 4u);
    EXPECT_EQ(bbIndex.rpo[0], entry);
    EXPECT_EQ(bbIndex.index.at(entry), 0u);

    auto& domInfo = AM.getResult<DominanceAnalysis>(func);
    EXPECT_EQ(domInfo.rpo.size(), 4u);
    // Entry dominates itself
    EXPECT_EQ(domInfo.idom[0], 0u);

    auto& loops = AM.getResult<LoopAnalysis>(func);
    EXPECT_TRUE(loops.empty());
}

TEST_F(AnalysisManagerTest, LoopCFGAnalysis) {
    // Build loop: entry -> header -> body -> header (back-edge)
    BasicBlock* entry = func.createBasicBlock("entry");
    BasicBlock* header = func.createBasicBlock("header");
    BasicBlock* body = func.createBasicBlock("body");
    BasicBlock* exit = func.createBasicBlock("exit");

    func.addEdge(entry, header);
    func.addEdge(header, body);
    func.addEdge(body, header);
    func.addEdge(header, exit);

    AnalysisManager AM;
    AM.registerPass<LoopAnalysis>();

    auto& loops = AM.getResult<LoopAnalysis>(func);
    EXPECT_EQ(loops.size(), 1u);
    EXPECT_EQ(loops[0].headerBB, header);
    EXPECT_EQ(loops[0].latchBB, body);
}

// -----------------------------------------------------------------------
// Dependency cascade invalidation
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, DependencyCascadeRecomputation) {
    // DependentAnalysis calls AM.getResult<CountingAnalysis> in its run().
    // Invalidate CountingAnalysis (the dependency), then request DependentAnalysis.
    // Both should recompute.
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<DependentAnalysis>();

    AM.getResult<DependentAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);
    EXPECT_EQ(DependentAnalysis::runCount, 1);

    // Invalidate both (none preserved)
    AM.invalidate(func, PreservedAnalyses::none());

    // Request DependentAnalysis again — triggers CountingAnalysis recompute too
    AM.getResult<DependentAnalysis>(func);
    EXPECT_EQ(CountingAnalysis::runCount, 2);
    EXPECT_EQ(DependentAnalysis::runCount, 2);
}

// -----------------------------------------------------------------------
// Pipeline ordering and mixed preservation
// -----------------------------------------------------------------------

TEST_F(AnalysisManagerTest, ThreePassPipelineMixedPreservation) {
    // PassA computes CountingAnalysis + SecondAnalysis, preserves both.
    // PassB preserves only CountingAnalysis (evicts SecondAnalysis).
    // PassC requests both — CountingAnalysis cached, SecondAnalysis recomputed.
    CountingAnalysis::runCount = 0;
    SecondAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();
    PM.getAnalysisManager().registerPass<SecondAnalysis>();

    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        AM.getResult<SecondAnalysis>(F);
        return PreservedAnalyses::all();
    }));
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        PreservedAnalyses PA;
        PA.preserve<CountingAnalysis>();
        return PA;  // evicts SecondAnalysis
    }));
    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        AM.getResult<SecondAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);  // cached through PassB
    EXPECT_EQ(SecondAnalysis::runCount, 2);    // evicted by PassB, recomputed in PassC
}

TEST_F(AnalysisManagerTest, MultiplePMRunsStartFresh) {
    // PM.run() calls clear() at start. Running PM twice should recompute.
    CountingAnalysis::runCount = 0;

    PassManager PM;
    PM.getAnalysisManager().registerPass<CountingAnalysis>();

    PM.addPass(std::make_unique<AnalysisUserPass>([](AnalysisManager& AM, Function& F) {
        AM.getResult<CountingAnalysis>(F);
        return PreservedAnalyses::all();
    }));

    PM.run(func);
    EXPECT_EQ(CountingAnalysis::runCount, 1);

    PM.run(func);  // clear() at start evicts everything
    EXPECT_EQ(CountingAnalysis::runCount, 2);
}

TEST_F(AnalysisManagerTest, GetCachedResultAfterInvalidation) {
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();

    AM.getResult<CountingAnalysis>(func);
    EXPECT_NE(AM.getCachedResult<CountingAnalysis>(), nullptr);

    AM.invalidate(func, PreservedAnalyses::none());
    EXPECT_EQ(AM.getCachedResult<CountingAnalysis>(), nullptr);
}

TEST_F(AnalysisManagerTest, InterleavedGetResults) {
    // Request A, then B, then A again — no interference
    AnalysisManager AM;
    AM.registerPass<CountingAnalysis>();
    AM.registerPass<SecondAnalysis>();

    auto& a1 = AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(a1.value, 42);

    auto& b = AM.getResult<SecondAnalysis>(func);
    EXPECT_EQ(b.value, 99);

    auto& a2 = AM.getResult<CountingAnalysis>(func);
    EXPECT_EQ(a2.value, 42);

    EXPECT_EQ(CountingAnalysis::runCount, 1);  // still cached
    EXPECT_EQ(SecondAnalysis::runCount, 1);
}
