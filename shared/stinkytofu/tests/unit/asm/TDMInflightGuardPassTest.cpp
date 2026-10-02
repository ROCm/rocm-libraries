// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for TDMInflightGuardPass (gfx1250).
//
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/TDMInflightGuardPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {
constexpr int kLimit = kDefaultTDMInflightLimit;
}  // namespace

class TDMInflightGuardPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        config.NumWaves = 4;

        func = std::make_unique<Function>("tdm_inflight_guard_test");
        setFunctionArch(*func, arch);
        // Unnamed, so the kernel start is not a label a branch could name.
        bb = func->createBasicBlock();
        func->setGemmTileConfig(config);
        registerAllAnalyses(am);
    }

    void TearDown() override {
        func.reset();
        bb = nullptr;
    }

    StinkyInstruction* tensorLoad() {
        return createTensorLoadInBlock(bb, arch, /*src0Reg=*/0, /*src1Reg=*/4);
    }

    std::vector<StinkyInstruction*> tensorLoads(int count) {
        std::vector<StinkyInstruction*> loads;
        for (int i = 0; i < count; ++i) loads.push_back(tensorLoad());
        return loads;
    }

    // `s_wait_tensorcnt <count>`. .stir input carries the count in the modifier
    // alone, which \p withLiteral = false reproduces.
    StinkyInstruction* waitTensorCnt(int count, bool withLiteral = true) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch));
        if (withLiteral) inst->addSrcReg(StinkyRegister(count));
        SWaitTensorCntData d;
        d.tlcnt = static_cast<int8_t>(count);
        inst->addModifier<SWaitTensorCntData>(d);
        return inst;
    }

    void createLabel(const char* name) {
        AsmIRBuilder builder(*bb, arch);
        builder.createLabel(name);
    }

    // `s_cmp_eq_u32 s<sgpr>, 0` and the branch that reads it.
    StinkyInstruction* createGuardedBranch(GFX opcode, int sgpr, const char* target) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* cmp = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));
        cmp->addSrcReg(StinkyRegister("s", sgpr, 1));
        cmp->addSrcReg(StinkyRegister(0));
        cmp->addDestReg(StinkyRegister::getSCCRegister());
        StinkyInstruction* br = builder.create(getMCIDByUOp(opcode, arch));
        br->addSrcReg(StinkyRegister(std::string(target)));
        br->addModifier<LabelData>(LabelData{target});
        return br;
    }

    // `s_swappc_b64`, with no CallTargetData naming the callee.
    StinkyInstruction* createCall() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_swappc_b64, arch));
        inst->addDestReg(StinkyRegister("s", 2, 2));
        inst->addSrcReg(StinkyRegister("s", 0, 2));
        return inst;
    }

    void createFiller() {
        createVAddInBlock(bb, arch, /*destReg=*/40, /*src0Reg=*/41, /*src1Reg=*/42);
    }

    // A callable function in the shape of an activation body.
    std::unique_ptr<Function> createCallee(bool issuesTensorOp) {
        auto callee = std::make_unique<Function>("label_Activation_Test");
        callee->setIsCallable(true);
        setFunctionArch(*callee, arch);
        BasicBlock* entry = callee->createBasicBlock("entry");
        createVAddInBlock(entry, arch, /*destReg=*/0, /*src0Reg=*/1, /*src1Reg=*/2);
        if (issuesTensorOp) createTensorLoadInBlock(entry, arch, /*src0Reg=*/0, /*src1Reg=*/4);
        return callee;
    }

    // Four issues before the loop, then six per trip and a wait down to six at the
    // bottom. The first trip enters with 4 in flight and peaks at 9 before an
    // issue; the back edge brings 6, so only the steady-state trip finds 11 before
    // its last issue. Returns the loop body's issues.
    std::vector<StinkyInstruction*> buildSteadyStateOverflowLoop() {
        tensorLoads(4);
        createLabel("label_LoopBeginL");
        createFiller();
        std::vector<StinkyInstruction*> body = tensorLoads(6);
        waitTensorCnt(6);
        createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/90, "label_LoopBeginL");
        createLabel("label_LoopEndL");
        waitTensorCnt(0);
        return body;
    }

    std::vector<StinkyInstruction*> stream() const {
        std::vector<StinkyInstruction*> insts;
        for (BasicBlock& block : *func)
            for (IRBase& ir : block)
                if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) insts.push_back(inst);
        return insts;
    }

    /// Runs the pass and returns what it added, in stream order. Everything that
    /// was there before has to come out in the same order.
    std::vector<StinkyInstruction*> runGuard(int limit = kLimit,
                                             std::vector<Function*> functions = {}) {
        const std::vector<StinkyInstruction*> before = stream();
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        createTDMInflightGuardPass(limit, std::move(functions))->run(*func, ctx, am);
        const std::unordered_set<StinkyInstruction*> old(before.begin(), before.end());
        std::vector<StinkyInstruction*> kept;
        std::vector<StinkyInstruction*> added;
        for (StinkyInstruction* inst : stream())
            (old.count(inst) != 0 ? kept : added).push_back(inst);
        EXPECT_EQ(kept, before) << "existing instructions were reordered or dropped";
        return added;
    }

    /// \p guard is `s_wait_tensorcnt <limit - 1>` and sits right before \p issue.
    void expectGuardBefore(const StinkyInstruction* guard, const StinkyInstruction* issue,
                           int limit = kLimit) const {
        ASSERT_NE(guard, nullptr);
        EXPECT_EQ(guard->getUnifiedOpcode(), GFX::s_wait_tensorcnt);
        ASSERT_FALSE(guard->getSrcRegs().empty());
        EXPECT_TRUE(guard->getSrcRegs()[0].dataType == StinkyRegister::Type::LiteralInt);
        EXPECT_EQ(guard->getSrcRegs()[0].getLiteralInt(), limit - 1);
        const auto* data = guard->getModifier<SWaitTensorCntData>();
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(static_cast<int>(data->tlcnt), limit - 1);
        const std::vector<StinkyInstruction*> insts = stream();
        const auto it = std::find(insts.begin(), insts.end(), guard);
        ASSERT_NE(it, insts.end());
        ASSERT_NE(std::next(it), insts.end());
        EXPECT_EQ(*std::next(it), issue);
    }
};

// Single-wave MX at PGR2 is the deepest per-wave shape generated today: four TDM
// ops per stage, two stages. The prologue issues both stages and every trip waits
// down to one stage before it issues the next, so at most 8 are ever in flight.
TEST_F(TDMInflightGuardPassTest, TodayShapedLoopIsLeftUntouched) {
    tensorLoads(8);
    createLabel("label_LoopBeginL");
    waitTensorCnt(4);
    createFiller();
    tensorLoads(4);
    createFiller();
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/90, "label_LoopBeginL");
    createLabel("label_LoopEndL");
    waitTensorCnt(0);
    EXPECT_TRUE(runGuard().empty());
}

// With no wait in between, the twelfth issue is the first to find 11 in flight.
TEST_F(TDMInflightGuardPassTest, StraightRunGuardsOnlyTheTwelfthIssue) {
    const std::vector<StinkyInstruction*> loads = tensorLoads(12);
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], loads[11]);
}

// A guard leaves the wave back at the limit after the issue, so every further
// issue without a wait needs one of its own.
TEST_F(TDMInflightGuardPassTest, EveryIssuePastTheLimitNeedsItsOwnGuard) {
    const std::vector<StinkyInstruction*> loads = tensorLoads(14);
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 3u);
    for (size_t i = 0; i < added.size(); ++i) expectGuardBefore(added[i], loads[11 + i]);
}

// Waits already in the stream count, including one whose count only the modifier
// carries.
TEST_F(TDMInflightGuardPassTest, ExistingWaitsCapTheBound) {
    tensorLoads(11);
    waitTensorCnt(5, /*withLiteral=*/false);
    tensorLoads(6);
    EXPECT_TRUE(runGuard().empty());
}

TEST_F(TDMInflightGuardPassTest, GuardLandsInsideTheLoopWhenOnlySteadyStateExceeds) {
    const std::vector<StinkyInstruction*> body = buildSteadyStateOverflowLoop();
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], body[5]);
}

// The backend runs the pass on a stream CFGBuilderPass has already split into
// blocks; the answer must not depend on where the block boundaries fall.
TEST_F(TDMInflightGuardPassTest, SameGuardAfterCFGBuilderSplitsTheStream) {
    const std::vector<StinkyInstruction*> body = buildSteadyStateOverflowLoop();
    PassContext ctx;
    ctx.setGemmTileConfig(config);
    createCFGBuilderPass()->run(*func, ctx, am);
    ASSERT_GT(func->size(), 1u);
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], body[5]);
}

// Only the path that skips the drain reaches the join with 6 in flight. A join
// takes the max over its incoming paths, so that path decides.
TEST_F(TDMInflightGuardPassTest, JoinTakesTheWorseIncomingPath) {
    tensorLoads(6);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_Join");
    waitTensorCnt(0);
    createLabel("label_Join");
    const std::vector<StinkyInstruction*> after = tensorLoads(6);
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], after[5]);
}

// A branch to a label this function does not define may land on any label, so
// the join sees the undrained count as well.
TEST_F(TDMInflightGuardPassTest, UnknownBranchTargetReachesEveryLabel) {
    tensorLoads(6);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_NotInThisFunction");
    waitTensorCnt(0);
    createLabel("label_Join");
    const std::vector<StinkyInstruction*> after = tensorLoads(6);
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], after[5]);
}

// Without the kernel's function list the callee is unknown, and so is the bound
// after the call.
TEST_F(TDMInflightGuardPassTest, UnknownCalleeSaturatesTheBound) {
    tensorLoads(2);
    createCall();
    StinkyInstruction* next = tensorLoad();
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], next);
}

// Activation bodies, the only callees generated today, issue no TDM op, so a call
// to one leaves the bound alone.
TEST_F(TDMInflightGuardPassTest, CallIsTransparentWhenNoCallableIssuesTensorOps) {
    std::unique_ptr<Function> callee = createCallee(/*issuesTensorOp=*/false);
    tensorLoads(2);
    createCall();
    tensorLoad();
    EXPECT_TRUE(runGuard(kLimit, {func.get(), callee.get()}).empty());
}

TEST_F(TDMInflightGuardPassTest, CallSaturatesWhenACallableIssuesTensorOps) {
    std::unique_ptr<Function> callee = createCallee(/*issuesTensorOp=*/true);
    tensorLoads(2);
    createCall();
    StinkyInstruction* next = tensorLoad();
    const std::vector<StinkyInstruction*> added = runGuard(kLimit, {func.get(), callee.get()});
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], next);
}

// What the caller of a callable function left in flight is unknown.
TEST_F(TDMInflightGuardPassTest, CallableFunctionStartsUnbounded) {
    func->setIsCallable(true);
    StinkyInstruction* first = tensorLoad();
    waitTensorCnt(0);
    tensorLoad();
    const std::vector<StinkyInstruction*> added = runGuard();
    ASSERT_EQ(added.size(), 1u);
    expectGuardBefore(added[0], first);
}

TEST_F(TDMInflightGuardPassTest, SmallerLimitGuardsEarlier) {
    const std::vector<StinkyInstruction*> loads = tensorLoads(6);
    const std::vector<StinkyInstruction*> added = runGuard(/*limit=*/4);
    ASSERT_EQ(added.size(), 2u);
    expectGuardBefore(added[0], loads[4], /*limit=*/4);
    expectGuardBefore(added[1], loads[5], /*limit=*/4);
}

TEST_F(TDMInflightGuardPassTest, LimitZeroDisablesThePass) {
    tensorLoads(12);
    EXPECT_TRUE(runGuard(/*limit=*/0).empty());
    EXPECT_TRUE(runGuard(/*limit=*/-1).empty());
}

TEST_F(TDMInflightGuardPassTest, SecondRunIsNoOp) {
    tensorLoads(14);
    EXPECT_EQ(runGuard().size(), 3u);
    EXPECT_TRUE(runGuard().empty());
}

TEST_F(TDMInflightGuardPassTest, RemarkReportsTheGuards) {
    tensorLoads(12);
    PassContext ctx;
    ctx.setGemmTileConfig(config);
    ctx.setRemarksEnabled(true);
    testing::internal::CaptureStderr();
    createTDMInflightGuardPass()->run(*func, ctx, am);
    const std::string remarks = testing::internal::GetCapturedStderr();
    EXPECT_NE(remarks.find("TDMInflightGuardPass"), std::string::npos) << remarks;
    EXPECT_NE(remarks.find("inserted 1 s_wait_tensorcnt 10"), std::string::npos) << remarks;
    EXPECT_NE(remarks.find("at most 11 TDM ops"), std::string::npos) << remarks;
}
