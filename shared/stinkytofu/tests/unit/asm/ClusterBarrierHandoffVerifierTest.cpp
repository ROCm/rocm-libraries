// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/ClusterBarrierHandoffVerifier.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;

namespace {
class ClusterBarrierHandoffVerifierTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func = std::make_unique<Function>("handoff_test");
    BasicBlock* bb = func->createBasicBlock("entry");

    StinkyInstruction* instruction(GFX opcode) {
        AsmIRBuilder builder(*bb, arch);
        return builder.create(getMCIDByUOp(opcode, arch));
    }
    StinkyInstruction* signal(int id) {
        auto* inst = instruction(GFX::s_barrier_signal);
        inst->addSrcReg(StinkyRegister(id));
        return inst;
    }
    StinkyInstruction* wait(int id) {
        auto* inst = instruction(GFX::s_barrier_wait);
        inst->addSrcReg(StinkyRegister(id));
        return inst;
    }
    void join() {
        signal(-1);
        wait(-1);
    }
    void label(const char* name) {
        AsmIRBuilder builder(*bb, arch);
        builder.createLabel(name);
    }
    void branch(const char* target, bool conditional = false) {
        auto* inst = instruction(conditional ? GFX::s_cbranch_scc1 : GFX::s_branch);
        inst->addSrcReg(StinkyRegister(std::string(target)));
        inst->addModifier<LabelData>(LabelData{target});
    }
    void compare(unsigned reg, int value, bool equal = true) {
        auto* inst = instruction(equal ? GFX::s_cmp_eq_u32 : GFX::s_cmp_lg_u32);
        inst->addSrcReg(StinkyRegister("s", reg, 1));
        inst->addSrcReg(StinkyRegister(value));
    }
    void clobberScc() {
        auto* inst = instruction(GFX::s_add_u32);
        inst->addDestReg(StinkyRegister("s", 90, 1));
        inst->addSrcReg(StinkyRegister("s", 90, 1));
        inst->addSrcReg(StinkyRegister(1));
        // Intentionally rely on the descriptor's implicit SCC write.
    }
    void zeroTripPrefix() {
        compare(11, 0);
        branch("zero", true);
        wait(-3);
        join();
        branch("shadow");
        label("zero");
        wait(-3);
        label("shadow");
        clobberScc();
    }
    void zeroTripExit() {
        compare(11, 0);
        branch("exit", true);
        signal(-3);
        label("exit");
        end();
    }
    void end() {
        instruction(GFX::s_endpgm);
    }
    std::string verify() {
        return verifyClusterBarrierHandoffs(*func);
    }
    void runPass() {
        PassContext context;
        AnalysisManager analyses;
        createClusterBarrierHandoffVerifierPass()->run(*func, context, analyses);
    }
};

TEST_F(ClusterBarrierHandoffVerifierTest, AcceptsImmediateTokenAnnotatedJoin) {
    wait(-3);
    signal(-1)->addModifier<MemTokenData>(MemTokenData{{0}});
    wait(-1)->addModifier<MemTokenData>(MemTokenData{{0}});
    signal(-3);
    EXPECT_TRUE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, RejectsOriginalEarlySignal) {
    wait(-3);
    signal(-3);
    join();
    EXPECT_NE(verify().find("without a workgroup signal/wait pair"), std::string::npos);
    EXPECT_DEATH(runPass(), "cluster handoff.*without a workgroup signal/wait pair");
}

TEST_F(ClusterBarrierHandoffVerifierTest, EarlierLocalArrivalDoesNotDischargeWait) {
    signal(-1);
    wait(-3);
    wait(-1);
    signal(-3);
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, LaterClusterWaitInvalidatesLocalArrival) {
    wait(-3);
    signal(-1);
    wait(-3);
    wait(-1);
    signal(-3);
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, AcceptsLaterJoinAcrossBlocks) {
    wait(-3);
    branch("other", true);
    instruction(GFX::s_nop)->addSrcReg(StinkyRegister(0));
    branch("joined");
    bb = func->createBasicBlock("other");
    branch("joined");
    bb = func->createBasicBlock("joined");
    signal(-1);
    instruction(GFX::s_nop)->addSrcReg(StinkyRegister(0));
    bb = func->createBasicBlock("local_wait");
    wait(-1);
    signal(-3);
    EXPECT_TRUE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, RejectsClonePathThatBypassesJoin) {
    wait(-3);
    branch("label_InitCIterWmma_label_LoopBeginL", true);
    join();
    branch("signal");
    label("label_InitCIterWmma_label_LoopBeginL");
    instruction(GFX::s_nop)->addSrcReg(StinkyRegister(0));
    label("signal");
    signal(-3);
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, ChecksLaterLoopPhasesAsWellAsEntry) {
    wait(-3);
    join();
    label("loop");
    signal(-3);
    wait(-3);
    // The loop's missing join must be caught even though entry is protected.
    branch("loop", true);
    end();
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, AcceptsJoinedLoopBackEdge) {
    label("loop");
    wait(-3);
    join();
    signal(-3);
    branch("loop", true);
    end();
    EXPECT_TRUE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, ReturnDoesNotFallThroughIntoOutOfLineCode) {
    wait(-3);
    end();
    label("unreachable_signal");
    signal(-3);
    EXPECT_TRUE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, FollowsAnnotatedIndirectBranches) {
    wait(-3);
    auto* jump = instruction(GFX::s_setpc_b64);
    jump->addSrcReg(StinkyRegister("s", 0, 2));
    jump->addModifier<LabelData>(LabelData{"target"});
    join();
    label("target");
    signal(-3);
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, UnresolvedBranchIsNotSilentlyAccepted) {
    wait(-3);
    branch("missing_target");
    EXPECT_NE(verify().find("unresolved branch"), std::string::npos);
}

TEST_F(ClusterBarrierHandoffVerifierTest, UnknownCallIsNotSilentlyAccepted) {
    wait(-3);
    instruction(GFX::s_swappc_b64);
    join();
    signal(-3);
    EXPECT_NE(verify().find("call whose barrier behavior is not proven"), std::string::npos);
}

TEST_F(ClusterBarrierHandoffVerifierTest, CrossesKnownBarrierFreeHelper) {
    wait(-3);
    auto* call = instruction(GFX::s_swappc_b64);
    call->addModifier<CallTargetData>(CallTargetData{{"helper"}});
    join();
    signal(-3);
    end();
    label("helper");
    instruction(GFX::s_nop)->addSrcReg(StinkyRegister(0));
    instruction(GFX::s_setpc_b64)->addSrcReg(StinkyRegister("s", 0, 2));
    EXPECT_TRUE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, RejectsSynchronizationInsidePendingCall) {
    wait(-3);
    auto* call = instruction(GFX::s_swappc_b64);
    call->addModifier<CallTargetData>(CallTargetData{{"helper"}});
    join();
    signal(-3);
    end();
    label("helper");
    signal(-3);
    instruction(GFX::s_setpc_b64)->addSrcReg(StinkyRegister("s", 0, 2));
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, ZeroTripWaitCannotReachMainLoopSignal) {
    // The zero-trip wait needs no join: its counter is still zero at the
    // later exit guard, despite intervening SCC writes and a CFG merge.
    zeroTripPrefix();
    zeroTripExit();
    EXPECT_TRUE(verify().empty());
}

// The nonzero entry path must not inherit the later zero-trip bypass. Once
// that counter is overwritten, both outcomes are possible and the gap is real.
class ClusterBarrierNonzeroGuardTest : public ClusterBarrierHandoffVerifierTest,
                                       public testing::WithParamInterface<bool> {};

TEST_P(ClusterBarrierNonzeroGuardTest, RequiresJoinOnEveryFeasiblePath) {
    compare(11, 0);
    branch("exit", true);
    wait(-3);
    clobberScc();
    if (GetParam()) {
        auto* mov = instruction(GFX::s_mov_b64);
        mov->addDestReg(StinkyRegister("s", 10, 2));
        mov->addSrcReg(StinkyRegister("s", 20, 2));
    }
    compare(11, 0);
    branch("next_signal", true);
    join();
    label("next_signal");
    signal(-3);
    label("exit");
    end();
    EXPECT_EQ(verify().empty(), !GetParam());
}

INSTANTIATE_TEST_SUITE_P(CounterWrites, ClusterBarrierNonzeroGuardTest,
                         testing::Values(false, true));

TEST_F(ClusterBarrierHandoffVerifierTest, CounterWriteInvalidatesZeroTripFact) {
    zeroTripPrefix();
    auto* mov = instruction(GFX::s_mov_b32);
    mov->addDestReg(StinkyRegister("s", 11, 1));
    mov->addSrcReg(StinkyRegister(4));
    zeroTripExit();
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, OverlappingWriteInvalidatesZeroTripFact) {
    zeroTripPrefix();
    auto* mov = instruction(GFX::s_mov_b64);
    mov->addDestReg(StinkyRegister("s", 10, 2));
    mov->addSrcReg(StinkyRegister("s", 20, 2));
    zeroTripExit();
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, OffsetDestinationInvalidatesScalarFacts) {
    zeroTripPrefix();
    auto* mov = instruction(GFX::s_mov_b32);
    mov->addDestReg(StinkyRegister("s", 12, 1, -1));
    mov->addSrcReg(StinkyRegister(4));
    zeroTripExit();
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, ImplicitSccWriteInvalidatesBranchResult) {
    compare(11, 0);
    branch("zero", true);
    end();
    label("zero");
    wait(-3);
    clobberScc();
    branch("exit", true);
    signal(-3);
    label("exit");
    end();
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, HelperClobbersScalarFacts) {
    zeroTripPrefix();
    auto* call = instruction(GFX::s_swappc_b64);
    call->addModifier<CallTargetData>(CallTargetData{{"helper"}});
    zeroTripExit();
    label("helper");
    auto* mov = instruction(GFX::s_mov_b32);
    mov->addDestReg(StinkyRegister("s", 11, 1));
    mov->addSrcReg(StinkyRegister(4));
    instruction(GFX::s_setpc_b64)->addSrcReg(StinkyRegister("s", 0, 2));
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, EqualityFactFromNotEqualFallthrough) {
    compare(11, 0, false);
    branch("exit", true);
    wait(-3);
    clobberScc();
    compare(11, 0);
    branch("exit", true);
    signal(-3);
    label("exit");
    end();
    EXPECT_TRUE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, LoopMergeDoesNotKeepFirstIterationFact) {
    compare(11, 0);
    branch("loop", true);
    end();
    label("loop");
    wait(-3);
    compare(11, 0);
    branch("next", true);
    signal(-3);
    label("next");
    auto* mov = instruction(GFX::s_mov_b32);
    mov->addDestReg(StinkyRegister("s", 11, 1));
    mov->addSrcReg(StinkyRegister(4));
    branch("loop");
    EXPECT_FALSE(verify().empty());
}

TEST_F(ClusterBarrierHandoffVerifierTest, InlineLabelDefinesActualBranchDestination) {
    wait(-3);
    branch("target");
    bb = func->createBasicBlock("target");
    join();  // A late insertion ahead of the inline label cannot cover the jump.
    label("target");
    signal(-3);
    EXPECT_NE(verify().find("without a workgroup signal/wait pair"), std::string::npos);
}

TEST_F(ClusterBarrierHandoffVerifierTest, PassDoesNotChangeInstructions) {
    wait(-3);
    join();
    signal(-3);
    std::vector<const IRBase*> before;
    for (const IRBase& inst : *bb) before.push_back(&inst);
    runPass();
    std::vector<const IRBase*> after;
    for (const IRBase& inst : *bb) after.push_back(&inst);
    EXPECT_EQ(after, before);
}
}  // namespace
