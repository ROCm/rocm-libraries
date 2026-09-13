/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * Unit tests: AsmMovePropagationPass must not copy-propagate into a source tied
 * to a read-write destination, and must still propagate ordinary mov chains.
 * ************************************************************************ */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/AsmMovePropagationPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// ===========================================================================
// Read-write (tied) destinations
//
// s_cmov_b32 d, s is "if (SCC) d = s; else d = d", so d is read as well as
// written. HwInstDesc marks that field RW and the IR carries d in both destRegs
// and srcRegs (AsmVerifierPass enforces it; see ReadWriteOperandTest). The
// emitter prints one operand per destination field and never prints an RW field
// as a source, so rewriting that source leaves the assembly text unchanged --
// but it hides the read from this pass's own liveness scan, which then erases
// the unconditional mov that seeded the register.
// ===========================================================================

class AsmMovePropagationPassTest : public ::testing::Test {
   protected:
    GfxArchID arch{};
    std::unique_ptr<Function> func;
    BasicBlock* bb{};
    AnalysisManager am;
    PassContext passCtx;

    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        func = std::make_unique<Function>("move_propagation_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    StinkyInstruction* emit(const char* mnemonic, const std::vector<StinkyRegister>& destRegs,
                            const std::vector<StinkyRegister>& srcRegs) {
        const uint16_t isaOp = getMnemonicToIsaOpcode(mnemonic, arch);
        EXPECT_NE(isaOp, GFX::INVALID) << mnemonic;
        const HwInstDesc* desc = getMCIDByIsaOp(isaOp, arch);
        EXPECT_NE(desc, nullptr) << mnemonic;
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(desc);
        for (const StinkyRegister& reg : destRegs) inst->addDestReg(reg);
        for (const StinkyRegister& reg : srcRegs) inst->addSrcReg(reg);
        return inst;
    }

    void runPass() {
        auto pass = createAsmMovePropagationPass();
        pass->run(*func, passCtx, am);
    }

    size_t countMnemonic(const std::string& mnemonic) const {
        size_t count = 0;
        for (const IRBase& node : *bb) {
            if (node.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&node);
            const HwInstDesc* desc = inst->getHwInstDesc();
            if (desc && desc->mnemonic && mnemonic == desc->mnemonic) ++count;
        }
        return count;
    }

    static bool reads(const StinkyInstruction& inst, const StinkyRegister& reg) {
        for (const StinkyRegister& src : inst.getSrcRegs()) {
            if (src == reg) return true;
        }
        return false;
    }
};

// The pre-fix shape of the TDMFuse=2 shared-set increment: a seed s_mov with
// s_cmov overrides. Propagating into the cmov's tied read makes the seed look
// dead, leaving the waves that do not take the cmov reading an undefined
// register -- so the fixture is right to construct this shape even though MX
// codegen no longer emits it (it emits chained s_cselect and no s_mov/s_cmov
// for that register, and that absence is correct on MX). This test guards the
// operand rule, which any opcode with a read-write destination still needs.
TEST_F(AsmMovePropagationPassTest, TiedCmovReadIsNotPropagatedAndSeedMovSurvives) {
    emit("s_mov_b32", {sgpr(21)}, {sgpr(23)});
    emit("s_cmp_eq_u32", {}, {sgpr(10), sgpr(11)});
    StinkyInstruction* cmov = emit("s_cmov_b32", {sgpr(21)}, {sgpr(25), sgpr(21)});

    runPass();

    EXPECT_TRUE(reads(*cmov, sgpr(21))) << "tied read-write source was rewritten";
    EXPECT_FALSE(reads(*cmov, sgpr(23)));
    EXPECT_EQ(countMnemonic("s_mov_b32"), 1u) << "the mov seeding s21 was erased";
}

// Only the tied operand is off limits. The cmov's real source is an ordinary
// read and still takes part in propagation.
TEST_F(AsmMovePropagationPassTest, UntiedCmovSourceIsStillPropagated) {
    emit("s_mov_b32", {sgpr(25)}, {sgpr(27)});
    StinkyInstruction* cmov = emit("s_cmov_b32", {sgpr(21)}, {sgpr(25), sgpr(21)});

    runPass();

    EXPECT_TRUE(reads(*cmov, sgpr(27))) << "propagation into the ordinary source was lost";
    EXPECT_TRUE(reads(*cmov, sgpr(21))) << "tied read-write source was rewritten";
    EXPECT_FALSE(reads(*cmov, sgpr(25)));
}

// The pass still does what it exists for: rewrite the use and drop the mov once
// the destination is redefined with no reader in between.
TEST_F(AsmMovePropagationPassTest, OrdinaryMovChainStillPropagatesAndDeadMovIsErased) {
    emit("s_mov_b32", {sgpr(21)}, {sgpr(23)});
    StinkyInstruction* addInst = emit("s_add_u32", {sgpr(30)}, {sgpr(21), sgpr(24)});
    emit("s_mov_b32", {sgpr(21)}, {sgpr(26)});

    runPass();

    EXPECT_TRUE(reads(*addInst, sgpr(23))) << "ordinary copy propagation stopped working";
    EXPECT_FALSE(reads(*addInst, sgpr(21)));
    EXPECT_EQ(countMnemonic("s_mov_b32"), 1u) << "the dead mov should still be erased";
}

// A destination that merely happens to name one of its own sources is not tied:
// s_add_u32 has no read-write field, so that source is an ordinary read and the
// fix must not reach it.
TEST_F(AsmMovePropagationPassTest, PlainDestinationReadingItselfIsStillPropagated) {
    emit("s_mov_b32", {sgpr(21)}, {sgpr(23)});
    StinkyInstruction* addInst = emit("s_add_u32", {sgpr(21)}, {sgpr(21), sgpr(24)});

    runPass();

    EXPECT_TRUE(reads(*addInst, sgpr(23))) << "the fix is too broad: it skipped a plain source";
    EXPECT_FALSE(reads(*addInst, sgpr(21)));
    EXPECT_EQ(countMnemonic("s_mov_b32"), 0u) << "the seed mov is dead once its only read is gone";
}

// Not specific to s_cmov: v_fmac_f32 accumulates into a read-write vdst, so its
// tied read must survive too.
TEST_F(AsmMovePropagationPassTest, TiedFmacAccumulatorIsNotPropagated) {
    emit("v_mov_b32", {vgpr(0)}, {vgpr(5)});
    StinkyInstruction* fmac = emit("v_fmac_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)});

    runPass();

    EXPECT_TRUE(reads(*fmac, vgpr(0))) << "tied accumulator read was rewritten";
    EXPECT_FALSE(reads(*fmac, vgpr(5)));
    EXPECT_EQ(countMnemonic("v_mov_b32"), 1u) << "the mov seeding v0 was erased";
}

// v_swap_b32 marks both of its fields read-write, so both reads are tied.
TEST_F(AsmMovePropagationPassTest, BothTiedSwapOperandsAreNotPropagated) {
    emit("v_mov_b32", {vgpr(0)}, {vgpr(5)});
    emit("v_mov_b32", {vgpr(1)}, {vgpr(6)});
    StinkyInstruction* swap = emit("v_swap_b32", {vgpr(0), vgpr(1)}, {vgpr(0), vgpr(1)});

    runPass();

    EXPECT_TRUE(reads(*swap, vgpr(0)));
    EXPECT_TRUE(reads(*swap, vgpr(1)));
    EXPECT_FALSE(reads(*swap, vgpr(5)));
    EXPECT_FALSE(reads(*swap, vgpr(6)));
    EXPECT_EQ(countMnemonic("v_mov_b32"), 2u);
}
