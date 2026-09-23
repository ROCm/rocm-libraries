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

// s_cmov_b32 d, s is "if (SCC) d = s; else d = d": d is tied RW.

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

// Seed s_mov + s_cmov: tied dest must not be rewritten (MX now emits s_cselect).
TEST_F(AsmMovePropagationPassTest, TiedCmovReadIsNotPropagatedAndSeedMovSurvives) {
    emit("s_mov_b32", {sgpr(21)}, {sgpr(23)});
    emit("s_cmp_eq_u32", {}, {sgpr(10), sgpr(11)});
    StinkyInstruction* cmov = emit("s_cmov_b32", {sgpr(21)}, {sgpr(25), sgpr(21)});

    runPass();

    EXPECT_TRUE(reads(*cmov, sgpr(21))) << "tied read-write source was rewritten";
    EXPECT_FALSE(reads(*cmov, sgpr(23)));
    EXPECT_EQ(countMnemonic("s_mov_b32"), 1u) << "the mov seeding s21 was erased";
}

// Untied cmov source still participates in propagation.
TEST_F(AsmMovePropagationPassTest, UntiedCmovSourceIsStillPropagated) {
    emit("s_mov_b32", {sgpr(25)}, {sgpr(27)});
    StinkyInstruction* cmov = emit("s_cmov_b32", {sgpr(21)}, {sgpr(25), sgpr(21)});

    runPass();

    EXPECT_TRUE(reads(*cmov, sgpr(27))) << "propagation into the ordinary source was lost";
    EXPECT_TRUE(reads(*cmov, sgpr(21))) << "tied read-write source was rewritten";
    EXPECT_FALSE(reads(*cmov, sgpr(25)));
}

// Ordinary mov chain: rewrite the use and drop the dead mov.
TEST_F(AsmMovePropagationPassTest, OrdinaryMovChainStillPropagatesAndDeadMovIsErased) {
    emit("s_mov_b32", {sgpr(21)}, {sgpr(23)});
    StinkyInstruction* addInst = emit("s_add_u32", {sgpr(30)}, {sgpr(21), sgpr(24)});
    emit("s_mov_b32", {sgpr(21)}, {sgpr(26)});

    runPass();

    EXPECT_TRUE(reads(*addInst, sgpr(23))) << "ordinary copy propagation stopped working";
    EXPECT_FALSE(reads(*addInst, sgpr(21)));
    EXPECT_EQ(countMnemonic("s_mov_b32"), 1u) << "the dead mov should still be erased";
}

// s_add_u32 has no RW field; a dest that names its own source is an ordinary read.
TEST_F(AsmMovePropagationPassTest, PlainDestinationReadingItselfIsStillPropagated) {
    emit("s_mov_b32", {sgpr(21)}, {sgpr(23)});
    StinkyInstruction* addInst = emit("s_add_u32", {sgpr(21)}, {sgpr(21), sgpr(24)});

    runPass();

    EXPECT_TRUE(reads(*addInst, sgpr(23))) << "the fix is too broad: it skipped a plain source";
    EXPECT_FALSE(reads(*addInst, sgpr(21)));
    EXPECT_EQ(countMnemonic("s_mov_b32"), 0u) << "the seed mov is dead once its only read is gone";
}

// v_fmac_f32 accumulator is tied RW.
TEST_F(AsmMovePropagationPassTest, TiedFmacAccumulatorIsNotPropagated) {
    emit("v_mov_b32", {vgpr(0)}, {vgpr(5)});
    StinkyInstruction* fmac = emit("v_fmac_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)});

    runPass();

    EXPECT_TRUE(reads(*fmac, vgpr(0))) << "tied accumulator read was rewritten";
    EXPECT_FALSE(reads(*fmac, vgpr(5)));
    EXPECT_EQ(countMnemonic("v_mov_b32"), 1u) << "the mov seeding v0 was erased";
}

// v_swap_b32 marks both fields RW.
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
