// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// StinkyRemoveWaitCntPass keeps a wait that carries KeepWaitData and removes the same wait without
// it, whatever its comment says. TensileLite sets keepWait on the waits that WaitCntInsertion does
// not re-derive (Components/TDMStoreInst.py).

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class StinkyRemoveWaitCntPassTest : public ::testing::Test {
   protected:
    static constexpr GfxArchID kArch = GfxArchID::Gfx1250;

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, kArch);
        bb = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    void addWait(GFX opcode, bool keep, const char* comment = nullptr) {
        AsmIRBuilder builder(*bb, kArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(opcode, kArch));
        inst->addSrcReg(StinkyRegister(0));
        if (opcode == GFX::s_wait_dscnt)
            inst->addModifier<SWaitCntData>(SWaitCntData{/*vlcnt=*/-1, /*vscnt=*/-1, /*dlcnt=*/0,
                                                         /*dscnt=*/0});
        if (keep) inst->addModifier(KeepWaitData{});
        if (comment != nullptr) inst->addModifier(CommentData(comment));
    }

    void runPass() {
        auto pass = createStinkyRemoveWaitCntPass();
        ASSERT_NE(pass, nullptr);
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        pass->run(*func, ctx, am);
    }

    // The waits of `opcode` left in the block, true for each one that carries KeepWaitData.
    std::vector<bool> waitsLeft(GFX opcode) {
        std::vector<bool> kept;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() != opcode) continue;
            kept.push_back(inst->getModifier<KeepWaitData>() != nullptr);
        }
        return kept;
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;
};

TEST_F(StinkyRemoveWaitCntPassTest, KeepsWaitsWithKeepWaitDataAndRemovesTheOthers) {
    addWait(GFX::s_wait_dscnt, /*keep=*/true, "staging stores done");
    addWait(GFX::s_wait_tensorcnt, /*keep=*/true);
    // Only the modifier keeps a wait; a comment does not, whatever it says.
    addWait(GFX::s_wait_dscnt, /*keep=*/false, "staging stores done [stinky-keep]");
    addWait(GFX::s_wait_tensorcnt, /*keep=*/false, "wait for the TDM store [stinky-keep]");
    addWait(GFX::s_wait_dscnt, /*keep=*/false);
    createVAddInBlock(bb, kArch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    runPass();

    EXPECT_EQ(waitsLeft(GFX::s_wait_dscnt), std::vector<bool>{true});
    EXPECT_EQ(waitsLeft(GFX::s_wait_tensorcnt), std::vector<bool>{true});
}

}  // namespace
