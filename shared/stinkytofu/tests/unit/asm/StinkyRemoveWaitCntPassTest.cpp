// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// StinkyRemoveWaitCntPass keeps a wait whose comment carries the stinky-keep tag and removes the
// same wait without it. TensileLite tags the waits that WaitCntInsertion does not re-derive
// (STINKY_KEEP in Components/TDMStore.py).

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

    void addWait(GFX opcode, const char* comment) {
        AsmIRBuilder builder(*bb, kArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(opcode, kArch));
        inst->addSrcReg(StinkyRegister(0));
        if (opcode == GFX::s_wait_dscnt)
            inst->addModifier<SWaitCntData>(SWaitCntData{/*vlcnt=*/-1, /*vscnt=*/-1, /*dlcnt=*/0,
                                                         /*dscnt=*/0});
        if (comment != nullptr) inst->addModifier(CommentData(comment));
    }

    void runPass() {
        auto pass = createStinkyRemoveWaitCntPass();
        ASSERT_NE(pass, nullptr);
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        pass->run(*func, ctx, am);
    }

    // The waits of `opcode` left in the block, true for each one whose comment carries the tag.
    std::vector<bool> waitsLeft(GFX opcode) {
        std::vector<bool> tagged;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() != opcode) continue;
            const auto* comment = inst->getModifier<CommentData>();
            tagged.push_back(comment != nullptr &&
                             comment->comment.find("stinky-keep") != std::string::npos);
        }
        return tagged;
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;
};

TEST_F(StinkyRemoveWaitCntPassTest, KeepsTaggedWaitsAndRemovesUntaggedOnes) {
    addWait(GFX::s_wait_dscnt, "staging stores done [stinky-keep]");
    addWait(GFX::s_wait_tensorcnt, "wait for the TDM load of C [stinky-keep]");
    addWait(GFX::s_wait_dscnt, "no tag");
    addWait(GFX::s_wait_tensorcnt, nullptr);
    createVAddInBlock(bb, kArch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    runPass();

    EXPECT_EQ(waitsLeft(GFX::s_wait_dscnt), std::vector<bool>{true});
    EXPECT_EQ(waitsLeft(GFX::s_wait_tensorcnt), std::vector<bool>{true});
}

}  // namespace
