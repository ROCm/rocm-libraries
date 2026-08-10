/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
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

#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/RemoveDscntPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class RemoveDscntPassTest : public ::testing::Test {
   protected:
    static constexpr GfxArchID kArch = GfxArchID::Gfx1250;

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, kArch);
        bb = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    void runPass() {
        auto pass = createRemoveDscntPass();
        ASSERT_NE(pass, nullptr);
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        pass->run(*func, ctx, am);
    }

    void runPass(int dsProximityThreshold) {
        auto pass = createRemoveDscntPass(dsProximityThreshold);
        ASSERT_NE(pass, nullptr);
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        pass->run(*func, ctx, am);
    }

    StinkyInstruction* addDsLoad(int destReg, int addrReg = 0) {
        return createDsReadB128InBlock(bb, kArch, destReg, addrReg);
    }

    StinkyInstruction* addWmmaWithSrcV8() {
        return addWmmaWithSrcBase(8);
    }

    StinkyInstruction* addWmmaWithSrcBase(int srcBaseReg) {
        AsmIRBuilder builder(*bb, kArch);
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, kArch));
        inst->addDestReg(StinkyRegister("v", 0, 8));
        inst->addSrcReg(StinkyRegister("v", srcBaseReg, 8));
        inst->addSrcReg(StinkyRegister("v", srcBaseReg, 8));
        inst->addSrcReg(StinkyRegister("v", 0, 8));
        return inst;
    }

    StinkyInstruction* addWaitDscnt(int keep) {
        AsmIRBuilder builder(*bb, kArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_wait_dscnt, kArch));
        inst->addSrcReg(StinkyRegister(keep));
        inst->addModifier<SWaitCntData>(SWaitCntData{/*vlcnt=*/-1, /*vscnt=*/-1, /*dlcnt=*/keep,
                                                     /*dscnt=*/keep});
        return inst;
    }

    int countOpcode(GFX opcode) const {
        int n = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() == opcode) ++n;
        }
        return n;
    }

    bool hasTextblockContaining(const std::string& needle) const {
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyAsmDirective) continue;
            const auto* dir = cast<AsmDirective>(&ir);
            if (dir->kind != AsmDirectiveKind::TEXTBLOCK) continue;
            if (dir->value.find(needle) != std::string::npos) return true;
        }
        return false;
    }

    struct LineInfo {
        bool isWaitDscnt = false;
        int waitKeep = -1;
    };

    int getWaitKeep(const StinkyInstruction& inst) const {
        const auto* w = inst.getModifier<SWaitCntData>();
        if (w) {
            if (w->dlcnt >= 0 && w->dscnt >= 0) return std::min(w->dlcnt, w->dscnt);
            if (w->dlcnt >= 0) return w->dlcnt;
            if (w->dscnt >= 0) return w->dscnt;
        }

        const auto& srcs = inst.getSrcRegs();
        if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt)
            return srcs[0].getLiteralInt();
        return -1;
    }

    std::vector<LineInfo> collectLineInfo() const {
        std::vector<LineInfo> lines;
        for (const IRBase& ir : *bb) {
            LineInfo info;
            if (ir.getType() == IRBase::IRType::StinkyTofu) {
                const auto* inst = cast<StinkyInstruction>(&ir);
                if (inst->getUnifiedOpcode() == GFX::s_wait_dscnt ||
                    inst->getUnifiedOpcode() == GFX::s_wait_loadcnt_dscnt) {
                    info.isWaitDscnt = true;
                    info.waitKeep = getWaitKeep(*inst);
                }
            }
            lines.push_back(info);
        }
        return lines;
    }

    void expectWaitsAreTightenedOrRemovedLineByLine(const std::vector<LineInfo>& before) const {
        std::vector<const IRBase*> after;
        for (const IRBase& ir : *bb) after.push_back(&ir);
        ASSERT_EQ(after.size(), before.size())
            << "Before/after line count mismatch, cannot do line-by-line comparison";

        for (size_t i = 0; i < before.size(); ++i) {
            if (!before[i].isWaitDscnt) continue;

            const IRBase& afterIr = *after[i];
            if (afterIr.getType() == IRBase::IRType::StinkyTofu) {
                const auto* afterInst = cast<StinkyInstruction>(&afterIr);
                ASSERT_TRUE(afterInst->getUnifiedOpcode() == GFX::s_wait_dscnt ||
                            afterInst->getUnifiedOpcode() == GFX::s_wait_loadcnt_dscnt)
                    << "Line " << i << ": expected wait or remove-comment replacement";
                const int afterKeep = getWaitKeep(*afterInst);
                ASSERT_GE(before[i].waitKeep, 0) << "Line " << i << ": invalid before wait value";
                ASSERT_GE(afterKeep, 0) << "Line " << i << ": invalid after wait value";
                EXPECT_LE(afterKeep, before[i].waitKeep)
                    << "Line " << i << ": wait keep should be tightened or equal";
                continue;
            }

            if (afterIr.getType() == IRBase::IRType::StinkyAsmDirective) {
                const auto* dir = cast<AsmDirective>(&afterIr);
                ASSERT_EQ(dir->kind, AsmDirectiveKind::TEXTBLOCK)
                    << "Line " << i << ": replacement should be TEXTBLOCK comment";
                EXPECT_TRUE(dir->value.find("removed") != std::string::npos)
                    << "Line " << i << ": replacement comment should indicate removal";
                continue;
            }

            FAIL() << "Line " << i
                   << ": before was wait_dscnt, after is neither wait nor remove-comment";
        }
    }

    std::vector<std::string> collectWaitLineTransitions(const std::vector<LineInfo>& before) const {
        std::vector<std::string> out;
        std::vector<const IRBase*> after;
        for (const IRBase& ir : *bb) after.push_back(&ir);
        if (after.size() != before.size()) return out;

        for (size_t i = 0; i < before.size(); ++i) {
            if (!before[i].isWaitDscnt) continue;
            const IRBase& afterIr = *after[i];
            if (afterIr.getType() == IRBase::IRType::StinkyTofu) {
                const auto* afterInst = cast<StinkyInstruction>(&afterIr);
                const int afterKeep = getWaitKeep(*afterInst);
                out.push_back(std::to_string(before[i].waitKeep) + "->" +
                              std::to_string(afterKeep));
            } else if (afterIr.getType() == IRBase::IRType::StinkyAsmDirective) {
                out.push_back(std::to_string(before[i].waitKeep) + "->removed");
            } else {
                out.push_back(std::to_string(before[i].waitKeep) + "->invalid");
            }
        }
        return out;
    }

    // Pattern requested by user:
    // dsload v0/v1/v2/v3
    // 10 unrelated wmma
    // dscnt 3, wmma(v0)
    // dscnt 2, wmma(v1)
    // dscnt 1, wmma(v2)
    // dscnt 0, wmma(v3)
    void buildBigBlockPattern() {
        addDsLoad(/*destReg=*/0);
        addDsLoad(/*destReg=*/8);
        addDsLoad(/*destReg=*/16);
        addDsLoad(/*destReg=*/24);

        for (int i = 0; i < 10; ++i) addWmmaWithSrcBase(/*srcBaseReg=*/200);

        addWaitDscnt(3);
        addWmmaWithSrcBase(/*srcBaseReg=*/0);

        addWaitDscnt(2);
        addWmmaWithSrcBase(/*srcBaseReg=*/8);

        addWaitDscnt(1);
        addWmmaWithSrcBase(/*srcBaseReg=*/16);

        addWaitDscnt(0);
        addWmmaWithSrcBase(/*srcBaseReg=*/24);
    }

    void expectBigBlockPatternTransitions(int dsProximityThreshold,
                                          const std::vector<std::string>& expected) {
        buildBigBlockPattern();
        const std::vector<LineInfo> before = collectLineInfo();
        runPass(dsProximityThreshold);
        const auto transitions = collectWaitLineTransitions(before);
        EXPECT_EQ(transitions, expected);
        expectWaitsAreTightenedOrRemovedLineByLine(before);
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;
};

TEST_F(RemoveDscntPassTest, RemovesSecondWaitWhenKeepReachesPrevPlusDsLoads) {
    addDsLoad(/*destReg=*/8);
    addWmmaWithSrcV8();
    addWaitDscnt(/*keep=*/1);  // Establish previous kept dscnt.
    addDsLoad(/*destReg=*/20);
    addWaitDscnt(/*keep=*/2);  // 2 >= (prev 1 + one ds_load), remove.

    const int before = countOpcode(GFX::s_wait_dscnt);
    ASSERT_GE(before, 2);

    runPass();

    EXPECT_LT(countOpcode(GFX::s_wait_dscnt), before);
    EXPECT_TRUE(hasTextblockContaining("dscnt is removed"));
}

TEST_F(RemoveDscntPassTest, DoesNotMarkRemovedWhenKeepBelowPrevPlusDsLoads) {
    addDsLoad(/*destReg=*/8);
    addWmmaWithSrcV8();
    addWaitDscnt(/*keep=*/1);  // Previous kept dscnt.
    addDsLoad(/*destReg=*/24);
    addWaitDscnt(/*keep=*/1);  // 1 < (prev 1 + one ds_load), should not hit remove rule.

    runPass();

    EXPECT_FALSE(hasTextblockContaining("dscnt is removed"));
}

TEST_F(RemoveDscntPassTest, RemoveCommentIncludesKeepValueWhenNoTightenNeeded) {
    addDsLoad(/*destReg=*/8);
    addWmmaWithSrcV8();
    addWaitDscnt(/*keep=*/1);
    addDsLoad(/*destReg=*/28);
    addWaitDscnt(/*keep=*/2);  // Remove path without tighten.

    runPass();

    EXPECT_TRUE(hasTextblockContaining("remove dscnt: keep=2 dscnt is removed"));
}

TEST_F(RemoveDscntPassTest, bigBlockPattern100) {
    expectBigBlockPatternTransitions(/*dsProximityThreshold=*/100,
                                     {"3->3", "2->2", "1->1", "0->0"});
}

TEST_F(RemoveDscntPassTest, bigBlockPattern64) {
    expectBigBlockPatternTransitions(/*dsProximityThreshold=*/64,
                                     {"3->0", "2->removed", "1->removed", "0->removed"});
}

TEST_F(RemoveDscntPassTest, bigBlockPattern84) {
    expectBigBlockPatternTransitions(/*dsProximityThreshold=*/84,
                                     {"3->3", "2->2", "1->0", "0->removed"});
}
