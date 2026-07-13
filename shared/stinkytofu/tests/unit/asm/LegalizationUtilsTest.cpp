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

#include <map>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class LegalizationUtilsTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    int countInstructions() const {
        int n = 0;
        for (auto it = bb->begin(); it != bb->end(); ++it) n++;
        return n;
    }

    int countByMnemonic(const std::string& mnemonic) const {
        int n = 0;
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(&*it);
            if (!inst) continue;
            const HwInstDesc* desc = inst->getHwInstDesc();
            if (desc && desc->mnemonic && std::string(desc->mnemonic) == mnemonic) n++;
        }
        return n;
    }

    StinkyInstruction* createInst(GFX opcode) {
        AsmIRBuilder builder(*bb, arch);
        return builder.create(getMCIDByUOp(opcode, arch));
    }
};

// ---------------------------------------------------------------------------
// legalizeVNop
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, VNopCountZeroRemovesInstruction) {
    StinkyInstruction* inst = createInst(GFX::v_nop);
    inst->addSrcReg(StinkyRegister(0));  // count = 0
    ASSERT_EQ(countInstructions(), 1);

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeVNop(inst, builder, arch);

    EXPECT_EQ(result.first, nullptr);
    EXPECT_EQ(result.last, nullptr);
    EXPECT_EQ(countInstructions(), 0);
}

TEST_F(LegalizationUtilsTest, VNopCountOneRemovesInstruction) {
    StinkyInstruction* inst = createInst(GFX::v_nop);
    inst->addSrcReg(StinkyRegister(1));  // count = 1 → just remove
    ASSERT_EQ(countInstructions(), 1);

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeVNop(inst, builder, arch);

    EXPECT_EQ(result.first, nullptr);
    EXPECT_EQ(result.last, nullptr);
    EXPECT_EQ(countInstructions(), 0);
}

TEST_F(LegalizationUtilsTest, VNopExpandsCountThree) {
    StinkyInstruction* inst = createInst(GFX::v_nop);
    inst->addSrcReg(StinkyRegister(3));  // count = 3 → 3 separate v_nops

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeVNop(inst, builder, arch);

    EXPECT_NE(result.first, nullptr);
    EXPECT_NE(result.last, nullptr);
    EXPECT_NE(result.first, result.last);
    EXPECT_EQ(countByMnemonic("v_nop"), 3);
}

TEST_F(LegalizationUtilsTest, VNopPreservesComment) {
    StinkyInstruction* inst = createInst(GFX::v_nop);
    inst->addSrcReg(StinkyRegister(2));
    inst->addModifier<CommentData>(CommentData{"nop comment"});

    AsmIRBuilder builder(*bb, arch);
    legalizeVNop(inst, builder, arch);

    int withComment = 0;
    for (auto it = bb->begin(); it != bb->end(); ++it) {
        auto* i = dyn_cast<StinkyInstruction>(&*it);
        if (i && i->getModifier<CommentData>()) withComment++;
    }
    EXPECT_EQ(withComment, 2);
}

// ---------------------------------------------------------------------------
// legalizeVCmpX
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, VCmpXNoOpWhenCMPXWritesSGPR) {
    // On an arch where CMPXWritesSGPR=1, no lowering is needed
    StinkyInstruction* inst = createInst(GFX::v_cmpx_lt_f32);
    inst->addDestReg(StinkyRegister(RegType::EXEC_LO, 0, 1));
    inst->addSrcReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    ASSERT_EQ(countInstructions(), 1);

    AsmIRBuilder builder(*bb, arch);
    std::map<std::string, int> archCaps = {{"CMPXWritesSGPR", 1}};
    auto result = legalizeVCmpX(inst, builder, arch, archCaps);

    EXPECT_EQ(result.first, nullptr);
    EXPECT_EQ(result.last, nullptr);
    EXPECT_EQ(countInstructions(), 1);  // original untouched
}

TEST_F(LegalizationUtilsTest, VCmpXExpandsWhenNoCMPXWritesSGPR) {
    StinkyInstruction* inst = createInst(GFX::v_cmpx_lt_f32);
    inst->addDestReg(StinkyRegister(RegType::EXEC_LO, 0, 1));
    inst->addSrcReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));

    AsmIRBuilder builder(*bb, arch);
    std::map<std::string, int> archCaps = {{"CMPXWritesSGPR", 0}};
    auto result = legalizeVCmpX(inst, builder, arch, archCaps);

    ASSERT_NE(result.first, nullptr);
    ASSERT_NE(result.last, nullptr);

    // Should expand to v_cmp + s_mov
    auto* cmpInst = dyn_cast<StinkyInstruction>(result.first);
    auto* movInst = dyn_cast<StinkyInstruction>(result.last);
    ASSERT_NE(cmpInst, nullptr);
    ASSERT_NE(movInst, nullptr);

    std::string cmpMnemonic(cmpInst->getHwInstDesc()->mnemonic);
    std::string movMnemonic(movInst->getHwInstDesc()->mnemonic);
    EXPECT_NE(cmpMnemonic.find("_cmp_"), std::string::npos) << "Expected v_cmp_* got: " << cmpMnemonic;
    EXPECT_NE(movMnemonic.find("s_mov"), std::string::npos) << "Expected s_mov_* got: " << movMnemonic;

    // Original v_cmpx removed
    EXPECT_EQ(countByMnemonic("v_cmpx_lt_f32"), 0);
    EXPECT_EQ(countInstructions(), 2);
}

// ---------------------------------------------------------------------------
// legalizeWaitCnt
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, WaitCntNonGfx1250IsNoOp) {
    // Only Gfx1250 is registered in this build; pass a synthetic non-1250 archId
    // to exercise the early-return path without needing a second arch setup.
    auto nonGfx1250 = static_cast<GfxArchID>(static_cast<uint32_t>(GfxArchID::Gfx1250) + 1);

    StinkyInstruction* inst = createInst(GFX::s_waitcnt);
    inst->addModifier<SWaitCntData>(SWaitCntData(/*vlcnt=*/0));

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeWaitCnt(inst, builder, nonGfx1250);
    EXPECT_EQ(result.first, nullptr);
    EXPECT_EQ(result.last, nullptr);
}

TEST_F(LegalizationUtilsTest, WaitCntVlcntOnlyEmitsLoadcnt) {
    StinkyInstruction* inst = createInst(GFX::s_waitcnt);
    inst->addModifier<SWaitCntData>(SWaitCntData(/*vlcnt=*/2));

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeWaitCnt(inst, builder, arch);

    ASSERT_NE(result.first, nullptr);
    auto* waitInst = dyn_cast<StinkyInstruction>(result.first);
    ASSERT_NE(waitInst, nullptr);
    std::string mnemonic(waitInst->getHwInstDesc()->mnemonic);
    EXPECT_EQ(mnemonic, "s_wait_loadcnt");
    EXPECT_EQ(countByMnemonic("s_waitcnt"), 0);
}

TEST_F(LegalizationUtilsTest, WaitCntVlcntAndDscntEmitsCombined) {
    // vlcnt + dscnt → s_wait_loadcnt_dscnt
    SWaitCntData data;
    data.vlcnt = 1;
    data.dscnt = 0;
    StinkyInstruction* inst = createInst(GFX::s_waitcnt);
    inst->addModifier<SWaitCntData>(data);

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeWaitCnt(inst, builder, arch);

    ASSERT_NE(result.first, nullptr);
    auto* waitInst = dyn_cast<StinkyInstruction>(result.first);
    ASSERT_NE(waitInst, nullptr);
    std::string mnemonic(waitInst->getHwInstDesc()->mnemonic);
    EXPECT_EQ(mnemonic, "s_wait_loadcnt_dscnt");
    EXPECT_EQ(countByMnemonic("s_waitcnt"), 0);
}

TEST_F(LegalizationUtilsTest, WaitCntKmcntOnlyEmitsKmcnt) {
    SWaitCntData data;
    data.kmcnt = 0;
    StinkyInstruction* inst = createInst(GFX::s_waitcnt);
    inst->addModifier<SWaitCntData>(data);

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeWaitCnt(inst, builder, arch);

    ASSERT_NE(result.first, nullptr);
    EXPECT_EQ(countByMnemonic("s_wait_kmcnt"), 1);
    EXPECT_EQ(countByMnemonic("s_waitcnt"), 0);
}

// ---------------------------------------------------------------------------
// legalizeDSLoadB192
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, DSLoadB192SplitsIntoB128AndB64) {
    StinkyInstruction* inst = createInst(GFX::ds_load_b192);
    inst->addDestReg(StinkyRegister("v", 0, 6));   // v[0:5]
    inst->addSrcReg(StinkyRegister("v", 40, 1));   // addr
    inst->addModifier<DSModifiers>(DSModifiers(1, /*offset=*/32));

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeDSLoadB192(inst, builder, arch, /*hasVgprMsb=*/false);

    ASSERT_NE(result.first, nullptr);
    ASSERT_NE(result.last, nullptr);
    EXPECT_EQ(countByMnemonic("ds_load_b128"), 1);
    EXPECT_EQ(countByMnemonic("ds_load_b64"), 1);
    EXPECT_EQ(countByMnemonic("ds_load_b192"), 0);
    EXPECT_EQ(countInstructions(), 2);

    // Check that b128 dst is 4 regs starting at 0 with base offset
    auto* load1 = dyn_cast<StinkyInstruction>(result.first);
    ASSERT_NE(load1, nullptr);
    EXPECT_EQ(load1->getDestReg(0).reg.idx, 0);
    EXPECT_EQ(load1->getDestReg(0).reg.num, 4);
    auto* ds1 = load1->getModifier<DSModifiers>();
    ASSERT_NE(ds1, nullptr);
    EXPECT_EQ(ds1->offset, 32);

    // Check that b64 dst starts at idx+4 with offset+16
    auto* load2 = dyn_cast<StinkyInstruction>(result.last);
    ASSERT_NE(load2, nullptr);
    EXPECT_EQ(load2->getDestReg(0).reg.idx, 4);
    EXPECT_EQ(load2->getDestReg(0).reg.num, 2);
    auto* ds2 = load2->getModifier<DSModifiers>();
    ASSERT_NE(ds2, nullptr);
    EXPECT_EQ(ds2->offset, 48);  // 32 + 16
}

// ---------------------------------------------------------------------------
// legalizeDSStoreB192
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, DSStoreB192SplitsIntoB128AndB64) {
    StinkyInstruction* inst = createInst(GFX::ds_store_b192);
    inst->addSrcReg(StinkyRegister("v", 50, 1));   // addr
    inst->addSrcReg(StinkyRegister("v", 0, 6));    // data v[0:5]
    inst->addModifier<DSModifiers>(DSModifiers(1, /*offset=*/0));

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeDSStoreB192(inst, builder, arch, /*hasVgprMsb=*/false);

    ASSERT_NE(result.first, nullptr);
    ASSERT_NE(result.last, nullptr);
    EXPECT_EQ(countByMnemonic("ds_store_b128"), 1);
    EXPECT_EQ(countByMnemonic("ds_store_b64"), 1);
    EXPECT_EQ(countByMnemonic("ds_store_b192"), 0);

    auto* store1 = dyn_cast<StinkyInstruction>(result.first);
    ASSERT_NE(store1, nullptr);
    EXPECT_EQ(store1->getSrcReg(1).reg.idx, 0);
    EXPECT_EQ(store1->getSrcReg(1).reg.num, 4);

    auto* store2 = dyn_cast<StinkyInstruction>(result.last);
    ASSERT_NE(store2, nullptr);
    EXPECT_EQ(store2->getSrcReg(1).reg.idx, 4);
    EXPECT_EQ(store2->getSrcReg(1).reg.num, 2);
    auto* ds2 = store2->getModifier<DSModifiers>();
    ASSERT_NE(ds2, nullptr);
    EXPECT_EQ(ds2->offset, 16);
}

// ---------------------------------------------------------------------------
// legalizeDSStoreB256
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, DSStoreB256SplitsIntoTwoB128) {
    // ds_store_b256 is not registered as a GFX opcode on gfx1250; use ds_store_b128
    // as a surrogate — legalizeDSStoreB256 only requires 0 dests and 2 srcs.
    StinkyInstruction* inst = createInst(GFX::ds_store_b128);
    inst->addSrcReg(StinkyRegister("v", 60, 1));   // addr
    inst->addSrcReg(StinkyRegister("v", 0, 8));    // data v[0:7]
    inst->addModifier<DSModifiers>(DSModifiers(1, /*offset=*/0));

    AsmIRBuilder builder(*bb, arch);
    auto result = legalizeDSStoreB256(inst, builder, arch, /*hasVgprMsb=*/false);

    ASSERT_NE(result.first, nullptr);
    ASSERT_NE(result.last, nullptr);
    EXPECT_EQ(countByMnemonic("ds_store_b128"), 2);
    EXPECT_EQ(countByMnemonic("ds_store_b256"), 0);

    auto* store1 = dyn_cast<StinkyInstruction>(result.first);
    ASSERT_NE(store1, nullptr);
    EXPECT_EQ(store1->getSrcReg(1).reg.idx, 0);
    EXPECT_EQ(store1->getSrcReg(1).reg.num, 4);

    auto* store2 = dyn_cast<StinkyInstruction>(result.last);
    ASSERT_NE(store2, nullptr);
    EXPECT_EQ(store2->getSrcReg(1).reg.idx, 4);
    EXPECT_EQ(store2->getSrcReg(1).reg.num, 4);
    auto* ds2 = store2->getModifier<DSModifiers>();
    ASSERT_NE(ds2, nullptr);
    EXPECT_EQ(ds2->offset, 16);
}

// ---------------------------------------------------------------------------
// legalizeImplicitSpecialRegisters
// ---------------------------------------------------------------------------

TEST_F(LegalizationUtilsTest, ImplicitSpecialRegistersNullIsNoOp) {
    // Must not crash on nullptr
    legalizeImplicitSpecialRegisters(nullptr, 64);
}

TEST_F(LegalizationUtilsTest, ImplicitSCCReadAdded) {
    // s_add_i32 implicitly reads SCC
    StinkyInstruction* inst = createInst(GFX::s_add_i32);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(StinkyRegister("s", 1, 1));
    inst->addSrcReg(StinkyRegister("s", 2, 1));

    legalizeImplicitSpecialRegisters(inst, 64);

    bool hasSCCDst = false;
    for (const auto& r : inst->getDestRegs()) {
        if (r.reg.type == RegType::SCC) hasSCCDst = true;
    }

    // s_add_i32 writes SCC (IF_ImplicitWriteSCC)
    EXPECT_TRUE(hasSCCDst) << "s_add_i32 should get implicit SCC dst";
}

TEST_F(LegalizationUtilsTest, ImplicitExecReadAndWriteAdded) {
    // s_and_saveexec_b32 has IF_ImplicitReadEXEC and IF_ImplicitWriteEXEC
    StinkyInstruction* inst = createInst(GFX::s_and_saveexec_b32);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(StinkyRegister("s", 1, 1));

    int srcsBefore = static_cast<int>(inst->getSrcRegs().size());
    int dstsBefore = static_cast<int>(inst->getDestRegs().size());
    legalizeImplicitSpecialRegisters(inst, 32);

    bool hasExecSrc = false;
    for (const auto& r : inst->getSrcRegs())
        if (r.reg.type == RegType::EXEC || r.reg.type == RegType::EXEC_LO) hasExecSrc = true;

    bool hasExecDst = false;
    for (const auto& r : inst->getDestRegs())
        if (r.reg.type == RegType::EXEC || r.reg.type == RegType::EXEC_LO) hasExecDst = true;

    EXPECT_TRUE(hasExecSrc) << "s_and_saveexec_b32 should get implicit EXEC src";
    EXPECT_TRUE(hasExecDst) << "s_and_saveexec_b32 should get implicit EXEC dst";
    EXPECT_GT(static_cast<int>(inst->getSrcRegs().size()), srcsBefore);
    EXPECT_GT(static_cast<int>(inst->getDestRegs().size()), dstsBefore);
}

TEST_F(LegalizationUtilsTest, ImplicitRegistersNotAddedTwice) {
    StinkyInstruction* inst = createInst(GFX::s_and_saveexec_b32);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(StinkyRegister("s", 1, 1));

    legalizeImplicitSpecialRegisters(inst, 32);
    int srcsAfterFirst = static_cast<int>(inst->getSrcRegs().size());
    int dstsAfterFirst = static_cast<int>(inst->getDestRegs().size());

    legalizeImplicitSpecialRegisters(inst, 32);

    EXPECT_EQ(static_cast<int>(inst->getSrcRegs().size()), srcsAfterFirst)
        << "Should not add duplicate implicit src registers";
    EXPECT_EQ(static_cast<int>(inst->getDestRegs().size()), dstsAfterFirst)
        << "Should not add duplicate implicit dst registers";
}
