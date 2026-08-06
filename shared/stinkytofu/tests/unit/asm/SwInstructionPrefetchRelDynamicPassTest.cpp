// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for SwInstructionPrefetchRelDynamicPass (Phase 1 + Phase 2).

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelDynamicPass.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

using namespace stinkytofu;
using stinkytofu::test::createVAddInBlock;
using stinkytofu::test::setFunctionArch;

namespace {
void appendAlignDirective(BasicBlock* bb, int64_t alignBytes) {
    AsmDirective* d = IRBase::createIR<AsmDirective>();
    d->kind = AsmDirectiveKind::ALIGN;
    d->name = ".align";
    d->symbol = std::to_string(alignBytes);
    d->intValue = alignBytes;
    bb->appendIR(d);
}

StinkyInstruction* createVWmmaBf16InBlock(BasicBlock* bb, GfxArchID arch) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch));
    inst->addDestReg(StinkyRegister("a", 0, 8));
    inst->addSrcReg(StinkyRegister("v", 8, 8));
    inst->addSrcReg(StinkyRegister("v", 16, 8));
    inst->addSrcReg(StinkyRegister("a", 0, 8));
    return inst;
}

/// Mirror sw_instruction_prefetch_rel_static.stir @at_p0_k0_wmma layout: P(0) inside 8 B WMMA.
/// Pre-insert total is exactly P(0)=32640 (pass no-op threshold).
void buildAtP0WmmaExactEndKernel(BasicBlock* bb, GfxArchID arch) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, 32632);
    createVWmmaBf16InBlock(bb, arch);
}

/// Same anchor as @at_p0_k0_wmma but totalLayoutBytes > P(0) so Phase 2 runs.
void buildAboveP0WmmaKernel(BasicBlock* bb, GfxArchID arch) {
    buildAtP0WmmaExactEndKernel(bb, arch);
    createVAddInBlock(bb, arch, 3, 4, 5);
}

int parsePlanInsertCount(const std::string& text) {
    const std::string key = "PLAN_INSERT=";
    const auto pos = text.find(key);
    if (pos == std::string::npos) return -1;
    return std::atoi(text.c_str() + pos + key.size());
}

int countPrefetchInstPcRel(const BasicBlock& bb) {
    int c = 0;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const IRBase* n = it.getNodePtr();
        if (n->getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
        const char* m = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;
        if (m && std::strcmp(m, "s_prefetch_inst_pc_rel") == 0) ++c;
    }
    return c;
}
}  // namespace

class SwInstructionPrefetchRelDynamicPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        func = std::make_unique<Function>("sw_prefetch_dynamic_test");
        bb = func->createBasicBlock("entry");
        setFunctionArch(*func, arch);

        gemmConfig.arch = {12, 5, 0};
        gemmConfig.NumWaves = 1;
        gemmConfig.TileA0 = 16;
        gemmConfig.TileB0 = 16;
        gemmConfig.TileM0 = 16;
        gemmConfig.NumGRA = 1;
        gemmConfig.NumGRB = 1;
        gemmConfig.NumGRM = 1;
    }

    GfxArchID arch{};
    std::unique_ptr<Function> func;
    BasicBlock* bb{};
    GemmTileConfig gemmConfig{};
};

TEST_F(SwInstructionPrefetchRelDynamicPassTest, SmallBlock_BelowThreshold_NoPrefetchInserted) {
    for (int i = 0; i < 8; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    pm.addPass(createSwInstructionPrefetchRelDynamicPass(std::string{}));
    pm.run(*func);

    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest, DebugFile_ContainsBelowThresholdMessage) {
    createVAddInBlock(bb, arch, 0, 1, 2);

    std::random_device rd;
    const std::filesystem::path outPath =
        std::filesystem::path(::testing::TempDir()) /
        ("st_sw_prefetch_dynamic_debug_" + std::to_string(rd()) + ".txt");

    {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchRelDynamicPass(outPath.string()));
        pm.run(*func);
    }

    std::ifstream in(outPath);
    ASSERT_TRUE(in) << "expected debug file at " << outPath;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    std::error_code ec;
    std::filesystem::remove(outPath, ec);

    EXPECT_NE(text.find("[SwInstructionPrefetchRelDynamicPass]"), std::string::npos);
    EXPECT_NE(text.find("planned insert sites"), std::string::npos);
    EXPECT_NE(text.find("Phase 1 accumulate"), std::string::npos);
    EXPECT_NE(text.find("layoutGlobal="), std::string::npos);
    EXPECT_NE(text.find("blockLocalBytesPostCp="), std::string::npos);
    EXPECT_NE(text.find("accumExit="), std::string::npos);
    EXPECT_NE(text.find("accumBeforeGlobal="), std::string::npos);
    EXPECT_NE(text.find("accumAfterGlobal="), std::string::npos);
    EXPECT_NE(text.find("gateBefore="), std::string::npos);
    EXPECT_NE(text.find("no-op"), std::string::npos);
    EXPECT_NE(text.find("32640"), std::string::npos);
    EXPECT_NE(text.find("P(0)"), std::string::npos);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest, Phase1Accum_LayoutGlobalPerInstruction) {
    for (int i = 0; i < 4; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    SwPrefetchRelPhase1Accum phase1;
    // No .set directives in this kernel; nullptr avoids linking collectAsmSetSymbolValues.
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);

    EXPECT_EQ(phase1.layoutStart[bb], 0);
    EXPECT_EQ(phase1.layoutGlobal.size(), 4u);
    EXPECT_EQ(phase1.accumByte[bb], 0);
    EXPECT_EQ(phase1.accumExit[bb], phase1.blockLocalBytesPostCp[bb]);
    EXPECT_EQ(phase1.blockLocalBytesPostCp[bb], 0);
    EXPECT_LT(phase1.totalLayoutBytes, kSwPrefetchFirstGlobalByte);

    int64_t expectedOffset = 0;
    for (auto it = bb->begin(); it != bb->end(); ++it) {
        if (it.getNodePtr()->getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction& inst = *cast<StinkyInstruction>(it.getNodePtr());
        if (inst.getUnifiedOpcode() == GFX::PHI || inst.getUnifiedOpcode() == GFX::LABEL) continue;
        auto found = phase1.layoutGlobal.find(const_cast<StinkyInstruction*>(&inst));
        ASSERT_NE(found, phase1.layoutGlobal.end());
        EXPECT_EQ(found->second, expectedOffset);
        expectedOffset += getEffectiveBaseSizeInBytes(inst);
    }
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest, ExactP0End_NoPrefetchInserted) {
    buildAtP0WmmaExactEndKernel(bb, arch);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    EXPECT_EQ(phase1.totalLayoutBytes, kSwPrefetchFirstGlobalByte);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    pm.addPass(createSwInstructionPrefetchRelDynamicPass(std::string{}));
    pm.run(*func);

    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest, AboveP0Wmma_InsertsOnePrefetch) {
    buildAboveP0WmmaKernel(bb, arch);

    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    pm.addPass(createSwInstructionPrefetchRelDynamicPass(std::string{}));
    pm.run(*func);

    EXPECT_EQ(countPrefetchInstPcRel(*bb), 1);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest, AboveP0Wmma_PlanInsertMatchesIrCount) {
    buildAboveP0WmmaKernel(bb, arch);

    std::random_device rd;
    const std::filesystem::path outPath =
        std::filesystem::path(::testing::TempDir()) /
        ("st_sw_prefetch_dynamic_plan_" + std::to_string(rd()) + ".txt");

    {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchRelDynamicPass(outPath.string()));
        pm.run(*func);
    }

    std::ifstream in(outPath);
    ASSERT_TRUE(in) << "expected debug file at " << outPath;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    std::error_code ec;
    std::filesystem::remove(outPath, ec);

    const int planInsert = parsePlanInsertCount(text);
    ASSERT_GE(planInsert, 0) << "missing PLAN_INSERT= in debug output";
    EXPECT_EQ(countPrefetchInstPcRel(*bb), planInsert);
    EXPECT_EQ(planInsert, 1);
    EXPECT_NE(text.find("Phase 2 complete"), std::string::npos);
    EXPECT_NE(text.find("totalPrefetchInserted=1"), std::string::npos);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest,
       InsertSwPrefetchLabelsDynamic_BelowThreshold_ReturnsZero) {
    for (int i = 0; i < 8; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);

    const int inserted = insertSwPrefetchLabelsDynamic(
        *bb, phase1.layoutStart.at(bb), phase1.accumByte.at(bb), 0, arch, nullptr, nullptr, true);
    EXPECT_EQ(inserted, 0);
    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);
}

// A small post-CP block reached via a large alignment gap: the only post-CP byte is at layout
// 50000. The layoutStart-based anchor estimate returns P(0)=32640 (assumes contiguous fill), so its
// 4 KiB grid lands at ...,49024,53120 and misses [50000,50004). Phase 1 must record the *actual*
// first post-CP byte (50000) so the per-BB anchor grid still inserts here.
TEST_F(SwInstructionPrefetchRelDynamicPassTest,
       PerBbAnchor_SmallPostCpBlockViaAlignmentGap_Inserts) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, 50000);
    createVAddInBlock(bb, arch, 3, 4, 5);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    pm.addPass(createSwInstructionPrefetchRelDynamicPass(std::string{}));
    pm.run(*func);

    EXPECT_EQ(countPrefetchInstPcRel(*bb), 1);
}

// A long fully-post-CP block (reached via alignment, so bbEntryAccum=0 diverges from the layout
// offset 35616) must get interior prefetches every 4 KiB (anchor, anchor+4096, ...), not one at the
// block start and one coalesced to the block end. This pins the layout-coordinate gating fix: the
// per-BB grid steps exactly 4096 B inside the block.
TEST_F(SwInstructionPrefetchRelDynamicPassTest, InteriorPerBbGridSteps4KiB) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, 35616);
    for (int i = 0; i < 1400; ++i) createVAddInBlock(bb, arch, 0, 1, 2);  // 4 B each => ~5600 B

    const std::filesystem::path outPath =
        std::filesystem::temp_directory_path() / "st_perbb_interior_step.txt";
    {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchRelDynamicPass(outPath.string()));
        pm.run(*func);
    }
    std::ifstream in(outPath);
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();

    // Anchor at the first post-CP byte (35616) and an interior point exactly +4096, both inserted
    // before a real instruction (never coalesced to the block end).
    EXPECT_NE(text.find("localK=0 P=35616"), std::string::npos);
    EXPECT_NE(text.find("localK=1 P=39712"), std::string::npos);
    EXPECT_EQ(text.find("bb_end_append action=PLAN_INSERT"), std::string::npos)
        << "interior grid point was wrongly coalesced to the block tail";
    EXPECT_EQ(countPrefetchInstPcRel(*bb), 2);
}

// Phase 1 records the real first post-CP layout byte (honoring the alignment gap), not the
// contiguous layoutStart estimate of 32640.
TEST_F(SwInstructionPrefetchRelDynamicPassTest,
       Phase1Accum_FirstPostCpLayoutByteHonorsAlignmentGap) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, 50000);
    createVAddInBlock(bb, arch, 3, 4, 5);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    EXPECT_EQ(phase1.firstPostCpLayoutByte.at(bb), 50000);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest,
       InsertSwPrefetchLabelsDynamicPerBbAnchor_InvalidAnchor_ReturnsZero) {
    for (int i = 0; i < 8; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);

    const int inserted = insertSwPrefetchLabelsDynamicPerBbAnchor(
        *bb, phase1.layoutStart.at(bb), phase1.accumByte.at(bb), kSwPrefetchNoPerBbGridAnchor, 0,
        arch, nullptr, nullptr, true);
    EXPECT_EQ(inserted, 0);
    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);
}

TEST_F(SwInstructionPrefetchRelDynamicPassTest,
       InsertSwPrefetchLabelsDynamicPerBbAnchor_AboveP0Wmma_InsertsPrefetch) {
    buildAboveP0WmmaKernel(bb, arch);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);

    const int64_t anchor = phase1.firstPostCpLayoutByte.at(bb);
    ASSERT_NE(anchor, kSwPrefetchNoPerBbGridAnchor);

    const int inserted = insertSwPrefetchLabelsDynamicPerBbAnchor(*bb, phase1.layoutStart.at(bb),
                                                                  phase1.accumByte.at(bb), anchor,
                                                                  0, arch, nullptr, nullptr, true);
    EXPECT_GE(inserted, 1);
    EXPECT_EQ(countPrefetchInstPcRel(*bb), inserted);
}
