/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Unit tests for SwInstructionPrefetchRelDynamicPass (P1 stub).
 * ************************************************************************ */

#include <gtest/gtest.h>

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
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelDynamicPass.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

using namespace stinkytofu;
using stinkytofu::test::createVAddInBlock;
using stinkytofu::test::setFunctionArch;

namespace {
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
