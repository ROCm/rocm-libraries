/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Unit tests for SwPrefetchInsertionPass (insertSwPrefetchLabels is exercised
 * only through the public pass API; this file does not include the .cpp).
 *
 * First SW threshold P(0) = 128*255 = 32640 bytes; small blocks stay below
 * P(0), so no mov+prefetch is inserted — we assert stability and debug dump
 * content.
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
#include "stinkytofu/transforms/asm/SwPrefetchInsertionPass.hpp"

using namespace stinkytofu;
using stinkytofu::test::createVAddInBlock;
using stinkytofu::test::setFunctionArch;

namespace {
int countStinkyInstructions(const BasicBlock& bb) {
    int c = 0;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        if (it.getNodePtr()->getType() == IRBase::IRType::StinkyTofu) ++c;
    }
    return c;
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

class SwPrefetchInsertionPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        func = std::make_unique<Function>("sw_prefetch_test");
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

// ---------------------------------------------------------------------------
// insertSwPrefetchLabels: block end < P(0) => no mov+prefetch IR
// ---------------------------------------------------------------------------

TEST_F(SwPrefetchInsertionPassTest, SmallBlock_BelowFirstThreshold_NoPrefetchInserted) {
    for (int i = 0; i < 8; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    const int before = countStinkyInstructions(*bb);
    EXPECT_EQ(before, 8);
    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(gemmConfig);
    pm.addPass(createSwPrefetchInsertionPass(std::string{}));
    pm.run(*func);

    EXPECT_EQ(countStinkyInstructions(*bb), before);
    EXPECT_EQ(countPrefetchInstPcRel(*bb), 0);
}

// ---------------------------------------------------------------------------
// Debug path: proposals section reports block below first threshold (32640)
// ---------------------------------------------------------------------------

TEST_F(SwPrefetchInsertionPassTest, DebugFile_ContainsBelowThresholdMessage) {
    createVAddInBlock(bb, arch, 0, 1, 2);

    std::random_device rd;
    const std::filesystem::path outPath = std::filesystem::path(::testing::TempDir()) /
                                          ("st_sw_prefetch_debug_" + std::to_string(rd()) + ".txt");

    {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwPrefetchInsertionPass(outPath.string()));
        pm.run(*func);
    }

    std::ifstream in(outPath);
    ASSERT_TRUE(in) << "expected debug file at " << outPath;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    std::error_code ec;
    std::filesystem::remove(outPath, ec);

    EXPECT_NE(text.find("[SwPrefetchInsertionPass]"), std::string::npos);
    EXPECT_NE(text.find("SW prefetch proposals"), std::string::npos);
    // insertSwPrefetchLabels + debugPrintSwPrefetchProposals: small block < P(0)
    EXPECT_NE(text.find("first threshold"), std::string::npos);
    EXPECT_NE(text.find("32640"), std::string::npos);
}
