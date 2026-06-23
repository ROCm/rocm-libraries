/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Unit tests for SwInstructionPrefetchAbsDynamicPass (P2 abs dynamic policy).
 *
 * Phase P2 currently ships a STUB: the dynamic pass no-ops for every kernel
 * size and emits a debug log when totalLayoutBytes > 64 KiB. These tests pin
 * the stub contract so the later real implementation (per-k targets + CFG
 * sites) replaces it deliberately, not by accident.
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
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsDynamicPass.hpp"
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

/// Build a kernel whose totalLayoutBytes lands just above `alignTo` (one
/// trailing 4-byte v_add). Use `alignTo` to target a size regime:
///   - <= 32640        : CP-only regime
///   - (32640, 65536]  : static regime (defer to static pass)
///   - > 65536         : dynamic regime (stub no-op + log)
void buildKernelAboveAlign(BasicBlock* bb, GfxArchID arch, int64_t alignTo) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, alignTo);
    createVAddInBlock(bb, arch, 3, 4, 5);  // 4 bytes at [alignTo, alignTo+4)
}

int countInstructions(const BasicBlock& bb, const char* mnemonic) {
    int c = 0;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const IRBase* n = it.getNodePtr();
        if (n->getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
        const char* m = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;
        if (m && std::strcmp(m, mnemonic) == 0) ++c;
    }
    return c;
}

int countSPrefetchInst(const Function& func) {
    int c = 0;
    for (const BasicBlock& bb : func) c += countInstructions(bb, "s_prefetch_inst");
    return c;
}

int countSGetpc(const Function& func) {
    int c = 0;
    for (const BasicBlock& bb : func) c += countInstructions(bb, "s_getpc_b64");
    return c;
}

/// True if any emitted label name contains "PrefetchAbs" (site or target).
/// The stub must never emit one.
bool hasAnyAbsPrefetchLabel(const Function& func) {
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            const IRBase* n = it.getNodePtr();
            if (n->getType() != IRBase::IRType::StinkyTofu) continue;
            const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
            if (inst.getUnifiedOpcode() != GFX::LABEL) continue;
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                if (ld->label.find("PrefetchAbs") != std::string::npos) return true;
            }
        }
    }
    return false;
}

}  // namespace

class SwInstructionPrefetchAbsDynamicPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        func = std::make_unique<Function>("sw_prefetch_abs_dynamic_test");
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

    PassManager makePm(int baseSgpr = 64) {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchAbsDynamicPass(baseSgpr));
        return pm;
    }

    std::string runWithDebug(int baseSgpr = 64) {
        std::random_device rd;
        const std::filesystem::path outPath =
            std::filesystem::path(::testing::TempDir()) /
            ("st_sw_prefetch_abs_dynamic_" + std::to_string(rd()) + ".txt");
        {
            PassManager pm;
            registerAllAnalyses(pm.getAnalysisManager());
            pm.setGemmTileConfig(gemmConfig);
            pm.addPass(createSwInstructionPrefetchAbsDynamicPass(baseSgpr, outPath.string()));
            pm.run(*func);
        }
        std::ifstream in(outPath);
        std::stringstream buf;
        buf << in.rdbuf();
        std::error_code ec;
        std::filesystem::remove(outPath, ec);
        return buf.str();
    }

    GfxArchID arch{};
    std::unique_ptr<Function> func;
    BasicBlock* bb{};
    GemmTileConfig gemmConfig{};
};

// ---------------------------------------------------------------------------
// Stub contract: the dynamic pass never mutates IR (no prefetch / getpc) for
// any size regime in P2.
// ---------------------------------------------------------------------------

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, BelowP0_NoInsert) {
    for (int i = 0; i < 8; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    auto pm = makePm();
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, StaticRegime_DefersToStatic_NoInsert) {
    // (32640, 65536] is the static pass's regime; the dynamic pass no-ops.
    buildKernelAboveAlign(bb, arch, 40000);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchFirstGlobalByte);
    ASSERT_LE(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm();
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_AboveIcache_StubNoInsert) {
    // > 65536 is the dynamic regime. P2 stub still inserts nothing.
    buildKernelAboveAlign(bb, arch, 70000);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm();
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
}

// Boundary guard: totalLayoutBytes == 65536 belongs to the STATIC regime
// (gate is `<= 65536`). If the dynamic gate ever regresses from `<=` to `<`,
// the stub would wrongly fire "dynamic pass not implemented" at the boundary.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, ExactIcacheBoundary_DefersToStatic) {
    buildKernelAboveAlign(bb, arch, 65532);  // align 65532 + 4-byte v_add -> 65536

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_EQ(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    const std::string text = runWithDebug();

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
    EXPECT_NE(text.find("no-op"), std::string::npos);
    EXPECT_EQ(text.find("dynamic pass not implemented"), std::string::npos)
        << "dynamic stub must defer to static at exactly 65536, not fire";
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, NoBaseSgpr_NoInsert) {
    buildKernelAboveAlign(bb, arch, 70000);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm(/*baseSgpr=*/-1);
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
}

// ---------------------------------------------------------------------------
// Debug output
// ---------------------------------------------------------------------------

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DebugFile_AboveIcache_NotImplementedMessage) {
    buildKernelAboveAlign(bb, arch, 70000);

    // baseSgpr=-1 mirrors the real pipeline (SwInstructionPrefetchAbsBaseSgpr
    // defaults to -1 until P5 Tensile wiring). The "not implemented" log must
    // still appear — it is the stub's one observable deliverable for >64K.
    const std::string text = runWithDebug(/*baseSgpr=*/-1);
    EXPECT_NE(text.find("SwInstructionPrefetchAbsDynamicPass"), std::string::npos);
    EXPECT_NE(text.find("dynamic pass not implemented"), std::string::npos);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DebugFile_BelowP0_NoOpMessage) {
    for (int i = 0; i < 4; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    const std::string text = runWithDebug();
    EXPECT_NE(text.find("no-op"), std::string::npos);
    EXPECT_NE(text.find("32640"), std::string::npos);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DebugFile_StaticRegime_DefersMessage) {
    buildKernelAboveAlign(bb, arch, 40000);

    const std::string text = runWithDebug();
    EXPECT_NE(text.find("no-op"), std::string::npos);
    EXPECT_NE(text.find("SwInstructionPrefetchAbsStaticPass"), std::string::npos);
}
