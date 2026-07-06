/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Unit tests for SwInstructionPrefetchAbsDynamicPass (abs CFG-target policy).
 *
 * Contract (D0/D1): no-op only when totalLayoutBytes <= P(0)=32640 (whole kernel
 * in the CP window). For totalLayoutBytes > 32640 the read-only CFG-target
 * DETECTOR runs (debug dump, no IR mutation). The Variant-1 ladder is EMITTED
 * only when totalLayoutBytes > 65536 AND a reserved baseSgpr is present AND the
 * GSU1 beta-split anchors + the label_MultiGemmEnd site exist AND the dispatch is
 * supported (sgprGSU + sgprBeta defined, not Stream-K). Most kernels here omit
 * those anchors so emission bails (no IR mutation); the DynamicRegime_ThreeArm*
 * tests build the fully emittable shape to pin the positive Variant-1 ladder and
 * the Stream-K bail-out.
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

/// Emit a `.set <symbol>, <value>` directive so collectAsmSetSymbolValues() sees the
/// symbol as defined (the emission guard keys on symbol presence, not the value).
void appendSetDirective(BasicBlock* bb, const std::string& symbol, const std::string& value) {
    AsmDirective* d = IRBase::createIR<AsmDirective>();
    d->kind = AsmDirectiveKind::SET;
    d->name = ".set";
    d->symbol = symbol;
    d->value = value;
    bb->appendIR(d);
}

/// Append a LABEL instruction (via the same builder path the pass uses) to \p bb.
void appendLabel(BasicBlock* bb, GfxArchID arch, const std::string& name) {
    AsmIRBuilder builder(*bb, arch);
    builder.createLabel(name);
}

int countLabel(const Function& func, const std::string& name) {
    int c = 0;
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            const IRBase* n = it.getNodePtr();
            if (n->getType() != IRBase::IRType::StinkyTofu) continue;
            const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
            if (inst.getUnifiedOpcode() != GFX::LABEL) continue;
            if (const LabelData* ld = inst.getModifier<LabelData>())
                if (ld->label == name) ++c;
        }
    }
    return c;
}

/// Build a kernel whose totalLayoutBytes lands just above `alignTo` (one
/// trailing 4-byte v_add). Use `alignTo` to target a size regime:
///   - <= 32640        : CP-only regime (pass no-ops)
///   - (32640, 65536]  : static regime (dynamic = detector-only, no emit)
///   - > 65536         : dynamic regime (emits only if GSU1 anchors + MGE site + baseSgpr present)
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
/// These synthetic kernels lack the GW/MultiGemmEnd anchors, so the pass must never emit one.
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
// No-mutation cases: for these synthetic kernels (no label_GW_B0_GSU1 /
// label_MultiGemmEnd anchors), the dynamic pass never inserts prefetch / getpc,
// regardless of size regime — emission bails on the missing dispatch.
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

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_AboveIcache_NoAnchors_NoInsert) {
    // > 65536: emission is reached (baseSgpr set), but this synthetic kernel has no
    // label_GW_B0_GSU1 / label_MultiGemmEnd, so emitVariant1Ladder bails → no IR mutation.
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

// Boundary guard: totalLayoutBytes == 65536 is post-CP, so the DETECTOR runs,
// but emission is strictly `> 65536` → static owns the (32640, 65536] regime,
// so the ladder must NOT be emitted at exactly 65536.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, ExactIcacheBoundary_DetectorOnly_NoEmit) {
    buildKernelAboveAlign(bb, arch, 65532);  // align 65532 + 4-byte v_add -> 65536

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_EQ(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    const std::string text = runWithDebug();

    // No emission at the boundary (static regime), regardless of detector running.
    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
    // Detector runs (post-CP), and the legacy "not implemented" stub message is gone.
    EXPECT_NE(text.find("D0 CFG-target detector"), std::string::npos);
    EXPECT_EQ(text.find("dynamic pass not implemented"), std::string::npos);
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

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DebugFile_AboveIcache_BaseSgprUnset_DetectorOnly) {
    buildKernelAboveAlign(bb, arch, 70000);

    // baseSgpr=-1 → no reserved SGPR triple, so D1 emission is skipped and the
    // pass runs the detector only. The legacy "not implemented" stub log is gone.
    const std::string text = runWithDebug(/*baseSgpr=*/-1);
    EXPECT_NE(text.find("SwInstructionPrefetchAbsDynamicPass"), std::string::npos);
    EXPECT_NE(text.find("D0 CFG-target detector"), std::string::npos);
    EXPECT_NE(text.find("detector-only"), std::string::npos);
    EXPECT_EQ(text.find("dynamic pass not implemented"), std::string::npos);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DebugFile_BelowP0_NoOpMessage) {
    for (int i = 0; i < 4; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    const std::string text = runWithDebug();
    EXPECT_NE(text.find("no-op"), std::string::npos);
    EXPECT_NE(text.find("32640"), std::string::npos);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DebugFile_StaticRegime_DetectorOnly_NoEmit) {
    // (32640, 65536]: detector runs (post-CP), but emission is gated `> 65536`,
    // so the dynamic pass never emits here — static owns this regime.
    buildKernelAboveAlign(bb, arch, 40000);

    const std::string text = runWithDebug();
    EXPECT_NE(text.find("D0 CFG-target detector"), std::string::npos);
    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
}

// ---------------------------------------------------------------------------
// Positive emission: dynamic regime (> 65536) with all D1 preconditions met —
// the reserved baseSgpr, the label_MultiGemmEnd site (with a following anchor),
// the 3-arm GSU/beta anchors (GW_B0_MB / GW_B0_GSU1 / GW_B1_GSU1), sgprGSU +
// sgprBeta defined, and NOT Stream-K. The pass must emit the Variant-1 ladder.
// ---------------------------------------------------------------------------

/// Build a > 65536-byte kernel that satisfies every D1 emission precondition and
/// produces the full 3-arm ladder (Case A=MB, B=GSU1 fall-through, C=B1_GSU1).
void buildThreeArmEmittableKernel(BasicBlock* bb, GfxArchID arch) {
    // Site: label_MultiGemmEnd followed by a real insn so the anchor (node after
    // the label) exists — the ladder is inserted before it, i.e. right after MGE.
    appendLabel(bb, arch, "label_MultiGemmEnd");
    createVAddInBlock(bb, arch, 0, 1, 2);

    // GSU/beta dispatch symbols must be defined (GSU0 / no-beta kernels bail), and
    // NOT Stream-K (no sgprSrdWS / sgprSynchronizer).
    appendSetDirective(bb, "sgprGSU", "54");
    appendSetDirective(bb, "sgprBeta", "40");

    // Push total layout past the 64 KiB I-cache split so the dynamic regime owns it.
    appendAlignDirective(bb, 70000);

    // 3-arm anchors, each followed by a real insn so buildLabelOffsets resolves them.
    appendLabel(bb, arch, "label_GW_B0_MB");  // Case A (GSU > 1)
    createVAddInBlock(bb, arch, 3, 4, 5);
    appendLabel(bb, arch, "label_GW_B0_GSU1");  // Case B (fall-through)
    createVAddInBlock(bb, arch, 6, 7, 8);
    appendLabel(bb, arch, "label_GW_B1_GSU1");  // Case C (beta split)
    createVAddInBlock(bb, arch, 9, 10, 11);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_ThreeArmLadder_Emits) {
    buildThreeArmEmittableKernel(bb, arch);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm(/*baseSgpr=*/64);
    pm.run(*func);

    // Three bursts (A, B, C), each = 1 getpc + 6 prefetch hints (fixed N=6).
    EXPECT_EQ(countSGetpc(*func), 3);
    EXPECT_EQ(countSPrefetchInst(*func), 18);

    // The full 3-arm ladder scaffolding must be present.
    EXPECT_TRUE(hasAnyAbsPrefetchLabel(*func));
    EXPECT_EQ(countLabel(*func, "label_Do_SW_PrefetchAbs_sel"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_caseA"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_caseC"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_end"), 1);
}

TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_StreamK_BailsNoEmit) {
    // Same emittable 3-arm shape, but Stream-K (sgprSynchronizer defined) → the
    // supported-dispatch guard must bail with no IR mutation.
    buildThreeArmEmittableKernel(bb, arch);
    appendSetDirective(bb, "sgprSynchronizer", "60");

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm(/*baseSgpr=*/64);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
}
