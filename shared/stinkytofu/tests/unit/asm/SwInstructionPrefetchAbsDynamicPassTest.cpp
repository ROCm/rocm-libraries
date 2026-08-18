// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for SwInstructionPrefetchAbsDynamicPass (abs CFG-target policy).
//
// Contract: no-op only when totalLayoutBytes <= P(0)=32640 (whole kernel
// in the CP window). For totalLayoutBytes > 32640 the read-only CFG-target
// analysis runs (debug dump, no IR mutation). The predicated ladder is EMITTED
// only when totalLayoutBytes > 65536 AND a reserved baseSgpr is present AND the
// GSU1 beta-split anchors + the label_MultiGemmEnd site exist AND the dispatch is
// supported (sgprGSU + sgprBeta defined, not Stream-K). Most kernels here omit
// those anchors so emission bails (no IR mutation); the DynamicRegime_ThreeArm*
// tests build the fully emittable shape to pin the positive predicated ladder and
// the Stream-K bail-out.

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

/// Byte offset of the first real instruction following LABEL \p name (== the label's offset,
/// since a label is 0 bytes), read from \p p.layoutGlobal. Returns -1 if not found.
int64_t labelByteOffset(const Function& func, const SwPrefetchRelPhase1Accum& p,
                        const std::string& name) {
    bool found = false;
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            const IRBase* n = it.getNodePtr();
            if (n->getType() != IRBase::IRType::StinkyTofu) continue;
            const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
            const auto op = inst.getUnifiedOpcode();
            if (op == GFX::LABEL) {
                const LabelData* ld = inst.getModifier<LabelData>();
                if (ld && ld->label == name) found = true;
                continue;
            }
            if (op == GFX::PHI) continue;
            if (found) {
                auto f = p.layoutGlobal.find(const_cast<StinkyInstruction*>(&inst));
                return f == p.layoutGlobal.end() ? -1 : f->second;
            }
        }
    }
    return -1;
}

/// True if the first `s_getpc_b64` in `func` occurs BEFORE label `name` in program order.
/// Used to assert the CP cover is emitted at kernel ENTRY (before label_MultiGemmEnd),
/// not at the MultiGemmEnd site.
bool firstGetpcBeforeLabel(const Function& func, const std::string& name) {
    int getpcPos = -1, labelPos = -1, pos = 0;
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it, ++pos) {
            const IRBase* n = it.getNodePtr();
            if (n->getType() != IRBase::IRType::StinkyTofu) continue;
            const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
            if (inst.getUnifiedOpcode() == GFX::LABEL) {
                if (labelPos < 0) {
                    const LabelData* ld = inst.getModifier<LabelData>();
                    if (ld && ld->label == name) labelPos = pos;
                }
                continue;
            }
            const char* m = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;
            if (getpcPos < 0 && m && std::strcmp(m, "s_getpc_b64") == 0) getpcPos = pos;
        }
    }
    return getpcPos >= 0 && labelPos >= 0 && getpcPos < labelPos;
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

/// Collect, in program order, the mnemonics of the burst-relevant instructions (getpc /
/// add-family / prefetch). Used to pin the INTRA-burst instruction ORDER, which the
/// count/label/offset assertions elsewhere cannot observe (they are order-invariant).
std::vector<std::string> burstMnemonicSequence(const Function& func) {
    static const char* kBurst[] = {"s_getpc_b64", "s_add_i32", "s_add_u32", "s_addc_u32",
                                   "s_prefetch_inst"};
    std::vector<std::string> seq;
    for (const BasicBlock& bb : func) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            const IRBase* n = it.getNodePtr();
            if (n->getType() != IRBase::IRType::StinkyTofu) continue;
            const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
            const char* m = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;
            if (!m) continue;
            for (const char* b : kBurst) {
                if (std::strcmp(m, b) == 0) {
                    seq.emplace_back(m);
                    break;
                }
            }
        }
    }
    return seq;
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

    PassManager makePm(int baseSgpr = 64, bool cpCover = false) {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchAbsDynamicPass(baseSgpr, {}, cpCover));
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

// Boundary guard: totalLayoutBytes == 65536 is post-CP, so the analysis runs,
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

    // baseSgpr=-1 → no reserved SGPR triple, so ladder emission is skipped and the
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
// Positive emission: dynamic regime (> 65536) with all emission preconditions met —
// the reserved baseSgpr, the label_MultiGemmEnd site (with a following anchor),
// the 3-arm GSU/beta anchors (GW_B0_MB / GW_B0_GSU1 / GW_B1_GSU1), sgprGSU +
// sgprBeta defined, and NOT Stream-K. The pass must emit the predicated ladder.
// ---------------------------------------------------------------------------

/// Build a > 65536-byte kernel that satisfies every emission precondition and produces the full
/// 3-arm ladder (Case A=MB, B=GSU1 fall-through, C=B1_GSU1). \p tailAlignTo controls the
/// CP-extend boundary: when > 0 a `label_TailLoopBeginL` is anchored at that byte offset (the
/// primary boundary), so coverN is deterministic; <= 0 omits it (boundary falls back to the
/// shallowest GW_* > P(0)).
static void buildThreeArmKernelTail(BasicBlock* bb, GfxArchID arch, int64_t tailAlignTo) {
    // Site: label_MultiGemmEnd followed by a real insn so the anchor (node after
    // the label) exists — the ladder is inserted before it, i.e. right after MGE.
    appendLabel(bb, arch, "label_MultiGemmEnd");
    createVAddInBlock(bb, arch, 0, 1, 2);

    // GSU/beta dispatch symbols must be defined (GSU0 / no-beta kernels bail), and
    // NOT Stream-K (no sgprSrdWS / sgprSynchronizer).
    appendSetDirective(bb, "sgprGSU", "54");
    appendSetDirective(bb, "sgprBeta", "40");

    // Once-through fast-path boundary (primary): label_TailLoopBeginL at a controlled offset.
    if (tailAlignTo > 0) {
        appendAlignDirective(bb, tailAlignTo);
        appendLabel(bb, arch, "label_TailLoopBeginL");
        createVAddInBlock(bb, arch, 15, 16, 17);
    }

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

/// Default emittable kernel: tail boundary at byte 40000 → coverN = 2 (armN stays 6). This
/// reproduces the earlier fixed-N=2 cover width, so the cover-on counts below are 2 + 3*6.
static void buildThreeArmEmittableKernel(BasicBlock* bb, GfxArchID arch) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/40000);
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
    // CP cover OFF by default: no CP-boundary target label, ladder counts unchanged.
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 0);
}

// Cover is emitted at KERNEL ENTRY (before label_MultiGemmEnd), not at the MGE site: the
// first s_getpc_b64 (the cover's) precedes label_MultiGemmEnd; the 3 arm getpc follow it.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CpCover_EmittedAtEntry) {
    buildThreeArmEmittableKernel(bb, arch);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 4);  // 1 cover (entry) + 3 arms (MGE)
    EXPECT_TRUE(firstGetpcBeforeLabel(*func, "label_MultiGemmEnd"));  // cover is at entry
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

// EMPTY entry-stub BB: real Tensile kernels have an empty CFG-stub as func.getEntryBlock()
// with code in later BBs. The cover must still emit (appended to the empty entry BB), NOT be
// skipped. (Regression guard: a prior emitBurst/anchor-based version silently skipped here.)
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CpCover_EmptyEntryStub_StillEmits) {
    BasicBlock* codeBB = func->createBasicBlock("code");  // 2nd BB; entry `bb` stays empty
    buildThreeArmEmittableKernel(codeBB, arch);
    ASSERT_EQ(bb->begin(), bb->end());  // entry stub is empty

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    // Cover emitted into the (formerly empty) entry stub + 3-arm ladder in codeBB.
    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 20);
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
    // The cover landed in the entry stub (now non-empty).
    EXPECT_NE(bb->begin(), bb->end());
}

// SGPR-safety gate: the cover is emitted only when label_MultiGemmEnd is present (the abs base
// triple is reserved across the prolog iff MGE/PreLoop is on). A >65536 kernel WITHOUT MGE must NOT
// emit the cover even with cpCover=true (no anchoring proxy → unsafe to use the triple at entry).
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CpCover_NoMGE_NoEmit) {
    buildKernelAboveAlign(bb, arch, 70000);  // >65536, but no label_MultiGemmEnd / no GW anchors

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
}

// baseSgpr<0 with the cover ENABLED must still no-op: the run()-level gate (baseSgpr>=0) guards
// emitVariant1Ladder before the cover is ever reached, so the cover cannot bypass the SGPR gate.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, NoBaseSgpr_WithCpCover_NoInsert) {
    buildThreeArmEmittableKernel(bb, arch);

    auto pm = makePm(/*baseSgpr=*/-1, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 0);
}

// CP-range-extend cover ENABLED: the same 3-arm kernel additionally gets ONE unconditional
// near-boundary burst (N=2) prepended before the sel ladder, plus the post-insertion-anchored
// label_SW_PrefetchAbs_CpBoundary. Ladder itself is unchanged (still 3 arms).
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_ThreeArmLadder_WithCpCover_Emits) {
    buildThreeArmEmittableKernel(bb, arch);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    // 3 arms (3 getpc + 18 prefetch) + CP cover (1 getpc + 2 prefetch, N=2) = 4 getpc / 20
    // prefetch.
    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 20);

    // 3-arm ladder scaffolding intact + the new CP-boundary target label present exactly once.
    EXPECT_EQ(countLabel(*func, "label_Do_SW_PrefetchAbs_sel"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_caseA"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_caseC"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_end"), 1);
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

/// Build a > 65536-byte kernel with the label_MultiGemmEnd site + a real insn <= P(0), but NO
/// GSU1 anchors and NO sgprGSU/sgprBeta — so the 3-arm ladder is unsupported. Used to prove the
/// CP cover is DECOUPLED (still emits) from the ladder's GSU/beta guard.
static void buildCoverOnlyKernel(BasicBlock* bb, GfxArchID arch) {
    appendLabel(bb, arch, "label_MultiGemmEnd");
    createVAddInBlock(bb, arch, 0, 1, 2);  // site anchor + a real insn at offset ~0 (<= P0)
    // Boundary (primary): tail at 40000 → coverN = 2. No GSU/beta and no GW anchors, so the
    // 3-arm ladder is unsupported — this isolates the DECOUPLED CP cover.
    appendAlignDirective(bb, 40000);
    appendLabel(bb, arch, "label_TailLoopBeginL");
    createVAddInBlock(bb, arch, 6, 7, 8);
    appendAlignDirective(bb, 70000);  // push total past the 64 KiB split
    createVAddInBlock(bb, arch, 3, 4, 5);
}

// DECOUPLE: no sgprGSU/sgprBeta and no GW anchors → the 3-arm ladder bails, but the CP cover
// (cpCover=true) must STILL emit its unconditional near-boundary burst + CpBoundary label.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CpCoverOnly_NoLadder_Emits) {
    buildCoverOnlyKernel(bb, arch);

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_GT(phase1.totalLayoutBytes, kSwPrefetchAbsStaticIcacheSizeBytes);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    // CP cover only: 1 getpc + 2 prefetch (N=2). No ladder scaffolding.
    EXPECT_EQ(countSGetpc(*func), 1);
    EXPECT_EQ(countSPrefetchInst(*func), 2);
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
    EXPECT_EQ(countLabel(*func, "label_Do_SW_PrefetchAbs_sel"), 0);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_caseA"), 0);
    EXPECT_EQ(countLabel(*func, "label_Do_PF_caseC"), 0);
}

// ORDER guard: the refactor moved burst emission from create(desc, anchor) to
// insertIR(at, createIR(...)) with a FIXED by-value iterator. Inserting successive nodes before a
// fixed `at` must PRESERVE creation order. Counts/labels/byte-offsets are order-invariant and stay
// green under a reversal, so pin the EXACT intra-burst mnemonic sequence on the single-burst
// cover-only kernel: getpc -> add_i32 -> add_u32 -> addc_u32 -> N(=2) * prefetch.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest,
       DynamicRegime_CpCover_BurstInstructionOrder_Preserved) {
    buildCoverOnlyKernel(bb, arch);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    const std::vector<std::string> seq = burstMnemonicSequence(*func);
    const std::vector<std::string> expected = {"s_getpc_b64", "s_add_i32",       "s_add_u32",
                                               "s_addc_u32",  "s_prefetch_inst", "s_prefetch_inst"};
    EXPECT_EQ(seq, expected);
}

// DECOUPLE + default-off: same cover-only kernel WITHOUT the cover toggle emits nothing
// (the ladder is unsupported and the cover is disabled).
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CpCoverOff_NoLadder_NoEmit) {
    buildCoverOnlyKernel(bb, arch);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/false);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
}

/// 3-arm emittable kernel with an extra real instruction anchored at byte 32000 (<= P(0)=32640),
/// so the CP-boundary label has a meaningful boundary instruction to anchor at (unlike the
/// tiny fixtures whose only <=P0 insn is at offset ~0).
static void buildThreeArmKernelWithBoundaryInsn(BasicBlock* bb, GfxArchID arch) {
    appendLabel(bb, arch, "label_MultiGemmEnd");
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendSetDirective(bb, "sgprGSU", "54");
    appendSetDirective(bb, "sgprBeta", "40");
    appendAlignDirective(bb,
                         32000);  // a real insn lands at byte 32000 (<= P(0)): CpBoundary anchor
    createVAddInBlock(bb, arch, 3, 4, 5);
    // Boundary (primary): tail at 40000 → coverN = 2 (the CpBoundary label still anchors at the
    // <= P0 insn at byte 32000, NOT at the boundary that only sizes the width).
    appendAlignDirective(bb, 40000);
    appendLabel(bb, arch, "label_TailLoopBeginL");
    createVAddInBlock(bb, arch, 18, 19, 20);
    appendAlignDirective(bb, 70000);  // push total past the 64 KiB split
    appendLabel(bb, arch, "label_GW_B0_MB");
    createVAddInBlock(bb, arch, 6, 7, 8);
    appendLabel(bb, arch, "label_GW_B0_GSU1");
    createVAddInBlock(bb, arch, 9, 10, 11);
    appendLabel(bb, arch, "label_GW_B1_GSU1");
    createVAddInBlock(bb, arch, 12, 13, 14);
}

// Byte-accurate anchoring: the CP-boundary target label must land at post-insertion offset
// <= P(0)=32640 (no gap past CP) and, here, at the boundary insn (byte 32000) — NOT at offset ~0.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CpCover_LabelAnchoredAtBoundary) {
    buildThreeArmKernelWithBoundaryInsn(bb, arch);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    // Re-accumulate the MUTATED kernel and read the label's final byte offset.
    SwPrefetchRelPhase1Accum p2;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, p2);
    const int64_t off = labelByteOffset(*func, p2, "label_SW_PrefetchAbs_CpBoundary");

    ASSERT_GE(off, 0);                           // label present + resolvable
    EXPECT_LE(off, kSwPrefetchFirstGlobalByte);  // <= P(0)=32640: no gap past CP
    EXPECT_EQ(off, 32000);                       // exact: anchored at the boundary insn, not ~0
    // Cover emitted; ladder intact.
    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 20);
}

// Boundary fixture, cover OFF: no CP-boundary label and standard 3-arm counts (negative twin of
// the anchoring test).
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_BoundaryKernel_CoverOff_NoLabel) {
    buildThreeArmKernelWithBoundaryInsn(bb, arch);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/false);
    pm.run(*func);

    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 0);
    EXPECT_EQ(countSGetpc(*func), 3);
    EXPECT_EQ(countSPrefetchInst(*func), 18);
}

// Stream-K must bail BOTH ladder and cover even when the cover toggle is on (its base triple
// is not reserved to MGE).
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_StreamK_WithCpCover_BailsNoEmit) {
    buildThreeArmEmittableKernel(bb, arch);
    appendSetDirective(bb, "sgprSynchronizer", "60");

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 0);
    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_FALSE(hasAnyAbsPrefetchLabel(*func));
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

// ---------------------------------------------------------------------------
// Dynamic coverN / armN regimes (cover ON). coverN is sized from the
// label_TailLoopBeginL byte offset: coverN = clamp(floor((tail+INSERT_UB-P0)/4096)+1, 0, 4),
// armN = min(6, 8-effectiveCoverN). Each test pins the (coverN, armN) split via the tail offset.
//   prefetch total = coverN (cover, if >0) + 3*armN (3 arms).   getpc = (coverN>0 ? 1 : 0) + 3.
// ---------------------------------------------------------------------------

// tail=31000 (<= P0=32640): coverN=0 → cover SKIPPED entirely (no burst, no CpBoundary label);
// arms keep the full armN=6. This is the "fast path already CP-resident" no-op path.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverN0_TailInCpWindow) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/31000);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 3);          // 3 arms, no cover
    EXPECT_EQ(countSPrefetchInst(*func), 18);  // 3 arms * armN(6); cover=0
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 0);
    EXPECT_EQ(countLabel(*func, "label_Do_SW_PrefetchAbs_sel"), 1);  // ladder still emits
}

// tail=33000 (gap = 33000+320-32640 = 680 -> coverN=1, armN=6): cover 1 + 3 arms * 6.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverN1_ArmN6) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/33000);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 4);  // 1 cover + 3 arms
    EXPECT_EQ(countSPrefetchInst(*func), 1 + 18);
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

// tail=40000 (gap 7680 -> coverN=2, armN=6): cover 2 + 3 arms * 6 = 20.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverN2_ArmN6) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/40000);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 2 + 18);
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

// tail=50976 (like f8f8s; rawCover 5 -> CLAMP to coverN=4, so armN shrinks to 4):
// cover 4 + 3 arms * 4 = 16. Pins the armFloor=4 clamp binding.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverN4_ClampBinds_ArmN4) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/50976);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 4 + 12);  // coverN=4, armN=4 (clamp)
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

// tail=44000 (gap 11680 -> coverN=3, armN=5): the ONLY split where armN is strictly between the
// floor (4) and the cap (6). cover 3 + 3 arms * 5 = 18.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverN3_ArmN5) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/44000);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 3 + 15);  // coverN=3, armN=5
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

// Fallback #2: NO label_TailLoopBegin* (tailAlignTo=0) → boundary falls back to the
// shallowest label_GW_* > P0 (~70000) → rawCover 10 -> clamp coverN=4, armN=4.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverBoundary_GwFallback) {
    buildThreeArmKernelTail(bb, arch, /*tailAlignTo=*/0);  // omit tail label

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countLabel(*func, "label_TailLoopBeginL"), 0);  // no primary boundary present
    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 4 + 12);  // GW boundary -> coverN=4, armN=4
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}

/// 3-arm emittable kernel whose fast-path boundary is label_OptNLL_End (fallback #1) — NO
/// label_TailLoopBegin* present, OptNLL_End anchored at \p optNllAlignTo.
static void buildThreeArmKernelOptNllBoundary(BasicBlock* bb, GfxArchID arch,
                                              int64_t optNllAlignTo) {
    appendLabel(bb, arch, "label_MultiGemmEnd");
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendSetDirective(bb, "sgprGSU", "54");
    appendSetDirective(bb, "sgprBeta", "40");
    appendAlignDirective(bb, optNllAlignTo);
    appendLabel(bb, arch, "label_OptNLL_End");
    createVAddInBlock(bb, arch, 15, 16, 17);
    appendAlignDirective(bb, 70000);
    appendLabel(bb, arch, "label_GW_B0_MB");
    createVAddInBlock(bb, arch, 3, 4, 5);
    appendLabel(bb, arch, "label_GW_B0_GSU1");
    createVAddInBlock(bb, arch, 6, 7, 8);
    appendLabel(bb, arch, "label_GW_B1_GSU1");
    createVAddInBlock(bb, arch, 9, 10, 11);
}

// Fallback #1: NO label_TailLoopBegin*, but label_OptNLL_End present at 44000 → boundary =
// OptNLL_End (beats GW_*) → gap 11680 -> coverN=3, armN=5.
TEST_F(SwInstructionPrefetchAbsDynamicPassTest, DynamicRegime_CoverBoundary_OptNllEndFallback) {
    buildThreeArmKernelOptNllBoundary(bb, arch, /*optNllAlignTo=*/44000);

    auto pm = makePm(/*baseSgpr=*/64, /*cpCover=*/true);
    pm.run(*func);

    EXPECT_EQ(countLabel(*func, "label_TailLoopBeginL"), 0);
    EXPECT_EQ(countSGetpc(*func), 4);
    EXPECT_EQ(countSPrefetchInst(*func), 3 + 15);  // OptNLL_End boundary -> coverN=3, armN=5
    EXPECT_EQ(countLabel(*func, "label_SW_PrefetchAbs_CpBoundary"), 1);
}
