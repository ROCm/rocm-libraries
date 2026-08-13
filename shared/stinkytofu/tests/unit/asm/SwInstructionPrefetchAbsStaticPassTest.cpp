// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for SwInstructionPrefetchAbsStaticPass (P1 abs static policy).

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsStaticPass.hpp"
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

/// Mirror sw_instruction_prefetch_rel_static.stir @at_p0_k0_wmma layout.
/// Pre-insert totalLayoutBytes = 32640 + 4 = 32644 > P(0). Static pass runs.
void buildAboveP0Kernel(BasicBlock* bb, GfxArchID arch) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, 32632);
    // v_add is at [0, 4); align pads to 32632; the 8-byte v_wmma then spans
    // [32632, 32640) so that the trailing v_add lands at [32640, 32644) and
    // totalLayoutBytes = 32644 > P(0) = 32640 (static pass runs).
    {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* wmma = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch));
        wmma->addDestReg(StinkyRegister("a", 0, 8));
        wmma->addSrcReg(StinkyRegister("v", 8, 8));
        wmma->addSrcReg(StinkyRegister("v", 16, 8));
        wmma->addSrcReg(StinkyRegister("a", 0, 8));
    }
    createVAddInBlock(bb, arch, 3, 4, 5);  // 4 bytes at [32640, 32644)
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

bool hasLabel(const BasicBlock& bb, std::string_view labelName) {
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        const IRBase* n = it.getNodePtr();
        if (n->getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
        if (inst.getUnifiedOpcode() != GFX::LABEL) continue;
        if (const LabelData* ld = inst.getModifier<LabelData>()) {
            if (ld->label == labelName) return true;
        }
    }
    return false;
}

bool hasSGetpc(const BasicBlock& bb) {
    return countInstructions(bb, "s_getpc_b64") > 0;
}

}  // namespace

class SwInstructionPrefetchAbsStaticPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        func = std::make_unique<Function>("sw_prefetch_abs_static_test");
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
        pm.addPass(createSwInstructionPrefetchAbsStaticPass(baseSgpr));
        return pm;
    }

    GfxArchID arch{};
    std::unique_ptr<Function> func;
    BasicBlock* bb{};
    GemmTileConfig gemmConfig{};
};

// ---------------------------------------------------------------------------
// Gate: kernel ≤ P(0) — no-op
// ---------------------------------------------------------------------------

TEST_F(SwInstructionPrefetchAbsStaticPassTest, SmallKernel_BelowP0_NoInsert) {
    for (int i = 0; i < 8; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    auto pm = makePm();
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_FALSE(hasLabel(*bb, kSwPrefetchAbsSiteLabel));
}

TEST_F(SwInstructionPrefetchAbsStaticPassTest, ExactP0End_NoInsert) {
    // totalLayoutBytes == 32640: <= P(0) gate fires, no insert.
    createVAddInBlock(bb, arch, 0, 1, 2);
    appendAlignDirective(bb, 32632);
    // 8-byte v_wmma: spans [32632, 32640). totalLayoutBytes = 32640.
    {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* wmma = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch));
        wmma->addDestReg(StinkyRegister("a", 0, 8));
        wmma->addSrcReg(StinkyRegister("v", 8, 8));
        wmma->addSrcReg(StinkyRegister("v", 16, 8));
        wmma->addSrcReg(StinkyRegister("a", 0, 8));
    }

    SwPrefetchRelPhase1Accum phase1;
    computeSwPrefetchRelPhase1Accum(*func, nullptr, phase1);
    ASSERT_EQ(phase1.totalLayoutBytes, kSwPrefetchFirstGlobalByte);

    auto pm = makePm();
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
}

// ---------------------------------------------------------------------------
// Gate: baseSgpr = -1 — no-op even when kernel is in range
// ---------------------------------------------------------------------------

TEST_F(SwInstructionPrefetchAbsStaticPassTest, NoBaseSgpr_NoInsert) {
    buildAboveP0Kernel(bb, arch);

    auto pm = makePm(/*baseSgpr=*/-1);
    pm.run(*func);

    EXPECT_EQ(countSPrefetchInst(*func), 0);
    EXPECT_FALSE(hasLabel(*bb, kSwPrefetchAbsSiteLabel));
}

// ---------------------------------------------------------------------------
// Main: kernel in (P(0), 64 KiB] with baseSgpr set
// ---------------------------------------------------------------------------

TEST_F(SwInstructionPrefetchAbsStaticPassTest, AboveP0_InsertsOnePrefetchAndLabels) {
    buildAboveP0Kernel(bb, arch);

    // Pre-check: no prefetch instructions yet.
    EXPECT_EQ(countSPrefetchInst(*func), 0);

    auto pm = makePm(64);
    pm.run(*func);

    // Exactly one s_prefetch_inst (N=1 since totalLayoutBytes just above P(0)).
    EXPECT_EQ(countSPrefetchInst(*func), 1);

    // Entry BB must have: site label, s_getpc_b64, s_prefetch_inst.
    EXPECT_TRUE(hasLabel(*bb, kSwPrefetchAbsSiteLabel));
    EXPECT_TRUE(hasSGetpc(*bb));
}

TEST_F(SwInstructionPrefetchAbsStaticPassTest, AboveP0_TargetLabelInserted) {
    buildAboveP0Kernel(bb, arch);

    auto pm = makePm(64);
    pm.run(*func);

    // label_SW_PrefetchAbs_0 must appear somewhere in the function.
    bool found = false;
    for (const BasicBlock& b : *func) {
        if (hasLabel(b, (std::string(kSwPrefetchAbsTargetLabelBase) + "0").c_str())) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "label_SW_PrefetchAbs_0 not found in any BB";
}

TEST_F(SwInstructionPrefetchAbsStaticPassTest, AboveP0_SiteBeforeGetpcBeforePrefetch) {
    buildAboveP0Kernel(bb, arch);

    auto pm = makePm(64);
    pm.run(*func);

    // Walk entry BB and verify order: siteLabel → s_getpc_b64 → s_add_i32 → s_add_u32 →
    // s_addc_u32 → s_prefetch_inst (at least once before any original code).
    enum class State {
        WantSiteLabel,
        WantGetpc,
        WantAddOff,
        WantAddLo,
        WantAddHi,
        WantPrefetch,
        Done
    };
    State s = State::WantSiteLabel;
    for (auto it = bb->begin(); it != bb->end(); ++it) {
        const IRBase* n = it.getNodePtr();
        if (n->getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction& inst = *cast<StinkyInstruction>(n);
        const char* m = inst.getHwInstDesc() ? inst.getHwInstDesc()->mnemonic : nullptr;

        switch (s) {
            case State::WantSiteLabel:
                if (inst.getUnifiedOpcode() == GFX::LABEL) {
                    if (const LabelData* ld = inst.getModifier<LabelData>()) {
                        if (ld->label == kSwPrefetchAbsSiteLabel) {
                            s = State::WantGetpc;
                            continue;
                        }
                    }
                }
                break;
            case State::WantGetpc:
                if (m && std::strcmp(m, "s_getpc_b64") == 0) {
                    s = State::WantAddOff;
                    continue;
                }
                break;
            case State::WantAddOff:
                if (m && std::strcmp(m, "s_add_i32") == 0) {
                    s = State::WantAddLo;
                    continue;
                }
                break;
            case State::WantAddLo:
                if (m && std::strcmp(m, "s_add_u32") == 0) {
                    s = State::WantAddHi;
                    continue;
                }
                break;
            case State::WantAddHi:
                if (m && std::strcmp(m, "s_addc_u32") == 0) {
                    s = State::WantPrefetch;
                    continue;
                }
                break;
            case State::WantPrefetch:
                if (m && std::strcmp(m, "s_prefetch_inst") == 0) {
                    s = State::Done;
                    continue;
                }
                break;
            case State::Done:
                break;
        }
    }
    EXPECT_EQ(s, State::Done) << "burst instruction sequence not found in entry BB";
}

// ---------------------------------------------------------------------------
// Debug output
// ---------------------------------------------------------------------------

TEST_F(SwInstructionPrefetchAbsStaticPassTest, DebugFile_ContainsExpectedStrings) {
    buildAboveP0Kernel(bb, arch);

    std::random_device rd;
    const std::filesystem::path outPath =
        std::filesystem::path(::testing::TempDir()) /
        ("st_sw_prefetch_abs_static_" + std::to_string(rd()) + ".txt");

    {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchAbsStaticPass(64, outPath.string()));
        pm.run(*func);
    }

    std::ifstream in(outPath);
    ASSERT_TRUE(in) << "expected debug file at " << outPath;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    std::error_code ec;
    std::filesystem::remove(outPath, ec);

    EXPECT_NE(text.find("SwInstructionPrefetchAbsStaticPass"), std::string::npos);
    EXPECT_NE(text.find("abs-static insert"), std::string::npos);
    EXPECT_NE(text.find("abs-static complete"), std::string::npos);
    EXPECT_NE(text.find("label_Do_SW_PrefetchAbs_entry"), std::string::npos);
    EXPECT_NE(text.find("label_SW_PrefetchAbs_0"), std::string::npos);
    EXPECT_NE(text.find("P(0)=32640"), std::string::npos);
}

TEST_F(SwInstructionPrefetchAbsStaticPassTest, DebugFile_BelowThreshold_NoOpMessage) {
    for (int i = 0; i < 4; ++i) createVAddInBlock(bb, arch, 0, 1, 2);

    std::random_device rd;
    const std::filesystem::path outPath =
        std::filesystem::path(::testing::TempDir()) /
        ("st_sw_prefetch_abs_static_noop_" + std::to_string(rd()) + ".txt");

    {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(gemmConfig);
        pm.addPass(createSwInstructionPrefetchAbsStaticPass(64, outPath.string()));
        pm.run(*func);
    }

    std::ifstream in(outPath);
    ASSERT_TRUE(in) << "expected debug file at " << outPath;
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    std::error_code ec;
    std::filesystem::remove(outPath, ec);

    EXPECT_NE(text.find("no-op"), std::string::npos);
    EXPECT_NE(text.find("32640"), std::string::npos);
}
