// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Covers Gfx1250HazardPass gate and rule behaviour across the four
// (RequiresXCntForVolatileVMEM, EnableXnackReplay) cap combinations:
//
//   requiresXCnt  enableReplay  pass runs  RuleAtomic  RuleOthers (SMEM/FLAT)
//   false         false         no         —           —
//   true          false         yes        yes         no
//   false         true          yes        yes         yes
//   true          true          yes        yes         yes
//
// Rule naming matches XcntDrainReason in Gfx1250HazardPass.cpp:
//   RuleAtomic  = AtomicRule4a4a  : s_wait_xcnt before first atomic after non-atomic VMEM
//   RuleOthers  = SmemRule33 / FlatRule22 : source-clobber protection for SMEM and FLAT groups
//
// The one case that aborts rather than inserting a drain (multi-DWORD SMEM
// self-overlap) is also covered here because FileCheck cannot express an abort.
// All other functional cases live in tests/filecheck/gfx1250_xnack_hazard_test.stir.

#include <gtest/gtest.h>

#include <array>
#include <string>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"
#include "stinkytofu/transforms/asm/Gfx1250ReplayGroup.hpp"

using namespace stinkytofu;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count s_wait_xcnt instructions inserted by the pass across all blocks.
static size_t countXcntDrains(const Function& func) {
    size_t n = 0;
    for (const BasicBlock& bb : func)
        for (const IRBase& ir : bb)
            if (ir.getType() == IRBase::IRType::StinkyTofu)
                if (cast<StinkyInstruction>(&ir)->getUnifiedOpcode() == GFX::s_wait_xcnt) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class Gfx1250HazardPassTest : public ::testing::Test {
   protected:
    std::array<int, 3> arch{12, 5, 0};

    /// Parse STIR, run the pass with the given caps, and return the function
    /// (still owned by \p converter).
    Function* parseAndRun(const std::string& irString, StinkyIRConverter& converter,
                          AsmCapsConfig caps) {
        Function* func = converter.convertToFunction(irString);
        if (!func) return nullptr;

        GemmTileConfig config;
        config.arch = arch;
        PassContext passCtx;
        passCtx.setGemmTileConfig(config);
        passCtx.setAsmCapsConfig(caps);
        AnalysisManager am;
        createGfx1250HazardPass()->run(*func, passCtx, am);
        return func;
    }

    void runPass(Function& func, AsmCapsConfig caps) {
        GemmTileConfig config;
        config.arch = arch;
        PassContext passCtx;
        passCtx.setGemmTileConfig(config);
        passCtx.setAsmCapsConfig(caps);
        AnalysisManager am;
        createGfx1250HazardPass()->run(func, passCtx, am);
    }
};

// ---------------------------------------------------------------------------
// Gate tests  (does the pass run at all?)
// ---------------------------------------------------------------------------

// Both caps false → pass is a no-op, no drains inserted.
TEST_F(Gfx1250HazardPassTest, GateBothFalse_NoOp) {
    // SMEM source clobber would normally fire RuleOthers (SmemRule3) under
    // enableXnackReplay; with both caps false the pass must not run at all.
    std::string ir = R"(
st.func @gate_both_false() {
^entry:
  s4 = "st.s_load_b32"(s[0:1])
  s0 = "st.s_load_b32"(s[2:3])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;  // both false
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_EQ(countXcntDrains(*func), 0u);
}

// requiresXCntForVolatileVMEM=true, enableXnackReplay=false → pass runs (RuleAtomic).
TEST_F(Gfx1250HazardPassTest, GateRequiresXCnt_PassRuns) {
    // buffer_load → global_atomic triggers RuleAtomic (AtomicRule4a).
    std::string ir = R"(
st.func @gate_requires_xcnt() {
^entry:
  v9 = "st.buffer_load_b32"(v0)
  v3 = "st.global_atomic_inc_u32"(v1, v2, s[14:15])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.requiresXCntForVolatileVMEM = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_GT(countXcntDrains(*func), 0u);
}

// requiresXCntForVolatileVMEM=false, enableXnackReplay=true → pass runs (RuleOthers).
TEST_F(Gfx1250HazardPassTest, GateEnableXnackReplay_PassRuns) {
    // SMEM source clobber triggers RuleOthers (SmemRule3) when enableXnackReplay is set.
    std::string ir = R"(
st.func @gate_xnack_replay() {
^entry:
  s4 = "st.s_load_b32"(s[0:1])
  s0 = "st.s_load_b32"(s[2:3])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.enableXnackReplay = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_GT(countXcntDrains(*func), 0u);
}

// ---------------------------------------------------------------------------
// RuleAtomic / RuleOthers gating tests
// ---------------------------------------------------------------------------

// requiresXCntForVolatileVMEM=true, enableXnackReplay=false:
// RuleAtomic fires, RuleOthers does not.
TEST_F(Gfx1250HazardPassTest, RequiresXCntOnly_RuleAtomicFires_RuleOthersSkipped) {
    // buffer_load (non-atomic) → global_atomic: RuleAtomic inserts one drain.
    // SMEM source clobber is also present but RuleOthers must NOT fire
    // (no enableXnackReplay). The VMEM→SMEM type switch also clears state.
    std::string ir = R"(
st.func @requires_xcnt_only() {
^entry:
  v9 = "st.buffer_load_b32"(v0)
  v3 = "st.global_atomic_inc_u32"(v1, v2, s[14:15])
  s4 = "st.s_load_b32"(s[0:1])
  s0 = "st.s_load_b32"(s[2:3])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.requiresXCntForVolatileVMEM = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    // Exactly one drain: RuleAtomic before the atomic.
    EXPECT_EQ(countXcntDrains(*func), 1u);
}

// requiresXCntForVolatileVMEM=false, enableXnackReplay=true:
// Both RuleAtomic and RuleOthers fire.
TEST_F(Gfx1250HazardPassTest, EnableXnackReplayOnly_RuleAtomicAndRuleOthersFire) {
    // RuleOthers (SmemRule3): SMEM source clobber → one drain.
    // RuleAtomic (AtomicRule4a): buffer_load → global_atomic → one drain.
    // The SMEM→VMEM type switch clears state between the two groups.
    std::string ir = R"(
st.func @xnack_replay_only() {
^entry:
  s4 = "st.s_load_b32"(s[0:1])
  s0 = "st.s_load_b32"(s[2:3])
  v9 = "st.buffer_load_b32"(v0)
  v3 = "st.global_atomic_inc_u32"(v1, v2, s[14:15])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.enableXnackReplay = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_EQ(countXcntDrains(*func), 2u);
}

// requiresXCntForVolatileVMEM=true, enableXnackReplay=true:
// Same result as enableXnackReplay-only — RuleAtomic and RuleOthers both fire.
TEST_F(Gfx1250HazardPassTest, BothTrue_RuleAtomicAndRuleOthersFire) {
    std::string ir = R"(
st.func @both_true() {
^entry:
  s4 = "st.s_load_b32"(s[0:1])
  s0 = "st.s_load_b32"(s[2:3])
  v9 = "st.buffer_load_b32"(v0)
  v3 = "st.global_atomic_inc_u32"(v1, v2, s[14:15])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.requiresXCntForVolatileVMEM = true;
    caps.enableXnackReplay = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_EQ(countXcntDrains(*func), 2u);
}

// RuleAtomic is absent when there are no atomics, even with requiresXCntForVolatileVMEM=true.
TEST_F(Gfx1250HazardPassTest, RequiresXCntOnly_NoAtomics_NoDrains) {
    std::string ir = R"(
st.func @requires_xcnt_no_atomics() {
^entry:
  v9 = "st.buffer_load_b32"(v0)
  v10 = "st.buffer_load_b32"(v1)
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.requiresXCntForVolatileVMEM = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_EQ(countXcntDrains(*func), 0u);
}

// RuleOthers (FlatRule2): FLAT source-clobber is only repaired under enableXnackReplay.
TEST_F(Gfx1250HazardPassTest, RuleOthers_FlatSourceClobber_RequiresXCntOnly_NoDrain) {
    std::string ir = R"(
st.func @flat_source_clobber_xcnt_only() {
^entry:
  v9 = "st.flat_load_b32"(v[0:1], s[16:17])
  v0 = "st.flat_load_b32"(v[2:3], s[16:17])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.requiresXCntForVolatileVMEM = true;
    // enableXnackReplay = false → FlatRule2 must NOT fire
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_EQ(countXcntDrains(*func), 0u);
}

TEST_F(Gfx1250HazardPassTest, RuleOthers_FlatSourceClobber_EnableXnackReplay_DrainInserted) {
    std::string ir = R"(
st.func @flat_source_clobber_replay() {
^entry:
  v9 = "st.flat_load_b32"(v[0:1], s[16:17])
  v0 = "st.flat_load_b32"(v[2:3], s[16:17])
}
)";
    StinkyIRConverter converter(arch);
    AsmCapsConfig caps;
    caps.enableXnackReplay = true;
    Function* func = parseAndRun(ir, converter, caps);
    ASSERT_NE(func, nullptr);
    EXPECT_EQ(countXcntDrains(*func), 1u);
}

// ---------------------------------------------------------------------------
// Abort test (debug only): multi-DWORD SMEM self-overlap under RuleOthers
// ---------------------------------------------------------------------------

// A release build compiles the assert away and only prints the error.
#ifndef NDEBUG
TEST_F(Gfx1250HazardPassTest, MultiDwordSmemSelfOverlapAborts) {
    std::string irString = R"(
st.func @smem_self_overlap() {
^entry:
  s[0:1] = "st.s_load_b64"(s[0:1])
}
)";
    // enableXnackReplay=true → RuleOthers (SmemRule3) runs → abort on unrepairable self-overlap.
    {
        StinkyIRConverter converter(arch);
        Function* func = converter.convertToFunction(irString);
        ASSERT_NE(func, nullptr);
        AsmCapsConfig caps;
        caps.enableXnackReplay = true;
        EXPECT_DEATH(runPass(*func, caps), "overwrites one of its own source registers");
    }
    // requiresXCntForVolatileVMEM=true only → RuleOthers is skipped, no abort.
    {
        StinkyIRConverter converter(arch);
        Function* func = converter.convertToFunction(irString);
        ASSERT_NE(func, nullptr);
        AsmCapsConfig caps;
        caps.requiresXCntForVolatileVMEM = true;
        runPass(*func, caps);  // must not abort
    }
}
#endif

// ---------------------------------------------------------------------------
// Replay-mode guard: the pass models single-group replay only
// ---------------------------------------------------------------------------

struct ModeWrite {
    uint16_t id;
    uint16_t offset;
    uint16_t size;
    int value;
};

static void prependSetreg(Function& func, GfxArchID archId, const ModeWrite& w) {
    BasicBlock& bb = *func.begin();
    StinkyInstruction* first = nullptr;
    for (IRBase& ir : bb) {
        auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst != nullptr && !isPseudoInst(inst)) {
            first = inst;
            break;
        }
    }
    ASSERT_NE(first, nullptr);
    AsmIRBuilder builder(bb, archId);
    StinkyInstruction* setreg =
        builder.create(getMCIDByUOp(GFX::s_setreg_IMM32_b32, archId), first);
    setreg->addDestReg(StinkyRegister::Hwreg(w.id, w.offset, w.size));
    setreg->addSrcReg(StinkyRegister(w.value));
}

TEST_F(Gfx1250HazardPassTest, MultiGroupReplayModeWriteAborts) {
    const std::string irString = R"(
st.func @replay_mode() {
^entry:
  s[8:9] = "st.s_load_b64"(s[0:1])
  v0 = "st.v_add_f32"(v1, v2)
}
)";
    const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
    AsmCapsConfig caps;
    caps.enableXnackReplay = true;

    for (const ModeWrite& w :
         {ModeWrite{HwReg::ID_MODE, 25, 1, 1}, ModeWrite{HwReg::ID_MODE, 24, 2, 0b10}}) {
        StinkyIRConverter converter(arch);
        Function* func = converter.convertToFunction(irString);
        ASSERT_NE(func, nullptr);
        prependSetreg(*func, archId, w);
        EXPECT_DEATH(runPass(*func, caps), "MODE.REPLAY_MODE");
    }

    for (const ModeWrite& w :
         {ModeWrite{HwReg::ID_MODE, 25, 1, 0}, ModeWrite{HwReg::ID_MODE, 24, 2, 0b01},
          ModeWrite{HwReg::ID_MODE, 0, 4, 0xf}, ModeWrite{HwReg::ID_SCHED_MODE_GFX12, 0, 2, 2}}) {
        StinkyIRConverter converter(arch);
        Function* func = converter.convertToFunction(irString);
        ASSERT_NE(func, nullptr);
        prependSetreg(*func, archId, w);
        runPass(*func, caps);  // single-group still holds; must not abort
    }
}

TEST(Gfx1250ReplayGroupTest, TextModeWriteDetection) {
    const char* const modeWrites[] = {
        "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1",
        "s_setreg_IMM32_b32 hwreg(1, 25, 1), 1",
        "s_setreg_b32 hwreg(1), s4",
    };
    for (const char* text : modeWrites) {
        EXPECT_TRUE(gfx1250::textMayWriteMode(text)) << text;
    }

    const char* const otherRegAccesses[] = {
        "s_setreg_IMM32_b32 hwreg(26, 0, 2), 2",
        "s_setreg_IMM32_b32 hwreg(12, 0, 2), 2",
        "s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)",
    };
    for (const char* text : otherRegAccesses) {
        EXPECT_FALSE(gfx1250::textMayWriteMode(text)) << text;
    }
}

}  // namespace
