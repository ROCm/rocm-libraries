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

#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// These tests exercise the pass as it ships today: kClusterBarrierDrainGateEnabled
// defaults to TRUE, i.e. the GATED scheme (kRule4SignalBeforeWaitEnabled defaults
// to false but is ignored while the drain gate is on). Expectations below describe
// that default gated scheme -- priming signals wrapped in a LoopCounterL gate,
// Rule 4's asymmetric wait/signal drain gates (s_cmp_le_i32), and Rule 6 disabled.
// The ungated / legacy signal-before-wait layouts are selected by flipping those
// constexprs and are not reachable from a unit test.

namespace {

// The literal split-barrier ids the pass keys off of (see InsertClusterBarrierPass.cpp).
constexpr int kClusterBarrierId = -3;    // s_barrier_{signal,wait} -3  (cluster scope)
constexpr int kWorkgroupBarrierId = -1;  // s_barrier_{signal,wait} -1  (workgroup scope)

// Anchor names / symbols the pass keys off of.
constexpr const char* kGSU1LabelName = "label_GSU_1";           // Rule 1 anchor
constexpr const char* kOpenLoopLLabelName = "label_openLoopL";  // Rule 3 anchor
constexpr const char* kLoopEndLLabelName = "label_LoopEndL";    // Rule 6 fallback anchor
constexpr const char* kTailLoopMarker = "/* Tail Loop */";  // Rule 5 anchor (TEXTBLOCK substring)
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";

class InsertClusterBarrierPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        func = std::make_unique<Function>("cluster_barrier_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    void TearDown() override {
        func.reset();
        bb = nullptr;
    }

    /// Create an s_barrier_wait with the given literal split-barrier id in `b`.
    StinkyInstruction* createBarrierWaitIn(BasicBlock* b, int id) {
        AsmIRBuilder builder(*b, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_wait, arch));
        inst->addSrcReg(StinkyRegister(id));
        return inst;
    }

    /// Create an s_barrier_wait with the given literal split-barrier id (in `bb`).
    StinkyInstruction* createBarrierWait(int id) {
        return createBarrierWaitIn(bb, id);
    }

    /// Create an s_barrier_signal with the given literal split-barrier id in `b`.
    StinkyInstruction* createBarrierSignalIn(BasicBlock* b, int id) {
        AsmIRBuilder builder(*b, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_signal, arch));
        inst->addSrcReg(StinkyRegister(id));
        return inst;
    }

    /// Create an s_barrier_signal with the given literal split-barrier id (in `bb`).
    StinkyInstruction* createBarrierSignal(int id) {
        return createBarrierSignalIn(bb, id);
    }

    /// Create an unconditional `s_branch <label>`. After CFGBuilder, branches
    /// terminate a basic block so Rule 4's backward scan does not cross them.
    StinkyInstruction* createBranch(const std::string& label) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
        inst->addSrcReg(StinkyRegister(label));
        return inst;
    }

    StinkyInstruction* createBranchIn(BasicBlock* b, const std::string& label) {
        AsmIRBuilder builder(*b, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
        inst->addSrcReg(StinkyRegister(label));
        return inst;
    }

    /// Create `s_cbranch_scc1 <label>` (reads SCC set by a preceding compare).
    StinkyInstruction* createCBranchScc1(const std::string& label) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cbranch_scc1, arch));
        inst->addSrcReg(StinkyRegister(label));
        return inst;
    }

    /// Create a LABEL pseudo carrying LabelData{name} (matches the pass's
    /// `isLabelNamed` anchor checks, e.g. `label_GSU_1`).
    StinkyInstruction* createLabel(const std::string& name) {
        AsmIRBuilder builder(*bb, arch);
        return builder.createLabel(name);
    }

    StinkyInstruction* createLabelIn(BasicBlock* b, const std::string& name) {
        AsmIRBuilder builder(*b, arch);
        return builder.createLabel(name);
    }

    /// Append a TEXTBLOCK AsmDirective whose `value` is `text` (Rule 5 anchors
    /// on the `Tail Loop` substring inside such a directive).
    AsmDirective* createTextblock(const std::string& text) {
        AsmDirective* d = IRBase::createIR<AsmDirective>();
        d->kind = AsmDirectiveKind::TEXTBLOCK;
        d->value = text;
        bb->appendIR(d);
        return d;
    }

    /// Create `s_cmp_eq_u32 s[sgprLoopCounterL], imm` (SCC-writing loop-exit
    /// compare). Rule 4's `findLiveSccCmpUpstream` keys off exactly this shape
    /// to detect a SIA-hoisted live LCL compare.
    StinkyInstruction* createLoopCounterLCmpEq(int imm) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));
        StinkyRegister lcl(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
        lcl.setSymbolicName(kLoopCounterLSymbol);
        inst->addSrcReg(lcl);
        inst->addSrcReg(StinkyRegister(imm));
        return inst;
    }

    StinkyInstruction* createLoopCounterLDecrement(int imm = 1) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_sub_u32, arch));
        StinkyRegister lcl(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
        lcl.setSymbolicName(kLoopCounterLSymbol);
        inst->addSrcReg(lcl);
        inst->addSrcReg(lcl);
        inst->addSrcReg(StinkyRegister(imm));
        return inst;
    }

    std::vector<int> collectRule4DrainGateImms() const {
        std::vector<int> drainGateImms;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() != GFX::s_cmp_le_i32) continue;
            const auto& srcs = inst->getSrcRegs();
            if (srcs.empty() || srcs[0].getSymbolicName() != kLoopCounterLSymbol) continue;
            if (srcs.size() < 2 || srcs[1].dataType != StinkyRegister::Type::LiteralInt) continue;
            drainGateImms.push_back(static_cast<int>(srcs[1].getLiteralInt()));
        }
        return drainGateImms;
    }

    /// Run the pass on `func` (whole-kernel IR, as in Gfx1250Backend).
    void runPass(int pgrValue = 1) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        auto pass = createInsertClusterBarrierPass(pgrValue);
        pass->run(*func, ctx, am);
    }

    /// Count instructions in `bb` matching `s_barrier_signal`/`s_barrier_wait`
    /// (selected by `wantSignal`) whose first source is the literal `id`.
    int countBarrier(bool wantSignal, int id) const {
        int count = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (wantSignal ? !isBarrierSignal(*inst) : !isBarrierWait(*inst)) continue;
            const auto& srcs = inst->getSrcRegs();
            if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
                srcs[0].getLiteralInt() == id) {
                ++count;
            }
        }
        return count;
    }

    /// Same as countBarrier but over every basic block in `func`.
    int countBarrierInFunc(bool wantSignal, int id) const {
        int count = 0;
        for (const BasicBlock& b : *func) {
            for (const IRBase& ir : b) {
                if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
                const auto* inst = cast<StinkyInstruction>(&ir);
                if (wantSignal ? !isBarrierSignal(*inst) : !isBarrierWait(*inst)) continue;
                const auto& srcs = inst->getSrcRegs();
                if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
                    srcs[0].getLiteralInt() == id) {
                    ++count;
                }
            }
        }
        return count;
    }

    int countTensorLoads() const {
        int count = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (isTensorLoad(*cast<StinkyInstruction>(&ir))) ++count;
        }
        return count;
    }

    /// 0-based position of `target` among StinkyTofu instructions in `bb`, or -1.
    int posOf(const StinkyInstruction* target) const {
        int pos = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (cast<StinkyInstruction>(&ir) == target) return pos;
            ++pos;
        }
        return -1;
    }

    /// Count `s_cmp_eq_{u32,i32}` whose first source is the symbolic register
    /// `name` (used to detect Rule 4's re-emitted "restore" loop-counter cmp).
    int countCmpEqWithSymbol(const std::string& name) const {
        int count = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            const auto uOp = inst->getUnifiedOpcode();
            if (uOp != GFX::s_cmp_eq_u32 && uOp != GFX::s_cmp_eq_i32) continue;
            const auto& srcs = inst->getSrcRegs();
            if (!srcs.empty() && srcs[0].getSymbolicName() == name) ++count;
        }
        return count;
    }

    /// Position of the first `s_barrier_{signal,wait}` (per `wantSignal`) whose
    /// first source is literal `id`, among StinkyTofu instructions in `bb`, or -1.
    int posOfBarrier(bool wantSignal, int id) const {
        int pos = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (wantSignal ? isBarrierSignal(*inst) : isBarrierWait(*inst)) {
                const auto& srcs = inst->getSrcRegs();
                if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
                    srcs[0].getLiteralInt() == id) {
                    return pos;
                }
            }
            ++pos;
        }
        return -1;
    }

    /// Position of the first `s_cmp_eq_*` whose first source is symbolic `name`, or -1.
    int posOfCmpEqWithSymbol(const std::string& name) const {
        int pos = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            const auto uOp = inst->getUnifiedOpcode();
            if (uOp == GFX::s_cmp_eq_u32 || uOp == GFX::s_cmp_eq_i32) {
                const auto& srcs = inst->getSrcRegs();
                if (!srcs.empty() && srcs[0].getSymbolicName() == name) return pos;
            }
            ++pos;
        }
        return -1;
    }

    /// Count `s_cmp_le_{u32,i32}` whose first source is the symbolic register
    /// `name` (the drain-gate compares emitted by the gated scheme).
    int countCmpLeWithSymbol(const std::string& name) const {
        int count = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            const auto uOp = inst->getUnifiedOpcode();
            if (uOp != GFX::s_cmp_le_u32 && uOp != GFX::s_cmp_le_i32) continue;
            const auto& srcs = inst->getSrcRegs();
            if (!srcs.empty() && srcs[0].getSymbolicName() == name) ++count;
        }
        return count;
    }

    /// 0-based position of `target` among StinkyTofu instructions in `func`, or -1.
    int posOfInFunc(const StinkyInstruction* target) const {
        int pos = 0;
        for (const BasicBlock& b : *func) {
            for (const IRBase& ir : b) {
                if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
                if (cast<StinkyInstruction>(&ir) == target) return pos;
                ++pos;
            }
        }
        return -1;
    }

    /// Position of the first barrier in `func` matching `wantSignal`/`id`, or -1.
    int posOfBarrierInFunc(bool wantSignal, int id) const {
        int pos = 0;
        for (const BasicBlock& b : *func) {
            for (const IRBase& ir : b) {
                if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
                const auto* inst = cast<StinkyInstruction>(&ir);
                if (wantSignal ? isBarrierSignal(*inst) : isBarrierWait(*inst)) {
                    const auto& srcs = inst->getSrcRegs();
                    if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
                        srcs[0].getLiteralInt() == id) {
                        return pos;
                    }
                }
                ++pos;
            }
        }
        return -1;
    }

    /// Position of the first bare cluster wait `s_barrier_wait -3`, or -1.
    int posOfClusterWait() const {
        return posOfBarrier(/*wantSignal=*/false, kClusterBarrierId);
    }

    /// Position of the first cluster signal `s_barrier_signal -3`, or -1.
    int posOfClusterSignal() const {
        return posOfBarrier(/*wantSignal=*/true, kClusterBarrierId);
    }
};

// ---------------------------------------------------------------------------
// Rules 1 + 2 -- prologue priming signal and first-load wait
// ---------------------------------------------------------------------------

// Typical prefetch prologue (no pre-existing cluster barriers): after
// `label_GSU_1:` Rule 1 emits a gated priming cluster signal; Rule 2 plants a
// bare `s_barrier_wait -3` immediately before the kernel's first
// `tensor_load_to_lds`. Input mirrors Tensile's GSU / loop-exit guard layout.
TEST_F(InsertClusterBarrierPassTest, Rule1AndRule2_PrologueHandshake) {
    createLabel("label_ASM_Start");
    createVAddInBlock(bb, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);  // LCL / GSU setup

    StinkyInstruction* gsu1Label = createLabel(kGSU1LabelName);

    createVAddInBlock(bb, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    StinkyInstruction* loopExitCmp = createLoopCounterLCmpEq(/*imm=*/0);
    createCBranchScc1(kLoopEndLLabelName);

    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass();

    // Rule 1: one gated priming cluster signal (no bare cluster wait from Rule 1).
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 1 emits exactly one cluster signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 2 emits exactly one bare cluster wait before the first load";
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kWorkgroupBarrierId), 1)
        << "gated Rule 1 emits a workgroup signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kWorkgroupBarrierId), 1)
        << "gated Rule 1 emits a workgroup wait";
    EXPECT_EQ(countCmpEqWithSymbol(kLoopCounterLSymbol), 2)
        << "Rule 1 LCL gate plus the input loop-exit LCL == 0 compare";
    EXPECT_EQ(countCmpEqWithSymbol(kWaveIdxSymbol), 1)
        << "gated Rule 1 emits the inner WaveIdx gate";

    // Rule 1 block sits right after label_GSU_1, before the loop-exit guard.
    const int lclGate = posOfCmpEqWithSymbol(kLoopCounterLSymbol);
    const int wgSignal = posOfBarrier(/*wantSignal=*/true, kWorkgroupBarrierId);
    const int wgWait = posOfBarrier(/*wantSignal=*/false, kWorkgroupBarrierId);
    const int waveGate = posOfCmpEqWithSymbol(kWaveIdxSymbol);
    const int clusterSignal = posOfClusterSignal();
    EXPECT_LT(posOf(gsu1Label), lclGate);
    EXPECT_LT(lclGate, wgSignal);
    EXPECT_LT(wgSignal, wgWait);
    EXPECT_LT(wgWait, waveGate);
    EXPECT_LT(waveGate, clusterSignal);
    EXPECT_LT(clusterSignal, posOf(loopExitCmp));

    // Rule 2: bare cluster wait directly precedes the first tensor load.
    EXPECT_EQ(posOfClusterWait() + 1, posOf(tl)) << "cluster wait must directly precede the load";
    EXPECT_LT(posOf(loopExitCmp), posOfClusterWait()) << "Rule 2 wait follows the loop-exit guard";
}

// PGR=0: Rule 2 is suppressed (no prologue prefetch wait before first load) and
// Rule 3 is suppressed (no publication-point priming signal). Rule 1 still runs.
TEST_F(InsertClusterBarrierPassTest, Pgr0_SkipsRule2AndRule3) {
    createLabel("label_ASM_Start");
    createVAddInBlock(bb, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    StinkyInstruction* gsu1Label = createLabel(kGSU1LabelName);
    createVAddInBlock(bb, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    StinkyInstruction* loopExitCmp = createLoopCounterLCmpEq(/*imm=*/0);
    createCBranchScc1(kLoopEndLLabelName);

    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    // Rule 3 cross-BB setup (would emit a priming signal at pgr=1).
    createBarrierWait(kWorkgroupBarrierId);
    createBranch("label_openLoopL");
    BasicBlock* openLoopBB = func->createBasicBlock("openLoop");
    func->addEdge(bb, openLoopBB);
    StinkyInstruction* openLoopLabel = createLabelIn(openLoopBB, kOpenLoopLLabelName);
    createVAddInBlock(openLoopBB, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    runPass(/*pgrValue=*/0);

    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 1 priming signal only -- Rule 3 suppressed at PGR=0";
    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/false, kClusterBarrierId), 0)
        << "Rule 2 suppressed at PGR=0 -- no bare wait before first load";
    EXPECT_EQ(posOfBarrierInFunc(/*wantSignal=*/false, kClusterBarrierId), -1)
        << "no cluster wait anywhere in the kernel at PGR=0";
}

// PGR=0 Rule 4: bare cluster wait (no LCL drain gate) before workgroup signal;
// signal half keeps the LCL drain gate.
TEST_F(InsertClusterBarrierPassTest, Pgr0_Rule4_BareWaitGatedSignal) {
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass(/*pgrValue=*/0);

    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "one bare Rule 4 wait (no Rule 2 leading wait at PGR=0)";
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1);
    EXPECT_EQ(countCmpLeWithSymbol(kLoopCounterLSymbol), 1)
        << "only the signal half is LCL-gated at PGR=0";

    const int clusterWait = posOfClusterWait();
    const int wgSignal = posOfBarrier(/*wantSignal=*/true, kWorkgroupBarrierId);
    const int wgWait = posOfBarrier(/*wantSignal=*/false, kWorkgroupBarrierId);
    const int clusterSignal = posOfClusterSignal();

    EXPECT_LT(clusterWait, wgSignal)
        << "bare cluster wait -3 sits immediately above workgroup signal -1";
    EXPECT_LT(wgWait, clusterSignal);
}

// ---------------------------------------------------------------------------
// Rule 3 -- priming signal at the LDS publication point before openLoopL
// ---------------------------------------------------------------------------

// Rule 3 cross-BB: after CFGBuilder the publication-point workgroup wait sits
// in the predecessor basic block while `label_openLoopL:` starts the loop BB.
// Rule 3 must walk that predecessor and emit its priming cluster signal after
// the existing wait in a predecessor basic block.
TEST_F(InsertClusterBarrierPassTest, Rule3_CrossBB_FindsWaitInPredecessor) {
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    StinkyInstruction* pubWait = createBarrierWait(kWorkgroupBarrierId);
    createBranch("label_openLoopL");
    BasicBlock* openLoopBB = func->createBasicBlock("openLoop");
    func->addEdge(bb, openLoopBB);
    StinkyInstruction* openLoopLabel = createLabelIn(openLoopBB, kOpenLoopLLabelName);
    createVAddInBlock(openLoopBB, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    runPass();

    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 3 must emit its priming cluster signal";
    EXPECT_LT(posOfInFunc(pubWait), posOfBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId))
        << "Rule 3 signal must follow the predecessor's publication wait";
    EXPECT_LT(posOfBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId),
              posOfInFunc(openLoopLabel))
        << "Rule 3 signal must precede label_openLoopL";
}

// Rule 3 synthesized-publication path: when no workgroup wait sits before the
// prefetch boundary, synthesize the publication sync pair before `label_openLoopL:`.
TEST_F(InsertClusterBarrierPassTest, Rule3_CrossBB_SynthesizesWgSyncAtLabel) {
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createBranch("label_openLoopL");
    BasicBlock* openLoopBB = func->createBasicBlock("openLoop");
    func->addEdge(bb, openLoopBB);
    StinkyInstruction* openLoopLabel = createLabelIn(openLoopBB, kOpenLoopLLabelName);
    createVAddInBlock(openLoopBB, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    runPass();

    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 3 must emit its priming cluster signal";
    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kWorkgroupBarrierId), 1)
        << "synthesized-publication path emits a workgroup signal before the cluster signal";
    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/false, kWorkgroupBarrierId), 1)
        << "synthesized-publication path emits a workgroup wait before the cluster signal";
    EXPECT_LT(posOfBarrierInFunc(/*wantSignal=*/false, kWorkgroupBarrierId),
              posOfInFunc(openLoopLabel))
        << "synthesized publication point sits immediately before label_openLoopL";
}

// ---------------------------------------------------------------------------
// Rule 4 -- per-iteration cluster handshake around a workgroup wait
// ---------------------------------------------------------------------------

// Rule 4 wait-move (random-hang fix): when the anchor wait's paired workgroup
// `s_barrier_signal -1` is present, the (gated) cluster `s_barrier_wait -3` is
// planted immediately BEFORE that workgroup signal, while the gated cluster
// `s_barrier_signal -3` stays AFTER the workgroup `s_barrier_wait -1`. This puts
// the workgroup signal/wait pair BETWEEN the cluster wait and cluster signal.
TEST_F(InsertClusterBarrierPassTest, Rule4_WaitMovesBeforeWorkgroupSignal) {
    createBarrierSignal(kWorkgroupBarrierId);  // workgroup signal (wait-move anchor)
    createBarrierWait(kWorkgroupBarrierId);    // anchor wait (Rule 4 trigger)
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass();

    // Rule 4's moved wait + Rule 2's leading wait before the first load.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 2);
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1);
    EXPECT_EQ(countCmpLeWithSymbol(kLoopCounterLSymbol), 2)
        << "gated Rule 4 emits two asymmetric LCL drain gates";

    // posOfClusterWait() returns the FIRST cluster wait, i.e. Rule 4's moved
    // wait (Rule 2's wait lands last, just before the load).
    const int clusterWait = posOfClusterWait();
    const int wgSignal = posOfBarrier(/*wantSignal=*/true, kWorkgroupBarrierId);
    const int wgWait = posOfBarrier(/*wantSignal=*/false, kWorkgroupBarrierId);
    const int clusterSignal = posOfClusterSignal();

    EXPECT_LT(clusterWait, wgSignal)
        << "cluster wait -3 must be moved immediately above the workgroup signal -1";
    EXPECT_LT(wgSignal, wgWait) << "workgroup signal -1 must precede workgroup wait -1";
    EXPECT_LT(wgWait, clusterSignal) << "cluster signal -3 must stay after the workgroup wait -1";
}

// Rule 4 multi-load: two `tensor_load_to_lds` that each have their OWN
// preceding `s_barrier_wait -1` each receive a separate cluster handshake.
TEST_F(InsertClusterBarrierPassTest, Rule4_DistinctWaits) {
    // Block 1: wait -1, load
    createBarrierWait(kWorkgroupBarrierId);
    StinkyInstruction* tl1 = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createBranch("label_next");
    // Block 1 continued: wait -1, load
    createBarrierWait(kWorkgroupBarrierId);
    StinkyInstruction* tl2 = createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);

    runPass();

    // One gated handshake per distinct anchor wait => two signals and two Rule-4
    // waits, plus Rule 2's leading wait before the function's first load (tl1).
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 2);
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 3);
    // Two drain gates per handshake => four LCL `s_cmp_le` gates.
    EXPECT_EQ(countCmpLeWithSymbol(kLoopCounterLSymbol), 4);
    // Both original loads survive.
    EXPECT_GE(posOf(tl1), 0);
    EXPECT_GE(posOf(tl2), 0);
}

// Rule 4 dedup: two `tensor_load_to_lds` in the SAME basic block that share a
// single preceding `s_barrier_wait -1` produce exactly ONE handshake -- the
// anchor wait is only gated once (seenTriggers dedup).
TEST_F(InsertClusterBarrierPassTest, Rule4_SharedWait) {
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);

    runPass();

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "loads sharing one anchor wait must yield a single signal";
    // One deduped Rule-4 gated wait + Rule 2's leading wait before the first load.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 2)
        << "one deduped Rule-4 wait plus Rule 2's leading wait";
    // The two loads are preserved.
    EXPECT_EQ(countTensorLoads(), 2);
}

// PGR=2 Rule 4: two distinct publication points get staggered WAIT drain
// thresholds (2 then 3) while both SIGNAL halves stay at 3.
TEST_F(InsertClusterBarrierPassTest, Rule4_Pgr2_StaggeredWaitDrainGates) {
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createBranch("label_next");
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);

    runPass(/*pgrValue=*/2);

    const std::vector<int> drainGateImms = collectRule4DrainGateImms();

    ASSERT_EQ(drainGateImms.size(), 4u);
    EXPECT_EQ(drainGateImms[0], 2) << "pub0 wait gate";
    EXPECT_EQ(drainGateImms[1], 3) << "pub0 signal gate";
    EXPECT_EQ(drainGateImms[2], 3) << "pub1 wait gate";
    EXPECT_EQ(drainGateImms[3], 3) << "pub1 signal gate";
}

// PGR=2 Rule 4: three publication points before `s_sub LCL` -- only the first
// pub keeps wait threshold 2; pubs 2 and 3 both use 3,3.
TEST_F(InsertClusterBarrierPassTest, Rule4_Pgr2_ThreePubsSameWindow) {
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createBranch("label_next");
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);
    createBranch("label_next2");
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/16, /*s1=*/20);

    runPass(/*pgrValue=*/2);

    const std::vector<int> drainGateImms = collectRule4DrainGateImms();

    ASSERT_EQ(drainGateImms.size(), 6u);
    EXPECT_EQ(drainGateImms[0], 2) << "pub0 wait gate";
    EXPECT_EQ(drainGateImms[1], 3) << "pub0 signal gate";
    EXPECT_EQ(drainGateImms[2], 3) << "pub1 wait gate";
    EXPECT_EQ(drainGateImms[3], 3) << "pub1 signal gate";
    EXPECT_EQ(drainGateImms[4], 3) << "pub2 wait gate";
    EXPECT_EQ(drainGateImms[5], 3) << "pub2 signal gate";
}

// PGR=1 Rule 4: three publication points with `s_sub LCL` after the window
// (TDM unroll) -- first pub uses 1,2; later pubs use 2,2.
TEST_F(InsertClusterBarrierPassTest, Rule4_Pgr1_ThreePubsWithDecAfterWindow) {
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createBranch("label_next");
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);
    createBranch("label_next2");
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/16, /*s1=*/20);
    createLoopCounterLDecrement(/*imm=*/1);

    runPass(/*pgrValue=*/1);

    const std::vector<int> drainGateImms = collectRule4DrainGateImms();

    ASSERT_EQ(drainGateImms.size(), 6u);
    EXPECT_EQ(drainGateImms[0], 1) << "pub0 wait gate";
    EXPECT_EQ(drainGateImms[1], 2) << "pub0 signal gate";
    EXPECT_EQ(drainGateImms[2], 2) << "pub1 wait gate";
    EXPECT_EQ(drainGateImms[3], 2) << "pub1 signal gate";
    EXPECT_EQ(drainGateImms[4], 2) << "pub2 wait gate";
    EXPECT_EQ(drainGateImms[5], 2) << "pub2 signal gate";
}

// Rule 4 SCC restore (mode d): when a live `s_cmp_eq s[sgprLoopCounterL]` is the
// first SCC writer above the anchor wait, the WaveIdx gate would clobber its
// SCC, so the pass re-clones that compare AFTER the whole handshake block. We
// therefore see the loop-counter compare twice (original + restore).
TEST_F(InsertClusterBarrierPassTest, Rule4_RestoresLclCmp) {
    createLoopCounterLCmpEq(/*imm=*/0);  // SIA-hoisted live loop-exit compare
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    EXPECT_EQ(countCmpEqWithSymbol(kLoopCounterLSymbol), 1);

    runPass();

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 4 still emits its cluster signal";
    EXPECT_EQ(countCmpEqWithSymbol(kLoopCounterLSymbol), 2)
        << "the live loop-counter compare must be cloned to restore SCC";
    // Rule 4's gated wait + Rule 2's leading wait before the first load.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 2)
        << "Rule 4's gated wait plus Rule 2's leading wait before the first load";
}

// Rule 4 is bounded by the `/* Tail Loop */` marker: a workgroup wait + tensor
// load that sit after the marker are owned exclusively by Rule 5, not Rule 4.
// The marker prevents the two rules from ever sharing an anchor wait, so the
// tail load gets exactly one cluster signal (Rule 5a) and one bare cluster
// wait (Rule 5b) -- no Rule-4 duplicate.
TEST_F(InsertClusterBarrierPassTest, Rule4_StopsAtTailMarker) {
    createTextblock(kTailLoopMarker);
    createBarrierWait(kWorkgroupBarrierId);  // same segment as the load (no branch)
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass();

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "tail load (after marker) gets exactly one Rule 5a cluster signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 4 must not also claim the tail wait -- only Rule 5b's bare wait remains";
}

// Rule 4 runs per basic block: two blocks that each contain a workgroup wait +
// tensor load each receive their own handshake. Rule 2 is function-wide and
// additionally plants a leading wait before the function's first load (block 1).
TEST_F(InsertClusterBarrierPassTest, Rule4_PerBasicBlock) {
    // Block 1 (entry / bb): wait -1, load
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    // Block 2: wait -1, load
    BasicBlock* bb2 = func->createBasicBlock("bb2");
    createBarrierWaitIn(bb2, kWorkgroupBarrierId);
    createTensorLoadInBlock(bb2, arch, /*s0=*/8, /*s1=*/12);

    runPass();

    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId), 2)
        << "each block's load must get its own Rule 4 cluster signal";
    // Two Rule-4 gated waits (one per block) + Rule 2's leading wait before
    // block 1's first load.
    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/false, kClusterBarrierId), 3)
        << "two Rule-4 waits plus Rule 2's leading wait before the first load";
}

// ---------------------------------------------------------------------------
// Rule 5 -- tail-loop handshake (5a signal + 5b wait)
// ---------------------------------------------------------------------------

// Rule 5 (5a + 5b): after the `/* Tail Loop */` TEXTBLOCK marker, with a
// preceding workgroup wait and a tail tensor load (split by a branch so Rule 4
// does not also claim the wait), 5a emits a signal-only handshake after the
// wait and 5b emits a bare cluster wait before the load.
TEST_F(InsertClusterBarrierPassTest, Rule5_EmitsSignalAndWait) {
    createTextblock(kTailLoopMarker);
    createBarrierWait(kWorkgroupBarrierId);  // tail load's publication point (5a anchor)
    createBranch("label_tail");              // keep Rule 4 from anchoring this wait
    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass();

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 5a emits exactly one cluster signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 5b emits exactly one bare cluster wait before the tail load";
    EXPECT_LT(posOfClusterWait(), posOf(tl));
}

// Rule 5a fallback: when no workgroup wait precedes the tail load, 5a
// synthesizes an `s_barrier_signal -1` / `s_barrier_wait -1` pair plus the
// WaveIdx-gated cluster signal right after the marker, then 5b gates the load.
TEST_F(InsertClusterBarrierPassTest, Rule5a_FallbackSynthesizesSync) {
    createTextblock(kTailLoopMarker);
    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass();

    // 5a fallback: synthesized workgroup-sync pair + cluster signal.
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kWorkgroupBarrierId), 1)
        << "fallback must synthesize a workgroup signal -1";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kWorkgroupBarrierId), 1)
        << "fallback must synthesize a workgroup wait -1";
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "fallback must emit the WaveIdx-gated cluster signal";
    // 5b still gates the tail load with a bare cluster wait.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 5b still gates the tail load with a bare cluster wait";
    EXPECT_LT(posOfClusterWait(), posOf(tl));
}

// Rule 5 cross-BB: after CFGBuilder the tail marker, publication wait, and
// tail tensor load can live in different basic blocks. Rule 5 must still pair
// 5a (signal after the wait) with 5b (wait before the load).
TEST_F(InsertClusterBarrierPassTest, Rule5_CrossBB_FindsWaitAndLoad) {
    createTextblock(kTailLoopMarker);
    createBranch("label_tail_body");
    BasicBlock* bodyBB = func->createBasicBlock("tail_body");
    func->addEdge(bb, bodyBB);
    createLabelIn(bodyBB, "label_tail_body");
    StinkyInstruction* pubWait = createBarrierWaitIn(bodyBB, kWorkgroupBarrierId);
    createBranchIn(bodyBB, "label_tail_load");
    BasicBlock* loadBB = func->createBasicBlock("tail_load");
    func->addEdge(bodyBB, loadBB);
    createLabelIn(loadBB, "label_tail_load");
    StinkyInstruction* tl = createTensorLoadInBlock(loadBB, arch, /*s0=*/0, /*s1=*/4);

    runPass();

    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 5a must emit its cluster signal across BB boundaries";
    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 5b must emit its cluster wait before the tail load";
    EXPECT_LT(posOfInFunc(pubWait), posOfBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId))
        << "Rule 5a signal must follow the tail publication wait";
    EXPECT_LT(posOfBarrierInFunc(/*wantSignal=*/false, kClusterBarrierId), posOfInFunc(tl))
        << "Rule 5b wait must precede the tail tensor load";
}

// ---------------------------------------------------------------------------
// Combined rules
// ---------------------------------------------------------------------------

// Combined-rule negative test: the load-driven rules (Rule 2 / Rule 4 / Rule 5)
// all key off a `tensor_load_to_lds`. With no tensor load and no Rule 1/3/5
// anchor label present, none of them may fire -- a lone `s_barrier_wait -1`
// must not be mistaken for an anchor -- and the pass leaves the function
// completely unchanged (existing wait kept).
TEST_F(InsertClusterBarrierPassTest, CombinedLoadDrivenRules_NoLoadIsNoOp) {
    createBarrierWait(kWorkgroupBarrierId);

    runPass();

    EXPECT_EQ(countTensorLoads(), 0);
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 0);
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 0);
    // The original workgroup wait is preserved.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kWorkgroupBarrierId), 1);
}

}  // namespace
