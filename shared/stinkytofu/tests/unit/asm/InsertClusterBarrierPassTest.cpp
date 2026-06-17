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

namespace {

// The literal split-barrier ids the pass keys off of (see InsertClusterBarrierPass.cpp).
constexpr int kClusterBarrierId = -3;    // s_barrier_{signal,wait} -3  (cluster scope)
constexpr int kWorkgroupBarrierId = -1;  // s_barrier_{signal,wait} -1  (workgroup scope)

// Anchor names / symbols the pass keys off of.
constexpr const char* kGSU1LabelName = "label_GSU_1";       // Rule 1 anchor
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

    /// Create an unconditional `s_branch <label>`. Branches are segment
    /// boundaries for Rule 4's backward scan, so they split the block into
    /// distinct anchor segments.
    StinkyInstruction* createBranch(const std::string& label) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
        inst->addSrcReg(StinkyRegister(label));
        return inst;
    }

    /// Create a LABEL pseudo carrying LabelData{name} (matches the pass's
    /// `isLabelNamed` anchor checks, e.g. `label_GSU_1`).
    StinkyInstruction* createLabel(const std::string& name) {
        AsmIRBuilder builder(*bb, arch);
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
    /// compare). Rule 4's `findLiveLoopCounterLCmpUpstream` keys off exactly
    /// this shape to detect a SIA-hoisted live LCL compare.
    StinkyInstruction* createLoopCounterLCmpEq(int imm) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));
        StinkyRegister lcl(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
        lcl.setSymbolicName(kLoopCounterLSymbol);
        inst->addSrcReg(lcl);
        inst->addSrcReg(StinkyRegister(imm));
        return inst;
    }

    /// Run the pass on `func`. `isKernelScope` toggles Rule 2 (kernel-scope-only
    /// bare `s_barrier_wait -3` before the first tensor load).
    void runPass(bool isKernelScope) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        auto pass = createInsertClusterBarrierPass(isKernelScope, /*pgrValue=*/1, /*plrValue=*/1);
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

    /// Position of the (single) bare cluster wait `s_barrier_wait -3`, or -1.
    int posOfClusterWait() const {
        int pos = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (isBarrierWait(*inst)) {
                const auto& srcs = inst->getSrcRegs();
                if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
                    srcs[0].getLiteralInt() == kClusterBarrierId) {
                    return pos;
                }
            }
            ++pos;
        }
        return -1;
    }
};

// ---------------------------------------------------------------------------
// Rule 1 -- signal-only handshake after `label_GSU_1:`
// ---------------------------------------------------------------------------

// Rule 1: a `label_GSU_1:` label gets a LoopCounterL-gated, WaveIdx-gated
// cluster SIGNAL plus a workgroup-sync pair (signal -1 / wait -1). It is
// signal-only: no bare cluster wait -3 is emitted by Rule 1.
TEST_F(InsertClusterBarrierPassTest, Rule1_EmitsGatedSignal) {
    createLabel(kGSU1LabelName);
    // A trailing instruction so the label has a concrete successor anchor.
    createVAddInBlock(bb, arch, /*dest=*/0, /*src0=*/1, /*src1=*/2);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 1 emits exactly one cluster signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 0)
        << "Rule 1 is signal-only -- no bare cluster wait";
    // The workgroup-sync pair planted between the two gates.
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kWorkgroupBarrierId), 1);
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kWorkgroupBarrierId), 1);

    // --- Gate structure ---
    // Outer LoopCounterL gate and inner WaveIdx gate are both present.
    EXPECT_EQ(countCmpEqWithSymbol(kLoopCounterLSymbol), 1)
        << "Rule 1 must emit the outer LoopCounterL gate (s_cmp_eq s[sgprLoopCounterL], 0)";
    EXPECT_EQ(countCmpEqWithSymbol(kWaveIdxSymbol), 1)
        << "Rule 1 must emit the inner WaveIdx gate (s_cmp_eq s[sgprWaveIdx], 0)";

    // Emitted order: LCL gate -> workgroup signal/wait pair -> WaveIdx gate ->
    // cluster signal.
    const int lclGate = posOfCmpEqWithSymbol(kLoopCounterLSymbol);
    const int wgSignal = posOfBarrier(/*wantSignal=*/true, kWorkgroupBarrierId);
    const int wgWait = posOfBarrier(/*wantSignal=*/false, kWorkgroupBarrierId);
    const int waveGate = posOfCmpEqWithSymbol(kWaveIdxSymbol);
    const int clusterSignal = posOfBarrier(/*wantSignal=*/true, kClusterBarrierId);
    EXPECT_LT(lclGate, wgSignal) << "LoopCounterL gate must precede the workgroup signal";
    EXPECT_LT(wgSignal, wgWait) << "workgroup signal must precede workgroup wait";
    EXPECT_LT(wgWait, waveGate) << "workgroup sync pair must precede the WaveIdx gate";
    EXPECT_LT(waveGate, clusterSignal) << "WaveIdx gate must precede the cluster signal";
}

// ---------------------------------------------------------------------------
// Rule 2 -- kernel-scope bare cluster wait before the first tensor load
// ---------------------------------------------------------------------------

// Rule 2: in kernel scope, a single bare `s_barrier_wait -3` is planted
// immediately before the first `tensor_load_to_lds` when no preceding
// workgroup wait makes Rule 4 fire.
TEST_F(InsertClusterBarrierPassTest, Rule2_InsertsWaitBeforeFirstLoad) {
    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 0);

    runPass(/*isKernelScope=*/true);

    // Exactly one cluster wait, and it sits immediately before the tensor load.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1);
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 0)
        << "Rule 2 emits only a wait, never a signal";
    EXPECT_EQ(posOfClusterWait() + 1, posOf(tl)) << "cluster wait must directly precede the load";
}

// Rule 2 idempotency: a pre-existing bare cluster wait in front of the first
// tensor load suppresses a second insertion in kernel scope.
TEST_F(InsertClusterBarrierPassTest, Rule2_SkipsExistingWait) {
    createBarrierWait(kClusterBarrierId);  // pretend a prior run already gated the load
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "existing cluster wait must not be duplicated";
}

// ---------------------------------------------------------------------------
// Rule 4 -- cluster handshake after a workgroup wait that publishes LDS
// ---------------------------------------------------------------------------

// Rule 4: a `tensor_load_to_lds` anchored by a preceding workgroup wait
// `s_barrier_wait -1` gets a full cluster handshake (WaveIdx-gated
// `s_barrier_signal -3` + bare `s_barrier_wait -3`). Kernel scope (the only
// production configuration): Rule 4 emits the handshake first, so Rule 2 sees
// the first load already gated by the cluster wait and self-suppresses --
// hence the counts stay at exactly one each.
TEST_F(InsertClusterBarrierPassTest, Rule4_EmitsHandshake) {
    createBarrierWait(kWorkgroupBarrierId);
    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 4 emits exactly one cluster signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 4 emits exactly one bare cluster wait (Rule 2 self-suppresses)";

    // The handshake lands between the workgroup wait and the tensor load:
    // the cluster wait must precede the load.
    EXPECT_LT(posOfClusterWait(), posOf(tl));
}

// Rule 4 multi-load: two `tensor_load_to_lds` that each have their OWN
// preceding `s_barrier_wait -1` (in distinct segments, split by a branch)
// each receive a separate cluster handshake.
TEST_F(InsertClusterBarrierPassTest, Rule4_DistinctWaits) {
    // Segment 1: wait -1, load
    createBarrierWait(kWorkgroupBarrierId);
    StinkyInstruction* tl1 = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    // Segment boundary
    createBranch("label_next");
    // Segment 2: wait -1, load
    createBarrierWait(kWorkgroupBarrierId);
    StinkyInstruction* tl2 = createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);

    runPass(/*isKernelScope=*/true);

    // One handshake per distinct anchor wait => two of each.
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 2);
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 2);
    // Both original loads survive.
    EXPECT_GE(posOf(tl1), 0);
    EXPECT_GE(posOf(tl2), 0);
}

// Rule 4 dedup: two `tensor_load_to_lds` in the SAME segment that share a
// single preceding `s_barrier_wait -1` produce exactly ONE handshake -- the
// anchor wait is only gated once (seenTriggers dedup).
TEST_F(InsertClusterBarrierPassTest, Rule4_SharedWait) {
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/12);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "loads sharing one anchor wait must yield a single signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "loads sharing one anchor wait must yield a single bare wait";
    // The two loads are preserved.
    EXPECT_EQ(countTensorLoads(), 2);
}

// Rule 4 segment boundary: a workgroup wait `-1` separated from the tensor
// load by a branch lives in a different segment, so it is NOT a valid anchor
// and Rule 4 emits no handshake (no cluster signal). Rule 2 still gates the
// first load with a bare wait.
TEST_F(InsertClusterBarrierPassTest, Rule4_BranchBreaksAnchor) {
    createBarrierWait(kWorkgroupBarrierId);
    createBranch("label_mid");  // segment boundary
    StinkyInstruction* tl = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 0)
        << "Rule 4 must not anchor across a segment boundary";
    // Rule 2 still inserts a bare cluster wait before the first load.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1);
    EXPECT_LT(posOfClusterWait(), posOf(tl));
}

// Rule 4 inherited-SCC restore (mode c): when a live `s_cmp_eq s[sgprLoopCounterL]`
// sits upstream of the anchor wait, the WaveIdx gate would clobber its SCC, so
// the pass re-emits a clone of that compare after the handshake. We therefore
// see the loop-counter compare twice (original + restore).
TEST_F(InsertClusterBarrierPassTest, Rule4_RestoresLclCmp) {
    createLoopCounterLCmpEq(/*imm=*/0);  // SIA-hoisted live loop-exit compare
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    EXPECT_EQ(countCmpEqWithSymbol(kLoopCounterLSymbol), 1);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "Rule 4 still emits its cluster signal";
    EXPECT_EQ(countCmpEqWithSymbol(kLoopCounterLSymbol), 2)
        << "the inherited loop-counter compare must be cloned to restore SCC";
    // The restore cmp is emitted BEFORE the bare wait, so the wait stays the
    // load's immediate predecessor and Rule 2 does not plant a redundant wait.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "exactly one cluster wait -- Rule 2 must not double-insert";
}

// Rule 4 is bounded by the `/* Tail Loop */` marker: a workgroup wait + tensor
// load that sit after the marker (even in the same segment, no branch between
// them) are owned exclusively by Rule 5, not Rule 4. The marker thus prevents
// the two rules from ever sharing an anchor wait, so the tail load gets exactly
// one cluster signal (Rule 5a) and one bare cluster wait (Rule 5b) -- no
// Rule-4 duplicate of either.
TEST_F(InsertClusterBarrierPassTest, Rule4_StopsAtTailMarker) {
    createTextblock(kTailLoopMarker);
    createBarrierWait(kWorkgroupBarrierId);  // same segment as the load (no branch)
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 1)
        << "tail load (after marker) gets exactly one Rule 5a cluster signal";
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "Rule 4 must not also claim the tail wait -- only Rule 5b's bare wait remains";
}

// Rule 4 runs per basic block: two blocks that each contain a workgroup wait +
// tensor load each receive their own handshake. Rule 2 is function-wide, so it
// only considers the first load (already gated by block 1's Rule 4 wait).
TEST_F(InsertClusterBarrierPassTest, Rule4_PerBasicBlock) {
    // Block 1 (entry / bb): wait -1, load
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    // Block 2: wait -1, load
    BasicBlock* bb2 = func->createBasicBlock("bb2");
    createBarrierWaitIn(bb2, kWorkgroupBarrierId);
    createTensorLoadInBlock(bb2, arch, /*s0=*/8, /*s1=*/12);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/true, kClusterBarrierId), 2)
        << "each block's load must get its own Rule 4 cluster signal";
    EXPECT_EQ(countBarrierInFunc(/*wantSignal=*/false, kClusterBarrierId), 2)
        << "each block's handshake adds one bare cluster wait (Rule 2 self-suppresses)";
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

    runPass(/*isKernelScope=*/true);

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

    runPass(/*isKernelScope=*/true);

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

// Rule 5b idempotency: a tail load already immediately preceded by a cluster
// wait is left alone (no duplicate wait).
TEST_F(InsertClusterBarrierPassTest, Rule5b_SkipsExistingWait) {
    createTextblock(kTailLoopMarker);
    createBarrierWait(kClusterBarrierId);  // pre-existing cluster wait
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 1)
        << "existing cluster wait before the tail load must not be duplicated";
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 0);
}

// ---------------------------------------------------------------------------
// Combined rules
// ---------------------------------------------------------------------------

// Combined-rule negative test: the load-driven rules (Rule 2 / Rule 4 / Rule 5)
// all key off a `tensor_load_to_lds`. With no tensor load present, none of them
// may fire -- a lone `s_barrier_wait -1` must not be mistaken for an anchor --
// and the pass leaves the function completely unchanged (existing wait kept).
TEST_F(InsertClusterBarrierPassTest, CombinedLoadDrivenRules_NoLoadIsNoOp) {
    createBarrierWait(kWorkgroupBarrierId);

    runPass(/*isKernelScope=*/true);

    EXPECT_EQ(countTensorLoads(), 0);
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kClusterBarrierId), 0);
    EXPECT_EQ(countBarrier(/*wantSignal=*/true, kClusterBarrierId), 0);
    // The original workgroup wait is preserved.
    EXPECT_EQ(countBarrier(/*wantSignal=*/false, kWorkgroupBarrierId), 1);
}

}  // namespace
