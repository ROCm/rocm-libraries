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
//
// Unit tests for InsertClusterBarrierPass (gfx1250), Rule 4.
//
// Rule 4 turns each `tensor_load_to_lds` (paired with a workgroup barrier
// `s_barrier_signal -1` / `s_barrier_wait -1`) into a cluster-scope handshake:
// a WaveIdx-gated `s_barrier_signal -3` (Rule 4a) plus a `s_barrier_wait -3`
// (Rule 4b). To hide cross-CU latency, Rule 4a is walked backward from its
// paired wait by up to `kRule4SignalLeadCycles` estimated cycles.
//
// The invariant these tests lock down is that the cluster barrier is a COUNTING
// barrier: it can only track one outstanding generation at a time, so the
// emitted `signal -3` / `wait -3` pairs must strictly alternate (never two
// signals in flight before a wait). The cycle-lead hoist must therefore never
// move a signal into or above a PRECEDING handshake -- otherwise the two phases
// overlap and the cluster deadlocks (observed as a tuning hang on hardware).
//
// These tests build a minimal `label_LoopBeginL` region so the cycle estimator
// populates per-instruction cycles and the cycle-lead hoist is actually active.
//
#include <gtest/gtest.h>

#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr int kClusterBarrierId = -3;
constexpr int kWorkgroupBarrierId = -1;

/// Classify a cluster-scope (`-3`) barrier: +1 for `s_barrier_signal -3`,
/// -1 for `s_barrier_wait -3`, 0 for anything else (incl. the `-1` workgroup
/// barriers the handshake reuses).
int clusterBarrierKind(const StinkyInstruction& inst) {
    const bool sig = isBarrierSignal(inst);
    const bool wait = isBarrierWait(inst);
    if (!sig && !wait) return 0;
    const auto& srcs = inst.getSrcRegs();
    if (srcs.empty()) return 0;
    if (srcs[0].dataType != StinkyRegister::Type::LiteralInt) return 0;
    if (srcs[0].getLiteralInt() != kClusterBarrierId) return 0;
    return sig ? 1 : -1;
}

}  // namespace

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
        // Fields the cycle estimator reads while modeling the loop region.
        config.TileA0 = 16;
        config.TileB0 = 16;
        config.TileM0 = 16;
        config.NumGRA = 4;
        config.NumGRB = 4;
        config.NumGRM = 4;
        config.NumWaves = 4;

        func = std::make_unique<Function>("cluster_barrier_test");
        setFunctionArch(*func, arch);
        // The estimator only models the block literally named `label_LoopBeginL`;
        // this is what makes the cycle-lead hoist active in Rule 4.
        bb = func->createBasicBlock("label_LoopBeginL");
        func->setGemmTileConfig(config);
        registerAllAnalyses(am);
    }

    void TearDown() override {
        func.reset();
        bb = nullptr;
    }

    StinkyInstruction* createBarrierSignal(int literal) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_signal, arch));
        inst->addSrcReg(StinkyRegister(literal));
        return inst;
    }

    StinkyInstruction* createBarrierWait(int literal) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_wait, arch));
        inst->addSrcReg(StinkyRegister(literal));
        return inst;
    }

    /// A WMMA to give the loop body some estimated-cycle length between the
    /// handshakes (so the cycle-lead hoist has room to move).
    StinkyInstruction* createWMMA(int destStart, int src0Start, int src1Start) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch);
        if (desc == nullptr) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("a", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src1Start, 8));
        inst->addSrcReg(StinkyRegister("a", destStart, 8));  // acc
        return inst;
    }

    /// Emit one workgroup-barrier + tensor-load "handshake" -- the exact shape
    /// Rule 4 triggers on: `s_barrier_signal -1`, `s_barrier_wait -1`, then a
    /// `tensor_load_to_lds`.
    void appendHandshake(int loadS0, int loadS1) {
        createBarrierSignal(kWorkgroupBarrierId);
        createBarrierWait(kWorkgroupBarrierId);
        createTensorLoadInBlock(bb, arch, loadS0, loadS1);
    }

    /// Run in REGION scope (`isKernelScope=false`) so Rule 2 -- the single
    /// kernel-opening `s_barrier_wait -3` planted before the function's very
    /// first tensor load -- stays out of the way. Rule 2 closes the PREVIOUS
    /// loop iteration's signal across the (cyclic) backedge, so in a
    /// straight-line unit-test body it would look like an unmatched leading
    /// wait; region scope keeps the sequence to the Rule 4 pairs under test.
    void runPass() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        auto pass = createInsertClusterBarrierPass(/*isKernelScope=*/false, /*pgrValue=*/1,
                                                   /*plrValue=*/1);
        pass->run(*func, ctx, am);
    }

    /// Cluster-barrier events (+1 signal / -1 wait) in program order.
    std::vector<int> clusterBarrierSequence() const {
        std::vector<int> seq;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const int kind = clusterBarrierKind(*cast<StinkyInstruction>(&ir));
            if (kind != 0) seq.push_back(kind);
        }
        return seq;
    }
};

// A single handshake yields exactly one balanced `signal -3` / `wait -3` pair,
// with the signal before the wait.
TEST_F(InsertClusterBarrierPassTest, SingleHandshakeEmitsOneBalancedPair) {
    createWMMA(32, 0, 8);
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(40, 8, 0);

    runPass();

    const std::vector<int> seq = clusterBarrierSequence();
    ASSERT_EQ(seq.size(), 2u) << "expected exactly one signal + one wait";
    EXPECT_EQ(seq[0], 1) << "signal -3 must come first";
    EXPECT_EQ(seq[1], -1) << "wait -3 must follow the signal";
}

// The core regression: two back-to-back handshakes must produce two cluster
// pairs that STRICTLY ALTERNATE (signal, wait, signal, wait). The cycle-lead
// hoist of the second signal must stop at the first handshake instead of
// crossing it -- otherwise both signals precede both waits (signal, signal,
// wait, wait), overlapping the two counting-barrier phases and deadlocking.
TEST_F(InsertClusterBarrierPassTest, TwoHandshakesDoNotOverlapClusterPhases) {
    // a little body before handshake 1 too, so its cycle-lead signal hoist has
    // somewhere to land ahead of its own wait.
    createWMMA(24, 0, 8);
    // handshake 1
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    // a little loop body between the two handshakes (well under the cycle lead,
    // so the backward walk for handshake 2 would reach handshake 1 if unguarded)
    createWMMA(32, 8, 16);
    createWMMA(40, 16, 8);
    // handshake 2
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createWMMA(56, 24, 32);

    runPass();

    const std::vector<int> seq = clusterBarrierSequence();
    ASSERT_EQ(seq.size(), 4u) << "expected two signal/wait pairs (one per handshake)";

    // Replay the sequence: a counting barrier can only have one outstanding
    // generation, so the running signal-minus-wait balance must never exceed 1
    // and must never go negative (a wait with no matching signal).
    int outstanding = 0;
    for (size_t i = 0; i < seq.size(); ++i) {
        outstanding += seq[i];
        EXPECT_GE(outstanding, 0) << "wait -3 with no outstanding signal at index " << i;
        EXPECT_LE(outstanding, 1)
            << "two cluster signals in flight before a wait at index " << i
            << " -- overlapping barrier phases will deadlock";
    }
    EXPECT_EQ(outstanding, 0) << "every signal -3 must be matched by a wait -3";

    // Spell the exact expected alternation out too, for a clearer failure.
    EXPECT_EQ(seq, (std::vector<int>{1, -1, 1, -1}))
        << "cluster signal/wait pairs must strictly alternate";
}

// Sanity: the pass leaves the workgroup (`-1`) barriers in place -- Rule 4
// reuses them, it does not delete or convert them.
TEST_F(InsertClusterBarrierPassTest, WorkgroupBarriersArePreserved) {
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(32, 8, 16);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);

    runPass();

    int wgSignals = 0;
    int wgWaits = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const auto& srcs = inst->getSrcRegs();
        const bool isMinusOne = !srcs.empty() &&
                                srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
                                srcs[0].getLiteralInt() == kWorkgroupBarrierId;
        if (!isMinusOne) continue;
        if (isBarrierSignal(*inst)) ++wgSignals;
        if (isBarrierWait(*inst)) ++wgWaits;
    }
    EXPECT_EQ(wgSignals, 2) << "both workgroup s_barrier_signal -1 must survive";
    EXPECT_EQ(wgWaits, 2) << "both workgroup s_barrier_wait -1 must survive";
}
