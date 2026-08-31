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

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "transforms/asm/dag/RegionDAG.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

static int countStinkyInstructions(const BasicBlock& bb) {
    int count = 0;
    for (const IRBase& ir : bb) {
        if (ir.getType() == IRBase::IRType::StinkyTofu) count++;
    }
    return count;
}

// Adds a hard scheduling constraint edge directly into \p graph/\p inDegree, mirroring
// what StinkyDAGSchedulerPass.cpp does: reject edges that would form a cycle, otherwise
// merge into the same graph used for the register-dependency DAG.
static bool addHardConstraint(std::vector<std::unordered_set<unsigned>>& graph,
                              std::vector<unsigned>& inDegree, unsigned predecessor,
                              unsigned successor) {
    if (graph[predecessor].contains(successor)) return true;
    if (dag::hasPath(graph, successor, predecessor)) return false;
    graph[predecessor].insert(successor);
    ++inDegree[successor];
    return true;
}

TEST(HardSchedulingConstraintMergeTest, MergesConstraintsIntoBaseDagAndPreservesReadinessOrder) {
    // Base DAG descendants: barrierAfter -> tensorLoad and barrierBefore -> dsLoad.
    std::vector<std::unordered_set<unsigned>> graph(4);
    graph[0].insert(1);
    graph[2].insert(3);
    std::vector<unsigned> inDegree{0, 1, 0, 1};

    ASSERT_TRUE(addHardConstraint(graph, inDegree, 0, 2));  // barrierAfter -> barrierBefore
    ASSERT_TRUE(addHardConstraint(graph, inDegree, 1, 2));  // tensorLoad -> barrierBefore
    ASSERT_TRUE(addHardConstraint(graph, inDegree, 1, 3));  // tensorLoad -> dsLoad

    EXPECT_EQ(inDegree[0], 0u);
    EXPECT_EQ(inDegree[1], 1u);
    EXPECT_EQ(inDegree[2], 2u);  // base 0->2 (none) + constraints 0->2, 1->2
    EXPECT_EQ(inDegree[3], 2u);  // base 2->3 + constraint 1->3

    // Simulate Kahn's algorithm and check the readiness order the constraints impose.
    unsigned order[] = {0, 1, 2, 3};
    for (unsigned i = 0; i < 4; ++i) {
        EXPECT_EQ(inDegree[order[i]], 0u)
            << "node " << order[i] << " should be ready at step " << i;
        for (unsigned successor : graph[order[i]]) --inDegree[successor];
    }
}

TEST(HardSchedulingConstraintMergeTest, SkipsCycleAndPreservesScheduleCompleteness) {
    std::vector<std::unordered_set<unsigned>> graph(3);
    graph[0].insert(1);
    graph[1].insert(2);
    std::vector<unsigned> inDegree{0, 1, 1};

    EXPECT_FALSE(addHardConstraint(graph, inDegree, 2, 0));  // Would close 0 -> 1 -> 2 -> 0.
    EXPECT_EQ(inDegree, (std::vector<unsigned>{0, 1, 1}));   // Rejected edge left no trace.

    unsigned scheduled = 0;
    for (unsigned node = 0; node < 3; ++node) {
        ASSERT_EQ(inDegree[node], 0u);
        ++scheduled;
        for (unsigned successor : graph[node]) --inDegree[successor];
    }
    EXPECT_EQ(scheduled, 3u);
}

TEST(HardSchedulingConstraintMergeTest, WaitsForBaseDagAndHardConstraintReadiness) {
    // C requires both the base edge A -> C and the independent hard link B -> C.
    std::vector<std::unordered_set<unsigned>> graph(3);
    graph[0].insert(2);
    std::vector<unsigned> inDegree{0, 0, 1};
    ASSERT_TRUE(addHardConstraint(graph, inDegree, 1, 2));
    EXPECT_EQ(inDegree[2], 2u);

    --inDegree[2];  // A completes first.
    EXPECT_NE(inDegree[2], 0u);
    --inDegree[2];  // B completes.
    EXPECT_EQ(inDegree[2], 0u);
}

class DAGSchedulerPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    std::unique_ptr<Pass> pass;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        func = std::make_unique<Function>("dag_sched_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
        pass = createStinkyDAGSchedulerPass();
        registerAllAnalyses(am);
    }

    void TearDown() override {
        pass.reset();
        func.reset();
        bb = nullptr;
    }

    void runPass() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        pass->run(*func, ctx, am);
    }

    void runPassWithUnrollGemm() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        ctx.setPassFeatureConfig(pfc);
        pass->run(*func, ctx, am);
    }

    // Run with the tensor_load_to_lds credit-pool throttle enabled.
    // distributeGlobalRead routes tensor loads into globalReadQueue; depth/latency
    // configure the in-flight credit pool.
    void runPassWithGlobalReadThrottle(int depth, int drainLatency) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        pfc.dagFeatures.distributeGlobalRead = true;
        pfc.dagFeatures.globalReadQueueDepth = depth;
        pfc.dagFeatures.globalReadDrainLatency = drainLatency;
        ctx.setPassFeatureConfig(pfc);
        pass->run(*func, ctx, am);
    }

    // Run with ds_read queue-depth + throttled-issue controls enabled.
    // perWmma is held generously high by default so the separate per-WMMA-window
    // ds cap never binds. throttleLatency drives queue-full pacing; drainLatency is
    // kept for paths that still model data-return/drain behavior (e.g. barrier timing).
    void runPassWithDsReadThrottle(int queueDepth, int throttleLatency, int perWmma = 100,
                                   int drainLatency = -1) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        pfc.dagFeatures.dsReadQueueDepth = queueDepth;
        pfc.dagFeatures.dsReadThrottleLatency = throttleLatency;
        if (drainLatency <= 0) drainLatency = throttleLatency;
        pfc.dagFeatures.dsReadDrainLatency = drainLatency;
        pfc.dagFeatures.dsReadPerWmma = perWmma;
        ctx.setPassFeatureConfig(pfc);
        pass->run(*func, ctx, am);
    }

    // Linearized mnemonic order of a block (skips PHIs and non-Stinky IR).
    static std::vector<std::string> mnemonicSequence(const BasicBlock& block) {
        std::vector<std::string> seq;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            const HwInstDesc* hw = inst->getHwInstDesc();
            if (!hw || !hw->mnemonic) continue;
            seq.push_back(hw->mnemonic);
        }
        return seq;
    }

    // Largest run of consecutive tensor_load_to_lds in a mnemonic sequence.
    static int maxConsecutiveTensorLoads(const std::vector<std::string>& seq) {
        int run = 0, best = 0;
        for (const std::string& m : seq) {
            if (m == "tensor_load_to_lds") {
                run++;
                best = std::max(best, run);
            } else {
                run = 0;
            }
        }
        return best;
    }

    // Largest run of consecutive ds_load_b128 in a mnemonic sequence
    // (gfx1250's actual mnemonic for this op; see Gfx1250Instructions.def).
    static int maxConsecutiveDsReads(const std::vector<std::string>& seq) {
        int run = 0, best = 0;
        for (const std::string& m : seq) {
            if (m == "ds_load_b128") {
                run++;
                best = std::max(best, run);
            } else {
                run = 0;
            }
        }
        return best;
    }

    // Build a single-BB self-loop so the scheduler uses the loop-aware CDNA5 path.
    // Returns the loop body BB with the branch already appended.
    BasicBlock* buildLoopBB(const char* label = "loop_body") {
        BasicBlock* body = func->createBasicBlock(label);
        body->addSuccessor(body);
        return body;
    }

    static StinkyInstruction* createSCbranchInBlock(BasicBlock* bb, GfxArchID arch) {
        AsmIRBuilder builder(*bb, arch);
        return builder.create(getMCIDByUOp(GFX::s_cbranch_scc0, arch));
    }

    StinkyInstruction* createWmmaF32_16x16x16_bf16_in(BasicBlock* targetBB, int destStart,
                                                      int src0Start) {
        AsmIRBuilder builder(*targetBB, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("v", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", destStart, 8));
        return inst;
    }

    StinkyInstruction* createWmmaF32_16x16x16_bf16(int destStart, int src0Start) {
        return createWmmaF32_16x16x16_bf16_in(bb, destStart, src0Start);
    }

    // v_wmma_scale_f32_16x16x128_f8f6f4 with F8 (FP8) input matrix formats. Its
    // cost is {issue=1, latency=8}, so the co-issue latency window stays open
    // right after issue, which is what exercises the co-exec hazard gate.
    // src VGPRs: v[src0Start:src0Start+8) (src0/src1) and v[destStart:destStart+8) (acc).
    StinkyInstruction* createWmmaScaleF8_in(BasicBlock* targetBB, int destStart, int src0Start) {
        AsmIRBuilder builder(*targetBB, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_scale_f32_16x16x128_f8f6f4, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("v", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", destStart, 8));
        MatrixFmtModifiers fmtMod;
        fmtMod.fmtA = MatrixFmt::FP8;
        fmtMod.fmtB = MatrixFmt::FP8;
        inst->addModifier(fmtMod);
        return inst;
    }

    StinkyInstruction* createWmmaScaleF8(int destStart, int src0Start) {
        return createWmmaScaleF8_in(bb, destStart, src0Start);
    }

    StinkyInstruction* createMovableDsLoad(int destReg, int addrReg, int ldsToken) {
        StinkyInstruction* inst = createDsReadB128InBlock(bb, arch, destReg, addrReg);
        inst->addSrcReg(StinkyRegister(RegType::LDS, ldsToken, 1));
        return inst;
    }

    // A tensor_load_to_lds with an LDS pseudo dest-reg so the DAG scheduler treats
    // it as movable (without LDS pseudo-regs hasSideEffect() makes it a region
    // boundary and it never enters globalReadQueue). Mirrors createMovableDsLoad.
    StinkyInstruction* createMovableTensorLoad(BasicBlock* targetBB, int src0Reg, int src1Reg,
                                               int ldsToken) {
        StinkyInstruction* inst = createTensorLoadInBlock(targetBB, arch, src0Reg, src1Reg);
        inst->addDestReg(StinkyRegister(RegType::LDS, ldsToken, 1));
        return inst;
    }

    // global_prefetch_b8 (gfx1250 gl2-prefetch): reads vaddr = v[vaddrReg:vaddrReg+2)
    // (64-bit vgpr) and saddr = s[saddrReg:saddrReg+2) (64-bit sreg). No destination,
    // not HasSideEffect, so the scheduler treats it as a movable op. Used to exercise
    // the ValuVgprToVmemAddr hazard rule against a prefetch consumer.
    StinkyInstruction* createGlobalPrefetchB8(BasicBlock* targetBB, int vaddrReg, int saddrReg) {
        AsmIRBuilder builder(*targetBB, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::global_prefetch_b8, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addSrcReg(StinkyRegister("v", vaddrReg, 2));
        inst->addSrcReg(StinkyRegister("s", saddrReg, 2));
        return inst;
    }

    // Run with the cluster-barrier SCC rule on/off. distributeGlobalRead mirrors the
    // gfx1250 pipeline so tensor loads take their normal queue.
    void runPassWithClusterBarrier(bool clusterBarrier) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        pfc.dagFeatures.distributeGlobalRead = true;
        pfc.dagFeatures.clusterBarrier = clusterBarrier;
        ctx.setPassFeatureConfig(pfc);
        if (testDumpEnabled()) {
            std::cerr << "\n=== INPUT (clusterBarrier=" << (clusterBarrier ? "on" : "off")
                      << "):" << scheduleOrder(*bb) << "\n";
        }
        pass->run(*func, ctx, am);
        if (testDumpEnabled()) {
            std::cerr << "\n=== OUTPUT (clusterBarrier=" << (clusterBarrier ? "on" : "off")
                      << "):" << scheduleOrder(*bb) << "\n";
        }
    }

    static bool testDumpEnabled() {
        static const bool enabled = std::getenv("STINKY_TEST_DUMP") != nullptr;
        return enabled;
    }

    // An `s_barrier_signal -1` / `s_barrier_wait -1` pair carrying LDS pseudo-regs, so
    // the DAG scheduler treats it as movable instead of a region boundary. Mirrors what
    // StinkyBuildImplicitDependencyPass derives from MemTokenData.
    std::pair<StinkyInstruction*, StinkyInstruction*> createMovableWorkgroupBarrier(
        BasicBlock* targetBB, int ldsToken) {
        AsmIRBuilder builder(*targetBB, arch);
        auto make = [&](GFX uop) {
            StinkyInstruction* inst = builder.create(getMCIDByUOp(uop, arch));
            inst->addSrcReg(StinkyRegister(-1));  // all-wave (workgroup) split barrier
            inst->addSrcReg(StinkyRegister(RegType::LDS, ldsToken, 1));
            inst->addDestReg(StinkyRegister(RegType::LDS, ldsToken, 1));
            return inst;
        };
        StinkyInstruction* signal = make(GFX::s_barrier_signal);
        StinkyInstruction* wait = make(GFX::s_barrier_wait);
        return {signal, wait};
    }

    // `s_cmp_eq_u32 s<srcSgpr>, 0` with its implicit SCC dest attached (the scheduler
    // test path does not run StinkyBuildImplicitDependencyPass).
    StinkyInstruction* createSCmpWritingScc(BasicBlock* targetBB, int srcSgpr) {
        AsmIRBuilder builder(*targetBB, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        inst->addSrcReg(StinkyRegister(0));
        inst->addDestReg(StinkyRegister::getSCCRegister());
        return inst;
    }

    StinkyInstruction* createSCbranchReadingScc(BasicBlock* targetBB) {
        AsmIRBuilder builder(*targetBB, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cbranch_scc0, arch));
        inst->addSrcReg(StinkyRegister::getSCCRegister());
        return inst;
    }

    // `s_sub_u32 s<sgpr>, s<sgpr>, 1` -- writes an SGPR *and* SCC (carry-out). The loop
    // counter decrement: its SGPR result is read across the back edge while its SCC
    // result is dead, killed by the compare that follows.
    StinkyInstruction* createSSubWritingSgprAndScc(BasicBlock* targetBB, int sgpr) {
        AsmIRBuilder builder(*targetBB, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_sub_u32, arch));
        inst->addDestReg(StinkyRegister("s", sgpr, 1));
        inst->addDestReg(StinkyRegister::getSCCRegister());
        inst->addSrcReg(StinkyRegister("s", sgpr, 1));
        inst->addSrcReg(StinkyRegister(1));
        return inst;
    }

    // `s_cselect_b32 s<dst>, s<src>, 0` -- an SCC reader that is ordinary SALU work, so
    // unlike a branch the scheduler is free to move it anywhere its deps allow.
    StinkyInstruction* createSCselectReadingScc(BasicBlock* targetBB, int destSgpr, int srcSgpr) {
        AsmIRBuilder builder(*targetBB, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cselect_b32, arch));
        inst->addDestReg(StinkyRegister("s", destSgpr, 1));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        inst->addSrcReg(StinkyRegister(0));
        inst->addSrcReg(StinkyRegister::getSCCRegister());
        return inst;
    }

    // The invariant the cluster-barrier SCC rule enforces: no workgroup barrier may sit
    // between the first and last scheduled member of an SCC def-use chain. Which side of
    // the barrier the chain ends up on is deliberately not constrained.
    static bool barrierSplitsChain(const BasicBlock& block,
                                   const std::vector<StinkyInstruction*>& chain) {
        int lo = -1, hi = -1, idx = 0;
        std::vector<int> barrierPositions;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (isBarrier(*inst)) barrierPositions.push_back(idx);
            for (const StinkyInstruction* member : chain) {
                if (inst != member) continue;
                if (lo < 0) lo = idx;
                hi = idx;
            }
            idx++;
        }
        for (int pos : barrierPositions)
            if (pos > lo && pos < hi) return true;
        return false;
    }

    // Scheduled index of the first `s_barrier_signal`, or -1.
    static int firstBarrierSignalPosition(const BasicBlock& block) {
        int idx = 0;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (isBarrier(*inst) && isBarrierSignal(*inst)) return idx;
            idx++;
        }
        return -1;
    }

    // Scheduled order as indexed IR lines, for assertion failure messages and dumps.
    static std::string scheduleOrder(const BasicBlock& block) {
        std::ostringstream os;
        int idx = 0;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            os << "\n  " << idx++ << ": ";
            cast<StinkyInstruction>(&ir)->dump(os);
        }
        return os.str();
    }

    // Scheduled index of \p target, or -1.
    static int positionOf(const BasicBlock& block, const StinkyInstruction* target) {
        int idx = 0;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (cast<StinkyInstruction>(&ir) == target) return idx;
            idx++;
        }
        return -1;
    }

    // Estimated cycles the schedule spends between \p from and \p to, counting neither
    // end. Same scale the scheduler plans on (a WMMA costs the co-issue window it opens,
    // anything else its issue cycles), so a distance measured here is comparable to the
    // cycle leads the passes are written against. -1 if the two are not in this order.
    static int cyclesBetween(const BasicBlock& block, const StinkyInstruction* from,
                             const StinkyInstruction* to) {
        int total = 0;
        bool started = false;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (inst == to) return started ? total : -1;
            if (started)
                total += isMatrixInstruction(*inst) ? inst->latencyCycles : inst->issueCycles;
            if (inst == from) started = true;
        }
        return -1;
    }

    // Scheduled index of the last `s_barrier_wait`, or -1.
    static int lastBarrierWaitPosition(const BasicBlock& block) {
        int idx = 0, last = -1;
        for (const IRBase& ir : block) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (isBarrier(*inst) && isBarrierWait(*inst)) last = idx;
            idx++;
        }
        return last;
    }

    StinkyInstruction* createExecNarrow(int srcSgpr) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
        inst->addDestReg(StinkyRegister::getEXECRegister(32));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        return inst;
    }

    StinkyInstruction* createExecReset() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
        inst->addDestReg(StinkyRegister::getEXECRegister(32));
        inst->addSrcReg(StinkyRegister(-1));
        return inst;
    }
};

// Integration check: collapse + schedule + expand together, through the real pass.
// See ExecMaskGroupingTest.cpp for isolated collapse/expand tests, and the
// ExecMaskGroup_* tests below for whether the scheduler treats a group as atomic.
TEST_F(DAGSchedulerPassTest, ExecMaskedRegion_PreservesSpanAndOrder) {
    createVAddInBlock(bb, arch, 40, 41, 42);
    createExecNarrow(10);
    createVAddInBlock(bb, arch, 0, 1, 2);
    createVAddInBlock(bb, arch, 3, 4, 5);
    createVAddInBlock(bb, arch, 6, 7, 8);
    createExecReset();
    createVAddInBlock(bb, arch, 50, 51, 52);

    const int n = countStinkyInstructions(*bb);
    runPass();

    EXPECT_EQ(countStinkyInstructions(*bb), n);
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        EXPECT_NE(cast<StinkyInstruction>(&ir)->getUnifiedOpcode(), GFX::EXEC_GROUP);
    }

    std::vector<int> destSeq;
    std::vector<bool> isExecWriteSeq;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (inst->getUnifiedOpcode() != GFX::s_mov_b32 &&
            inst->getUnifiedOpcode() != GFX::v_add_f32)
            continue;
        ASSERT_FALSE(inst->getDestRegs().empty());
        destSeq.push_back(static_cast<int>(inst->getDestReg(0).reg.idx));
        isExecWriteSeq.push_back(inst->getDestReg(0).reg.type == RegType::EXEC_LO);
    }

    ASSERT_EQ(destSeq.size(), 7u);
    EXPECT_EQ(destSeq[0], 40);
    EXPECT_TRUE(isExecWriteSeq[1]);
    EXPECT_EQ(destSeq[2], 0);
    EXPECT_EQ(destSeq[3], 3);
    EXPECT_EQ(destSeq[4], 6);
    EXPECT_TRUE(isExecWriteSeq[5]);
    EXPECT_EQ(destSeq[6], 50);
}

// Layer 2: does the scheduler treat a hand-built ExecMaskGroup (bypassing
// collapseExecMaskedRegions() entirely) as a single atomic node?

// runPass() always runs collapseExecMaskedRegions()/expandExecMaskedGroups() around
// scheduling (see StinkyDAGSchedulerPass::run()'s scheduleBlock lambda), so any node
// with GFX::EXEC_GROUP -- hand-built or not -- gets unzipped via its ExecGroupData at
// the end. So a hand-built group under a real pass run needs a real (if minimal)
// ExecGroupData child to unzip into; that child's own registers are irrelevant here
// since ordering is driven by the group's own declared src/dest, set explicitly below.
TEST_F(DAGSchedulerPassTest, ExecMaskGroup_TreatedAsSingleAtomicNode) {
    createVAddInBlock(bb, arch, 20, 21, 22);
    StinkyInstruction* consumer = createVAddInBlock(bb, arch, 40, 30, 31);

    StinkyInstruction* child = createVAddInBlock(bb, arch, 60, 61, 62);
    bb->removeIR(child);

    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* group = builder.createExecMaskGroup(consumer);
    group->addSrcReg(StinkyRegister("v", 20, 1));
    group->addDestReg(StinkyRegister("v", 30, 1));
    group->issueCycles = 4;
    group->latencyCycles = 4;
    group->addModifier<ExecGroupData>(ExecGroupData{{child}});

    runPass();

    std::vector<int> destSeq;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        ASSERT_FALSE(inst->getDestRegs().empty());
        destSeq.push_back(static_cast<int>(inst->getDestReg(0).reg.idx));
    }
    ASSERT_EQ(destSeq.size(), 3u);
    EXPECT_EQ(destSeq[0], 20);  // producer
    EXPECT_EQ(destSeq[1], 60);  // group's child, unzipped in its place
    EXPECT_EQ(destSeq[2], 40);  // consumer
}

TEST_F(DAGSchedulerPassTest, ExecMaskGroup_NotMisclassified) {
    StinkyInstruction* anchor = createVAddInBlock(bb, arch, 0, 1, 2);

    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* group = builder.createExecMaskGroup(anchor);
    group->addModifier<ExecGroupData>(ExecGroupData{{}});

    EXPECT_TRUE(isExecMaskGroup(*group));
    EXPECT_FALSE(isMatrixInstruction(*group));
    EXPECT_FALSE(isDSRead(*group));
    EXPECT_FALSE(isDSWrite(*group));
    EXPECT_FALSE(isBarrier(*group));
    EXPECT_FALSE(isVectorALU(*group));
    EXPECT_FALSE(isTensorLoad(*group));
    EXPECT_FALSE(hasSideEffect(*group));

    runPassWithUnrollGemm();
    // The group has no children to unzip into, so it's simply dropped; only the
    // anchor v_add remains.
    EXPECT_EQ(countStinkyInstructions(*bb), 1);
}

TEST_F(DAGSchedulerPassTest, ExecMaskGroup_InheritsSideEffectFromChildren) {
    StinkyInstruction* sideEffecting = createDSWriteInBlock(bb, arch, 0, 1);  // no MemTokenData
    bb->removeIR(sideEffecting);

    StinkyInstruction* anchor = createVAddInBlock(bb, arch, 0, 1, 2);
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* group = builder.createExecMaskGroup(anchor);
    group->addModifier<ExecGroupData>(ExecGroupData{{sideEffecting}});

    EXPECT_TRUE(hasSideEffect(*group));
}

// Empty block: pass should not crash
TEST_F(DAGSchedulerPassTest, EmptyBlock_DoesNotCrash) {
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), 0);
}

// Single instruction: pass should not crash
TEST_F(DAGSchedulerPassTest, SingleInstruction_DoesNotCrash) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    int n = countStinkyInstructions(*bb);
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), n);
}

// A few independent instructions: pass should not crash, count unchanged
TEST_F(DAGSchedulerPassTest, IndependentInstructions_DoesNotCrash) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    createVAddInBlock(bb, arch, 3, 4, 5);
    createVAddInBlock(bb, arch, 6, 7, 8);
    int n = countStinkyInstructions(*bb);
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), n);
}

// Chain of dependencies: pass should not crash, count unchanged
TEST_F(DAGSchedulerPassTest, DependentInstructions_DoesNotCrash) {
    createVAddInBlock(bb, arch, 0, 1, 2);  // v0 = v1 + v2
    createVAddInBlock(bb, arch, 3, 0, 4);  // v3 = v0 + v4
    createVAddInBlock(bb, arch, 5, 3, 6);  // v5 = v3 + v6
    int n = countStinkyInstructions(*bb);
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), n);
}

// DS reads + WMMAs: scheduler must not issue WMMAs back-to-back when other instructions exist.
// With real ds_load latency, WMMAs are not latency-free until ds_reads are issued and latency
// elapses, so we get: 4 ds_load, then 2 wmma. The rule "lastPickedWasWMMA => prefer other"
// ensures that when both WMMA and other are ready we interleave (no consecutive WMMAs).
TEST_F(DAGSchedulerPassTest, DSReadAndWMMA_NoConsecutiveWMMA) {
    const int addrReg = 24;
    createDsReadB128InBlock(bb, arch, 8, addrReg);                 // v[8:11]
    createDsReadB128InBlock(bb, arch, 12, addrReg);                // v[12:15]
    createDsReadB128InBlock(bb, arch, 16, addrReg);                // v[16:19]
    createDsReadB128InBlock(bb, arch, 20, addrReg);                // v[20:23]
    StinkyInstruction* wmma1 = createWmmaF32_16x16x16_bf16(0, 8);  // v[0:7] v[8:15] v[8:15] v[0:7]
    StinkyInstruction* wmma2 =
        createWmmaF32_16x16x16_bf16(0, 16);  // v[0:7] v[16:23] v[16:23] v[0:7]
    ASSERT_NE(wmma1, nullptr);
    ASSERT_NE(wmma2, nullptr);

    runPassWithUnrollGemm();

    // With real latency, all 4 ds_load are issued first, then 2 wmma. No two WMMAs in a row.
    std::vector<std::pair<std::string, int>> sequence;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction* inst = cast<StinkyInstruction>(&ir);
        const char* mnem = inst->getHwInstDesc() ? inst->getHwInstDesc()->mnemonic : nullptr;
        if (!mnem) continue;
        if (std::string(mnem) == "ds_load_b128") {
            if (!inst->getDestRegs().empty() && inst->getDestRegs()[0].isRegister())
                sequence.push_back(
                    {"ds_load_b128", static_cast<int>(inst->getDestRegs()[0].reg.idx)});
        } else if (std::string(mnem) == "v_wmma_f32_16x16x16_bf16") {
            if (!inst->getSrcRegs().empty() && inst->getSrcRegs()[0].isRegister())
                sequence.push_back(
                    {"v_wmma_f32_16x16x16_bf16", static_cast<int>(inst->getSrcRegs()[0].reg.idx)});
        }
    }

    ASSERT_EQ(sequence.size(), 6u) << "Expected 6 instructions (4 ds_load + 2 wmma)";
    // All 4 ds_load first (real latency: WMMAs not ready until latency elapses)
    EXPECT_EQ(sequence[0].first, "ds_load_b128");
    EXPECT_EQ(sequence[0].second, 8);
    EXPECT_EQ(sequence[1].first, "ds_load_b128");
    EXPECT_EQ(sequence[1].second, 12);
    EXPECT_EQ(sequence[2].first, "ds_load_b128");
    EXPECT_EQ(sequence[2].second, 16);
    EXPECT_EQ(sequence[3].first, "ds_load_b128");
    EXPECT_EQ(sequence[3].second, 20);
    // Then 2 wmma; rule ensures we never issue two WMMAs in a row when other work exists
    EXPECT_EQ(sequence[4].first, "v_wmma_f32_16x16x16_bf16");
    EXPECT_EQ(sequence[4].second, 8);
    EXPECT_EQ(sequence[5].first, "v_wmma_f32_16x16x16_bf16");
    EXPECT_EQ(sequence[5].second, 16);
    // When other instructions exist, scheduler prefers them after a WMMA (no back-to-back WMMA).
    // Here with real latency only WMMAs are left at the end so they are issued consecutively.
}

// ---------------------------------------------------------------------------
// Property: when all independent, WMMA fires first (Phase B), then ds_loads
// and VALU fill the WMMA latency window.
// Within that window, ds_load has priority over VALU.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, IndependentWMMAFirst_ThenDsThenVALU) {
    const int addrReg = 80;
    // 3 independent ds_loads (LDS pseudo-reg makes them movable in DAG)
    createMovableDsLoad(0, addrReg, 1);
    createMovableDsLoad(4, addrReg, 2);
    createMovableDsLoad(8, addrReg, 3);
    // 3 independent VALUs
    createVAddInBlock(bb, arch, 40, 41, 42);
    createVAddInBlock(bb, arch, 43, 44, 45);
    createVAddInBlock(bb, arch, 46, 47, 48);
    // 1 independent WMMA
    createWmmaF32_16x16x16_bf16(12, 50);

    runPassWithUnrollGemm();

    int firstWmmaPos = -1;
    int firstDsPos = -1;
    int firstValuPos = -1;
    int pos = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        if (mnem.find("wmma") != std::string_view::npos) {
            if (firstWmmaPos < 0) firstWmmaPos = pos;
        } else if (mnem.find("ds_load") != std::string_view::npos) {
            if (firstDsPos < 0) firstDsPos = pos;
        } else if (mnem.find("v_add") != std::string_view::npos) {
            if (firstValuPos < 0) firstValuPos = pos;
        }
        pos++;
    }

    ASSERT_GE(firstWmmaPos, 0) << "No WMMA found";
    ASSERT_GE(firstDsPos, 0) << "No ds_load found";
    ASSERT_GE(firstValuPos, 0) << "No VALU found";
    EXPECT_LT(firstWmmaPos, firstDsPos) << "WMMA should fire before ds_load (Phase B)";
    EXPECT_LT(firstDsPos, firstValuPos)
        << "DS loads should be prioritized before VALU during WMMA latency";
}

// ---------------------------------------------------------------------------
// Co-execution hazard (regression test for destOverlapsActiveWmmaSrc):
// a ds_load whose dest VGPRs overlap the in-flight WMMA's src VGPRs must NOT be
// issued inside that WMMA's latency window, because the load could clobber a
// source register the WMMA is still reading.
//
// Setup: WMMA #0 reads v[50:58); the ds_load writes v[52:56) (overlap). Four
// more independent WMMAs (disjoint registers) are available. While WMMA #0 is
// in flight the ds_load is held back by the hazard gate, so the scheduler
// issues the next independent WMMA (D#100) first and only then the ds_load,
// once a WMMA whose sources it does not touch is the active one:
//
//   wmma D#12  ->  wmma D#100  ->  ds_load D#52  ->  wmma D#108/116/124
//
// Without the hazard gate the ds_load would issue right after WMMA #0
// (wmma D#12 -> ds_load -> wmma D#100 -> ...), clobbering v[52:56) mid-read.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, WmmaSrcOverlap_HazardDsLoadDeferredPastWindow) {
    const int addrReg = 80;
    // F8 MX WMMA fires first (Phase B). Its src VGPRs are v[50:58) (src0/src1)
    // and v[12:20) (acc); see createWmmaScaleF8. cost latency=8 keeps the
    // co-issue window open so the hazard gate is exercised.
    createWmmaScaleF8(/*destStart=*/12, /*src0Start=*/50);
    // ds_load dest v[52:56) overlaps the WMMA's src0 v[50:58): co-exec hazard.
    createMovableDsLoad(/*destReg=*/52, addrReg, /*ldsToken=*/1);
    // Independent WMMAs (registers disjoint from the hazard pair and from each
    // other) to fill the latency window ahead of the deferred ds_load.
    for (int i = 0; i < 4; i++)
        createWmmaScaleF8(/*destStart=*/100 + i * 8, /*src0Start=*/200 + i * 8);

    int beforeCount = countStinkyInstructions(*bb);
    runPassWithUnrollGemm();
    EXPECT_EQ(countStinkyInstructions(*bb), beforeCount)
        << "hazard deferral must not drop instructions";

    // Collect (mnemonic-kind, first-dest-vgpr) in scheduled order.
    std::vector<std::pair<std::string, int>> seq;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        std::string kind = mnem.find("wmma") != std::string_view::npos      ? "wmma"
                           : mnem.find("ds_load") != std::string_view::npos ? "ds"
                                                                            : std::string(mnem);
        int dst = (!inst->getDestRegs().empty() && inst->getDestRegs()[0].isRegister())
                      ? static_cast<int>(inst->getDestRegs()[0].reg.idx)
                      : -1;
        seq.push_back({kind, dst});
    }

    const std::vector<std::pair<std::string, int>> expected = {
        {"wmma", 12}, {"wmma", 100}, {"ds", 52}, {"wmma", 108}, {"wmma", 116}, {"wmma", 124},
    };
    EXPECT_EQ(seq, expected)
        << "hazardous ds_load must be deferred until an independent WMMA (D#100) has "
           "issued; it must not co-issue inside WMMA D#12's latency window";
}

// ---------------------------------------------------------------------------
// Co-execution hazard, VALU variant (regression test for destOverlapsActiveWmmaSrc
// on the VALU path): a VALU whose dest VGPR overlaps the in-flight WMMA's src VGPRs
// must NOT be issued inside that WMMA's latency window, because it could clobber a
// source register the WMMA is still reading.
//
// Unlike the ds_load variant, a VALU only becomes co-issue pickable at the positions
// set in the WMMA's co-issue window (MXWMMA_SCALE = 0x00C0, i.e. positions 6/7). Right
// after issue the position is 1, where isValuPickable() is already false, so extra
// non-hazardous fillers are needed to advance the co-issue timeline into a pickable
// position while WMMA #0 is still in flight. That is exactly the moment the hazard gate
// must fire:
//   - 3 non-hazardous ds_loads (v[300:], v[320:], v[340:]) fill the per-WMMA DS cap and
//     advance positions 1 -> 4.
//   - 3 independent scalar ops advance positions 4 -> 7; at position 6 the VALU becomes
//     co-issue pickable while WMMA #0 (v[50:58)) is still the active window.
//
// With the gate the hazardous VALU (dst v52) is skipped at position 6 (inside WMMA #0's
// window) and deferred until that window closes; it then issues right after D#100 opens a
// non-overlapping window. Without the gate it would co-issue at position 6, clobbering v52.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, WmmaSrcOverlap_HazardValuDeferredPastWindow) {
    const int addrReg = 400;
    auto createScalarOp = [&](int dst, int src0, int src1) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_add_u32, arch));
        inst->addDestReg(StinkyRegister("s", dst, 1));
        inst->addSrcReg(StinkyRegister("s", src0, 1));
        inst->addSrcReg(StinkyRegister("s", src1, 1));
        return inst;
    };
    // F8 MX WMMA fires first (Phase B). Its src VGPRs are v[50:58) (src0/src1) and
    // v[12:20) (acc); latency=8 keeps the co-issue window open so the gate is exercised.
    createWmmaScaleF8(/*destStart=*/12, /*src0Start=*/50);
    // Hazardous VALU: dst v52 overlaps the WMMA's src0 v[50:58): co-exec hazard.
    createVAddInBlock(bb, arch, /*destReg=*/52, /*src0Reg=*/60, /*src1Reg=*/61);
    // Non-hazardous fillers to advance the co-issue timeline into a VALU-pickable
    // position (6) while WMMA #0 is still in flight.
    createMovableDsLoad(/*destReg=*/300, addrReg, /*ldsToken=*/1);
    createMovableDsLoad(/*destReg=*/320, addrReg, /*ldsToken=*/2);
    createMovableDsLoad(/*destReg=*/340, addrReg, /*ldsToken=*/3);
    createScalarOp(/*dst=*/10, 11, 12);
    createScalarOp(/*dst=*/13, 14, 15);
    createScalarOp(/*dst=*/16, 17, 18);
    // Independent WMMAs (registers disjoint from the hazard pair and each other) to fill
    // the latency windows ahead of the deferred VALU.
    for (int i = 0; i < 4; i++)
        createWmmaScaleF8(/*destStart=*/100 + i * 16, /*src0Start=*/200 + i * 16);

    int beforeCount = countStinkyInstructions(*bb);
    runPassWithUnrollGemm();
    EXPECT_EQ(countStinkyInstructions(*bb), beforeCount)
        << "hazard deferral must not drop instructions";

    // Collect (mnemonic-kind, first-dest-reg) in scheduled order.
    std::vector<std::pair<std::string, int>> seq;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        std::string kind = mnem.find("wmma") != std::string_view::npos      ? "wmma"
                           : mnem.find("ds_load") != std::string_view::npos ? "ds"
                           : mnem.rfind("v_", 0) == 0                       ? "valu"
                           : mnem.rfind("s_", 0) == 0                       ? "s"
                                                                            : std::string(mnem);
        int dst = (!inst->getDestRegs().empty() && inst->getDestRegs()[0].isRegister())
                      ? static_cast<int>(inst->getDestRegs()[0].reg.idx)
                      : -1;
        seq.push_back({kind, dst});
    }

    // Hazardous VALU (dst v52) is deferred past WMMA D#12's window, then issues right after
    // the first independent WMMA (D#100), whose window no longer overlaps v52.
    const std::vector<std::pair<std::string, int>> expected = {
        {"wmma", 12}, {"ds", 300},   {"ds", 320},  {"ds", 340},   {"s", 10},     {"s", 13},
        {"s", 16},    {"wmma", 100}, {"valu", 52}, {"wmma", 116}, {"wmma", 132}, {"wmma", 148},
    };
    EXPECT_EQ(seq, expected)
        << "hazardous VALU (dst v52) must not co-issue inside WMMA D#12's latency window; "
           "it must be deferred until that window closes, then issue once a non-overlapping "
           "WMMA window is open";
}

// ---------------------------------------------------------------------------
// Hidden-stall window fill (pickFreeBest allowHiddenStall path): a SALU that is
// only blocked by a src RAW hazard whose remaining wait fits under the active
// WMMA's latency shadow may be co-issued *inside* that window — the stall we pay
// waiting for its src is hidden by the in-flight WMMA, so it costs no extra
// cycles. It must therefore be preferred over starting the next independent WMMA.
//
// Setup (region, not a loop, so no loop-head deferral):
//   - WMMA #0 (v[12:20)) fires first (Phase B) and opens an 8-cycle window.
//   - A chain of inter-dependent SALUs a0 -> a1 -> a2 -> a3, each writing s(100+i)
//     with latency=2 > issue=1 so issuing it stamps a 1-cycle data-ready latency
//     on its dest (the src RAW gate for the next link). a0 is free; a1..a3 are
//     each RAW-blocked for 1 cycle, which fits under WMMA #0's remaining latency
//     shadow, so every link is a valid hidden-stall fill. The chain is strict, so
//     only one link is ready at a time — their relative order is forced by the
//     DAG; the test is purely about whether each link lands inside the window.
//   - WMMA #1 (v[200:208)) is independent and ready.
//
// Expected (new behavior):  wmma#0, a0, a1, a2, a3, wmma#1
//   Every chain link is co-issued inside wmma#0's window, ahead of wmma#1.
// Old behavior would issue wmma#1 as soon as a0's consumer was RAW-blocked
// (wmma#0, a0, wmma#1, a1, a2, a3), because a RAW-blocked SALU was never pickable
// inside the window.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, HiddenStallSaluFillsWmmaWindowBeforeNextWmma) {
    auto createScalarAdd = [&](int dst, int src0, int src1) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_add_u32, arch));
        inst->addDestReg(StinkyRegister("s", dst, 1));
        inst->addSrcReg(StinkyRegister("s", src0, 1));
        inst->addSrcReg(StinkyRegister("s", src1, 1));
        return inst;
    };

    // WMMA #0 fires first (Phase B), latency=8 keeps its co-issue window open.
    createWmmaScaleF8(/*destStart=*/12, /*src0Start=*/50);
    // Chain a0 -> a1 -> a2 -> a3: a_i writes s(100+i), a_(i+1) reads it (RAW).
    // Each 1-cycle wait fits the shrinking window (positions 2,4,6,8), so all four
    // are hidden-stall filled inside WMMA #0's window.
    const int kChain = 4;
    for (int i = 0; i < kChain; i++) {
        const int src0 = (i == 0) ? 0 : (100 + i - 1);  // previous link's dest
        StinkyInstruction* a = createScalarAdd(/*dst=*/100 + i, src0, /*src1=*/1);
        a->issueCycles = 1;
        a->latencyCycles = 2;
    }
    // WMMA #1: independent (disjoint regs) and ready.
    createWmmaScaleF8(/*destStart=*/200, /*src0Start=*/220);

    int beforeCount = countStinkyInstructions(*bb);
    runPassWithUnrollGemm();
    EXPECT_EQ(countStinkyInstructions(*bb), beforeCount)
        << "hidden-stall fill must not drop instructions";

    std::vector<std::pair<std::string, int>> seq;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        std::string kind = mnem.find("wmma") != std::string_view::npos ? "wmma"
                           : mnem.rfind("s_", 0) == 0                  ? "s"
                                                                       : std::string(mnem);
        int dst = (!inst->getDestRegs().empty() && inst->getDestRegs()[0].isRegister())
                      ? static_cast<int>(inst->getDestRegs()[0].reg.idx)
                      : -1;
        seq.push_back({kind, dst});
    }

    const std::vector<std::pair<std::string, int>> expected = {
        {"wmma", 12}, {"s", 100}, {"s", 101}, {"s", 102}, {"s", 103}, {"wmma", 200},
    };
    EXPECT_EQ(seq, expected)
        << "every link of the RAW-dependent SALU chain must be co-issued inside WMMA #0's latency "
           "window (each 1-cycle wait hidden by the in-flight WMMA), ahead of the independent "
           "WMMA #1";
}

// ---------------------------------------------------------------------------
// Property: per-WMMA-window DS cap — after a WMMA fires,
// at most floor((latency - issue) / 2) = 3 ds_loads can issue in its window
// because back-to-back ds_load issue cost doubles.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, DSWindowCap_VALUInterleaveAfter3) {
    const int addrReg = 80;
    // 5 independent ds_loads (LDS pseudo-reg makes them movable in DAG)
    createMovableDsLoad(0, addrReg, 1);
    createMovableDsLoad(4, addrReg, 2);
    createMovableDsLoad(8, addrReg, 3);
    createMovableDsLoad(12, addrReg, 4);
    createMovableDsLoad(16, addrReg, 5);
    // 2 independent VALUs
    createVAddInBlock(bb, arch, 60, 61, 62);
    createVAddInBlock(bb, arch, 63, 64, 65);
    // 2 independent WMMAs (fire first, create co-issue window)
    createWmmaF32_16x16x16_bf16(20, 28);
    createWmmaF32_16x16x16_bf16(36, 44);

    runPassWithUnrollGemm();

    int consecutiveDs = 0;
    int maxConsecutiveDs = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        if (mnem.find("ds_load") != std::string_view::npos) {
            consecutiveDs++;
            maxConsecutiveDs = std::max(maxConsecutiveDs, consecutiveDs);
        } else {
            consecutiveDs = 0;
        }
    }

    EXPECT_LE(maxConsecutiveDs, 3) << "DS window cap violated: found " << maxConsecutiveDs
                                   << " consecutive ds_loads (max 3 per WMMA window)";
}

// ---------------------------------------------------------------------------
// Property: all original instructions are preserved.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, DSWindowCap_InstructionCountPreserved) {
    const int addrReg = 100;
    // 6 independent ds_loads (LDS pseudo-reg makes them movable in DAG)
    for (int i = 0; i < 6; i++) createMovableDsLoad(i * 4, addrReg, i + 1);
    // 4 independent VALUs
    for (int i = 0; i < 4; i++) createVAddInBlock(bb, arch, 30 + i, 40 + i, 50 + i);
    // 3 independent WMMAs
    for (int i = 0; i < 3; i++) createWmmaF32_16x16x16_bf16(60 + i * 8, 84 + i * 8);

    int beforeCount = countStinkyInstructions(*bb);
    runPassWithUnrollGemm();
    int afterCount = countStinkyInstructions(*bb);

    EXPECT_EQ(afterCount, beforeCount) << "Scheduler must preserve instruction count";
}

// ---------------------------------------------------------------------------
// tensor_load_to_lds bounded in-flight credit pool (dagFeatures.globalReadQueueDepth
// / globalReadDrainLatency). The throttle is loop-only (uses the CDNA5 loop path),
// so all of these build a self-loop body with 4 movable tensor_loads + 10 VALU.
//
// Model: each issued tensor_load holds a credit for `drainLatency` cycles, decayed
// once per issued instruction (~1 cycle/pick here). With depth D, at most D credits
// may be in flight, so once D loads have issued the scheduler must interleave VALU
// until a credit drains before issuing the next load.
//
// NOTE on the chosen numbers: a credit must survive long enough to still be in
// flight when the next load wants to issue, so the throttle only engages when
// drainLatency > depth. At 1 issue/cycle a credit that drains in <= depth cycles
// frees before the cap is reached, so the hardware sustains D loads/cycle with no
// stall. We therefore use drainLatency comfortably above depth and assert the
// back-to-back run equals exactly depth (D loads, then forced interleave).
// ---------------------------------------------------------------------------

// Depth 2: exactly 2 tensor_loads issue back-to-back, then VALU must interleave.
TEST_F(DAGSchedulerPassTest, GlobalReadThrottle_Depth2_RespectsQueueDepth) {
    // Self-loop on the entry block so it is in RPO (scheduled) AND detected as a
    // loop (the cross-BB credit carry is loop-only). buildLoopBB makes an
    // unreachable second block, so reuse the entry block here.
    BasicBlock* body = bb;
    body->addSuccessor(body);
    for (int i = 0; i < 4; i++)
        createMovableTensorLoad(body, /*s0=*/i * 12, /*s1=*/i * 12 + 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    runPassWithGlobalReadThrottle(/*depth=*/2, /*drainLatency=*/8);

    std::vector<std::string> seq = mnemonicSequence(*body);
    EXPECT_EQ(maxConsecutiveTensorLoads(seq), 2)
        << "depth=2: at most 2 tensor_loads in flight before an interleave is forced";
}

// Depth 3: exactly 3 tensor_loads issue back-to-back, then VALU must interleave.
TEST_F(DAGSchedulerPassTest, GlobalReadThrottle_Depth3_RespectsQueueDepth) {
    // Self-loop on the entry block so it is in RPO (scheduled) AND detected as a
    // loop (the cross-BB credit carry is loop-only). buildLoopBB makes an
    // unreachable second block, so reuse the entry block here.
    BasicBlock* body = bb;
    body->addSuccessor(body);
    for (int i = 0; i < 4; i++)
        createMovableTensorLoad(body, /*s0=*/i * 12, /*s1=*/i * 12 + 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    runPassWithGlobalReadThrottle(/*depth=*/3, /*drainLatency=*/8);

    std::vector<std::string> seq = mnemonicSequence(*body);
    EXPECT_EQ(maxConsecutiveTensorLoads(seq), 3)
        << "depth=3: at most 3 tensor_loads in flight before an interleave is forced";
}

// Depth 1: degenerate cap — every tensor_load must be separated by other work.
TEST_F(DAGSchedulerPassTest, GlobalReadThrottle_Depth1_SeparatesEveryLoad) {
    // Self-loop on the entry block so it is in RPO (scheduled) AND detected as a
    // loop (the cross-BB credit carry is loop-only). buildLoopBB makes an
    // unreachable second block, so reuse the entry block here.
    BasicBlock* body = bb;
    body->addSuccessor(body);
    for (int i = 0; i < 4; i++)
        createMovableTensorLoad(body, /*s0=*/i * 12, /*s1=*/i * 12 + 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    runPassWithGlobalReadThrottle(/*depth=*/1, /*drainLatency=*/8);

    std::vector<std::string> seq = mnemonicSequence(*body);
    EXPECT_EQ(maxConsecutiveTensorLoads(seq), 1) << "depth=1: no two tensor_loads may be adjacent";
}

// All instructions are preserved regardless of throttle (count invariant).
TEST_F(DAGSchedulerPassTest, GlobalReadThrottle_PreservesInstructionCount) {
    // Self-loop on the entry block so it is in RPO (scheduled) AND detected as a
    // loop (the cross-BB credit carry is loop-only). buildLoopBB makes an
    // unreachable second block, so reuse the entry block here.
    BasicBlock* body = bb;
    body->addSuccessor(body);
    for (int i = 0; i < 4; i++)
        createMovableTensorLoad(body, /*s0=*/i * 12, /*s1=*/i * 12 + 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    int beforeCount = countStinkyInstructions(*body);
    runPassWithGlobalReadThrottle(/*depth=*/2, /*drainLatency=*/8);
    EXPECT_EQ(countStinkyInstructions(*body), beforeCount) << "throttle must not drop instructions";
}

// Throttle off (depth=0): feature is opt-in and does not perturb the default path.
TEST_F(DAGSchedulerPassTest, GlobalReadThrottle_Disabled_PreservesAll) {
    // Self-loop on the entry block so it is in RPO (scheduled) AND detected as a
    // loop (the cross-BB credit carry is loop-only). buildLoopBB makes an
    // unreachable second block, so reuse the entry block here.
    BasicBlock* body = bb;
    body->addSuccessor(body);
    for (int i = 0; i < 4; i++)
        createMovableTensorLoad(body, /*s0=*/i * 12, /*s1=*/i * 12 + 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    int beforeCount = countStinkyInstructions(*body);
    runPassWithGlobalReadThrottle(/*depth=*/0, /*drainLatency=*/0);
    EXPECT_EQ(countStinkyInstructions(*body), beforeCount);
}

// SGPR->tensor_load hazard: a SALU that writes an SGPR a tensor_load reads must be
// separated from that tensor_load by the fixed hardware gap (kCdna5HazardRules'
// SaluSgprToMemAddr entry, 8 cycles). Mirrors the real case (wmma/ds fill around the
// SALU): the scheduler hoists the SALU and/or holds the tensor_load so >= 8 cycles of
// work sit between them. We assert the cycle invariant, not an exact order. A WMMA
// counts as its latencyCycles (the co-issue window it opens, 8 here), other ops as
// issueCycles.
TEST_F(DAGSchedulerPassTest, SgprToTensorLoadHazard_AtLeast8CycleGap) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    // Movable ds_loads (LDS token -> stay in one region, no side-effect boundary) as the
    // fill work, a SALU writing s0, and a tensor_load reading s[0:4) so s0 is the hazard
    // register. Enough ds fill (>= hazard) so the gap is filled by real work, observable
    // in the emitted order rather than an invisible stall.
    for (int i = 0; i < 12; i++)
        createMovableDsLoad(/*destReg=*/8 + i * 4, /*addrReg=*/60, /*ldsToken=*/i + 2);

    AsmIRBuilder builder(*body, arch);
    StinkyInstruction* salu = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
    salu->addDestReg(StinkyRegister("s", 0, 1));
    salu->addSrcReg(StinkyRegister(0));

    createMovableTensorLoad(body, /*s0=*/0, /*s1=*/4, /*ldsToken=*/1);

    runPassWithGlobalReadThrottle(/*depth=*/4, /*drainLatency=*/8);

    // Locate the SALU and the tensor_load in the scheduled order, and total the cycles
    // of the work between them (WMMA -> latency window, else issue cycles).
    int saluPos = -1, tensorPos = -1, idx = 0;
    std::vector<int> cyclesAt;
    for (const IRBase& ir : *body) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        cyclesAt.push_back(isMatrixInstruction(*inst) ? inst->latencyCycles : inst->issueCycles);
        if (inst == salu) saluPos = idx;
        if (isTensorLoad(*inst)) tensorPos = idx;
        idx++;
    }
    ASSERT_GE(saluPos, 0);
    ASSERT_GE(tensorPos, 0);
    ASSERT_LT(saluPos, tensorPos) << "SALU must be scheduled before the tensor_load it feeds";

    int gap = 0;
    for (int i = saluPos + 1; i < tensorPos; i++) gap += cyclesAt[i];
    EXPECT_GE(gap, 8) << "tensor_load must be >= 8 cycles after the SALU writing its SGPR";
}

// Two-instruction SGPR address (e.g. low/high 64-bit split) where a second WMMA
// becomes ready right as WMMA #0's window closes, stealing the slot the address
// SALUs needed -- both must still end up >= 8 cycles ahead of the tensor_load.
TEST_F(DAGSchedulerPassTest, SgprPairToTensorLoadHazard_StolenWindow_AtLeast8CycleGap) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    auto createScalarAdd = [&](int dst, int src0, int src1) {
        AsmIRBuilder builder(*body, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_add_u32, arch));
        inst->addDestReg(StinkyRegister("s", dst, 1));
        inst->addSrcReg(StinkyRegister("s", src0, 1));
        inst->addSrcReg(StinkyRegister("s", src1, 1));
        return inst;
    };

    // WMMA #0 fires first (Phase B); latency=8 keeps its co-issue window open.
    createWmmaScaleF8_in(body, /*destStart=*/12, /*src0Start=*/50);

    // RAW chain a0->a1->a2->a3: fills positions 2,4,6,8 exactly, saturating the window.
    const int kChain = 4;
    for (int i = 0; i < kChain; i++) {
        const int src0 = (i == 0) ? 20 : (100 + i - 1);
        StinkyInstruction* a = createScalarAdd(/*dst=*/100 + i, src0, /*src1=*/21);
        a->issueCycles = 1;
        a->latencyCycles = 2;
    }

    // Independent WMMA #1: ready the instant WMMA #0's window closes.
    createWmmaScaleF8_in(body, /*destStart=*/200, /*src0Start=*/220);

    // The hazard pair: address low/high, independently computed (no RAW between them),
    // both feeding the same tensor_load's 4-SGPR base address s[150:154).
    StinkyInstruction* addLow = createScalarAdd(/*dst=*/150, /*src0=*/160, /*src1=*/161);
    StinkyInstruction* addHigh = createScalarAdd(/*dst=*/151, /*src0=*/162, /*src1=*/163);

    createMovableTensorLoad(body, /*s0=*/150, /*s1=*/158, /*ldsToken=*/1);

    int beforeCount = countStinkyInstructions(*body);
    runPassWithUnrollGemm();
    EXPECT_EQ(countStinkyInstructions(*body), beforeCount)
        << "scheduling must not drop instructions";

    int addLowPos = -1, addHighPos = -1, tensorPos = -1, idx = 0;
    std::vector<int> cyclesAt;
    for (const IRBase& ir : *body) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        cyclesAt.push_back(isMatrixInstruction(*inst) ? inst->latencyCycles : inst->issueCycles);
        if (inst == addLow) addLowPos = idx;
        if (inst == addHigh) addHighPos = idx;
        if (isTensorLoad(*inst)) tensorPos = idx;
        idx++;
    }
    ASSERT_GE(addLowPos, 0);
    ASSERT_GE(addHighPos, 0);
    ASSERT_GE(tensorPos, 0);
    ASSERT_LT(addLowPos, tensorPos) << "low-address SALU must precede the tensor_load";
    ASSERT_LT(addHighPos, tensorPos) << "high-address SALU must precede the tensor_load";

    const int laterProducerPos = std::max(addLowPos, addHighPos);
    int gap = 0;
    for (int i = laterProducerPos + 1; i < tensorPos; i++) gap += cyclesAt[i];
    EXPECT_GE(gap, 8) << "tensor_load must be >= 8 cycles after BOTH address SALUs, "
                         "including whichever one was scheduled later";
}

// Forces Phase G to fire while a SaluSgprToMemAddr gate is still live. Phase G's
// wait is a clock-only advance with no emitted instruction, so this asserts on
// the PASS_DEBUG trace instead of instruction order.
TEST_F(DAGSchedulerPassTest, QueueFullDuringHazard_PhaseGPaysHazardWait) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    // Occupies the single global-read credit slot (unrelated address).
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    AsmIRBuilder builder(*body, arch);
    StinkyInstruction* salu = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
    salu->addDestReg(StinkyRegister("s", 0, 1));
    salu->addSrcReg(StinkyRegister(0));

    // Hazarded: address s[0:4) written by the SALU above; credit pool still full.
    createMovableTensorLoad(body, /*s0=*/0, /*s1=*/4, /*ldsToken=*/2);

    PassManagerDebugConfig::addDebugOnly("StinkyDAGSchedulerPass");
    std::ostringstream captured;
    std::streambuf* oldBuf = std::cerr.rdbuf(captured.rdbuf());
    runPassWithGlobalReadThrottle(/*depth=*/1, /*drainLatency=*/6);
    std::cerr.rdbuf(oldBuf);
    PassManagerDebugConfig::clearDebugOnly();

    const std::string trace = captured.str();
    const std::string marker = "Phase G fallback pick";
    const size_t phaseGPos = trace.find(marker);
    ASSERT_NE(phaseGPos, std::string::npos)
        << "expected the credit-pool-exhaustion scenario to force Phase G; trace:\n"
        << trace;

    const std::string waitMarker = "wait=";
    const size_t waitPos = trace.find(waitMarker, phaseGPos);
    ASSERT_NE(waitPos, std::string::npos)
        << "Phase G must record the hazard wait it paid before issuing (regression: it used "
           "to pay only the credit-pool drain wait and skip the hazard gate entirely); trace:\n"
        << trace;
    const int paidWait = std::stoi(trace.substr(waitPos + waitMarker.size()));

    // Safe clock is >= 9 (gate stamped at clock 1); Phase G is reached at clock 2, so
    // it must pay >= 7 -- the pre-fix code paid only the credit-pool wait (6).
    EXPECT_GE(paidWait, 7) << "Phase G must pay the full outstanding SaluSgprToMemAddr hazard "
                              "wait, not just the credit-pool drain wait";
}

// VGPR->global_prefetch_b8 address hazard: a VALU that writes a VGPR the prefetch reads
// as its vaddr must be separated from that prefetch by the ValuVgprToVmemAddr gap (16
// cycles). Same structure as SgprToTensorLoadHazard, but exercises the vgpr-address rule
// against a global_prefetch_b8 consumer. Regression for the bug where the prefetch was
// missing IF_GLOBALLoad, so isBufferMemLoad (the rule's consumer predicate) never matched
// it and the gate was skipped -- the prefetch could sit < 16 cycles after its address VALU.
TEST_F(DAGSchedulerPassTest, VgprToGlobalPrefetchHazard_AtLeast16CycleGap) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    // Movable ds_loads as fill work so the 16-cycle gap is paid by real intervening
    // instructions (observable in emitted order rather than an invisible stall). Each
    // ds_load_b128 is issueCycles=1, so >= 16 are needed to cover the 16-cycle gate; use
    // the default ds in-flight queue depth (16) worth so they all pack between.
    for (int i = 0; i < 16; i++)
        createMovableDsLoad(/*destReg=*/8 + i * 4, /*addrReg=*/60, /*ldsToken=*/i + 2);

    // VALU writes v100; prefetch reads v[100:102) as vaddr, so v100 is the hazard register.
    StinkyInstruction* valu = createVAddInBlock(body, arch, /*destReg=*/100, /*src0Reg=*/101,
                                                /*src1Reg=*/102);
    StinkyInstruction* prefetch = createGlobalPrefetchB8(body, /*vaddrReg=*/100, /*saddrReg=*/0);

    runPassWithGlobalReadThrottle(/*depth=*/4, /*drainLatency=*/8);

    int valuPos = -1, prefetchPos = -1, idx = 0;
    std::vector<int> cyclesAt;
    for (const IRBase& ir : *body) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        cyclesAt.push_back(isMatrixInstruction(*inst) ? inst->latencyCycles : inst->issueCycles);
        if (inst == valu) valuPos = idx;
        if (inst == prefetch) prefetchPos = idx;
        idx++;
    }
    ASSERT_GE(valuPos, 0);
    ASSERT_GE(prefetchPos, 0);
    ASSERT_LT(valuPos, prefetchPos) << "VALU must be scheduled before the prefetch it feeds";

    int gap = 0;
    for (int i = valuPos + 1; i < prefetchPos; i++) gap += cyclesAt[i];
    EXPECT_GE(gap, 16) << "global_prefetch_b8 must be >= 16 cycles after the VALU writing its "
                          "vaddr VGPR";
}

// ---------------------------------------------------------------------------
// dsReadQueueDepth / dsReadThrottleLatency / dsReadPerWmma: queue-full pacing
// and in-flight depth control for ds_read_b128 (analogous to global-read
// queue throttling, but with DS-specific queue + WMMA interactions). Unlike global-read
// throttling, the ds_read gate additionally requires a WMMA to have been
// picked at least once (it seeds maxDsPerWmmaWindow_), so each test below
// includes one WMMA read (with dest/src registers disjoint from the ds_reads,
// so its DS-latency gate is trivially satisfied and it issues first).
// ---------------------------------------------------------------------------

// Depth 2: exactly 2 ds_reads issue back-to-back, then VALU must interleave.
TEST_F(DAGSchedulerPassTest, DsReadThrottle_Depth2_RespectsQueueDepth) {
    BasicBlock* body = bb;
    body->addSuccessor(body);
    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 4; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i * 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    runPassWithDsReadThrottle(/*queueDepth=*/2, /*throttleLatency=*/8);

    std::vector<std::string> seq = mnemonicSequence(*body);
    EXPECT_EQ(maxConsecutiveDsReads(seq), 2)
        << "depth=2: at most 2 ds_reads in flight before an interleave is forced";
}

// Depth 1: degenerate cap — every ds_read must be separated by other work.
TEST_F(DAGSchedulerPassTest, DsReadThrottle_Depth1_SeparatesEveryLoad) {
    BasicBlock* body = bb;
    body->addSuccessor(body);
    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 4; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i * 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    runPassWithDsReadThrottle(/*queueDepth=*/1, /*throttleLatency=*/8);

    std::vector<std::string> seq = mnemonicSequence(*body);
    EXPECT_EQ(maxConsecutiveDsReads(seq), 1) << "depth=1: no two ds_reads may be adjacent";
}

// Queue-full ds_read pacing is controlled by dsReadThrottleLatency. In a
// barrier-free region, changing dsReadDrainLatency alone should not change the
// ds_read interleave pattern.
TEST_F(DAGSchedulerPassTest, DsReadThrottle_QueuePacingIgnoresDrainLatency) {
    auto runCase = [&](int drainLatency) {
        // Reusing `am` across a new Function: without clearing, cached analysis
        // results (e.g. loop info) from the previous (now-destroyed) Function can
        // be reused if the allocator hands the new Function the same address —
        // undetectable in isolation, but corrupts scheduling amid a full suite run.
        am.clear();
        func = std::make_unique<Function>("dag_sched_test_ds_throttle_drain_split");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("loop_body");
        bb->addSuccessor(bb);

        createWmmaF32_16x16x16_bf16_in(bb, /*destStart=*/200, /*src0Start=*/204);
        for (int i = 0; i < 4; i++)
            createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i * 4, /*ldsToken=*/i + 1);
        for (int i = 0; i < 30; i++) createVAddInBlock(bb, arch, 40 + i, 80 + i, 100 + i);

        runPassWithDsReadThrottle(/*queueDepth=*/2, /*throttleLatency=*/8, /*perWmma=*/100,
                                  /*drainLatency=*/drainLatency);
        return mnemonicSequence(*bb);
    };

    const std::vector<std::string> seqDrain8 = runCase(/*drainLatency=*/8);
    const std::vector<std::string> seqDrain80 = runCase(/*drainLatency=*/80);
    EXPECT_EQ(maxConsecutiveDsReads(seqDrain8), 2);
    EXPECT_EQ(maxConsecutiveDsReads(seqDrain80), 2);
    EXPECT_EQ(seqDrain8, seqDrain80)
        << "drainLatency must not change queue pacing order when throttleLatency is fixed";
}

// Smaller dsReadThrottleLatency should allow more aggressive ds_read bursts
// under the same queue depth.
TEST_F(DAGSchedulerPassTest, DsReadThrottle_ThrottleLatencyControlsBurstLength) {
    auto runCase = [&](int throttleLatency) {
        // See the am.clear() comment in DsReadThrottle_QueuePacingIgnoresDrainLatency
        // above — same reused-`am`-across-a-new-Function hazard.
        am.clear();
        func = std::make_unique<Function>("dag_sched_test_ds_throttle_interval");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("loop_body");
        bb->addSuccessor(bb);

        createWmmaF32_16x16x16_bf16_in(bb, /*destStart=*/200, /*src0Start=*/204);
        for (int i = 0; i < 4; i++)
            createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i * 4, /*ldsToken=*/i + 1);
        for (int i = 0; i < 30; i++) createVAddInBlock(bb, arch, 40 + i, 80 + i, 100 + i);

        runPassWithDsReadThrottle(/*queueDepth=*/2, /*throttleLatency=*/throttleLatency,
                                  /*perWmma=*/100, /*drainLatency=*/80);
        return mnemonicSequence(*bb);
    };

    const std::vector<std::string> seqSlow = runCase(/*throttleLatency=*/8);
    const std::vector<std::string> seqFast = runCase(/*throttleLatency=*/2);
    const int burstSlow = maxConsecutiveDsReads(seqSlow);
    const int burstFast = maxConsecutiveDsReads(seqFast);
    EXPECT_EQ(burstSlow, 2);
    EXPECT_GE(burstFast, burstSlow)
        << "smaller throttleLatency should not reduce ds_read burst capacity";
    EXPECT_GT(burstFast, burstSlow)
        << "smaller throttleLatency should increase ds_read burst length under same queue depth";
}

// NOTE: the former DsReadThrottle_PerWmmaCap_RespectsCap test isolated the
// per-WMMA-window cap with a single WMMA. That predated the ds_load in-flight
// queue; now the cap only binds while a WMMA window is active (covered by
// DSWindowCap_VALUInterleaveAfter3, which keeps two WMMAs pending), and the
// no-WMMA case is bounded by the in-flight queue (covered by
// DsReadThrottle_Depth2_RespectsQueueDepth). No standalone single-WMMA cap test
// is kept — it would assert behavior the cap no longer has.

// Regression (see image(2).png bug): with no WMMA to issue and a chain of
// ds_loads each consumed by a VALU (RAW), the scheduler must NOT interleave
// ds,ds,valu,ds,ds,valu — that pattern forces an s_wait_dscnt per pair and
// tanks the kernel. Loads have their own in-flight queue, so they should drain
// (up to queue depth) before the consumer VALUs run: the consumers RAW-depend on
// the loads and are hazard-deferred until the load latency clears. Assert the
// loads front-load ahead of every consumer VALU.
TEST_F(DAGSchedulerPassTest, DsReadThrottle_NoWmma_LoadsDrainBeforeConsumerValu) {
    BasicBlock* body = bb;
    body->addSuccessor(body);
    // 6 ds_loads (movable via LDS token) on the same address register; each VALU
    // consumes the matching load's dest (RAW), mirroring the image's
    // ds_load_u8 -> v_lshl_or_b32 dependency chain. No WMMA in the region.
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/200, /*ldsToken=*/i + 1);
    for (int i = 0; i < 6; i++)
        createVAddInBlock(body, arch, /*dst=*/100 + i, /*src0=*/i * 4, /*src1=*/i * 4 + 1);

    // Queue depth 6 so all loads can be in flight at once; perWmma irrelevant (no WMMA).
    runPassWithDsReadThrottle(/*queueDepth=*/6, /*throttleLatency=*/8, /*perWmma=*/100);

    std::vector<std::string> seq = mnemonicSequence(*body);
    // Every ds_load must precede every v_add: find the last load and first valu.
    int lastLoad = -1, firstValu = -1;
    for (int i = 0; i < (int)seq.size(); i++) {
        if (seq[i] == "ds_load_b128") lastLoad = i;
        if (seq[i] == "v_add_f32" && firstValu < 0) firstValu = i;
    }
    ASSERT_GE(lastLoad, 0);
    ASSERT_GE(firstValu, 0);
    EXPECT_LT(lastLoad, firstValu)
        << "no-WMMA: all ds_loads must drain before consumer VALUs (no ds,valu,ds interleave)";
}

// Type-A WAR via elapse-time ordering (replaces the old dsAddrReadLatencyCounters):
// a VALU that overwrites the ds_load's address reg must be deferred behind other
// independent VALUs, because that reg was just touched (small elapse) — even though
// the overwrite is EARLIEST in program order (smallest DAG id, which plain
// pop()-by-id would pick first). This proves the read->write gap comes from elapse
// ordering, not a hard counter.
TEST_F(DAGSchedulerPassTest, WarOverwriteOfDsAddrDeferredByElapse) {
    BasicBlock* body = bb;
    body->addSuccessor(body);
    // ds_load reads address v200 (single ds_load so no load-drain effects dominate).
    createMovableDsLoad(/*destReg=*/8, /*addrReg=*/200, /*ldsToken=*/1);
    // Overwrite of v200 — created FIRST after the load, so it has the smallest DAG id
    // among the VALUs. Independent VALUs (disjoint regs) created after it.
    StinkyInstruction* overwrite = createVAddInBlock(body, arch, /*dst=*/200, /*src0=*/101,
                                                     /*src1=*/102);
    for (int i = 0; i < 3; i++)
        createVAddInBlock(body, arch, /*dst=*/50 + i, /*src0=*/60 + i, /*src1=*/70 + i);

    runPassWithUnrollGemm();

    // Find the scheduled position of the overwrite vs. the independent VALUs.
    int overwritePos = -1, firstIndependentPos = -1, idx = 0;
    for (const IRBase& ir : *body) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (inst == overwrite)
            overwritePos = idx;
        else if (inst->getUnifiedOpcode() == GFX::v_add_f32 && firstIndependentPos < 0)
            firstIndependentPos = idx;
        idx++;
    }
    ASSERT_GE(overwritePos, 0);
    ASSERT_GE(firstIndependentPos, 0);
    EXPECT_GT(overwritePos, firstIndependentPos)
        << "WAR overwrite of the ds_load address must be deferred behind independent VALUs "
           "by elapse-time ordering, despite having the smallest DAG id";
}

// MSB bank affinity: among equal-priority free VALU candidates, the same-bank one wins
// even with a larger DAG id, so no s_set_vgpr_msb switch is inserted. (Keys off VALU;
// a pure SALU has no VGPR MSB opinion.)
TEST_F(DAGSchedulerPassTest, MsbAffinity_SameBankPreferredAmongEqualPriority) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    // Anchor establishes currentMsb_=0; then a bank-1 op (smaller id) and a bank-0 op
    // (larger id). Plain id-order picks bank-1 next; affinity pulls bank-0 ahead.
    createVAddInBlock(body, arch, /*dst=*/10, /*src0=*/11, /*src1=*/12);
    StinkyInstruction* bank1 = createVAddInBlock(body, arch, /*dst=*/260, /*src0=*/261,
                                                 /*src1=*/262);
    StinkyInstruction* bank0 = createVAddInBlock(body, arch, /*dst=*/20, /*src0=*/21,
                                                 /*src1=*/22);

    runPass();

    int bank0Pos = -1, bank1Pos = -1, idx = 0;
    for (const IRBase& ir : *body) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (inst == bank0) bank0Pos = idx;
        if (inst == bank1) bank1Pos = idx;
        idx++;
    }
    ASSERT_GE(bank0Pos, 0);
    ASSERT_GE(bank1Pos, 0);
    EXPECT_LT(bank0Pos, bank1Pos)
        << "same-bank VALU (matching currentMsb_) must be scheduled before the different-bank "
           "VALU despite its larger DAG id, so no s_set_vgpr_msb switch is inserted between them";
}

// --- Cluster-barrier SCC rule (see applyClusterBarrierSccRule) ---
//
// InsertClusterBarrierPass later plants an SCC-clobbering handshake at or before the
// workgroup barrier that guards a tensor_load. The SCC def consumed by the region
// terminator (here: the loop-close compare feeding s_cbranch_scc0) must therefore stay
// below the last workgroup barrier, where no handshake anchor can reach it.

// Each case builds the same shape: WMMA/ds fill, a workgroup barrier guarding a
// tensor_load, then the loop-close compare + branch. The compare reads an SGPR nothing
// in the region writes, so it is ready from the first pick and would otherwise drift
// far above the barrier.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_PinsLiveOutSccDefBelowLastBarrier) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);
    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/220, /*src0Start=*/228);

    auto [barrierSignal, barrierWait] = createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    (void)barrierSignal;
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    createSCbranchReadingScc(body);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount)
        << "the SCC rule must not drop instructions";

    const int barrierPos = lastBarrierWaitPosition(*body);
    const int sccDefPos = positionOf(*body, sccDef);
    ASSERT_GE(barrierPos, 0);
    ASSERT_GE(sccDefPos, 0);
    EXPECT_GT(sccDefPos, barrierPos)
        << "the live-out SCC def must be scheduled after the last workgroup barrier, so the "
           "cluster-barrier handshake cannot clobber SCC inside its live range";
}

// The pin above says how early the live-out compare may go, not how late, and on its own
// it puts the compare right behind the barrier: it is a one-cycle SALU the barrier frees,
// so the queue takes it at once. That is the whole schedule away from the branch whenever
// the barrier has work behind it, which is the ordinary shape of an unrolled body -- the
// barrier opens the next buffer and the loads and WMMA that read it follow.
//
// So the region below hangs a tail off the barrier: loads carrying the barrier's LDS
// token, and a WMMA per load reading what it brought in. None of it can be scheduled
// until the barrier issues, and all of it costs far more than the lead the rule allows.
//
// The lead these two are written against is applyClusterBarrierSccRule's
// kLiveOutSccDefLeadCycles.
constexpr int kSccDefLeadCycles = 50;

IF_RULE3_CROSS_LOOP(TEST_F(DAGSchedulerPassTest,
                           ClusterBarrierSccRule_LiveOutSccDefLandsNearItsBranch) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);

    for (int i = 0; i < 12; i++) {
        createMovableDsLoad(/*destReg=*/100 + i * 8, /*addrReg=*/320 + i, /*ldsToken=*/1);
        createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/500 + i * 8,
                                       /*src0Start=*/100 + i * 8);
    }

    StinkyInstruction* branch = createSCbranchReadingScc(body);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount)
        << "the SCC rule must not drop instructions";

    const int barrierPos = lastBarrierWaitPosition(*body);
    const int sccDefPos = positionOf(*body, sccDef);
    ASSERT_GE(barrierPos, 0);
    ASSERT_GE(sccDefPos, 0);
    EXPECT_GT(sccDefPos, barrierPos)
        << "the live-out def must still be scheduled below the last workgroup barrier:"
        << scheduleOrder(*body);

    const int lead = cyclesBetween(*body, sccDef, branch);
    ASSERT_GE(lead, 0) << "the compare must be scheduled before the branch that reads it";
    EXPECT_LE(lead, kSccDefLeadCycles)
        << "the compare must also wait for the branch to come within " << kSccDefLeadCycles
        << " cycles instead of issuing the moment the barrier frees it:" << scheduleOrder(*body);
})

// With cluster barrier on and kRule3CrossLoop false, only the barrier pin applies.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_CrossLoopOffLeavesSccDefFarFromItsBranch) {
    if (cluster_barrier::kRule3CrossLoop) GTEST_SKIP() << "requires kRule3CrossLoop == false";

    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);

    for (int i = 0; i < 12; i++) {
        createMovableDsLoad(/*destReg=*/100 + i * 8, /*addrReg=*/320 + i, /*ldsToken=*/1);
        createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/500 + i * 8,
                                       /*src0Start=*/100 + i * 8);
    }

    StinkyInstruction* branch = createSCbranchReadingScc(body);

    runPassWithClusterBarrier(/*clusterBarrier=*/true);

    const int lead = cyclesBetween(*body, sccDef, branch);
    ASSERT_GE(lead, 0);
    EXPECT_GT(lead, kSccDefLeadCycles)
        << "kRule3CrossLoop off keeps only the barrier pin:" << scheduleOrder(*body);
}

// The same region with the rule off, which is also what the pin alone used to give: the
// compare goes as early as it can and its value then spans the barrier's whole tail. That
// is the distance the ceiling above closes.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_DisabledLeavesSccDefFarFromItsBranch) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);

    for (int i = 0; i < 12; i++) {
        createMovableDsLoad(/*destReg=*/100 + i * 8, /*addrReg=*/320 + i, /*ldsToken=*/1);
        createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/500 + i * 8,
                                       /*src0Start=*/100 + i * 8);
    }

    StinkyInstruction* branch = createSCbranchReadingScc(body);

    runPassWithClusterBarrier(/*clusterBarrier=*/false);

    const int lead = cyclesBetween(*body, sccDef, branch);
    ASSERT_GE(lead, 0);
    EXPECT_GT(lead, kSccDefLeadCycles)
        << "with the rule off nothing keeps the compare near its branch:" << scheduleOrder(*body);
}

// A live-out def written between two barriers. Following the barrier above it is not
// enough: its reader is the region terminator, so the range runs to the end of the region
// and the barrier below would fall inside it. The def has to be pushed under that one as
// well, which is an edge pointing back up the program order.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_PushesLiveOutSccDefBelowTheBarrierAfterIt) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/2);
    createSCbranchReadingScc(body);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount)
        << "the SCC rule must not drop instructions";

    const int lastWaitPos = lastBarrierWaitPosition(*body);
    const int sccDefPos = positionOf(*body, sccDef);
    ASSERT_GE(lastWaitPos, 0);
    ASSERT_GE(sccDefPos, 0);
    EXPECT_GT(sccDefPos, lastWaitPos)
        << "the live-out def must end up below every barrier in the region:"
        << scheduleOrder(*body);
}

// Same IR with the rule off: the compare is free to drift above the barrier. This is
// what makes the assertion above meaningful — it isolates the rule as the cause.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_DisabledLeavesSccDefFree) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);
    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/220, /*src0Start=*/228);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    createSCbranchReadingScc(body);

    runPassWithClusterBarrier(/*clusterBarrier=*/false);

    const int barrierPos = lastBarrierWaitPosition(*body);
    const int sccDefPos = positionOf(*body, sccDef);
    ASSERT_GE(barrierPos, 0);
    ASSERT_GE(sccDefPos, 0);
    EXPECT_LT(sccDefPos, barrierPos)
        << "without the rule the compare is an unconstrained SALU and drifts above the barrier";
}

// A def-use chain (one write, two ordinary SALU readers) ahead of a barrier that guards
// a tensor_load: the barrier may not be scheduled into it, so neither reader can be left
// between the signal and the wait nor pushed past the wait.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_GuardingBarrierNeverSplitsChain) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* reader1 = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* reader2 = createSCselectReadingScc(body, /*destSgpr=*/93, /*srcSgpr=*/94);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount);

    EXPECT_FALSE(barrierSplitsChain(*body, {sccDef, reader1, reader2}))
        << "the handshake anchors on the barrier, so it may not land inside the chain";
}

// Negative control for the test above: with the rule off the scheduler hoists the barrier
// signal into the middle of the chain, leaving the def above the signal and both readers
// below it. That ordering is what InsertClusterBarrierPass would later corrupt, since its
// handshake clobbers SCC at the barrier.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_DisabledLetsBarrierSplitChain) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* reader1 = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* reader2 = createSCselectReadingScc(body, /*destSgpr=*/93, /*srcSgpr=*/94);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    runPassWithClusterBarrier(/*clusterBarrier=*/false);

    const int signalPos = firstBarrierSignalPosition(*body);
    const int waitPos = lastBarrierWaitPosition(*body);
    ASSERT_GE(signalPos, 0);
    ASSERT_GE(waitPos, 0);
    EXPECT_LT(signalPos, waitPos);
    EXPECT_LT(positionOf(*body, sccDef), signalPos);
    EXPECT_GT(positionOf(*body, reader1), signalPos);
    EXPECT_GT(positionOf(*body, reader2), signalPos);
    EXPECT_TRUE(barrierSplitsChain(*body, {sccDef, reader1, reader2}))
        << "nothing keeps the chain together once the rule is disabled";
}

// The same chain starting behind the guarding barrier. The rule must not pin it there:
// the chain is still free to be hoisted, as long as it is hoisted whole.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_ChainBehindBarrierStaysWhole) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* reader1 = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* reader2 = createSCselectReadingScc(body, /*destSgpr=*/93, /*srcSgpr=*/94);

    runPassWithClusterBarrier(/*clusterBarrier=*/true);

    EXPECT_FALSE(barrierSplitsChain(*body, {sccDef, reader1, reader2}))
        << "hoisting the chain above the barrier is allowed, but only as a whole";
}

// A chain that starts entirely behind the guarding barrier, with the tensor_load trailing
// it. Both cases below build this program order:
//
//     s_barrier_signal -1 / s_barrier_wait -1 / s_cmp_eq_u32 / s_cselect_b32 / tensor_load
//
// The compare reads an SGPR nothing in the region writes, so it is ready from the first
// pick and wants to hoist above the barrier; its reader cannot follow until the compare
// has issued. That is what pulls the chain apart when nothing holds it together.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_DisabledSplitsChainBehindBarrier) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* sccReader = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* tensorLoad = createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48,
                                                            /*ldsToken=*/1);

    runPassWithClusterBarrier(/*clusterBarrier=*/false);

    const std::string order = scheduleOrder(*body);
    // The compare hoists above the barrier and leaves its reader behind, so the barrier is
    // scheduled straight through the chain's live range.
    EXPECT_LT(positionOf(*body, sccDef), firstBarrierSignalPosition(*body)) << order;
    EXPECT_LT(firstBarrierSignalPosition(*body), positionOf(*body, sccReader)) << order;
    EXPECT_LT(positionOf(*body, sccReader), lastBarrierWaitPosition(*body)) << order;
    EXPECT_LT(lastBarrierWaitPosition(*body), positionOf(*body, tensorLoad)) << order;
    EXPECT_TRUE(barrierSplitsChain(*body, {sccDef, sccReader}))
        << "nothing keeps the chain together once the rule is disabled:" << order;
}

// Same IR with the rule on. The chain is still free to hoist above the barrier -- the rule
// does not pin it behind -- but it may only do so as a whole.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_ChainBehindBarrierHoistsWhole) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* sccReader = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* tensorLoad = createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48,
                                                            /*ldsToken=*/1);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount);

    const std::string order = scheduleOrder(*body);
    EXPECT_FALSE(barrierSplitsChain(*body, {sccDef, sccReader}))
        << "no guarding barrier may land between the SCC def and its reader:" << order;
    EXPECT_LT(lastBarrierWaitPosition(*body), positionOf(*body, tensorLoad))
        << "the barrier must still guard the tensor_load:" << order;
}

// Same chain, same barrier, but nothing behind the barrier to guard. The handshake anchor
// is picked by a cycle-lead climb and can come to rest on any workgroup barrier, so having
// nothing to guard buys this one no exemption: the chain must still be kept whole.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_BarrierWithNothingToGuardKeepsChainWhole) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* reader1 = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* reader2 = createSCselectReadingScc(body, /*destSgpr=*/93, /*srcSgpr=*/94);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount);

    EXPECT_FALSE(barrierSplitsChain(*body, {sccDef, reader1, reader2}))
        << "a barrier with nothing to guard is still a place the handshake may land:"
        << scheduleOrder(*body);
}

// The loop counter decrement writes SCC as a carry-out that the compare right after it
// immediately kills, so its SCC is dead and the rule must leave it alone. Its *SGPR*
// result is read across the back edge, which must not be mistaken for a live SCC value
// -- doing so pins the decrement behind the barrier and blocks a hoist the scheduler
// would otherwise make.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_DeadSccCarryOutStaysFree) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

    StinkyInstruction* decrement = createSSubWritingSgprAndScc(body, /*sgpr=*/90);
    createSCmpWritingScc(body, /*srcSgpr=*/90);
    createSCbranchReadingScc(body);
    // Past the branch, so in a later region: this is the cross-back-edge SGPR read that
    // makes the decrement look live-out unless the check is narrowed to SCC readers.
    {
        AsmIRBuilder builder(*body, arch);
        StinkyInstruction* carry = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
        carry->addDestReg(StinkyRegister("s", 95, 1));
        carry->addSrcReg(StinkyRegister("s", 90, 1));
    }

    runPassWithClusterBarrier(/*clusterBarrier=*/true);

    const int signalPos = firstBarrierSignalPosition(*body);
    ASSERT_GE(signalPos, 0);
    EXPECT_LT(positionOf(*body, decrement), signalPos)
        << "the decrement's SCC is dead, so nothing stops it from being hoisted";
}

// The same pin with no tensor_load in the region: the live-out def belongs below the
// barrier either way, since what it has to stay clear of is the handshake, not the load.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_PinsLiveOutSccDefBelowBarePlainBarrier) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);
    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/220, /*src0Start=*/228);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    createSCbranchReadingScc(body);

    runPassWithClusterBarrier(/*clusterBarrier=*/true);

    const int barrierPos = lastBarrierWaitPosition(*body);
    const int sccDefPos = positionOf(*body, sccDef);
    ASSERT_GE(barrierPos, 0);
    ASSERT_GE(sccDefPos, 0);
    EXPECT_GT(sccDefPos, barrierPos)
        << "the handshake can land on any workgroup barrier, so the live-out def has to "
           "follow this one too";
}

// An SCC chain that is born and dies inside the region, sitting ahead of the barrier that
// guards a tensor_load. Without protection the scheduler drops the barrier's signal between
// the def and its reader, which is where InsertClusterBarrierPass anchors its SCC-clobbering
// handshake.
TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_InRegionChainSurvivesGuardingBarrier) {
    BasicBlock* body = bb;
    body->addSuccessor(body);

    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 6; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i, /*ldsToken=*/i + 10);

    StinkyInstruction* sccDef = createSCmpWritingScc(body, /*srcSgpr=*/90);
    StinkyInstruction* sccReader = createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);

    createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
    StinkyInstruction* tensorLoad = createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48,
                                                            /*ldsToken=*/1);

    const int beforeCount = countStinkyInstructions(*body);
    runPassWithClusterBarrier(/*clusterBarrier=*/true);
    ASSERT_EQ(countStinkyInstructions(*body), beforeCount);

    EXPECT_FALSE(barrierSplitsChain(*body, {sccDef, sccReader}))
        << "no guarding barrier may land between the SCC def and its reader:"
        << scheduleOrder(*body);
    EXPECT_LT(lastBarrierWaitPosition(*body), positionOf(*body, tensorLoad))
        << "the barrier must still guard the tensor_load:" << scheduleOrder(*body);
}

TEST_F(DAGSchedulerPassTest, ClusterBarrierSccRule_LiveInSccReaderWithoutDefAborts) {
    EXPECT_DEATH(
        {
            BasicBlock* body = bb;

            // Keep region in "normal schedulable" shape.
            createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
            createMovableDsLoad(/*destReg=*/0, /*addrReg=*/300, /*ldsToken=*/10);

            // SCC live-in reader: no SCC writer in this region before this point.
            createSCselectReadingScc(body, /*destSgpr=*/91, /*srcSgpr=*/92);

            // Barrier comes after the SCC reader.
            createMovableWorkgroupBarrier(body, /*ldsToken=*/1);
            createMovableTensorLoad(body, /*s0=*/40, /*s1=*/48, /*ldsToken=*/1);

            runPassWithClusterBarrier(/*clusterBarrier=*/true);
        },
        "region has SCC reader\\(s\\) but no SCC writer");
}

// All instructions are preserved regardless of throttle (count invariant).
TEST_F(DAGSchedulerPassTest, DsReadThrottle_PreservesInstructionCount) {
    BasicBlock* body = bb;
    body->addSuccessor(body);
    createWmmaF32_16x16x16_bf16_in(body, /*destStart=*/200, /*src0Start=*/204);
    for (int i = 0; i < 4; i++)
        createMovableDsLoad(/*destReg=*/i * 4, /*addrReg=*/300 + i * 4, /*ldsToken=*/i + 1);
    for (int i = 0; i < 30; i++) createVAddInBlock(body, arch, 40 + i, 80 + i, 100 + i);

    int beforeCount = countStinkyInstructions(*body);
    runPassWithDsReadThrottle(/*queueDepth=*/2, /*throttleLatency=*/8);
    EXPECT_EQ(countStinkyInstructions(*body), beforeCount) << "throttle must not drop instructions";
}
