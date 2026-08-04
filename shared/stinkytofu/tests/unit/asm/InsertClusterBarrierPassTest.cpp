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
// Unit tests for InsertClusterBarrierPass (gfx1250).
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
constexpr const char* kGSU1LabelName = "label_GSU_1";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";

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

int countClusterSignals(const std::vector<int>& seq) {
    int n = 0;
    for (int e : seq)
        if (e == 1) ++n;
    return n;
}

int countClusterWaits(const std::vector<int>& seq) {
    int n = 0;
    for (int e : seq)
        if (e == -1) ++n;
    return n;
}

bool isClusterBarrierWithLiteral(const StinkyInstruction& inst, bool wantSignal) {
    const bool sig = isBarrierSignal(inst);
    const bool wait = isBarrierWait(inst);
    if (wantSignal ? !sig : !wait) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
           srcs[0].getLiteralInt() == kClusterBarrierId;
}

bool isImmediatelyPrecededByClusterBarrierWait(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return false;
    auto it = BasicBlock::iterator(anchor);
    while (it != parent->begin()) {
        --it;
        auto* prev = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (prev == nullptr) continue;
        if (isPseudoInst(prev)) continue;
        return isClusterBarrierWithLiteral(*prev, /*wantSignal=*/false);
    }
    return false;
}

StinkyInstruction* firstRealInstAfter(StinkyInstruction* anchor) {
    BasicBlock* parent = anchor->getParent();
    if (parent == nullptr) return nullptr;
    for (auto it = std::next(BasicBlock::iterator(anchor)); it != parent->end(); ++it) {
        auto* next = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (next == nullptr || isPseudoInst(next)) continue;
        return next;
    }
    return nullptr;
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
        config.TileA0 = 16;
        config.TileB0 = 16;
        config.TileM0 = 16;
        config.NumGRA = 4;
        config.NumGRB = 4;
        config.NumGRM = 4;
        config.NumWaves = 4;

        func = std::make_unique<Function>("cluster_barrier_test");
        setFunctionArch(*func, arch);
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

    StinkyInstruction* createWMMA(int destStart, int src0Start, int src1Start) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch);
        if (desc == nullptr) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("a", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src1Start, 8));
        inst->addSrcReg(StinkyRegister("a", destStart, 8));
        return inst;
    }

    void appendHandshake(int loadS0, int loadS1) {
        createBarrierSignal(kWorkgroupBarrierId);
        createBarrierWait(kWorkgroupBarrierId);
        createTensorLoadInBlock(bb, arch, loadS0, loadS1);
    }

    void createLabel(const char* name) {
        AsmIRBuilder builder(*bb, arch);
        builder.createLabel(name);
    }

    StinkyInstruction* findFirstTensorLoad() {
        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (isTensorLoad(*inst)) return inst;
        }
        return nullptr;
    }

    StinkyInstruction* findLabelNamed(const char* name) {
        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (!isLabel(*inst)) continue;
            const auto* labelData = inst->getModifier<LabelData>();
            if (labelData != nullptr && labelData->label == name) return inst;
        }
        return nullptr;
    }

    void runPass() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        auto pass = createInsertClusterBarrierPass();
        pass->run(*func, ctx, am);
    }

    void buildTwoHandshakeBody() {
        createWMMA(24, 0, 8);
        appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
        createWMMA(32, 8, 16);
        createWMMA(40, 16, 8);
        appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
        createWMMA(56, 24, 32);
    }

    void expectNoClusterPhaseOverlap(int expectedSignals) {
        const std::vector<int> seq = clusterBarrierSequence();
        EXPECT_EQ(countClusterSignals(seq), expectedSignals)
            << "exactly one Rule 3 signal -3 per handshake";
        int outstanding = 0;
        for (size_t i = 0; i < seq.size(); ++i) {
            outstanding += seq[i];
            EXPECT_LE(outstanding, 1)
                << "two cluster signals in flight before a wait at index " << i;
        }
    }

    std::vector<int> clusterBarrierSequence() const {
        std::vector<int> seq;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const int kind = clusterBarrierKind(*cast<StinkyInstruction>(&ir));
            if (kind != 0) seq.push_back(kind);
        }
        return seq;
    }

    std::pair<int, int> clusterBarrierCounts() const {
        const std::vector<int> seq = clusterBarrierSequence();
        return {countClusterSignals(seq), countClusterWaits(seq)};
    }

    size_t indexOf(const StinkyInstruction* target) const {
        size_t idx = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (cast<StinkyInstruction>(&ir) == target) return idx;
            ++idx;
        }
        return static_cast<size_t>(-1);
    }

    static bool isWorkgroupBarrierSignalInst(const StinkyInstruction& inst) {
        if (!isBarrierSignal(inst)) return false;
        const auto& srcs = inst.getSrcRegs();
        return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
               srcs[0].getLiteralInt() == kWorkgroupBarrierId;
    }

    static bool isWorkgroupBarrierWaitInst(const StinkyInstruction& inst) {
        if (!isBarrierWait(inst)) return false;
        const auto& srcs = inst.getSrcRegs();
        return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
               srcs[0].getLiteralInt() == kWorkgroupBarrierId;
    }

    StinkyInstruction* findClusterWaveCmpAfter(size_t startIdx) const {
        size_t idx = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (idx >= startIdx && inst->getUnifiedOpcode() == GFX::s_cmp_eq_u32 &&
                !inst->getSrcRegs().empty() &&
                inst->getSrcRegs()[0].getSymbolicName() == "sgprWaveIdx") {
                return const_cast<StinkyInstruction*>(inst);
            }
            ++idx;
        }
        return nullptr;
    }
};

TEST_F(InsertClusterBarrierPassTest, SingleHandshakeEmitsOneSignalBeforeItsWaits) {
    createWMMA(32, 0, 8);
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(40, 8, 0);

    runPass();

    const std::vector<int> seq = clusterBarrierSequence();
    ASSERT_FALSE(seq.empty()) << "expected at least one cluster barrier";
    EXPECT_EQ(countClusterSignals(seq), 1) << "exactly one Rule 3 signal -3 per handshake";
    EXPECT_EQ(seq.front(), 1) << "the signal must come before any cluster wait";
}

TEST_F(InsertClusterBarrierPassTest, TwoHandshakesDoNotOverlapClusterPhases) {
    buildTwoHandshakeBody();
    runPass();
    expectNoClusterPhaseOverlap(/*expectedSignals=*/2);
}

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

TEST_F(InsertClusterBarrierPassTest, Rule1PostGsu1InsertsLclGatedClusterSignal) {
    createLabel(kGSU1LabelName);
    createVAddInBlock(bb, arch, /*destReg=*/0, /*src0Reg=*/4, /*src1Reg=*/8);

    runPass();

    StinkyInstruction* gsu1 = findLabelNamed(kGSU1LabelName);
    ASSERT_NE(gsu1, nullptr);
    StinkyInstruction* next = firstRealInstAfter(gsu1);
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->getUnifiedOpcode(), GFX::s_cmp_eq_u32);
    ASSERT_GE(next->getSrcRegs().size(), 1u);
    EXPECT_EQ(next->getSrcRegs()[0].getSymbolicName(), kLoopCounterLSymbol);

    bool sawClusterSignal = false;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (isClusterBarrierWithLiteral(*inst, /*wantSignal=*/true)) {
            sawClusterSignal = true;
            break;
        }
    }
    EXPECT_TRUE(sawClusterSignal) << "Rule 1 must emit a cluster signal -3";
}

TEST_F(InsertClusterBarrierPassTest, Rule2InsertsWaitBeforeFirstTensorLoad) {
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);

    runPass();

    StinkyInstruction* firstLoad = findFirstTensorLoad();
    ASSERT_NE(firstLoad, nullptr);
    EXPECT_TRUE(isImmediatelyPrecededByClusterBarrierWait(firstLoad))
        << "Rule 2 must insert s_barrier_wait -3 immediately before the first load";
}

TEST_F(InsertClusterBarrierPassTest, IdempotencySecondRunIsNoOp) {
    buildTwoHandshakeBody();
    runPass();
    const auto [signalsAfterFirst, waitsAfterFirst] = clusterBarrierCounts();

    runPass();
    const auto [signalsAfterSecond, waitsAfterSecond] = clusterBarrierCounts();

    EXPECT_EQ(signalsAfterSecond, signalsAfterFirst)
        << "a second pass must not insert additional cluster signals";
    EXPECT_EQ(waitsAfterSecond, waitsAfterFirst)
        << "a second pass must not insert additional cluster waits";
}

TEST_F(InsertClusterBarrierPassTest, Rule3ForwardsPastWorkgroupBarriers) {
    for (int i = 0; i < 80; ++i) {
        createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    }
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*loadS0=*/0, /*loadS1=*/4);

    runPass();

    size_t clusterSignalIdx = static_cast<size_t>(-1);
    size_t firstWgSignalIdx = static_cast<size_t>(-1);
    size_t idx = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (isClusterBarrierWithLiteral(*inst, /*wantSignal=*/true) &&
            clusterSignalIdx == static_cast<size_t>(-1)) {
            clusterSignalIdx = idx;
        }
        if (firstWgSignalIdx == static_cast<size_t>(-1) && isWorkgroupBarrierSignalInst(*inst)) {
            firstWgSignalIdx = idx;
        }
        ++idx;
    }
    ASSERT_NE(clusterSignalIdx, static_cast<size_t>(-1));
    ASSERT_NE(firstWgSignalIdx, static_cast<size_t>(-1));
    EXPECT_LT(clusterSignalIdx, firstWgSignalIdx)
        << "cluster signal must forward past intervening workgroup barriers";
}

TEST_F(InsertClusterBarrierPassTest, Wait3StopAnchorsAfterFollowingWorkgroupBarrier) {
    createWMMA(24, 0, 8);
    createBarrierWait(kClusterBarrierId);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);
    createWMMA(40, 16, 8);
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);

    runPass();

    StinkyInstruction* preexistingClusterWait = nullptr;
    for (IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        auto* inst = cast<StinkyInstruction>(&ir);
        if (isClusterBarrierWithLiteral(*inst, /*wantSignal=*/false)) {
            preexistingClusterWait = inst;
            break;
        }
    }
    ASSERT_NE(preexistingClusterWait, nullptr);

    StinkyInstruction* wgWaitAfterPreexisting = nullptr;
    for (StinkyInstruction* fwd = firstRealInstAfter(preexistingClusterWait); fwd != nullptr;
         fwd = firstRealInstAfter(fwd)) {
        if (!isWorkgroupBarrierSignalInst(*fwd)) continue;
        StinkyInstruction* maybeWait = firstRealInstAfter(fwd);
        ASSERT_NE(maybeWait, nullptr);
        ASSERT_TRUE(isWorkgroupBarrierWaitInst(*maybeWait));
        wgWaitAfterPreexisting = maybeWait;
        break;
    }
    ASSERT_NE(wgWaitAfterPreexisting, nullptr);

    const size_t anchorFloor = indexOf(firstRealInstAfter(wgWaitAfterPreexisting));
    ASSERT_NE(anchorFloor, static_cast<size_t>(-1));

    StinkyInstruction* rule3ClusterCmp = findClusterWaveCmpAfter(indexOf(preexistingClusterWait));
    ASSERT_NE(rule3ClusterCmp, nullptr);
    EXPECT_GE(indexOf(rule3ClusterCmp), anchorFloor)
        << "scan hitting wait-3 must anchor after the following workgroup barrier";
}

TEST_F(InsertClusterBarrierPassTest, Rule3SegmentBoundaryFallbackAnchorsAtSegBegin) {
    // Short segment: one instruction after the label, then the handshake.  With
    // kRule3SignalLeadCycles = 900 the cycle lead cannot match, so the backward
    // scan falls back to segBegin (label + 1) rather than co-locating with wait.
    createLabel("label_SegmentStart");
    StinkyInstruction* segBeginInst =
        createVAddInBlock(bb, arch, /*destReg=*/0, /*src0Reg=*/4, /*src1Reg=*/8);
    ASSERT_NE(segBeginInst, nullptr);
    StinkyInstruction* labelBeforePass = findLabelNamed("label_SegmentStart");
    ASSERT_NE(labelBeforePass, nullptr);
    ASSERT_EQ(segBeginInst, firstRealInstAfter(labelBeforePass));
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);

    runPass();

    StinkyInstruction* wgSignal = nullptr;
    for (IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        auto* inst = cast<StinkyInstruction>(&ir);
        if (isWorkgroupBarrierSignalInst(*inst)) {
            wgSignal = inst;
            break;
        }
    }
    ASSERT_NE(wgSignal, nullptr);

    StinkyInstruction* rule3ClusterCmp = findClusterWaveCmpAfter(0);
    ASSERT_NE(rule3ClusterCmp, nullptr);

    EXPECT_LT(indexOf(rule3ClusterCmp), indexOf(segBeginInst))
        << "cluster signal must anchor before segBegin, not co-locate with wait";
    EXPECT_LT(indexOf(segBeginInst), indexOf(wgSignal))
        << "segBegin must precede the workgroup signal";
}
