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

#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPassTestSupport.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr int kClusterBarrierId = -3;
constexpr int kWorkgroupBarrierId = -1;
constexpr const char* kGSU1LabelName = "label_GSU_1";
constexpr const char* kLoopCounterLSymbol = "sgprLoopCounterL";
constexpr const char* kWaveGateLabelPrefix = "label_skipCBPreSignal";

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

// Instructions the pass emits carry no explicit SCC destination -- like the pass itself,
// go by the descriptor flag as well as the operand list.
bool writesScc(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitWriteSCC)) return true;
    for (const StinkyRegister& reg : inst.getDestRegs())
        if (reg.isRegister() && reg.reg.type == RegType::SCC) return true;
    return false;
}

// The compare the Rule 3 handshake emits ahead of its `s_barrier_signal -3`. It is the
// instruction that clobbers SCC, so it is what a live SCC value has to survive.
bool isClusterWaveCmp(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_cmp_eq_u32) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].getSymbolicName() == "sgprWaveIdx";
}

// Set STINKY_TEST_DUMP=1 to have a test print the block before and after the pass. Off by
// default so the suite stays quiet.
bool testDumpEnabled() {
    static const bool enabled = std::getenv("STINKY_TEST_DUMP") != nullptr;
    return enabled;
}

std::string blockListing(const BasicBlock& block) {
    std::ostringstream os;
    int idx = 0;
    for (const IRBase& ir : block) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        os << "\n  " << idx++ << ": ";
        // AsmPrinter treats labels as block boundaries and prints nothing for them, which
        // would leave a listing that says where the pass put things without saying what it
        // put them relative to. Spell the label out instead.
        if (inst->getUnifiedOpcode() == GFX::LABEL) {
            const auto* labelData = inst->getModifier<LabelData>();
            os << (labelData != nullptr ? labelData->label : std::string("<label>")) << ":\n";
            continue;
        }
        inst->dump(os);
    }
    return os.str();
}

/// Cluster signals and waits have to alternate along every *path*, which is not the same
/// thing as alternating down the page. A branch makes the printed order and the executed
/// order two different stories: an edge that leaves holding a token and lands where none is
/// expected drops it, and one that leaves empty-handed and lands where a token is assumed
/// posts a second signal on top of the first. Neither shows up in a straight read of the
/// block, and the second one hangs the kernel.
///
/// So walk the edges instead, carrying the token count and recording what each instruction
/// was reached holding. Two paths that reach the same instruction disagreeing is the whole
/// bug class in one check: it is what an unbalanced exit, a missing drain, or a loop whose
/// head and latch differ all come out as. The back edge is included, so a loop that does not
/// hand the next trip what it promised the first one is caught here too.
///
/// The pass's own wave-id gates are the one exception: they exist to jump over the signal
/// that only wave 0 posts, so their two sides genuinely disagree and the wave-0 side is the
/// one that describes the token.
/// \p completeProgram says the block is a whole kernel rather than a fragment, which adds
/// the two checks that only make sense end to end: every wait has a signal to consume, and
/// no path runs out of block still holding one. Most tests here build a fragment that starts
/// mid-stream and stops before the loop is closed, so the producer or consumer of a token is
/// legitimately absent and those two would fire on the input, not on the pass.
std::string clusterTokenPathProblems(const BasicBlock& block, bool completeProgram = false) {
    std::vector<const StinkyInstruction*> insts;
    std::unordered_map<std::string, size_t> labelIndex;
    for (const IRBase& ir : block) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (isLabel(*inst)) {
            if (const auto* labelData = inst->getModifier<LabelData>())
                labelIndex.emplace(labelData->label, insts.size());
        }
        insts.push_back(inst);
    }
    if (insts.empty()) return {};

    std::ostringstream problems;
    std::vector<int> reachedHolding(insts.size(), -1);
    std::vector<std::pair<size_t, int>> work{{0, 0}};
    while (!work.empty()) {
        const auto [idx, incoming] = work.back();
        work.pop_back();
        if (idx >= insts.size()) {
            if (completeProgram && incoming != 0)
                problems << "\n  a path runs off the end of the block still holding a token";
            continue;
        }
        if (reachedHolding[idx] != -1) {
            if (reachedHolding[idx] != incoming)
                problems << "\n  index " << idx << " is reached holding " << incoming
                         << " on one path and " << reachedHolding[idx] << " on another";
            continue;
        }
        reachedHolding[idx] = incoming;

        const StinkyInstruction& inst = *insts[idx];
        int outgoing = incoming;
        const int kind = clusterBarrierKind(inst);
        if (kind == 1) {
            if (incoming == 1)
                problems << "\n  index " << idx << ": a cluster signal with one already in flight";
            outgoing = 1;
        } else if (kind == -1) {
            if (completeProgram && incoming == 0)
                problems << "\n  index " << idx << ": a cluster wait with nothing to consume";
            outgoing = 0;
        }

        if (!isBranch(inst)) {
            work.push_back({idx + 1, outgoing});
            continue;
        }
        const std::string target = getBranchTarget(inst);
        const bool waveGate = target.rfind(kWaveGateLabelPrefix, 0) == 0;
        if (!waveGate) {
            const auto found = labelIndex.find(target);
            if (found != labelIndex.end()) work.push_back({found->second, outgoing});
        }
        if (waveGate || !isUnconditionalBranch(inst)) work.push_back({idx + 1, outgoing});
    }
    return problems.str();
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
        expectClusterTokensBalanceOnEveryPath();
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

    StinkyInstruction* createWaitTensorCnt(int count) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch));
        inst->addSrcReg(StinkyRegister(count));
        SWaitTensorCntData d;
        d.tlcnt = count;
        inst->addModifier<SWaitTensorCntData>(d);
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

    // `s_swappc_b64` -- a call, which is a segment boundary that falls through rather than
    // naming a label, so the code below it is reached the ordinary way.
    StinkyInstruction* createCall() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_swappc_b64, arch));
        inst->addDestReg(StinkyRegister("s", 2, 2));
        inst->addSrcReg(StinkyRegister("s", 0, 2));
        return inst;
    }

    // `s_cmp_eq_u32 s<srcSgpr>, 0` -- writes SCC and nothing else.
    StinkyInstruction* createSCmpWritingScc(int srcSgpr) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        inst->addSrcReg(StinkyRegister(0));
        inst->addDestReg(StinkyRegister::getSCCRegister());
        return inst;
    }

    // `s_sub_u32 s<sgpr>, s<sgpr>, 1` -- writes an SGPR *and* SCC (carry-out). The loop
    // counter decrement, and an SCC def the pass cannot rematerialize: re-running it would
    // decrement the counter a second time.
    StinkyInstruction* createSSubWritingSgprAndScc(int sgpr) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_sub_u32, arch));
        inst->addDestReg(StinkyRegister("s", sgpr, 1));
        inst->addDestReg(StinkyRegister::getSCCRegister());
        inst->addSrcReg(StinkyRegister("s", sgpr, 1));
        inst->addSrcReg(StinkyRegister(1));
        return inst;
    }

    // `s_cselect_b32 s<destSgpr>, s<srcSgpr>, 0` -- consumes SCC as ordinary SALU work.
    StinkyInstruction* createSCselectReadingScc(int destSgpr, int srcSgpr) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cselect_b32, arch));
        inst->addDestReg(StinkyRegister("s", destSgpr, 1));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        inst->addSrcReg(StinkyRegister(0));
        inst->addSrcReg(StinkyRegister::getSCCRegister());
        return inst;
    }

    StinkyInstruction* createDsRead(int destReg, int addrReg) {
        return createDSLoadInBlock(bb, arch, destReg, addrReg);
    }

    // The last instruction before \p beforeIdx that writes SCC, or null when there is none.
    const StinkyInstruction* lastSccWriterBefore(size_t beforeIdx) const {
        const StinkyInstruction* found = nullptr;
        size_t idx = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (idx >= beforeIdx) break;
            const auto* inst = cast<StinkyInstruction>(&ir);
            if (writesScc(*inst)) found = inst;
            ++idx;
        }
        return found;
    }

    // Returns the workgroup barrier signal, which is the trigger Rule 3 reads.
    StinkyInstruction* appendHandshake(int loadS0, int loadS1) {
        StinkyInstruction* trigger = createBarrierSignal(kWorkgroupBarrierId);
        createBarrierWait(kWorkgroupBarrierId);
        createTensorLoadInBlock(bb, arch, loadS0, loadS1);
        return trigger;
    }

    // The run-up every real kernel opens with: the GSU_1 label Rule 1 posts its signal below,
    // and the function's first tensor load, which is where Rule 2 plants the wait that drinks
    // it. The two only make sense together -- a wait with no signal above it is a hang -- so
    // no test builds one without the other.
    void appendGsu1Preheader() {
        createLabel(kGSU1LabelName);
        createWMMA(24, 0, 8);
        createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
        createWMMA(32, 8, 16);
        // Real kernels always rejoin at a label between that load and the loop head -- the
        // PGR2 join points -- and a run-up with no label in it at all sends the pre-loop
        // signal down a path they never take.
        createLabel("label_PreLoopJoin");
        createWMMA(40, 16, 24);
    }

    // Rule 3 speaks for the loop body and nowhere else, so anything asked of it has to sit
    // inside a loop.
    void openLoop() {
        createLabel("label_TestLoop");
    }

    // The body's way out, then the latch. The exit branch is part of the shape rather than
    // decoration: it is what names the exit, and the exit is where a token carried out of the
    // body has to be drained.
    void closeLoop() {
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");
        createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
        createLabel("label_TestLoopEnd");
        createWMMA(8, 0, 8);
    }

    void createLabel(const char* name) {
        AsmIRBuilder builder(*bb, arch);
        builder.createLabel(name);
    }

    // A compare feeding the branch that consumes it, so the SCC live range stays confined to
    // the pair and does not push the pass's anchors around.
    StinkyInstruction* createGuardedBranch(GFX opcode, int sgpr, const char* target) {
        createSCmpWritingScc(sgpr);
        return createBranchReadingScc(opcode, target);
    }

    // `s_branch <target>` -- a back edge that reads no SCC of its own, so what the climb
    // carries across it came from the body rather than from the branch.
    StinkyInstruction* createUnconditionalBranch(const char* target) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
        inst->addSrcReg(StinkyRegister(std::string(target)));
        inst->addModifier<LabelData>(LabelData{target});
        return inst;
    }

    // A branch with no compare of its own: it reads whatever SCC value is already live, which
    // is what lets a live range reach across a segment boundary.
    StinkyInstruction* createBranchReadingScc(GFX opcode, const char* target) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(opcode, arch));
        inst->addSrcReg(StinkyRegister(std::string(target)));
        inst->addModifier<LabelData>(LabelData{target});
        return inst;
    }

    StinkyInstruction* findLastClusterSignalBefore(size_t limitIdx) const {
        StinkyInstruction* found = nullptr;
        size_t idx = 0;
        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (idx >= limitIdx) break;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (isClusterBarrierWithLiteral(*inst, /*wantSignal=*/true)) found = inst;
            ++idx;
        }
        return found;
    }

    /// First cluster signal strictly between \p afterIdx and \p beforeIdx. Names a signal by
    /// the window it has to land in, for blocks holding more than one.
    StinkyInstruction* findClusterSignalBetween(size_t afterIdx, size_t beforeIdx) const {
        size_t idx = 0;
        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const size_t here = idx++;
            if (here <= afterIdx) continue;
            if (here >= beforeIdx) break;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (isClusterBarrierWithLiteral(*inst, /*wantSignal=*/true)) return inst;
        }
        return nullptr;
    }

    StinkyInstruction* realInstBefore(const StinkyInstruction* anchor) const {
        StinkyInstruction* prev = nullptr;
        for (IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (inst == anchor) return prev;
            if (isPseudoInst(inst)) continue;
            prev = inst;
        }
        return nullptr;
    }

    // Cluster tokens outstanding just before \p limitIdx, read straight down the block. The
    // preheader signal is the first cluster instruction there, so the sweep starts empty and
    // every later position is the state a branch standing there would leave in.
    int inFlightAt(size_t limitIdx) const {
        int outstanding = 0;
        size_t idx = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (idx >= limitIdx) break;
            outstanding += clusterBarrierKind(*cast<StinkyInstruction>(&ir));
            ++idx;
        }
        return outstanding;
    }

    // Every path through the block has to hand the cluster barrier a balanced sequence.
    // Cheap enough that TearDown runs it for every test, and most of the ways this pass can
    // go wrong end up looking like a path that disagrees with another about what is
    // outstanding.
    void expectClusterTokensBalanceOnEveryPath(bool completeProgram = false) {
        if (bb == nullptr) return;
        const std::string problems = clusterTokenPathProblems(*bb, completeProgram);
        EXPECT_TRUE(problems.empty())
            << "cluster tokens do not balance along every path:" << problems << blockListing(*bb);
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

    // Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
    void runPass() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        auto pass = createInsertClusterBarrierPass();
        if (testDumpEnabled()) {
            std::cerr << "\n=== INPUT (before InsertClusterBarrierPass):" << blockListing(*bb)
                      << "\n";
        }
        pass->run(*func, ctx, am);
        if (testDumpEnabled()) {
            std::cerr << "\n=== OUTPUT (after InsertClusterBarrierPass):" << blockListing(*bb)
                      << "\n";
        }
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

    BasicBlock::iterator segBeginAfter(StinkyInstruction* boundary) const {
        return std::next(BasicBlock::iterator(boundary));
    }

    bool anchorInWaitSegment(const IRBase* anchor, BasicBlock::iterator segBegin,
                             const StinkyInstruction* trigger) const {
        if (anchor == trigger) return true;
        for (auto it = segBegin;
             it != BasicBlock::iterator(const_cast<StinkyInstruction*>(trigger)); ++it) {
            if (it.getNodePtr() == anchor) return true;
        }
        return false;
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

// The smallest shape the pass has an answer for, and it already needs all three rules to hold
// together: Rule 1 posts a token below GSU_1, Rule 2's wait in front of the run-up's load
// drinks it, the body's handshake sends its signal across the back edge, and the pre-loop
// signal that owes the first trip a token has to fit between Rule 2's wait and the loop head.
// Take any one rule away and what is left either hangs or leaks. Run with STINKY_TEST_DUMP=1
// to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           SingleHandshakeInALoopIsFedByRule1AndRule2) {
    appendGsu1Preheader();
    openLoop();
    createWMMA(32, 0, 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(40, 8, 0);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);

    // Rule 1 only ever signals, so the first cluster wait in the block is Rule 2's.
    StinkyInstruction* rule2Wait = nullptr;
    for (IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        auto* inst = cast<StinkyInstruction>(&ir);
        if (isClusterBarrierWithLiteral(*inst, /*wantSignal=*/false)) {
            rule2Wait = inst;
            break;
        }
    }
    ASSERT_NE(rule2Wait, nullptr) << blockListing(*bb);
    EXPECT_LT(indexOf(rule2Wait), indexOf(loopHead))
        << "Rule 2's wait belongs to the run-up:" << blockListing(*bb);
    EXPECT_EQ(inFlightAt(indexOf(rule2Wait)), 1)
        << "Rule 2's wait has nothing to drink unless Rule 1 posted first:" << blockListing(*bb);

    StinkyInstruction* preSignalCmp = findClusterWaveCmpAfter(indexOf(rule2Wait));
    ASSERT_NE(preSignalCmp, nullptr)
        << "the body's signal crossed the back edge, so the run-up owes the first trip one:"
        << blockListing(*bb);
    EXPECT_LT(indexOf(preSignalCmp), indexOf(loopHead))
        << "the pre-loop signal has to stand below Rule 2's wait and above the loop head:"
        << blockListing(*bb);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 1)
        << "the first trip enters holding the token its wait drinks:" << blockListing(*bb);

    EXPECT_TRUE(isImmediatelyPrecededByClusterBarrierWait(trigger))
        << "the body's handshake puts its wait in front of the trigger:" << blockListing(*bb);
    StinkyInstruction* bodyWait = realInstBefore(trigger);
    ASSERT_NE(bodyWait, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(bodyWait)), 1)
        << "a wait reached with nothing posted above it is a hang:" << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

TEST_F(InsertClusterBarrierPassTest, TwoHandshakesDoNotOverlapClusterPhases) {
    appendGsu1Preheader();
    openLoop();
    buildTwoHandshakeBody();
    closeLoop();

    runPass();

    const std::vector<int> seq = clusterBarrierSequence();
    int outstanding = 0;
    for (size_t i = 0; i < seq.size(); ++i) {
        outstanding += seq[i];
        EXPECT_LE(outstanding, 1) << "two cluster signals in flight before a wait at index " << i
                                  << blockListing(*bb);
        EXPECT_GE(outstanding, 0) << "a wait with nothing in flight at index " << i
                                  << blockListing(*bb);
    }
    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

TEST_F(InsertClusterBarrierPassTest, WorkgroupBarriersArePreserved) {
    appendGsu1Preheader();
    openLoop();
    StinkyInstruction* firstTrigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(32, 8, 16);
    StinkyInstruction* secondTrigger = appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    closeLoop();

    runPass();

    for (StinkyInstruction* trigger : {firstTrigger, secondTrigger}) {
        ASSERT_NE(indexOf(trigger), static_cast<size_t>(-1))
            << "a workgroup s_barrier_signal -1 the body already had was removed:"
            << blockListing(*bb);
        StinkyInstruction* paired = firstRealInstAfter(trigger);
        ASSERT_NE(paired, nullptr) << blockListing(*bb);
        EXPECT_TRUE(isWorkgroupBarrierWaitInst(*paired))
            << "the pass must not come between a workgroup barrier and its wait:"
            << blockListing(*bb);
    }
}

// Rule 1 and Rule 2 are one mechanism read from two ends: the signal below GSU_1 and the wait
// in front of the first tensor load that drinks it. Neither is testable alone -- a signal
// nobody waits on leaves a token in flight forever, and a wait with nothing above it hangs --
// so this covers both and checks the token actually crosses from one to the other.
TEST_F(InsertClusterBarrierPassTest, Rule1SignalBelowGsu1IsDrunkByRule2Wait) {
    appendGsu1Preheader();

    runPass();

    StinkyInstruction* gsu1 = findLabelNamed(kGSU1LabelName);
    ASSERT_NE(gsu1, nullptr);
    StinkyInstruction* next = firstRealInstAfter(gsu1);
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->getUnifiedOpcode(), GFX::s_cmp_eq_u32);
    ASSERT_GE(next->getSrcRegs().size(), 1u);
    EXPECT_EQ(next->getSrcRegs()[0].getSymbolicName(), kLoopCounterLSymbol)
        << "Rule 1's signal is gated on the trip count:" << blockListing(*bb);

    StinkyInstruction* firstLoad = findFirstTensorLoad();
    ASSERT_NE(firstLoad, nullptr);
    EXPECT_TRUE(isImmediatelyPrecededByClusterBarrierWait(firstLoad))
        << "Rule 2 must insert s_barrier_wait -3 immediately before the first load:"
        << blockListing(*bb);

    StinkyInstruction* rule2Wait = realInstBefore(firstLoad);
    ASSERT_NE(rule2Wait, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(rule2Wait)), 1)
        << "the wait has to have Rule 1's token to drink:" << blockListing(*bb);
}

TEST_F(InsertClusterBarrierPassTest, IdempotencySecondRunIsNoOp) {
    appendGsu1Preheader();
    openLoop();
    buildTwoHandshakeBody();
    closeLoop();
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
    appendGsu1Preheader();
    openLoop();
    for (int i = 0; i < 80; ++i) {
        createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    }
    StinkyInstruction* firstWgSignal = createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*loadS0=*/0, /*loadS1=*/4);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    StinkyInstruction* rule3ClusterCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(rule3ClusterCmp, nullptr);
    EXPECT_LT(indexOf(rule3ClusterCmp), indexOf(firstWgSignal))
        << "cluster signal must forward past intervening workgroup barriers:" << blockListing(*bb);
}

TEST_F(InsertClusterBarrierPassTest, Wait3StopAnchorsAfterFollowingWorkgroupBarrier) {
    appendGsu1Preheader();
    openLoop();
    createWMMA(24, 0, 8);
    StinkyInstruction* preexistingClusterWait = createBarrierWait(kClusterBarrierId);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);
    createWMMA(40, 16, 8);
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    closeLoop();

    runPass();

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

// Segments too short to hold the lead, and only one hop to spend on reaching back for it.
// The climb crosses the label above its own segment, finds the segment there just as short,
// and then runs into the loop head with nothing left to spend. What it settles for is the
// start of the segment it got to -- not the wait's own position, which would buy no lead at
// all.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           Rule3SegmentBoundaryFallbackAnchorsAtSegBegin) {
    appendGsu1Preheader();
    openLoop();
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    createLabel("label_SegmentStart");
    StinkyInstruction* segBeginInst =
        createVAddInBlock(bb, arch, /*destReg=*/0, /*src0Reg=*/4, /*src1Reg=*/8);
    ASSERT_NE(segBeginInst, nullptr);
    StinkyInstruction* labelBeforePass = findLabelNamed("label_SegmentStart");
    ASSERT_NE(labelBeforePass, nullptr);
    ASSERT_EQ(segBeginInst, firstRealInstAfter(labelBeforePass));
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    StinkyInstruction* rule3ClusterCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(rule3ClusterCmp, nullptr);

    EXPECT_LT(indexOf(rule3ClusterCmp), indexOf(segBeginInst))
        << "cluster signal must anchor at a segment start, not co-locate with wait:"
        << blockListing(*bb);
    EXPECT_GT(indexOf(rule3ClusterCmp), indexOf(loopHead))
        << "the hop budget runs out at the loop head, so the signal stays inside the body:"
        << blockListing(*bb);
    EXPECT_LT(indexOf(segBeginInst), indexOf(trigger))
        << "segBegin must precede the workgroup signal:" << blockListing(*bb);
})

// StinkyWaitCntInsertionPass runs before this pass and anchors its counter
// drains on the same workgroup signal Rule 3(b) targets, so the slot right
// before that signal is already taken by an s_wait_tensorcnt. The cluster wait
// must be planted ABOVE that drain:
//
//     s_barrier_wait -3
//     s_wait_tensorcnt N
//     s_barrier_signal -1
//
// The inverted order is equally correct but measured slower. This test pins the
// ordering only; see hoistAboveLeadingWaitCnts for why no mechanism is claimed.
TEST_F(InsertClusterBarrierPassTest, Rule3ClusterWaitIsHoistedAboveTensorDrain) {
    appendGsu1Preheader();
    openLoop();
    createWMMA(24, 0, 8);
    StinkyInstruction* drain = createWaitTensorCnt(0);
    StinkyInstruction* trigger = createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    closeLoop();

    runPass();

    // Rule 2's wait belongs to the run-up, so the handshake's own wait is named by
    // the drain it stands above rather than by being the first one in the block.
    StinkyInstruction* clusterWait = realInstBefore(drain);
    ASSERT_NE(clusterWait, nullptr) << blockListing(*bb);
    EXPECT_TRUE(isClusterBarrierWithLiteral(*clusterWait, /*wantSignal=*/false))
        << "s_barrier_wait -3 must stand directly above the drain it was hoisted over:"
        << blockListing(*bb);
    EXPECT_EQ(realInstBefore(trigger), drain)
        << "the hoist moves the cluster wait, not the drain:" << blockListing(*bb);
}

// Re-running the pass must not plant a second cluster wait just because the
// first one now sits above the drain rather than adjacent to the anchor.
TEST_F(InsertClusterBarrierPassTest, HoistedClusterWaitStaysIdempotent) {
    appendGsu1Preheader();
    openLoop();
    createWMMA(24, 0, 8);
    createWaitTensorCnt(0);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4);
    closeLoop();

    runPass();
    const auto afterFirst = clusterBarrierCounts();
    runPass();
    EXPECT_EQ(clusterBarrierCounts(), afterFirst) << "re-running the pass must be a no-op";
}

// A call ends the climb whatever the hop budget says. It is the one boundary that gets there
// by falling through rather than by naming a label, so the segment below it is reached the
// ordinary way and nothing about the listing warns that the code above ran under a callee's
// register state:
//
//     label_TestLoop:
//     v_wmma ... (x150)     <- where the lead would be met, if the climb were allowed up there
//     s_swappc_b64          <- the call
//     v_wmma ... (x3)       <- all the climb may have, and not enough for the lead
//     s_barrier_signal -1   <- the wait
//
// Asked with a hop to spare, so the answer is about the call rather than about the budget:
// the anchor has to be the first spot below it, with no hop billed and no crossing claimed.
TEST_F(InsertClusterBarrierPassTest, CallStopsTheClimbEvenWithAHopToSpare) {
    const int kLead = 500;
    const int kMaxLead = 900;

    appendGsu1Preheader();
    openLoop();
    // Long enough to hold the lead, which is what makes the call the reason the climb stops.
    for (int i = 0; i < 150; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* call = createCall();
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    closeLoop();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    StinkyInstruction* belowCall = firstRealInstAfter(call);
    ASSERT_NE(belowCall, nullptr);
    StinkyInstruction* aboveCall = firstRealInstAfter(loopHead);
    ASSERT_NE(aboveCall, nullptr);

    PassContext ctx;
    ctx.setGemmTileConfig(config);
    const auto cycleMap = computeEstimatedCyclesPerInstruction(*func, ctx);
    const auto cyclesAt = [&](const StinkyInstruction* inst) -> int64_t {
        auto it = cycleMap.find(inst);
        return (it == cycleMap.end()) ? -1 : static_cast<int64_t>(it->second);
    };
    ASSERT_GE(cyclesAt(trigger), 0);
    ASSERT_GE(cyclesAt(belowCall), 0);
    ASSERT_GE(cyclesAt(aboveCall), 0);

    // Both halves of the premise. Below the call there is not enough room for the lead, so the
    // climb has every reason to keep going; above it there is, so a climb that crossed would
    // have come back with something. Either one missing and the test would prove nothing.
    ASSERT_LT(cyclesAt(trigger) - cyclesAt(belowCall), kLead)
        << "the segment below the call must be too short to hold the lead:" << blockListing(*bb);
    ASSERT_GT(cyclesAt(trigger) - cyclesAt(aboveCall), kLead)
        << "the stretch above the call must be able to hold the lead:" << blockListing(*bb);

    const auto found = cluster_barrier::test::findRule3SignalAnchorByCycleLeadForUnitTest(
        trigger, segBeginAfter(call), trigger, cycleMap, kLead, kMaxLead,
        /*priorWaitAnchors=*/{}, /*maxHops=*/1, loopHead);

    EXPECT_EQ(found.anchor, static_cast<IRBase*>(belowCall))
        << "the climb had a hop left and a better spot above, and still may not pass a call:"
        << blockListing(*bb);
    EXPECT_EQ(found.hops, 0) << "nothing was crossed, so nothing may be billed:"
                             << blockListing(*bb);
    EXPECT_FALSE(found.crossedLoopHead)
        << "a call is not a back edge and asks for no compensation:" << blockListing(*bb);
}

// The cycle lead alone would drop the Rule 3 signal anchor inside a live SCC range:
//
//     v_wmma ...              <- padding, so "in front of the def" is not the segment start
//     s_sub_u32 s90, s90, 1   <- SCC def (carry-out)
//     v_wmma ... / ds_read    <- where the 500-cycle lead point falls
//     v_wmma ...
//     s_cselect_b32           <- reads the value the def computed
//     s_barrier_signal -1 / s_barrier_wait -1 / tensor_load_to_lds
//
// The handshake opens with `s_cmp_eq_u32 sgprWaveIdx, 0`, so planting it at the lead point
// would leave the s_cselect_b32 consuming the wave-id comparison. The anchor scan has to
// keep climbing and come to rest in front of the def instead. Nothing rewrites SCC for it
// afterwards: this def also writes an SGPR, so replaying it would decrement the counter a
// second time. Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
TEST_F(InsertClusterBarrierPassTest, Rule3SignalAnchorClimbsOutOfLiveSccRange) {
    appendGsu1Preheader();
    openLoop();
    for (int i = 0; i < 4; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    // The WMMA counts put the 500-cycle lead point just past the ds_read, i.e. between the
    // def and the reader, which is the placement the scan has to reject.
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    for (int i = 0; i < 16; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    createDsRead(/*destReg=*/100, /*addrReg=*/104);
    for (int i = 0; i < 64; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* sccReader = createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    const size_t defIdx = indexOf(sccDef);
    const size_t readerIdx = indexOf(sccReader);
    ASSERT_NE(defIdx, static_cast<size_t>(-1));
    ASSERT_NE(readerIdx, static_cast<size_t>(-1));

    // Past the loop head, so this is the body's own handshake rather than Rule 1's signal.
    StinkyInstruction* handshakeCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(handshakeCmp, nullptr) << "the pass planted no Rule 3 handshake";
    const size_t handshakeIdx = indexOf(handshakeCmp);

    EXPECT_LT(handshakeIdx, defIdx)
        << "the handshake must climb above the SCC def rather than split its live range:"
        << blockListing(*bb);
    EXPECT_GT(handshakeIdx, indexOf(loopHead))
        << "the scan stopped in front of the def, not by leaving the segment altogether:"
        << blockListing(*bb);

    // The value the reader consumes is whatever the last SCC write before it left behind.
    const StinkyInstruction* lastWriter = lastSccWriterBefore(readerIdx);
    ASSERT_NE(lastWriter, nullptr);
    EXPECT_FALSE(isClusterWaveCmp(*lastWriter))
        << "the handshake's wave-id compare is the last SCC write before the reader, so the "
           "reader consumes it instead of the carry-out s_sub_u32 s90, s90, 1 computed:"
        << blockListing(*bb);

    // The lead still has to buy something: the signal must sit ahead of the barrier it
    // was derived from, not collapse onto it.
    EXPECT_LT(handshakeIdx, indexOf(trigger))
        << "the cluster signal must still lead its workgroup barrier:" << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

// Same shape, except the reader is behind a branch, where reading the block top to bottom
// never arrives:
//
//     label_TestLoop:
//     s_sub_u32 s90, s90, 1   <- SCC def
//     v_wmma ... (x81)        <- the lead point, and where a fall-through reading settles
//     s_barrier_signal -1     <- the wait
//     s_barrier_wait -1 / tensor_load_to_lds
//     s_branch label_SccUser  <- the only way on
//     s_cmp_eq_u32 s93, 0     <- the trap: a rewrite nothing reaches
//     label_SccUser:
//     s_cselect_b32           <- the reader, and it wants what the def left
//
// Read as a straight line the def looks dead from the lead point on, because the rewrite that
// appears to close its range is the trap -- so the handshake settles there and its
// `s_cmp_eq_u32 sgprWaveIdx, 0` is what the s_cselect_b32 ends up consuming. Following the
// branch is the only way to see that the range is live throughout, and the anchor then has to
// clear it the same way it does when the reader is in plain sight.
TEST_F(InsertClusterBarrierPassTest, Rule3SignalAnchorFollowsBranchesToFindTheSccReader) {
    appendGsu1Preheader();
    openLoop();
    // Padding, so that "in front of the def" is not just the segment start by default.
    for (int i = 0; i < 4; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    // The same stretch Rule3SignalAnchorClimbsOutOfLiveSccRange uses: long enough to put the
    // lead point between the def and the wait, short enough that clearing the range upward is
    // still within kRule3SignalMaxLeadCycles.
    for (int i = 0; i < 16; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    createDsRead(/*destReg=*/100, /*addrReg=*/104);
    for (int i = 0; i < 64; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createUnconditionalBranch("label_SccUser");
    StinkyInstruction* unreachableRewrite = createSCmpWritingScc(/*srcSgpr=*/93);
    createLabel("label_SccUser");
    StinkyInstruction* sccReader = createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    const size_t defIdx = indexOf(sccDef);
    ASSERT_NE(defIdx, static_cast<size_t>(-1));
    ASSERT_NE(indexOf(sccReader), static_cast<size_t>(-1));

    // The premise: the trap really does sit between the wait and the reader, so a walk that
    // read straight down the block would stop there and call the value dead.
    ASSERT_LT(indexOf(trigger), indexOf(unreachableRewrite));
    ASSERT_LT(indexOf(unreachableRewrite), indexOf(sccReader));

    StinkyInstruction* handshakeCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(handshakeCmp, nullptr) << "the pass planted no Rule 3 handshake";
    const size_t handshakeIdx = indexOf(handshakeCmp);

    EXPECT_LT(handshakeIdx, defIdx)
        << "the reader is behind a branch, so the def's range is live at the wait and the "
           "handshake must climb above the def rather than settle inside it:"
        << blockListing(*bb);
    EXPECT_GT(handshakeIdx, indexOf(loopHead))
        << "the scan stopped in front of the def, not by leaving the segment altogether:"
        << blockListing(*bb);
}

// Same shape, but with the def..reader range stretched until climbing out of it would put
// the signal more than kRule3SignalMaxLeadCycles ahead of its wait. Clearing the range then
// costs more overlap than it buys, so the anchor drops below the reader instead and ends up
// nearer the wait than the nominal lead would have placed it.
TEST_F(InsertClusterBarrierPassTest, Rule3SignalAnchorSinksBelowOverlongSccRange) {
    appendGsu1Preheader();
    openLoop();
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    for (int i = 0; i < 60; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    createDsRead(/*destReg=*/100, /*addrReg=*/104);
    for (int i = 0; i < 64; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* sccReader = createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    // Room below the range for the anchor to land on.
    for (int i = 0; i < 10; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    const size_t defIdx = indexOf(sccDef);
    const size_t readerIdx = indexOf(sccReader);
    ASSERT_NE(defIdx, static_cast<size_t>(-1));
    ASSERT_NE(readerIdx, static_cast<size_t>(-1));

    StinkyInstruction* handshakeCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(handshakeCmp, nullptr) << "the pass planted no Rule 3 handshake";
    const size_t handshakeIdx = indexOf(handshakeCmp);

    EXPECT_GT(handshakeIdx, readerIdx)
        << "an overlong range must be settled below, not by climbing over the def:"
        << blockListing(*bb);

    // Sinking below the range is only worth doing if it still leaves a real lead.
    EXPECT_LT(handshakeIdx, indexOf(trigger))
        << "the signal must not collapse onto its workgroup barrier:" << blockListing(*bb);

    const StinkyInstruction* lastWriter = lastSccWriterBefore(readerIdx);
    ASSERT_NE(lastWriter, nullptr);
    EXPECT_FALSE(isClusterWaveCmp(*lastWriter))
        << "the reader must still see the carry-out s_sub_u32 s90, s90, 1 computed:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

// A boundary decides the anchor before the cycle lead ever gets a say, and the spot it
// picks is inside a live SCC range:
//
//     s_barrier_wait -3       <- boundary: the scan may not climb past this
//     s_sub_u32 s90, s90, 1   <- SCC def, stranded above the reachable region
//     v_wmma ...
//     s_barrier_signal -1     <- the scan resumes below this pair, which is where the
//     s_barrier_wait -1          boundary hands back an anchor -- inside the live range
//     v_wmma ...
//     s_cselect_b32           <- reads the value the def computed
//     v_wmma ...
//     s_barrier_signal -1 / s_barrier_wait -1 / tensor_load_to_lds
//
// Climbing is not an option here, so the only legal correction is the other direction:
// drop below the reader. The boundary itself still has to hold.
TEST_F(InsertClusterBarrierPassTest, Rule3BoundaryForcedAnchorSinksOutOfLiveSccRange) {
    appendGsu1Preheader();
    openLoop();
    StinkyInstruction* clusterWait = createBarrierWait(kClusterBarrierId);
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    for (int i = 0; i < 2; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    for (int i = 0; i < 20; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* sccReader = createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    // Room below the range for the anchor to land on.
    for (int i = 0; i < 5; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/0, /*src1Reg=*/4);
    closeLoop();

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    const size_t defIdx = indexOf(sccDef);
    const size_t readerIdx = indexOf(sccReader);
    ASSERT_NE(defIdx, static_cast<size_t>(-1));
    ASSERT_NE(readerIdx, static_cast<size_t>(-1));

    StinkyInstruction* handshakeCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(handshakeCmp, nullptr) << "the pass planted no Rule 3 handshake";
    const size_t handshakeIdx = indexOf(handshakeCmp);

    EXPECT_GT(handshakeIdx, indexOf(clusterWait))
        << "the cluster wait is a hard boundary and must not be crossed:" << blockListing(*bb);
    EXPECT_GT(handshakeIdx, readerIdx)
        << "the anchor cannot climb over the def here, so it must settle below the reader "
           "rather than split the range:"
        << blockListing(*bb);
    EXPECT_LT(handshakeIdx, indexOf(trigger))
        << "the signal must not collapse onto its workgroup barrier:" << blockListing(*bb);

    const StinkyInstruction* lastWriter = lastSccWriterBefore(readerIdx);
    ASSERT_NE(lastWriter, nullptr);
    EXPECT_FALSE(isClusterWaveCmp(*lastWriter))
        << "the reader must still see the carry-out s_sub_u32 s90, s90, 1 computed:"
        << blockListing(*bb);
}

// The one shape the downward correction has no answer for: the live range runs past the wait
// itself, so no spot between the lead point and the wait is safe and neither is the wait's own
// spot, which is where the search otherwise gives up.
//
//     label_TestLoop:
//     v_wmma ...                   <- long enough to blow past maxLeadCycles
//     s_barrier_signal -1          <- the wait, and SCC is live in front of it
//     s_barrier_wait -1 / tensor_load_to_lds
//     s_cselect_b32                <- the only reader, and it is below the wait
//
// The pass may not quietly settle for the wait here: the handshake it plants opens with
// `s_cmp_eq_u32`, which would clobber the value that s_cselect_b32 still wants. Nor is there
// anywhere else to go. So this is a bug report about the caller rather than a case to recover
// from, and a real block never gets here -- a range that reaches the wait is closed by
// something the caller already looked at. The abort is what keeps a future caller from
// discovering that the hard way.
TEST_F(InsertClusterBarrierPassTest, Rule3SignalAnchorAbortsWhenSccIsLiveAtItsWait) {
    appendGsu1Preheader();
    openLoop();
    // The climb has to get past maxLeadCycles while still standing in the range: that is what
    // sends it back down looking for a spot below, which is the search that comes up empty.
    for (int i = 0; i < 200; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    // Below the wait with no def between the two, so SCC reads live everywhere above it.
    createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    closeLoop();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);

    PassContext ctx;
    ctx.setGemmTileConfig(config);
    const auto cycleMap = computeEstimatedCyclesPerInstruction(*func, ctx);
    ASSERT_NE(cycleMap.find(trigger), cycleMap.end())
        << "trigger must be present in the estimated cycle map:" << blockListing(*bb);
    const BasicBlock::iterator segBegin = segBeginAfter(loopHead);

    EXPECT_DEATH(
        {
            (void)cluster_barrier::test::findRule3SignalAnchorByCycleLeadForUnitTest(
                trigger, segBegin, trigger, cycleMap, /*leadCycles=*/500, /*maxLeadCycles=*/900,
                /*priorWaitAnchors=*/{}, /*maxHops=*/0, loopHead);
        },
        "SCC live at the wait");
}

// The downward correction is allowed through one kind of wall, and this is the shape that asks
// it to be. The range is held open by the exit branch itself -- a branch with no compare of
// its own, reading what the def above it left -- so the first spot outside the range is the
// one the branch falls through to, on the far side of a segment boundary:
//
//     label_TestLoop:
//     s_sub_u32 s90, s90, 1     <- def: the exit predicate
//     v_wmma ... (x150)         <- climbed, but every spot here is inside the range
//     s_cbranch_scc1 label_TestLoopEnd   <- the reader, and a segment boundary
//     v_wmma ...                <- the only spot left, and it is below the boundary
//     s_barrier_signal -1       <- the wait
//
// Refusing to step over that branch would leave the scan with nothing and send the signal
// onto the wait's own spot, giving up a lead it was entitled to. Stepping over it is safe
// because the walk only moves towards the wait, so it stays on the path the wait is on.
//
// What must not happen is the anchor landing in the stretch above the branch: the handshake
// opens with `s_cmp_eq_u32 sgprWaveIdx, 0`, and that would hand the exit branch the wave
// comparison instead of the loop's own predicate.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           DownwardScanLandsOnTheFallThroughSideOfAnExitBranch) {
    appendGsu1Preheader();
    openLoop();
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    // Long enough that climbing clear of the range would cost more than the ceiling allows,
    // which is what turns the climb around and puts the downward scan in charge.
    for (int i = 0; i < 150; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* exitBranch =
        createBranchReadingScc(GFX::s_cbranch_scc1, "label_TestLoopEnd");
    StinkyInstruction* landing = createWMMA(16, 0, 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);

    // The premise, before the pass moves anything: taking the whole stretch above the branch
    // would break the ceiling. On a shorter one the climb would settle up there legitimately
    // and this test would be watching the wrong thing.
    {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        const auto cycleMap = computeEstimatedCyclesPerInstruction(*func, ctx);
        const auto cyclesAt = [&](const StinkyInstruction* inst) -> int64_t {
            auto it = cycleMap.find(inst);
            return (it == cycleMap.end()) ? -1 : static_cast<int64_t>(it->second);
        };
        ASSERT_GE(cyclesAt(trigger), 0);
        ASSERT_GE(cyclesAt(sccDef), 0);
        ASSERT_GT(cyclesAt(trigger) - cyclesAt(sccDef), 900)
            << "the range has to be deeper than the ceiling for the climb to turn around:"
            << blockListing(*bb);
    }

    runPass();

    StinkyInstruction* handshakeCmp = findClusterWaveCmpAfter(indexOf(loopHead));
    ASSERT_NE(handshakeCmp, nullptr) << "the pass planted no Rule 3 handshake";
    const size_t handshakeIdx = indexOf(handshakeCmp);

    EXPECT_GT(handshakeIdx, indexOf(exitBranch))
        << "the only spot outside the range is below the exit branch, so the scan had to step "
           "over it to get there:"
        << blockListing(*bb);
    EXPECT_LT(handshakeIdx, indexOf(trigger))
        << "the signal must still lead its workgroup barrier rather than collapse onto it:"
        << blockListing(*bb);
    EXPECT_LT(handshakeIdx, indexOf(landing))
        << "the handshake goes in front of the spot the scan picked, which is the branch's "
           "fall-through:"
        << blockListing(*bb);

    // What the step-over is not allowed to cost: the branch has to keep reading the def.
    const StinkyInstruction* lastWriter = lastSccWriterBefore(indexOf(exitBranch));
    ASSERT_NE(lastWriter, nullptr);
    EXPECT_FALSE(isClusterWaveCmp(*lastWriter))
        << "the exit branch must still see the predicate s_sub_u32 s90, s90, 1 computed:"
        << blockListing(*bb);

    // The anchor stayed in the wait's own segment, so no edge leaves holding a token.
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 0)
        << "nothing crossed the back edge, so the loop must not be entered expecting a token:"
        << blockListing(*bb);
    EXPECT_EQ(findLabelNamed("label_TestLoopEnd_skipCBWait"), nullptr)
        << "no token leaves this loop, so there is no drain to route anything around:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// A climb can cross edges and still come back empty-handed. Here the opening segment's signal
// follows the latch across the back edge, lands in the tail segment, and finds SCC live from
// the moment it arrives until the latch that reads it -- the whole segment is one live range,
// with no safe spot in it and no room below it. The only legal answer left is the caller's
// default, which is the wait's own position, so the signal does not move at all:
//
//     label_TestLoop:
//     <short segment>                  <- its signal wants to cross the back edge
//     s_cmp / s_cbranch label_TestLoopEnd
//     <long segment>
//     s_cmp / s_cbranch label_TestLoopEnd
//     <short segment>                  <- this one really does hoist, across the exit above it
//     s_cmp_eq_u32 s93, 0              <- SCC def
//     s_cbranch_scc1 label_TestLoopEnd
//     v_wmma x2                        <- tail segment, SCC live throughout
//     s_cbranch_scc0 label_TestLoop    <- the latch reads it, so the range never closes
//
// What the loop is billed for has to describe where the signal ended up, not how far the
// search travelled to get there. A crossing that was given up on leaves no signal on the far
// side, so charging the loop for it buys a preheader signal that nothing in the body ever
// consumes -- the loop head would then be entered holding a token on the first trip and empty
// on every later one. The third segment is there to keep that visible: it hoists across an
// exit for real, so the loop needs a drain, and the bogus preheader signal survives to be
// caught instead of being discarded together with a compensation the loop never needed.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           ClimbThatGivesUpIsNotBilledForCrossingTheBackEdge) {
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);

    createLabel("label_TestLoop");
    // Short: nothing above it inside the body, so its signal leaves across the back edge.
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    StinkyInstruction* firstExit =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    StinkyInstruction* secondExit =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");
    // A segment that really does hoist, so some edge out of this loop really does carry a
    // token. Without it the loop would need no drain at all, and a preheader signal emitted
    // on a false crossing would be dropped along with everything else instead of showing up.
    for (int i = 0; i < 2; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/16, /*loadS1=*/20);
    // This compare opens the range that swallows the tail segment.
    StinkyInstruction* thirdExit =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/93, "label_TestLoopEnd");
    for (int i = 0; i < 2; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    // No compare of its own, so the range above stays live all the way down to here.
    createBranchReadingScc(GFX::s_cbranch_scc0, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);

    // The premise: the signal really did stay put. A handshake that did not move plants its
    // signal directly on top of its own wait, with only the label that closes the wave-0 gate
    // between them.
    StinkyInstruction* firstLoopWait = nullptr;
    size_t idx = 0;
    for (IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        auto* inst = cast<StinkyInstruction>(&ir);
        if (idx > indexOf(loopHead) && isClusterBarrierWithLiteral(*inst, /*wantSignal=*/false)) {
            firstLoopWait = inst;
            break;
        }
        ++idx;
    }
    ASSERT_NE(firstLoopWait, nullptr)
        << "the pass planted no handshake in the loop:" << blockListing(*bb);
    StinkyInstruction* pairedSignal = findLastClusterSignalBefore(indexOf(firstLoopWait));
    ASSERT_NE(pairedSignal, nullptr);
    EXPECT_EQ(indexOf(firstLoopWait) - indexOf(pairedSignal), 2u)
        << "this test is only meaningful while the climb gives up and the signal sits on its "
           "own wait:"
        << blockListing(*bb);

    // The Rule 1 signal at GSU_1 also sits above the loop head, so what says a preheader
    // signal was emitted is not the presence of one but whether anything is still outstanding
    // by the time the loop head is reached.
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 0)
        << "the signal never made it across the back edge, so nothing must be posted ahead of "
           "the loop for it:"
        << blockListing(*bb);

    // The other half of the premise: a different segment did hoist, so this loop genuinely
    // needs a drain. Without that the whole compensation would be skipped and a preheader
    // signal emitted on the false crossing would never become visible.
    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* drainWait = firstRealInstAfter(exitLabel);
    ASSERT_NE(drainWait, nullptr);
    ASSERT_TRUE(isClusterBarrierWithLiteral(*drainWait, /*wantSignal=*/false))
        << "this test is only meaningful while one segment really does hoist across an exit:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*secondExit), "label_TestLoopEnd")
        << "the segment below this exit hoisted across it, so this edge leaves holding a token "
           "and must land on the drain:"
        << blockListing(*bb);
    const char* kBypassLabel = "label_TestLoopEnd_skipCBWait";
    EXPECT_EQ(getBranchTarget(*firstExit), kBypassLabel)
        << "this edge leaves empty-handed and must be routed past the drain:" << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*thirdExit), kBypassLabel)
        << "this edge leaves empty-handed and must be routed past the drain:" << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// The preheader is not always willing to take a signal. Here a live SCC range runs from the
// preheader across the loop head into the body, so every spot the climb may settle on sits
// between a def and its reader, and there is no room below the range either -- the range ends
// inside the loop:
//
//     s_sub_u32 s90, s90, 1     <- SCC def
//     s_barrier_signal -1 / s_barrier_wait -1
//     v_wmma ...                <- where the preheader signal wants to go, but SCC is live
//     label_TestLoop:
//     s_cselect_b32             <- the reader, on the far side of the loop head
//
// Climbing above the barrier is not a way out. Wave 0 issues the signal for the whole group,
// so it may not run ahead of the barrier that gathers the group, and every spot below the
// barrier belongs to the range.
//
// The opening segment is short enough that its signal would rather climb across the back
// edge. It must not: doing so hands the signal to the *next* trip, and the first trip is then
// left waiting on a token that the preheader was never able to post. Crossing the back edge
// and placing a preheader signal are one decision, not two, so when the preheader cannot be
// served the signal gives up its lead and settles between the reader and its own wait, inside
// the loop.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           SignalStaysInLoopWhenThePreheaderHasNoSafeSccSpot) {
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    // The climb out of the loop head stops behind this barrier, which puts it inside the
    // range the def opened, and it may not step above the barrier to get out.
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);

    createLabel("label_TestLoop");
    // The reader sits below the loop head, so the range covers every candidate spot in the
    // preheader and there is nowhere below it to drop to either.
    StinkyInstruction* sccReader = createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    StinkyInstruction* exitBranch =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/93, "label_TestLoopEnd");
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 0)
        << "the preheader had nowhere safe to post a signal, so the loop must not be entered "
           "expecting one:"
        << blockListing(*bb);

    // The premise: the range really does cover the preheader and end inside the body.
    ASSERT_LT(indexOf(sccDef), indexOf(loopHead));
    ASSERT_GT(indexOf(sccReader), indexOf(loopHead));
    const StinkyInstruction* lastWriter = lastSccWriterBefore(indexOf(sccReader));
    ASSERT_NE(lastWriter, nullptr);
    EXPECT_FALSE(isClusterWaveCmp(*lastWriter))
        << "the reader must still see what the carry-out computed:" << blockListing(*bb);

    // Where the signal ended up instead: below the reader that closed the range, and above the
    // wait it belongs to. No lead, but a pair the first trip can actually complete.
    StinkyInstruction* loopWait = nullptr;
    size_t idx = 0;
    for (IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        auto* inst = cast<StinkyInstruction>(&ir);
        if (idx > indexOf(loopHead) && isClusterBarrierWithLiteral(*inst, /*wantSignal=*/false)) {
            loopWait = inst;
            break;
        }
        ++idx;
    }
    ASSERT_NE(loopWait, nullptr);
    StinkyInstruction* pairedSignal = findLastClusterSignalBefore(indexOf(loopWait));
    ASSERT_NE(pairedSignal, nullptr);
    EXPECT_GT(indexOf(pairedSignal), indexOf(sccReader))
        << "the signal had to sink below the reader that keeps the range live:"
        << blockListing(*bb);
    EXPECT_LT(indexOf(pairedSignal), indexOf(loopWait))
        << "the signal has to stay above the wait that consumes it:" << blockListing(*bb);

    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* afterExit = firstRealInstAfter(exitLabel);
    ASSERT_NE(afterExit, nullptr);
    EXPECT_FALSE(isClusterBarrierWithLiteral(*afterExit, /*wantSignal=*/false))
        << "no signal crosses an edge out of this loop, so there is nothing to drain:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*exitBranch), "label_TestLoopEnd")
        << "an exit branch must be left alone when there is no drain to route around:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// This signal says the run-up is finished, so it belongs at the end of the run-up rather than
// the start of it: the search climbs from the loop head and takes the first workgroup barrier
// it meets, not the first one below the cluster wait. A preheader with two of them says which
// reading is in force:
//
//     tensor_load_to_lds        <- Rule 2's wait goes in front of this
//     s_barrier_signal -1       <- a barrier the signal climbs past
//     s_barrier_wait -1
//     v_wmma x3
//     s_barrier_signal -1       <- the barrier closest to the loop: the signal goes below its
//     s_barrier_wait -1            wait
//     v_wmma x3
//     label_TestLoop:
//
// Sitting behind the upper pair instead would announce this workgroup ready while the work
// between the two barriers is still ahead of it.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           PreheaderSignalSitsBehindTheBarrierClosestToTheLoop) {
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* lowerBarrierSignal = createBarrierSignal(kWorkgroupBarrierId);
    StinkyInstruction* lowerBarrierWait = createBarrierWait(kWorkgroupBarrierId);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);

    createLabel("label_TestLoop");
    // Short, so its signal leaves across the back edge and the preheader has to serve the
    // first trip.
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 1)
        << "the opening segment's signal left across the back edge, so the first trip has to "
           "be handed a token by the preheader:"
        << blockListing(*bb);

    StinkyInstruction* preSignal = findLastClusterSignalBefore(indexOf(loopHead));
    ASSERT_NE(preSignal, nullptr);
    EXPECT_GT(indexOf(preSignal), indexOf(lowerBarrierWait))
        << "the signal belongs at the end of the run-up, behind the barrier closest to the "
           "loop:"
        << blockListing(*bb);
    EXPECT_GT(indexOf(preSignal), indexOf(lowerBarrierSignal))
        << "wave 0 may not announce the group ready before the group has gathered:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// The same preheader with the barriers taken out. Rule 2's wait is still there, and below it
// nothing but plain work all the way to the loop:
//
//     tensor_load_to_lds        <- Rule 2's wait goes in front of this
//     v_wmma x3
//     label_PreLoopTail:        <- the last label of the preheader
//     v_wmma x3
//     label_TestLoop:
//
// The first trip still needs a token, and wave 0 still may not announce the group ready while
// its other waves are behind, so the pass has to bring a barrier of its own. It goes below
// the last label, which is where every path into the loop passes through, and the signal sits
// behind it. Unlike Rule 1's signal this one carries no trip-count gate: its wait is below the
// loop, and the two are reached on exactly the same paths.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           PreheaderWithNoBarrierBringsOneBelowItsLastLabel) {
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    createLabel("label_PreLoopTail");
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);

    createLabel("label_TestLoop");
    // Short, so its signal leaves across the back edge and the preheader has to serve the
    // first trip.
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 1)
        << "the opening segment's signal left across the back edge, so the first trip has to "
           "be handed a token by the preheader:"
        << blockListing(*bb);

    StinkyInstruction* tail = findLabelNamed("label_PreLoopTail");
    ASSERT_NE(tail, nullptr);
    StinkyInstruction* preSignal = findLastClusterSignalBefore(indexOf(loopHead));
    ASSERT_NE(preSignal, nullptr);
    EXPECT_GT(indexOf(preSignal), indexOf(tail))
        << "the barrier the signal needs goes below the preheader's last label, and the signal "
           "below that:"
        << blockListing(*bb);

    // The barrier the pass brought: the closest workgroup pair above the signal, which has to
    // be one it planted below the label rather than anything that was already there.
    StinkyInstruction* wgWait = nullptr;
    StinkyInstruction* wgSignal = nullptr;
    size_t idx = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        if (idx >= indexOf(preSignal)) break;
        auto* inst = const_cast<StinkyInstruction*>(cast<StinkyInstruction>(&ir));
        if (isWorkgroupBarrierSignalInst(*inst)) wgSignal = inst;
        if (isWorkgroupBarrierWaitInst(*inst)) wgWait = inst;
        ++idx;
    }
    ASSERT_NE(wgSignal, nullptr) << blockListing(*bb);
    ASSERT_NE(wgWait, nullptr) << blockListing(*bb);
    EXPECT_GT(indexOf(wgSignal), indexOf(tail))
        << "the preheader had no barrier of its own, so the pass must have planted this one "
           "below the last label:"
        << blockListing(*bb);
    EXPECT_LT(indexOf(wgSignal), indexOf(wgWait)) << "signal then wait:" << blockListing(*bb);

    // Rule 1's signal is gated on the trip count; this one must not be, or the paths that
    // reach its wait below the loop would not all have posted it.
    idx = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const size_t here = idx++;
        if (here < indexOf(wgWait) || here >= indexOf(preSignal)) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        if (inst->getUnifiedOpcode() != GFX::s_cmp_eq_u32) continue;
        ASSERT_FALSE(inst->getSrcRegs().empty());
        EXPECT_NE(inst->getSrcRegs()[0].getSymbolicName(), kLoopCounterLSymbol)
            << "the preheader signal must not be gated on the trip count:" << blockListing(*bb);
    }

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// No workgroup barrier and no preheader label before the loop. The nearest run-up anchor
// is the last tensor load, so the signal goes below it with a planted workgroup barrier:
//
//     tensor_load_to_lds            <- nearest anchor
//     s_barrier_signal -1           <- pass-planted
//     s_barrier_wait -1
//     s_barrier_signal -3
//     label_TestLoop:
//
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           PreheaderSkipsGsu1AndFallsBackToLastRunUpLoad) {
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* lastLoad = createTensorLoadInBlock(bb, arch, /*src0Reg=*/68, /*src1Reg=*/72);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);

    createLabel("label_TestLoop");
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 1)
        << "the opening segment's signal left across the back edge, so the first trip has to "
           "be handed a token by the preheader:"
        << blockListing(*bb);

    StinkyInstruction* preSignal = findLastClusterSignalBefore(indexOf(loopHead));
    ASSERT_NE(preSignal, nullptr);
    EXPECT_GT(indexOf(preSignal), indexOf(lastLoad))
        << "with no preheader label, the signal falls back below the last run-up load:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// A preheader whose last stop before the loop is a cluster wait rather than a barrier:
//
//     s_barrier_signal -1
//     s_barrier_wait -1         <- the workgroup barrier, further from the loop
//     v_wmma x3
//     s_barrier_signal -3
//     s_barrier_wait -3         <- closer to the loop, and it drinks a cluster token
//     v_wmma x3
//     label_TestLoop:
//
// The signal the first trip is owed has to go *below* that wait. Above it the wait would
// drink it on the way in, leaving the loop head empty-handed again while the exit still
// drains a token nobody has -- the compensation would pay for itself twice and land nothing.
// A cluster wait says nothing about where the workgroup's other waves are, so unlike a
// s_barrier_wait -1 this stop still needs a barrier planted with the signal.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           PreheaderSignalStaysBelowAClusterWaitThatWouldDrinkIt) {
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    StinkyInstruction* wgSignal = createBarrierSignal(kWorkgroupBarrierId);
    StinkyInstruction* wgWait = createBarrierWait(kWorkgroupBarrierId);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    // Paired, so the token this wait drinks is one the preheader itself posted and the test
    // is about placement rather than about an unbalanced block.
    createBarrierSignal(kClusterBarrierId);
    StinkyInstruction* clusterWait = createBarrierWait(kClusterBarrierId);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);

    createLabel("label_TestLoop");
    // Short, so its signal leaves across the back edge and the preheader has to serve the
    // first trip.
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 1)
        << "the opening segment's signal left across the back edge, so the first trip has to "
           "be handed a token by the preheader:"
        << blockListing(*bb);

    // Named by where it is rather than by being the last one before the loop: the preheader
    // already holds cluster signals of its own -- Rule 1's and this test's -- so "the last
    // one above the loop head" would name whichever of those the compensation failed to get
    // below, and the assertion would then be reading its own premise.
    StinkyInstruction* compensation =
        findClusterSignalBetween(indexOf(clusterWait), indexOf(loopHead));
    ASSERT_NE(compensation, nullptr)
        << "no compensating signal between the cluster wait and the loop: one placed above "
           "that wait is drunk by it instead of reaching the first trip:"
        << blockListing(*bb);
    EXPECT_GT(indexOf(compensation), indexOf(wgWait))
        << "the cluster wait is nearer the loop than the workgroup barrier, so stopping at "
           "the barrier would put the signal above it:"
        << blockListing(*bb);

    // The cluster wait gathers nothing, so the pass owes this signal a barrier of its own,
    // planted between the wait it stopped at and the signal.
    StinkyInstruction* plantedSignal = nullptr;
    size_t idx = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        if (idx >= indexOf(compensation)) break;
        auto* inst = const_cast<StinkyInstruction*>(cast<StinkyInstruction>(&ir));
        if (isWorkgroupBarrierSignalInst(*inst)) plantedSignal = inst;
        ++idx;
    }
    ASSERT_NE(plantedSignal, nullptr) << blockListing(*bb);
    EXPECT_GT(indexOf(plantedSignal), indexOf(clusterWait))
        << "wave 0 may not announce the group ready before the group has gathered, and the "
           "cluster wait it stopped at does not gather it:"
        << blockListing(*bb);
    EXPECT_NE(plantedSignal, wgSignal)
        << "the barrier above the signal must be a planted one, not the pre-existing pair:"
        << blockListing(*bb);
    StinkyInstruction* plantedWait = firstRealInstAfter(plantedSignal);
    ASSERT_NE(plantedWait, nullptr);
    EXPECT_TRUE(isWorkgroupBarrierWaitInst(*plantedWait))
        << "the planted barrier is a pair, and the signal goes below its wait:"
        << blockListing(*bb);
    EXPECT_LT(indexOf(plantedWait), indexOf(compensation))
        << "the signal goes below the wait of the pair the pass planted:" << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// Loop-body handshakes always include a cluster wait at the trigger (independent of Rule 2,
// which waits on the kernel's first load in the preheader).
TEST_F(InsertClusterBarrierPassTest, LoopBodyHandshakeAlwaysWaitsAtTrigger) {
    appendGsu1Preheader();
    openLoop();
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(32, 8, 16);
    closeLoop();

    runPass();

    EXPECT_TRUE(isImmediatelyPrecededByClusterBarrierWait(trigger))
        << "Rule 3 always waits at the trigger regardless of Rule 2:" << blockListing(*bb);
    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

// The other half of the loop story: a body whose segments are all long enough to hold the
// 500-cycle lead on their own. Every signal comes to rest between its own segment's start
// and its wait, so no trip ever hands the next one a token and the loop needs no wrapping at
// all -- no preheader signal, no drain below the exit, no exit branch sent anywhere new.
//
// This is the case that says what the compensation costs: everything the hoisting tests
// assert the pass emits has to be absent here, or the pass is paying for a carried signal
// that no segment actually carries.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
TEST_F(InsertClusterBarrierPassTest, SegmentsLongEnoughToHoldTheLeadNeedNoLoopCompensation) {
    // ~8 cycles apiece, so this clears the 500-cycle lead with room to spare and the climb
    // stops well short of the segment start.
    const auto fillSegment = [&] {
        for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    };

    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);

    createLabel("label_TestLoop");
    fillSegment();
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    StinkyInstruction* firstExit =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    fillSegment();
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    StinkyInstruction* secondExit =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");
    fillSegment();
    appendHandshake(/*loadS0=*/16, /*loadS1=*/20);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 0)
        << "no segment carries a signal across a trip, so the loop must not be entered "
           "holding one:"
        << blockListing(*bb);

    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* afterExit = firstRealInstAfter(exitLabel);
    ASSERT_NE(afterExit, nullptr);
    EXPECT_FALSE(isClusterBarrierWithLiteral(*afterExit, /*wantSignal=*/false))
        << "there is nothing left in flight at the exit, so there is nothing to drain:"
        << blockListing(*bb);
    EXPECT_EQ(findLabelNamed("label_TestLoopEnd_skipCBWait"), nullptr)
        << "a bypass label with no drain to bypass:" << blockListing(*bb);

    EXPECT_EQ(getBranchTarget(*firstExit), "label_TestLoopEnd")
        << "an exit branch must be left alone when nothing is in flight to route around:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*secondExit), "label_TestLoopEnd")
        << "an exit branch must be left alone when nothing is in flight to route around:"
        << blockListing(*bb);

    // One handshake per segment, plus the Rule 1 signal at GSU_1 and the Rule 2 wait that
    // consumes it in front of the first load.
    const auto [signals, waits] = clusterBarrierCounts();
    EXPECT_EQ(signals, 4) << "expected one signal per segment and one at GSU_1:"
                          << blockListing(*bb);
    EXPECT_EQ(waits, 4) << "expected one wait per segment and one before the first load:"
                        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

// A loop whose first handshake climbs out of its opening segment carries a cluster signal
// from one trip into the next, so the pass wraps it: a signal in the preheader to feed the
// first trip, and a wait below the exit label to swallow the last one.
//
// Two branches leave the body for that exit label, and each one meets the drain in a
// different state:
//
//     S-1                             <- preheader, feeds the first trip
//     label_TestLoop:
//     W-1
//     s_cbranch_scc1 label_TestLoopEnd    <- nothing in flight: must skip the drain wait
//     S-2
//     s_cbranch_scc1 label_TestLoopEnd
//     W-2
//     S-3
//     s_cbranch_scc1 label_TestLoopEnd    <- S-3 outstanding: must reach the drain wait
//     W-3
//     S-1
//     s_cbranch_scc0 label_TestLoop
//     label_TestLoopEnd:
//     W-1                             <- drains the last trip's carried signal
//     label_TestLoopEnd_skipCBWait:
//
// All three branches sit in the same loop and leave for the same label, so nothing about
// where they stand tells them apart; only what is outstanding there does. What puts S-2
// below the first branch rather than above it is the edge right underneath: a handshake may
// climb one segment, and S-2 spends that hop on that edge before it ever reaches the first
// branch. The handshake at the bottom has no such edge in the way, so its signal climbs
// straight over the branch above it.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           ExitBranchSkipsDrainWaitOnlyWithNoTokenInFlight) {
    // Preheader: the compensating signal comes to rest behind this workgroup barrier.
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);

    createLabel("label_TestLoop");
    // Every segment here is far too short to hold the 500-cycle lead, so every handshake
    // climbs one segment. The first one climbs across the loop head, which is what leaves a
    // signal carried from trip to trip and puts the loop up for compensation.
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(40, 16, 8);
    StinkyInstruction* branchWithEmptyHand =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");

    // A second edge right below that branch spends the next handshake's one hop before it
    // reaches the exit branch, so that handshake's signal comes to rest between the two.
    // That is what leaves the exit branch above holding nothing. It leaves for the exit as
    // well: an edge that jumped forward over a wait instead would strand the signal above it
    // and put two in flight, which is not a shape this pass claims to handle.
    StinkyInstruction* secondExit =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/93, "label_TestLoopEnd");
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createWMMA(56, 24, 32);
    StinkyInstruction* branchWithTokenInFlight =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");

    // Nothing stands between this branch and the handshake below it, so that handshake's
    // signal spends its hop climbing over the branch and leaves it holding a token.
    appendHandshake(/*loadS0=*/16, /*loadS1=*/20);
    createWMMA(64, 32, 40);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* drainWait = firstRealInstAfter(exitLabel);
    ASSERT_NE(drainWait, nullptr);
    ASSERT_TRUE(isClusterBarrierWithLiteral(*drainWait, /*wantSignal=*/false))
        << "a hoisted loop must open its exit with the wait that drains the carried signal:"
        << blockListing(*bb);

    const char* kBypassLabel = "label_TestLoopEnd_skipCBWait";
    StinkyInstruction* bypassLabel = findLabelNamed(kBypassLabel);
    ASSERT_NE(bypassLabel, nullptr) << "no drain-bypass label was emitted:" << blockListing(*bb);
    EXPECT_EQ(indexOf(bypassLabel), indexOf(drainWait) + 1)
        << "the bypass label must sit just below the drain wait:" << blockListing(*bb);

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    ASSERT_NE(findLastClusterSignalBefore(indexOf(loopHead)), nullptr)
        << "no preheader signal was emitted:" << blockListing(*bb);

    // Both branches are inside the body, so only the state where they stand separates them.
    // Spell that state out before reading the targets, so a body that came out shaped
    // differently fails as a broken premise rather than as a wrong answer.
    ASSERT_GT(indexOf(branchWithEmptyHand), indexOf(loopHead))
        << "the first branch has to be inside the loop:" << blockListing(*bb);
    EXPECT_EQ(inFlightAt(indexOf(branchWithEmptyHand)), 0)
        << "the first branch is only interesting while it leaves with nothing outstanding:"
        << blockListing(*bb);
    EXPECT_EQ(inFlightAt(indexOf(branchWithTokenInFlight)), 1)
        << "the second branch is only interesting while it leaves with a token outstanding:"
        << blockListing(*bb);

    EXPECT_EQ(getBranchTarget(*branchWithEmptyHand), kBypassLabel)
        << "a branch leaving with nothing in flight must skip the drain wait:" << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*branchWithTokenInFlight), "label_TestLoopEnd")
        << "a branch leaving with a token in flight must reach the drain wait:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*secondExit), "label_TestLoopEnd")
        << "the edge below the first branch also leaves with a token and must drain it:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// Segments in one loop need not agree about hoisting. Here the first two are long enough to
// hold the lead on their own and the last is not, so only the last one's signal moves, and it
// moves over the exit branch above it rather than over the loop head:
//
//     label_TestLoop:
//     S-1 ... W-1
//     s_cbranch_scc1 label_TestLoopEnd    <- nothing in flight: skips the drain
//     S-2 ... W-2
//     S-3                                 <- climbed out of the short tail segment
//     s_cbranch_scc1 label_TestLoopEnd    <- carries a token: must drain
//     W-3
//     s_cbranch_scc0 label_TestLoop
//     s_branch label_TestLoopEnd_skipCBWait   <- the fall-through, now spelled out
//     label_TestLoopEnd:
//     W
//     label_TestLoopEnd_skipCBWait:
//
// Nothing crosses the back edge, so no trip hands the next one anything and the preheader
// stays empty -- yet one exit still leaves holding a token, so the drain is needed anyway.
// That splits the two things the old all-or-nothing rule had welded together.
//
// The last trip then runs off the end of the body having already spent its token on W-3, and
// that edge is spelled by no instruction at all. Left alone it would fall straight into a
// drain with nothing to drain and hang there, so it is the one edge that has to be given a
// jump rather than have one rewritten.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           ShortTailSegmentDrainsItsExitAndSendsTheFallThroughPast) {
    const auto fillSegment = [&] {
        for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    };

    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);

    createLabel("label_TestLoop");
    fillSegment();
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    StinkyInstruction* exitWithEmptyHand =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    fillSegment();
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    StinkyInstruction* exitWithTokenInFlight =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");
    // Too short to hold the lead, so this one's signal climbs over the branch just above.
    createWMMA(56, 24, 32);
    appendHandshake(/*loadS0=*/16, /*loadS1=*/20);
    createWMMA(64, 32, 40);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 0)
        << "no signal crossed the back edge, so the loop must not be entered holding one:"
        << blockListing(*bb);

    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* drainWait = firstRealInstAfter(exitLabel);
    ASSERT_NE(drainWait, nullptr);
    EXPECT_TRUE(isClusterBarrierWithLiteral(*drainWait, /*wantSignal=*/false))
        << "one exit leaves holding a token, so the drain is still needed:" << blockListing(*bb);

    const char* kBypassLabel = "label_TestLoopEnd_skipCBWait";
    ASSERT_NE(findLabelNamed(kBypassLabel), nullptr)
        << "no drain-bypass label was emitted:" << blockListing(*bb);

    EXPECT_EQ(inFlightAt(indexOf(exitWithEmptyHand)), 0)
        << "the first exit is only interesting while it leaves with nothing outstanding:"
        << blockListing(*bb);
    EXPECT_EQ(inFlightAt(indexOf(exitWithTokenInFlight)), 1)
        << "the second exit is only interesting while it leaves with a token outstanding:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*exitWithEmptyHand), kBypassLabel)
        << "a branch leaving with nothing in flight must skip the drain wait:" << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*exitWithTokenInFlight), "label_TestLoopEnd")
        << "a branch leaving with a token in flight must reach the drain wait:"
        << blockListing(*bb);

    StinkyInstruction* beforeExit = realInstBefore(exitLabel);
    ASSERT_NE(beforeExit, nullptr);
    EXPECT_TRUE(isUnconditionalBranch(*beforeExit) && getBranchTarget(*beforeExit) == kBypassLabel)
        << "the body runs off its end with nothing in flight, so that edge needs a jump of "
           "its own to get past the drain:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// maxLeadCycles bounds the answer, and after a wrap the answer's distance from the wait is
// not a distance in the listing: the anchor sits textually *below* its wait, and what
// separates them is the way the loop actually runs -- wait back to the loop head, then latch
// back to the anchor. That sum is what has to stay under the ceiling.
//
//     label_TestLoop:
//     v_wmma x3                        <- wait to loop head: the first half of the distance
//     s_barrier_signal -1 / wait -1 / tensor_load_to_lds   <- the wait
//     s_cmp / s_cbranch label_TestLoopEnd
//     v_wmma x200                      <- far longer than the ceiling on its own
//     s_cmp / s_cbranch_scc0 label_TestLoop   <- the latch, where the wrap arrives
//
// The tail is deliberately long enough that the segment start is out of reach: an answer that
// took the whole segment because a boundary said stop would be over the ceiling, so this also
// says the ceiling is not something only the upward climb consults.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           WrapAroundAnchorStaysWithinMaxLeadCyclesOfItsWait) {
    const int kLead = 500;
    const int kMaxLead = 900;

    appendGsu1Preheader();
    openLoop();
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    for (int i = 0; i < 200; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* latch =
        createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    StinkyInstruction* segBeginInst = firstRealInstAfter(loopHead);
    ASSERT_NE(segBeginInst, nullptr);

    PassContext ctx;
    ctx.setGemmTileConfig(config);
    const auto cycleMap = computeEstimatedCyclesPerInstruction(*func, ctx);
    const auto cyclesAt = [&](const StinkyInstruction* inst) -> int64_t {
        auto it = cycleMap.find(inst);
        return (it == cycleMap.end()) ? -1 : static_cast<int64_t>(it->second);
    };
    ASSERT_GE(cyclesAt(trigger), 0);
    ASSERT_GE(cyclesAt(segBeginInst), 0);
    ASSERT_GE(cyclesAt(latch), 0);

    const auto found = cluster_barrier::test::findRule3SignalAnchorByCycleLeadForUnitTest(
        trigger, segBeginAfter(loopHead), trigger, cycleMap, kLead, kMaxLead,
        /*priorWaitAnchors=*/{}, /*maxHops=*/1, loopHead);

    ASSERT_TRUE(found.crossedLoopHead)
        << "this test only says anything while the climb wraps:" << blockListing(*bb);
    auto* anchorInst = dyn_cast<StinkyInstruction>(found.anchor);
    ASSERT_NE(anchorInst, nullptr) << blockListing(*bb);
    ASSERT_GE(cyclesAt(anchorInst), 0) << blockListing(*bb);

    // wait -> loop head, then latch -> anchor: the run-time distance, not the listing one.
    const int64_t toLoopHead = cyclesAt(trigger) - cyclesAt(segBeginInst);
    const int64_t fromLatch = cyclesAt(latch) - cyclesAt(anchorInst);
    const int64_t wrapDistance = toLoopHead + fromLatch;

    // Without this the test would still pass on a tail too short to reach the ceiling, and
    // would then be asserting nothing.
    const int64_t toSegmentStart = toLoopHead + (cyclesAt(latch) - cyclesAt(segBeginInst));
    ASSERT_GT(toSegmentStart, kMaxLead)
        << "the tail has to be long enough that taking all of it would break the ceiling, or "
           "this test cannot tell the ceiling is being applied:"
        << blockListing(*bb);

    EXPECT_LE(wrapDistance, kMaxLead)
        << "wait->loop head plus latch->anchor is what the ceiling bounds, and this anchor is "
           "beyond it:"
        << blockListing(*bb);
    EXPECT_GE(wrapDistance, kLead)
        << "the anchor gave up lead it was entitled to:" << blockListing(*bb);
})

// kRule3CrossLoop off: lead met while SCC stays live below the trigger; the climb must not
// cross the loop head into the preheader. It scans down from the lead point toward the wait
// instead, leaving the signal in the body segment.
TEST_F(InsertClusterBarrierPassTest,
       CrossLoopOffDoesNotPlaceRule3SignalInPreheaderWhenLeadMetAtLoopHead) {
    if (cluster_barrier::kRule3CrossLoop) GTEST_SKIP() << "requires kRule3CrossLoop == false";

    appendGsu1Preheader();
    StinkyInstruction* preheaderSccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    openLoop();
    for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    // Above the trigger: backward climb steps over this reader, but forward SCC scans from
    // the wait do not, so downwardFromLeadMet can still find a dead point before the wait.
    createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    for (int i = 0; i < 50; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    closeLoop();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);

    PassContext ctx;
    ctx.setGemmTileConfig(config);
    const auto cycleMap = computeEstimatedCyclesPerInstruction(*func, ctx);
    const BasicBlock::iterator segBegin = segBeginAfter(loopHead);
    const auto found = cluster_barrier::test::findRule3SignalAnchorByCycleLeadForUnitTest(
        trigger, segBegin, trigger, cycleMap, /*leadCycles=*/500, /*maxLeadCycles=*/900,
        /*priorWaitAnchors=*/{}, /*maxHops=*/0, loopHead);
    ASSERT_NE(cycleMap.find(trigger), cycleMap.end())
        << "trigger must be present in the estimated cycle map:" << blockListing(*bb);
    EXPECT_TRUE(anchorInWaitSegment(found.anchor, segBegin, trigger))
        << "signal anchor must stay inside the wait segment:" << blockListing(*bb);
    EXPECT_EQ(found.anchor, static_cast<IRBase*>(trigger))
        << "downward SCC scan should fall back to co-locating with the wait:" << blockListing(*bb);
    EXPECT_EQ(found.outOfSegmentNomination, nullptr)
        << "must not nominate a preheader anchor once loop-head continue is removed:"
        << blockListing(*bb);
    (void)preheaderSccDef;

    runPass();

    const size_t headIdx = indexOf(loopHead);
    const size_t triggerIdx = indexOf(trigger);
    ASSERT_NE(headIdx, static_cast<size_t>(-1));
    ASSERT_NE(triggerIdx, static_cast<size_t>(-1));

    StinkyInstruction* rule3Cmp = findClusterWaveCmpAfter(headIdx);
    ASSERT_NE(rule3Cmp, nullptr) << "Rule 3 handshake missing:" << blockListing(*bb);

    EXPECT_GT(indexOf(rule3Cmp), headIdx)
        << "Rule 3 signal must not anchor at or above the loop head (preheader):"
        << blockListing(*bb);
    EXPECT_LE(indexOf(rule3Cmp), triggerIdx)
        << "with no hop budget the signal stays in its segment near the trigger:"
        << blockListing(*bb);
    EXPECT_EQ(inFlightAt(headIdx), 0)
        << "no preheader compensation when kRule3CrossLoop is off:" << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

// With kRule3CrossLoop false, segments too short to hold the lead stay inside their segment.
TEST_F(InsertClusterBarrierPassTest, CrossLoopOffKeepsSignalsInsideTheirSegments) {
    if (cluster_barrier::kRule3CrossLoop) GTEST_SKIP() << "requires kRule3CrossLoop == false";

    const auto fillSegment = [&] {
        for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    };

    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);

    createLabel("label_TestLoop");
    fillSegment();
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    fillSegment();
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");
    createWMMA(56, 24, 32);
    appendHandshake(/*loadS0=*/16, /*loadS1=*/20);
    createWMMA(64, 32, 40);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 0)
        << "kRule3CrossLoop off must not post a preheader signal:" << blockListing(*bb);

    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* afterExit = firstRealInstAfter(exitLabel);
    ASSERT_NE(afterExit, nullptr);
    EXPECT_FALSE(isClusterBarrierWithLiteral(*afterExit, /*wantSignal=*/false))
        << "kRule3CrossLoop off must not drain at the exit:" << blockListing(*bb);
    EXPECT_EQ(findLabelNamed("label_TestLoopEnd_skipCBWait"), nullptr)
        << "kRule3CrossLoop off must not rewrite exits around a drain:" << blockListing(*bb);
}

// The same mixture, but with the *first* segment short instead of the last. Its signal has no
// segment above it inside the body, so it climbs over the loop head and follows the latch,
// landing near the tail where it feeds the next trip's wait. That is what asks for a signal in
// the preheader, and the middle segment holding its own lead does not change it:
//
//     S                                   <- preheader, feeds the first trip
//     label_TestLoop:
//     W-1
//     s_cbranch_scc1 label_TestLoopEnd    <- nothing in flight: skips the drain
//     S-2 ... W-2
//     S-3
//     s_cbranch_scc1 label_TestLoopEnd    <- carries a token: must drain
//     W-3
//     S-1                                 <- climbed across the back edge
//     s_cbranch_scc0 label_TestLoop
//     label_TestLoopEnd:                  <- fall-through arrives holding S-1: no jump needed
//
// So the preheader signal answers to one question only -- did anything cross the back edge --
// while the drain answers to another, and this loop says yes to both for different segments.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
IF_RULE3_CROSS_LOOP(TEST_F(InsertClusterBarrierPassTest,
                           PreheaderSignalFollowsOnlyTheSegmentCrossingTheBackEdge) {
    const auto fillSegment = [&] {
        for (int i = 0; i < 70; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    };

    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    createBarrierSignal(kWorkgroupBarrierId);
    createBarrierWait(kWorkgroupBarrierId);
    createWMMA(32, 8, 16);

    createLabel("label_TestLoop");
    // Short: nothing above it inside the body, so its signal leaves across the back edge.
    appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createWMMA(40, 16, 8);
    StinkyInstruction* exitWithEmptyHand =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/90, "label_TestLoopEnd");
    fillSegment();
    appendHandshake(/*loadS0=*/48, /*loadS1=*/52);
    StinkyInstruction* exitWithTokenInFlight =
        createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/91, "label_TestLoopEnd");
    createWMMA(56, 24, 32);
    appendHandshake(/*loadS0=*/16, /*loadS1=*/20);
    createWMMA(64, 32, 40);
    createGuardedBranch(GFX::s_cbranch_scc0, /*sgpr=*/92, "label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    runPass();

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    // The Rule 1 signal at GSU_1 is above the loop head too, so a signal being there says
    // nothing. What says the preheader fed this loop is that one is still outstanding here.
    EXPECT_EQ(inFlightAt(indexOf(loopHead)), 1)
        << "a signal crossed the back edge, so the preheader must feed the first trip:"
        << blockListing(*bb);

    StinkyInstruction* exitLabel = findLabelNamed("label_TestLoopEnd");
    ASSERT_NE(exitLabel, nullptr);
    StinkyInstruction* drainWait = firstRealInstAfter(exitLabel);
    ASSERT_NE(drainWait, nullptr);
    EXPECT_TRUE(isClusterBarrierWithLiteral(*drainWait, /*wantSignal=*/false))
        << "the last trip carries a signal out, so the exit must open with the drain:"
        << blockListing(*bb);

    const char* kBypassLabel = "label_TestLoopEnd_skipCBWait";
    ASSERT_NE(findLabelNamed(kBypassLabel), nullptr)
        << "no drain-bypass label was emitted:" << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*exitWithEmptyHand), kBypassLabel)
        << "a branch leaving with nothing in flight must skip the drain wait:" << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*exitWithTokenInFlight), "label_TestLoopEnd")
        << "a branch leaving with a token in flight must reach the drain wait:"
        << blockListing(*bb);

    StinkyInstruction* beforeExit = realInstBefore(exitLabel);
    ASSERT_NE(beforeExit, nullptr);
    EXPECT_FALSE(isUnconditionalBranch(*beforeExit))
        << "the fall-through arrives holding the signal that crossed the back edge, so it "
           "belongs in the drain and must not be routed around it:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
})

// The climb carries an SCC flag of its own while `clearScc` reads liveness off the code
// below the anchor, and everywhere the climb walks the text the two agree. This is the shape
// that separates them, and the only kind that can: a range the loop closes on the next trip.
//
//     label_TestLoop:
//     s_cselect_b32 s91, s92, 0        <- reads what the previous trip's tail computed
//     v_wmma ... (x40)                 <- too short to hold the lead
//     s_barrier_signal -1              <- the wait
//     s_cmp_eq_u32 / s_cbranch_scc1    <- the body's way out
//     v_wmma ... (x20)
//     s_sub_u32 s90, s90, 1            <- the def, read only after the back edge
//     v_wmma ... (x55)                 <- where the lead is met, inside the range
//     s_branch label_TestLoop          <- unconditional, so it contributes no read of its own
//
// The wait's segment cannot hold the lead, so the climb follows the latch and keeps going up
// the tail. The lead falls among the wmma below the def -- and a forward scan from there sees
// SCC written by nobody and read by nobody, because the reader is not below that spot, it is
// beyond the back edge. Only the flag the climb carried around the loop knows the range is
// still open. Planting the handshake at the lead point would put its `s_cmp_eq_u32 waveIdx, 0`
// between the def and its reader, and nothing puts SCC back: the signal block is the compare,
// its branch, `s_barrier_signal -3`, and the skip label.
//
// The latch has to be unconditional. An `s_cbranch_scc*` back edge reads SCC by construction,
// which sets the flag whatever the body did with it, and then the flag agreeing is no evidence.
//
// This also depends on the climb being allowed to cross with SCC live, which it is: the
// `!targetMet` arm of the boundary logic asks about hops and the kind of boundary, never about
// SCC. A rule that stopped a live climb from crossing would take this shape away.
TEST_F(InsertClusterBarrierPassTest, WrappedClimbClearsARangeOnlyTheBackEdgeReaches) {
    appendGsu1Preheader();
    // Gives the preheader an SCC-dead spot, without which the climb refuses to cross at all.
    createSCmpWritingScc(/*srcSgpr=*/94);
    openLoop();
    StinkyInstruction* sccReader = createSCselectReadingScc(/*destSgpr=*/91, /*srcSgpr=*/92);
    for (int i = 0; i < 40; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* trigger = appendHandshake(/*loadS0=*/0, /*loadS1=*/4);
    createGuardedBranch(GFX::s_cbranch_scc1, /*sgpr=*/93, "label_TestLoopEnd");
    for (int i = 0; i < 20; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* sccDef = createSSubWritingSgprAndScc(/*sgpr=*/90);
    for (int i = 0; i < 55; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* latch = createUnconditionalBranch("label_TestLoop");
    createLabel("label_TestLoopEnd");
    createWMMA(8, 0, 8);

    StinkyInstruction* loopHead = findLabelNamed("label_TestLoop");
    ASSERT_NE(loopHead, nullptr);
    StinkyInstruction* belowDef = firstRealInstAfter(sccDef);
    ASSERT_NE(belowDef, nullptr);

    PassContext ctx;
    ctx.setGemmTileConfig(config);
    const auto cycleMap = computeEstimatedCyclesPerInstruction(*func, ctx);
    const auto cyclesAt = [&](const StinkyInstruction* inst) -> int64_t {
        auto it = cycleMap.find(inst);
        return (it == cycleMap.end()) ? -1 : static_cast<int64_t>(it->second);
    };
    constexpr int kLead = 500;
    constexpr int kMaxLead = 900;
    ASSERT_GE(cyclesAt(trigger), 0);
    ASSERT_GE(cyclesAt(latch), 0);
    ASSERT_GE(cyclesAt(sccDef), 0);
    ASSERT_GE(cyclesAt(belowDef), 0);
    const int64_t headPart = cyclesAt(trigger) - cyclesAt(sccReader);
    const int64_t tailPart = cyclesAt(latch) - cyclesAt(sccDef);
    const int64_t atBelowDef = headPart + (cyclesAt(latch) - cyclesAt(belowDef));

    // The premise, in three parts: the wait's own segment cannot hold the lead, so the climb
    // has to wrap; the lead is then met below the def, so the two notions of liveness are
    // asked about a spot they disagree on; and the answer stays inside the ceiling, so what
    // turns the climb around is the flag rather than maxLeadCycles.
    ASSERT_LT(headPart, kLead) << "the wait segment must be too short to hold the lead:"
                               << blockListing(*bb);
    ASSERT_GE(atBelowDef, kLead) << "the lead has to be met below the def, or the climb never "
                                    "stands anywhere the two disagree:"
                                 << blockListing(*bb);
    ASSERT_LT(headPart + tailPart, kMaxLead)
        << "the ceiling must not be what turns the climb around:" << blockListing(*bb);

    const auto found = cluster_barrier::test::findRule3SignalAnchorByCycleLeadForUnitTest(
        trigger, segBeginAfter(loopHead), trigger, cycleMap, kLead, kMaxLead,
        /*priorWaitAnchors=*/{}, /*maxHops=*/1, loopHead);
    const auto* anchorInst = dyn_cast<StinkyInstruction>(found.anchor);
    ASSERT_NE(anchorInst, nullptr) << blockListing(*bb);

    EXPECT_LE(indexOf(anchorInst), indexOf(sccDef))
        << "the anchor has to stand at or above the def; below it is inside the range the next "
           "trip reads, which a forward scan from there cannot see:"
        << blockListing(*bb);
    EXPECT_EQ(found.hops, 1) << "the climb left the wait's segment and has to be billed for it:"
                             << blockListing(*bb);
    EXPECT_TRUE(found.crossedLoopHead)
        << "it left by the back edge, so the first trip is owed a signal:" << blockListing(*bb);
}
