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
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

#define SKIP_UNLESS_RULE3_CROSS_LOOP()                                                   \
    do {                                                                                 \
        if (!cluster_barrier::kRule3CrossLoop)                                         \
            GTEST_SKIP() << "requires cluster_barrier::kRule3CrossLoop == true";         \
    } while (0)

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
    void openLoop() { createLabel("label_TestLoop"); }

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
TEST_F(InsertClusterBarrierPassTest, SingleHandshakeInALoopIsFedByRule1AndRule2) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
}

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
        << "cluster signal must forward past intervening workgroup barriers:"
        << blockListing(*bb);
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
TEST_F(InsertClusterBarrierPassTest, Rule3SegmentBoundaryFallbackAnchorsAtSegBegin) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
TEST_F(InsertClusterBarrierPassTest, ClimbThatGivesUpIsNotBilledForCrossingTheBackEdge) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
    ASSERT_NE(firstLoopWait, nullptr) << "the pass planted no handshake in the loop:"
                                     << blockListing(*bb);
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
        << "this edge leaves empty-handed and must be routed past the drain:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*thirdExit), kBypassLabel)
        << "this edge leaves empty-handed and must be routed past the drain:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

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
TEST_F(InsertClusterBarrierPassTest, SignalStaysInLoopWhenThePreheaderHasNoSafeSccSpot) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
}

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
TEST_F(InsertClusterBarrierPassTest, PreheaderSignalSitsBehindTheBarrierClosestToTheLoop) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
}

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
TEST_F(InsertClusterBarrierPassTest, PreheaderWithNoBarrierBringsOneBelowItsLastLabel) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
    EXPECT_LT(indexOf(wgSignal), indexOf(wgWait))
        << "signal then wait:" << blockListing(*bb);

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
}

// The last shape the preheader can take: no barrier and no label either, so neither of the
// two spots above is available and the run-up is nothing but its own loads and work:
//
//     tensor_load_to_lds        <- Rule 2's wait goes in front of this
//     v_wmma x3
//     tensor_load_to_lds        <- the last load of the run-up
//     v_wmma x3
//     label_TestLoop:
//
// The signal brings its own barrier again, and the pair goes below the last load. Those loads
// are the work this signal announces as finished, so a signal put over them would be telling
// the cluster the run-up is done while it is still fetching.
// Run with STINKY_TEST_DUMP=1 to print the block before and after the pass.
TEST_F(InsertClusterBarrierPassTest, PreheaderWithNoBarrierAndNoLabelSitsBelowItsLastLoad) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
    createLabel(kGSU1LabelName);
    createWMMA(24, 0, 8);
    createTensorLoadInBlock(bb, arch, /*src0Reg=*/60, /*src1Reg=*/64);
    for (int i = 0; i < 3; ++i) createWMMA(8 + (i % 8) * 8, (i % 8) * 8, ((i + 1) % 8) * 8);
    StinkyInstruction* lastLoad =
        createTensorLoadInBlock(bb, arch, /*src0Reg=*/68, /*src1Reg=*/72);
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
    EXPECT_GT(indexOf(preSignal), indexOf(lastLoad))
        << "the run-up's own loads are the work the signal announces as finished, so it may "
           "not be posted above them:"
        << blockListing(*bb);

    // The barrier the pass brought: the closest workgroup pair above the signal, which the
    // preheader did not have to begin with.
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
    EXPECT_GT(indexOf(wgSignal), indexOf(lastLoad))
        << "the preheader had no barrier of its own, so the pass must have planted this one "
           "below the last load:"
        << blockListing(*bb);
    EXPECT_LT(indexOf(wgSignal), indexOf(wgWait))
        << "signal then wait:" << blockListing(*bb);

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
TEST_F(InsertClusterBarrierPassTest, ExitBranchSkipsDrainWaitOnlyWithNoTokenInFlight) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
        << "a branch leaving with nothing in flight must skip the drain wait:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*branchWithTokenInFlight), "label_TestLoopEnd")
        << "a branch leaving with a token in flight must reach the drain wait:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*secondExit), "label_TestLoopEnd")
        << "the edge below the first branch also leaves with a token and must drain it:"
        << blockListing(*bb);

    expectClusterTokensBalanceOnEveryPath(/*completeProgram=*/true);
}

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
TEST_F(InsertClusterBarrierPassTest, ShortTailSegmentDrainsItsExitAndSendsTheFallThroughPast) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
        << "a branch leaving with nothing in flight must skip the drain wait:"
        << blockListing(*bb);
    EXPECT_EQ(getBranchTarget(*exitWithTokenInFlight), "label_TestLoopEnd")
        << "a branch leaving with a token in flight must reach the drain wait:"
        << blockListing(*bb);

    StinkyInstruction* beforeExit = realInstBefore(exitLabel);
    ASSERT_NE(beforeExit, nullptr);
    EXPECT_TRUE(isUnconditionalBranch(*beforeExit) &&
                getBranchTarget(*beforeExit) == kBypassLabel)
        << "the body runs off its end with nothing in flight, so that edge needs a jump of "
           "its own to get past the drain:"
        << blockListing(*bb);

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
TEST_F(InsertClusterBarrierPassTest, PreheaderSignalFollowsOnlyTheSegmentCrossingTheBackEdge) {
    SKIP_UNLESS_RULE3_CROSS_LOOP();
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
        << "a branch leaving with nothing in flight must skip the drain wait:"
        << blockListing(*bb);
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
}
