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

// ----------------------------------------------------------------------------
// RemoveDscntPass
//
// Runs after StinkyWaitCntInsertionPass and only operates on basic blocks that
// are themselves loops (a block with a back-edge to itself).
//
// For each such block it performs a linear scan of the StinkyTofu IR while
// tracking the current cycle count. The cycle model accounts for WMMA
// co-execution: while a matrix instruction's latency window is open, VALU
// instructions can co-issue into it without advancing the global cycle
// counter.
//
//   * On a ds_load (LDS read) the current cycle and the load's destination
//     register(s) are pushed onto an in-flight FIFO.
//   * On an s_wait_dscnt (LDS-load wait) the wait's count N is read and the
//     oldest in-flight loads are popped until only N remain outstanding.
//
// The actual instruction removal is not implemented yet.
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/RemoveDscntPass.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "RemoveDscntPass"

namespace {
using namespace stinkytofu;

/// Extra cycles a load must already have been back for before it counts as finished.
/// Padding every modeled drain latency by this much keeps the estimate on the safe
/// side: under-counting finished loads just leaves a wait as the insertion pass
/// emitted it, while over-counting would tighten one past what the hardware has
/// actually retired.
constexpr int kDsProximityThreshold = 512;

/// An outstanding LDS read: the cycle it was issued at, its modeled return
/// latency, and the destination register(s) it will eventually write.
struct DsLoadEntry {
    int cycle = 0;
    int latency = 0;
    std::vector<StinkyRegister> dests;
};

/// Scan state carried across basic blocks so the walk behaves as one
/// continuous stream rather than restarting per block.
struct ScanState {
    int cycles = 0;
    int hwMFMA = -99;

    // Currently active WMMA co-issue window.
    int activeWmmaStartCycle = -1;
    int activeWmmaCoExecAdvance = 0;
    std::vector<bool> activeWmmaValuSlots;

    // Outstanding LDS reads, oldest at the front.
    std::deque<DsLoadEntry> inFlightDsLoads;

    // Dormant until the first WMMA whose source VGPRs overlap an in-flight
    // ds_load destination is seen.
    bool waitCheckActive = false;

    // The keep value of the last dscnt wait that remains in IR. -1 means none yet.
    int prevKeptDscnt = -1;

    // Number of ds_load instructions seen since the last kept dscnt wait.
    int dsLoadsSinceLastKeptDscnt = 0;

    // Number of DS ops accumulated for pre-activation dscnt tightening.
    int numDsLoadsBeforeActivation = 0;

    // Return latency of the last ds_read scanBlock walked over. scanBlockHead runs
    // after it and only counts its loads rather than walking them, so this is the
    // one real measurement it has to work from. 0 means no ds_read was seen.
    int lastDsLoadLatency = 0;
};

/// VALU co-issue profile for an in-flight WMMA/matrix instruction.
///
/// `valuCoExecSlots[i]` is true when a VALU instruction may co-issue at the
/// i-th cycle of the matrix instruction's latency window (relative to its
/// start). The first cycle (the execute cycle) is never insertable.
struct WmmaCoExecProfile {
    std::vector<bool> valuCoExecSlots;
};

std::string toUpperASCII(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c >= 'a' && c <= 'z')
            out.push_back(static_cast<char>(c - ('a' - 'A')));
        else
            out.push_back(c);
    }
    return out;
}

/// Build the VALU co-issue window for a matrix instruction from its latency.
/// Non-WMMA / unknown instructions get an empty (no co-exec) window.
WmmaCoExecProfile getWmmaCoExecProfile(const StinkyInstruction& inst) {
    WmmaCoExecProfile profile;
    if (!inst.getHwInstDesc()) return profile;

    const std::string mnemonic = toUpperASCII(inst.getHwInstDesc()->mnemonic);
    if (!mnemonic.starts_with("V_WMMA_")) return profile;
    if (inst.latencyCycles <= 0) return profile;

    profile.valuCoExecSlots.assign(static_cast<size_t>(inst.latencyCycles), true);
    // The first (execute) cycle of the window is not insertable.
    profile.valuCoExecSlots[0] = false;
    return profile;
}

int getActiveWmmaElapsedCycle(int cycles, int activeWmmaStartCycle, int activeWmmaCoExecAdvance) {
    if (activeWmmaStartCycle < 0) return -1;
    return (cycles - activeWmmaStartCycle) + activeWmmaCoExecAdvance;
}

/// Decide whether a VALU instruction can co-issue into the currently active
/// WMMA window at `cycles`. When it can, `activeWmmaCoExecAdvance` is bumped so
/// the next co-issued instruction lands on the following free slot.
bool canCoExecAtCurrentCycle(int cycles, int activeWmmaStartCycle, int& activeWmmaCoExecAdvance,
                             const std::vector<bool>& valuSlots) {
    if (activeWmmaStartCycle < 0 || valuSlots.empty()) return false;

    int elapsed = getActiveWmmaElapsedCycle(cycles, activeWmmaStartCycle, activeWmmaCoExecAdvance);
    if (elapsed < 0 || elapsed >= static_cast<int>(valuSlots.size())) return false;
    if (valuSlots[static_cast<size_t>(elapsed)]) return true;

    // Skip forward to the next insertable co-exec slot.
    for (int idx = elapsed + 1; idx < static_cast<int>(valuSlots.size()); ++idx) {
        if (valuSlots[static_cast<size_t>(idx)]) {
            activeWmmaCoExecAdvance += (idx - elapsed);
            return true;
        }
    }
    return false;
}

/// Cycles the front end needs to issue `numDsLoads` LDS ops, scaled by 3.
///
/// Empirical model from experiment data. With A = numDsLoads and B = issue time:
///            { A,                                A <= 16
/// B(A) =     { 16 + (A - 16) * 7 / 3,       16 < A <= 43
///            { 16 + 27 * 7 / 3 + (A - 43) * 4,   A > 43
///
/// The result is scaled by 3 so those slopes stay integral:
/// - branch1: B*3 = 3*A
/// - branch2: B*3 = 48 + 7*(A-16)
/// - branch3: B*3 = 12*A - 279
int computeDsIssueTimeTimes3(size_t numDsLoads) {
    // conservative constant is 20, which is the expirimental result from the experiment.
    const int conservativeConstant = 20;
    const int n = static_cast<int>(numDsLoads) + conservativeConstant;
    if (n <= 16) return 3 * n;
    if (n <= 43) return 48 + (n - 16) * 7;
    return 12 * n - 279;
}

/// computeDsIssueTimeTimes3() in whole cycles, for callers that do not need the
/// extra precision and would otherwise repeat the scaling.
int computeDsIssueTime(size_t numDsLoads) {
    return computeDsIssueTimeTimes3(numDsLoads) / 3;
}

/// If `inst` is an LDS-load wait (s_wait_dscnt / s_wait_loadcnt_dscnt), return
/// the number of LDS loads that are allowed to remain outstanding after it.
/// Returns nullopt for any other instruction.
std::optional<int> getDsWaitCount(const StinkyInstruction& inst) {
    if (!isWaitCnt(inst)) return std::nullopt;

    const uint16_t op = inst.getUnifiedOpcode();
    const bool isDsWait = (op == GFX::s_wait_dscnt || op == GFX::s_wait_loadcnt_dscnt);
    if (!isDsWait) return std::nullopt;

    // On gfx1250 the insertion pass stores the LDS-load count in `dlcnt`; after
    // legalization the ds count may live in `dlcnt` and/or `dscnt`.
    if (const SWaitCntData* w = inst.getModifier<SWaitCntData>()) {
        const int dl = w->dlcnt;
        const int ds = w->dscnt;
        if (dl >= 0 && ds >= 0) return std::min(dl, ds);
        if (dl >= 0) return dl;
        if (ds >= 0) return ds;
    }

    // Fallback: read the immediate carried as the first source literal.
    const auto& srcs = inst.getSrcRegs();
    if (!srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt) {
        const int imm = srcs[0].getLiteralInt();
        // s_wait_loadcnt_dscnt packs {loadcnt << 8 | dscnt}.
        if (op == GFX::s_wait_loadcnt_dscnt) return imm & 0xFF;
        return imm;
    }

    return std::nullopt;
}

/// Overwrite the LDS-load wait count of `inst` with `newVal`, keeping both the
/// emitted immediate (first literal source) and the semantic SWaitCntData
/// modifier in sync. Only meaningful for the ds-bearing wait opcodes.
void setDsWaitCount(StinkyInstruction& inst, int newVal) {
    const uint16_t op = inst.getUnifiedOpcode();

    // The emitter prints the immediate from the first literal source operand.
    const auto& srcs = inst.getSrcRegs();
    for (size_t i = 0; i < srcs.size(); ++i) {
        if (srcs[i].dataType != StinkyRegister::Type::LiteralInt) continue;
        int literal = newVal;
        if (op == GFX::s_wait_loadcnt_dscnt) {
            // Preserve the packed loadcnt high byte; only replace the dscnt byte.
            const int loadcnt = (srcs[i].getLiteralInt() >> 8) & 0xFF;
            literal = (loadcnt << 8) | (newVal & 0xFF);
        }
        inst.setSrcReg(i, StinkyRegister(literal));
        break;
    }

    // Keep the semantic modifier consistent with the new count.
    if (SWaitCntData* w = inst.getModifier<SWaitCntData>()) {
        if (w->dlcnt >= 0) w->dlcnt = newVal;
        if (w->dscnt >= 0) w->dscnt = newVal;
        if (w->dlcnt < 0 && w->dscnt < 0) w->dlcnt = newVal;
    }
}

/// True iff any source register of `inst` overlaps a destination register of
/// any load currently outstanding in `inFlight`.
bool srcOverlapsInFlight(const StinkyInstruction& inst, const std::deque<DsLoadEntry>& inFlight) {
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (!src.isRegister()) continue;
        for (const DsLoadEntry& entry : inFlight) {
            for (const StinkyRegister& dst : entry.dests) {
                if (dst.isRegister() && src.isOverlap(dst)) return true;
            }
        }
    }
    return false;
}

/// Append `note` to the instruction's trailing comment (creating one if none
/// exists), separated from any existing text by a space.
void appendComment(StinkyInstruction& inst, const std::string& note) {
    if (CommentData* c = inst.getModifier<CommentData>()) {
        if (!c->comment.empty()) c->comment += " ";
        c->comment += note;
    } else {
        inst.addModifier<CommentData>(CommentData{note});
    }
}

AsmDirective* createTextCommentDirective(const std::string& comment) {
    AsmDirective* directive = IRBase::createIR<AsmDirective>();
    directive->kind = AsmDirectiveKind::TEXTBLOCK;
    directive->value = "// " + comment + "\n";
    return directive;
}

int recomputePrefetchInFlightDsLoads(const BasicBlock& bb) {
    int inFlight = 0;
    for (const IRBase& node : bb) {
        if (node.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&node);
        if (isLabel(*inst) || isPseudoInst(inst) || !inst->getHwInstDesc()) continue;
        if (isBranch(*inst) || isMatrixInstruction(*inst)) break;

        if (isDSRead(*inst) || isDSWrite(*inst)) {
            ++inFlight;
        } else if (std::optional<int> keep = getDsWaitCount(*inst)) {
            inFlight = std::min(inFlight, std::max(0, *keep));
        }
    }
    return inFlight;
}

class RemoveDscntPass : public StinkyInstPass {
   public:
    explicit RemoveDscntPass(int dsProximityThreshold)
        : dsProximityThreshold_(std::max(0, dsProximityThreshold)) {}

    static char ID;

    const char* getName() const override {
        return "RemoveDscntPass";
    }

    PassID getPassID() const override {
        return &RemoveDscntPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        // This pass is scheduled inside the region-scoped pipeline for the
        // {"loopWithPrefetch", "noLoadLoopBody"} regions, so shouldProcessBasicBlock
        // already restricts us to the loop blocks of interest.
        //
        // The processed blocks are scanned as one continuous stream: `state` is
        // carried across block boundaries so cycle counting and the in-flight
        // ds_load FIFO persist across BBs.
        hw_ = &passCtx.getHWModel();
        // Passed through as-is: computeDynamicDrainLatency() clamps it to the range
        // the drain model is defined over.
        numWaves_ = static_cast<int>(passCtx.getGemmTileConfig().NumWaves);
        PASS_DEBUG(std::cerr << "[RemoveDscnt] numWaves=" << numWaves_ << "\n");

        ScanState state;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            scanBlock(bb, state);
            if (std::string_view(bb.getLabel()).starts_with("label_LoopBeginL")) {
                // No need to remove the blocks after the loop begin label,
                scanBlockHead(bb, state.cycles, state.numDsLoadsBeforeActivation,
                              state.lastDsLoadLatency);
                return PreservedAnalyses::none();
            }
            const int recomputedPrefetchInFlightDsLoads =
                std::max(0, recomputePrefetchInFlightDsLoads(bb));
            if (recomputedPrefetchInFlightDsLoads > 0) {
                state.numDsLoadsBeforeActivation = recomputedPrefetchInFlightDsLoads;
            }
        }
        return PreservedAnalyses::none();
    }

   private:
    // Safety margin added to every modeled drain latency; see kDsProximityThreshold.
    int dsProximityThreshold_ = kDsProximityThreshold;

    // Hardware facts and occupancy for the function being scanned, captured in run().
    const HWModel* hw_ = nullptr;
    int numWaves_ = 0;

    /// How many of `numDsLoads` back-to-back LDS reads have returned by the time the
    /// last of them has been issued. Used for the pre-activation run of loads, where
    /// there is no scanned cycle count to measure against: the issue-time model stands
    /// in for the clock, and a load counts as finished when its issue time, its drain
    /// latency and the dsProximityThreshold_ margin still fit inside the run's total
    /// issue time.
    int computeNumDsFinished(size_t numDsLoads, int targetDsLoadLatency, int numWaves) const {
        if (!hw_) return 0;

        const int totalIssueTime = computeDsIssueTime(numDsLoads);

        int drained = 0;
        for (int count = 1; count <= static_cast<int>(numDsLoads); ++count) {
            const int issueTime = computeDsIssueTime(count);
            const int landsAt =
                issueTime + computeDynamicDrainLatency(*hw_, count, targetDsLoadLatency, numWaves);
            if (landsAt + dsProximityThreshold_ > totalIssueTime) break;
            drained = count;
        }
        PASS_DEBUG(std::cerr << "[RemoveDscnt] computeNumDsFinished: numDsLoads=" << numDsLoads
                             << " targetDsLoadLatency=" << targetDsLoadLatency
                             << " numWaves=" << numWaves << " totalIssueTime=" << totalIssueTime
                             << " drained=" << drained << "\n");
        return drained;
    }

    /// How many of the outstanding LDS reads the hardware should have retired by
    /// `cycles`. Does not touch the FIFO.
    ///
    /// computeDynamicDrainLatency() answers, for a burst of N loads, how many
    /// cycles pass between the burst's first issue and the N-th load landing. Both that
    /// curve and the FIFO's issue cycles grow with N, so the loads that have come back
    /// are the prefix whose drain latency, plus the dsProximityThreshold_ margin, still
    /// fits in the time since each one issued.
    int estimateDrainedDsLoads(const std::deque<DsLoadEntry>& inFlight, int cycles) const {
        if (!hw_ || inFlight.empty()) return 0;

        const DsLoadEntry& oldest = inFlight.front();
        // The oldest load has the most time behind it, so if even it cannot have
        // landed, nothing in the FIFO has.
        if (cycles - oldest.cycle <= 0) return 0;

        // A ds_write carries no return latency of its own, so fall back to the
        // arch's static figure rather than letting a write at the front of the
        // FIFO report an instant drain.
        const int targetDsLoadLatency =
            oldest.latency > 0 ? oldest.latency : hw_->lds.readDrainLatency;

        int drained = 0;
        for (int count = 1; count <= static_cast<int>(inFlight.size()); ++count) {
            const int elapsedForCount = cycles - inFlight[count - 1].cycle;
            const int landedCyclesAgo =
                elapsedForCount -
                computeDynamicDrainLatency(*hw_, count, targetDsLoadLatency, numWaves_);
            if (landedCyclesAgo < dsProximityThreshold_) break;
            drained = count;
        }
        return drained;
    }

    void scanBlockHead(BasicBlock& bb, int cycles, int& numDsLoadsBeforeActivation,
                       int lastDsLoadLatency) {
        // Second pass: handle dscnt before waitCheckActive becomes true.
        bool seenFirstDscntBeforeActivation = false;
        // Log carried pre-activation DS-op count at the beginning of scanBlockHead
        PASS_DEBUG({
            std::cerr << "[RemoveDscnt] scanBlockHead: numDsLoadsBeforeActivation="
                      << numDsLoadsBeforeActivation << ", cycles=" << cycles
                      << ", lastDsLoadLatency=" << lastDsLoadLatency << '\n';
        });
        if (numDsLoadsBeforeActivation == 0) return;

        // scanBlock ran first and measured a real ds_read; only fall back to the
        // arch's static figure when it never saw one.
        const int targetDsLoadLatency =
            lastDsLoadLatency > 0 ? lastDsLoadLatency : hw_->lds.readDrainLatency;
        const int numDsFinished =
            computeNumDsFinished(static_cast<size_t>(std::max(0, numDsLoadsBeforeActivation)),
                                 targetDsLoadLatency, numWaves_);
        PASS_DEBUG(std::cerr << "[RemoveDscnt] pre-activation numDsFinished=" << numDsFinished
                             << " from numDsLoads=" << numDsLoadsBeforeActivation << "\n");
        for (auto it = bb.begin(); it != bb.end();) {
            IRBase& node = *it.getNodePtr();
            PASS_DEBUG({
                std::cerr << "[RemoveDscnt] scanBlockHead: node type="
                          << static_cast<int>(node.getType()) << ", node ptr=" << &node;
                if (node.getType() == IRBase::IRType::StinkyTofu) {
                    auto* inst = cast<StinkyInstruction>(&node);
                    if (inst->getHwInstDesc()) {
                        std::cerr << ", mnemonic=" << inst->getHwInstDesc()->mnemonic;
                    }
                }
                std::cerr << '\n';
            });
            if (node.getType() != IRBase::IRType::StinkyTofu) {
                ++it;
                continue;
            }
            auto* inst = cast<StinkyInstruction>(&node);
            if (isLabel(*inst) || isPseudoInst(inst) || !inst->getHwInstDesc()) {
                ++it;
                continue;
            }
            if (isBranch(*inst)) break;

            bool removeWaitInst = false;
            std::string removalComment;

            if (isDSRead(*inst) || isDSWrite(*inst)) {
                numDsLoadsBeforeActivation++;
            } else if (std::optional<int> keep = getDsWaitCount(*inst)) {
                const int newVal = (numDsLoadsBeforeActivation - numDsFinished);
                PASS_DEBUG(std::cerr << "[RemoveDscnt]   reduce dscnt: tighten wait " << *keep
                                     << "->" << newVal
                                     << " numDsLoadsBeforeActivation=" << numDsLoadsBeforeActivation
                                     << " numDsFinished=" << numDsFinished << "\n");
                if (*keep >= newVal) {
                    std::string comment =
                        "reduce dscnt:" + std::to_string(*keep) + "---->" + std::to_string(newVal);
                    if (seenFirstDscntBeforeActivation) {
                        comment += " to be removed";
                        removeWaitInst = true;
                        removalComment = comment;
                    } else {
                        setDsWaitCount(*inst, newVal);
                        appendComment(*inst, comment);
                        PASS_DEBUG(std::cerr << "[RemoveDscnt]   reduce dscnt: set wait " << *keep
                                             << "->" << newVal << " to comment\n");
                    }
                } else {
                    break;
                }
                seenFirstDscntBeforeActivation = true;
            }
            if (removeWaitInst) {
                PASS_DEBUG(std::cerr << "[RemoveDscnt]   remove wait and replace with comment: "
                                     << removalComment << "\n");
                bb.insertIR(it, createTextCommentDirective(removalComment));
                it = bb.eraseIR(it);
                continue;
            }
            ++it;
        }
    }

    /// Linear cycle-tracking scan of a loop block that continues from `state`.
    /// Builds the in-flight LDS-read FIFO and drains it at each s_wait_dscnt.
    void scanBlock(BasicBlock& bb, ScanState& state) {
        int& cycles = state.cycles;
        int& hwMFMA = state.hwMFMA;
        int& activeWmmaStartCycle = state.activeWmmaStartCycle;
        int& activeWmmaCoExecAdvance = state.activeWmmaCoExecAdvance;
        std::vector<bool>& activeWmmaValuSlots = state.activeWmmaValuSlots;
        std::deque<DsLoadEntry>& inFlightDsLoads = state.inFlightDsLoads;
        bool& waitCheckActive = state.waitCheckActive;
        int& prevKeptDscnt = state.prevKeptDscnt;
        int& dsLoadsSinceLastKeptDscnt = state.dsLoadsSinceLastKeptDscnt;
        int& lastDsLoadLatency = state.lastDsLoadLatency;

        for (auto it = bb.begin(); it != bb.end();) {
            IRBase& node = *it.getNodePtr();
            if (node.getType() != IRBase::IRType::StinkyTofu) {
                ++it;
                continue;
            }
            auto* inst = cast<StinkyInstruction>(&node);
            if (isLabel(*inst) || isPseudoInst(inst) || !inst->getHwInstDesc()) {
                ++it;
                continue;
            }
            if (isBranch(*inst)) break;

            // --- advance the cycle counter (WMMA co-exec aware) ---
            const bool isCoIssued =
                isVectorALU(*inst) && !isMatrixInstruction(*inst) &&
                canCoExecAtCurrentCycle(cycles, activeWmmaStartCycle, activeWmmaCoExecAdvance,
                                        activeWmmaValuSlots);
            if (isCoIssued) {
                // Co-issued VALU is packed into the active WMMA window and does
                // not advance the global cycle counter.
                activeWmmaCoExecAdvance += std::max(1, inst->issueCycles);
            } else if (isMatrixInstruction(*inst)) {
                const int mfmaLatency = inst->latencyCycles;
                if (cycles - hwMFMA >= (mfmaLatency - 1)) {
                    cycles += inst->issueCycles;
                } else {
                    cycles = hwMFMA + mfmaLatency;
                }
                hwMFMA = cycles;

                // Open a fresh co-issue window for this matrix instruction.
                activeWmmaStartCycle = cycles;
                activeWmmaCoExecAdvance = 0;
                activeWmmaValuSlots = getWmmaCoExecProfile(*inst).valuCoExecSlots;

                // Activate the dscnt wait check once a WMMA actually consumes a
                // register produced by an in-flight ds_load.
                if (!waitCheckActive && srcOverlapsInFlight(*inst, inFlightDsLoads)) {
                    waitCheckActive = true;
                    PASS_DEBUG(std::cerr << "[RemoveDscnt] bb=\"" << bb.getLabel()
                                         << "\" cycle=" << cycles
                                         << " wait check activated by WMMA src/ds overlap\n");
                }
            } else {
                cycles += inst->issueCycles;
            }

            // --- track in-flight LDS ops / drain on dscnt waits ---
            if (isDSRead(*inst) || isDSWrite(*inst)) {
                if (isDSRead(*inst)) {
                    inFlightDsLoads.push_back(DsLoadEntry{.cycle = cycles,
                                                          .latency = inst->latencyCycles,
                                                          .dests = inst->getDestRegs()});
                    lastDsLoadLatency = inst->latencyCycles;
                } else {
                    // DS writes contribute to dscnt accounting but have no produced VGPR dest.
                    inFlightDsLoads.push_back(
                        DsLoadEntry{.cycle = cycles, .latency = inst->latencyCycles, .dests = {}});
                }
                ++dsLoadsSinceLastKeptDscnt;
            } else if (std::optional<int> keep =
                           waitCheckActive ? getDsWaitCount(*inst) : std::nullopt) {
                bool removeWaitInst = false;
                std::string removalComment;
                const size_t remaining = static_cast<size_t>(std::max(0, *keep));
                PASS_DEBUG(std::cerr << "[RemoveDscnt] bb=\"" << bb.getLabel()
                                     << "\" cycle=" << cycles << " s_wait_dscnt keep=" << *keep
                                     << " inFlight=" << inFlightDsLoads.size()
                                     << " remaining=" << remaining << "\n");
                while (inFlightDsLoads.size() > remaining) {
                    inFlightDsLoads.pop_front();
                }

                // After draining to the wait count, estimate how many of the
                // remaining loads the hardware has had time to return. The FIFO is
                // deliberately left alone - only the count is taken - so a load that
                // the model considers landed is still visible to srcOverlapsInFlight
                // and to the next wait's own estimate.
                const int drainedDsLoads = estimateDrainedDsLoads(inFlightDsLoads, cycles);

                // If the original wait count exceeds the number of loads that are
                // actually still in flight, tighten it to that number and record the
                // change as an "X->Y" note in the instruction comment.
                const int newVal =
                    std::max(0, static_cast<int>(inFlightDsLoads.size()) - drainedDsLoads);
                PASS_DEBUG(std::cerr << "[RemoveDscnt]   drained=" << drainedDsLoads << " of "
                                     << inFlightDsLoads.size() << " inFlight -> newVal=" << newVal
                                     << "\n");
                const bool canRemoveByDscntHistory =
                    (prevKeptDscnt >= 0) && (*keep >= (prevKeptDscnt + dsLoadsSinceLastKeptDscnt));
                const bool needTighten = (*keep > newVal);

                if (needTighten) {
                    PASS_DEBUG(std::cerr << "[RemoveDscnt]   tighten wait " << *keep << "->"
                                         << newVal << "\n");
                }

                if (canRemoveByDscntHistory) {
                    if (needTighten) {
                        removalComment = std::to_string(*keep) + "---->" + std::to_string(newVal);
                    } else {
                        removalComment = "remove dscnt: keep=" + std::to_string(*keep);
                    }
                    removalComment += " dscnt is removed";
                    PASS_DEBUG(std::cerr << "[RemoveDscnt]   remove condition met: keep=" << *keep
                                         << " prevKeptDscnt=" << prevKeptDscnt
                                         << " dsLoadsSinceLastKeptDscnt="
                                         << dsLoadsSinceLastKeptDscnt << "\n");
                    removeWaitInst = true;
                } else {
                    if (needTighten) {
                        std::string comment =
                            std::to_string(*keep) + "---->" + std::to_string(newVal);
                        setDsWaitCount(*inst, newVal);
                        appendComment(*inst, comment);
                        prevKeptDscnt = newVal;
                    } else {
                        prevKeptDscnt = *keep;
                    }
                    dsLoadsSinceLastKeptDscnt = 0;
                }
                if (removeWaitInst) {
                    PASS_DEBUG(std::cerr << "[RemoveDscnt]   remove wait and replace with comment: "
                                         << removalComment << "\n");
                    bb.insertIR(it, createTextCommentDirective(removalComment));
                    it = bb.eraseIR(it);
                    continue;
                }
            }
            ++it;
        }
    }
};

char RemoveDscntPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRemoveDscntPass() {
    return createRemoveDscntPass(kDsProximityThreshold);
}

std::unique_ptr<Pass> createRemoveDscntPass(int dsProximityThreshold) {
    return std::make_unique<RemoveDscntPass>(dsProximityThreshold);
}
}  // namespace stinkytofu
