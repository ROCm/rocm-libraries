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

#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

namespace {
using namespace stinkytofu;

// ----------------------------------------------------------------------------
// Free-standing helpers
// ----------------------------------------------------------------------------

/// Recursively collect register sources of @p inst, flattening PHI operands. Non-PHI
/// operands are inserted directly; each PHI is replaced by its incoming sources.
/// @p seenPhi avoids infinite recursion on cyclic PHI webs (e.g., loop-carried deps).
static void collectSourcesRec(StinkyInstruction* inst,
                              std::unordered_set<StinkyInstruction*>& seenPhi,
                              std::unordered_set<StinkyInstruction*>& out,
                              const std::function<bool(StinkyInstruction*)>& filter = nullptr) {
    if (inst == nullptr) {
        return;
    }

    if (inst->getUnifiedOpcode() == GFX::PHI) {
        if (!seenPhi.insert(inst).second) {
            return;
        }
    }

    for (StinkyInstruction* src : inst->getSources()) {
        if (src == nullptr) {
            continue;
        }
        if (src->getUnifiedOpcode() == GFX::PHI) {
            collectSourcesRec(src, seenPhi, out, filter);
        } else {
            if (filter == nullptr || filter(src)) {
                out.insert(src);
            }
        }
    }
}

/// Entry point for source collection. PHIs are flattened to their incoming values.
std::unordered_set<StinkyInstruction*> collectSources(
    StinkyInstruction* inst, const std::function<bool(StinkyInstruction*)>& filter = nullptr) {
    std::unordered_set<StinkyInstruction*> sources;
    std::unordered_set<StinkyInstruction*> seenPhi;
    if (inst != nullptr) {
        collectSourcesRec(inst, seenPhi, sources, filter);
    }
    return sources;
}

bool isDSMemoryOp(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst);
}

bool isBufferMemoryOp(const StinkyInstruction& inst) {
    return isGlobalMemLoad(inst) || isGlobalMemStore(inst);
}

/// Returns true if any element of @p a is also present in @p b.
template <typename ContainerA, typename ContainerB>
bool hasTokenOverlap(const ContainerA& a, const ContainerB& b) {
    return std::any_of(a.begin(), a.end(),
                       [&b](int token) { return std::find(b.begin(), b.end(), token) != b.end(); });
}

/// True for LDS writers (tensor_load_to_lds / ds_write) carrying MemTokenData.
bool isLdsWriterWithTokens(const StinkyInstruction& inst) {
    if (!isTensorLoad(inst) && !isDSWrite(inst)) {
        return false;
    }
    return inst.getModifier<MemTokenData>() != nullptr;
}

/// True when @p a and @p b share the same hardware memory pipeline, so the
/// hardware-side FIFO retirement orders them implicitly and no synthetic wait
/// is required between them.
///
/// Today this only covers the DS (dlcnt) pipeline: ds_read, ds_write and
/// ds_atomic all issue and retire FIFO on dlcnt. tensor_load_to_lds is NOT a
/// DS op (it lives on tlcnt), so a tensor_load_to_lds / ds_read pair is not
/// on the same pipeline.
bool isOnSameDSPipeline(const StinkyInstruction& a, const StinkyInstruction& b) {
    return isDSMemoryOp(a) && isDSMemoryOp(b);
}

// ----------------------------------------------------------------------------
// Data structures
// ----------------------------------------------------------------------------

/// Descriptor for the wait counter immediate(s) to emit before an anchor instruction.
/// Each field is either a non-negative count immediate or @c kUnused if not needed.
struct WaitCountSpec {
    /// Sentinel: this counter is not used for this wait record.
    static constexpr int kUnused = -1;

    /// Immediate for @c s_wait_dscnt (dlcnt); @c kUnused if omitted.
    int dsCount;
    /// Immediate for @c s_wait_loadcnt (vlcnt); @c kUnused if omitted.
    int bufferCount;
    /// Immediate for @c s_wait_tensorcnt (tlcnt); @c kUnused if omitted.
    int tensorCount;

    WaitCountSpec(int dsCount = kUnused, int bufferCount = kUnused, int tensorCount = kUnused)
        : dsCount(dsCount), bufferCount(bufferCount), tensorCount(tensorCount) {}

    bool isValid() const {
        return dsCount != kUnused || bufferCount != kUnused || tensorCount != kUnused;
    }
};

/// Tracks in-flight (not yet waited-on) memory operations as ordered queues. Each
/// processed basic block has an associated tracker representing its exit state.
struct PendingMemOpTracker {
    /// DS reads/writes in issue order.
    std::deque<StinkyInstruction*> pendingDSOps;
    /// MemTokenData tokens accumulated from pending DS ops (used for barrier conflict).
    std::unordered_set<int> activeDSTokens;
    /// Global loads/stores in issue order.
    std::deque<StinkyInstruction*> pendingBufferOps;

    int pendingDSCount() const {
        return static_cast<int>(pendingDSOps.size());
    }

    int pendingBufferCount() const {
        return static_cast<int>(pendingBufferOps.size());
    }

    /// Number of DS ops from @p inst (inclusive) to the end of the queue. Returns 0 if
    /// @p inst is not in the queue. Maps directly to the hardware wait-count immediate.
    int pendingDSCountFrom(StinkyInstruction* inst) const {
        auto it = std::find(pendingDSOps.begin(), pendingDSOps.end(), inst);
        if (it != pendingDSOps.end()) {
            return static_cast<int>(std::distance(it, pendingDSOps.end()));
        }
        return 0;
    }

    /// Number of buffer ops from @p inst (inclusive) to the end of the queue. Returns 0
    /// if @p inst is not in the queue.
    int pendingBufferCountFrom(StinkyInstruction* inst) const {
        auto it = std::find(pendingBufferOps.begin(), pendingBufferOps.end(), inst);
        if (it != pendingBufferOps.end()) {
            return static_cast<int>(std::distance(it, pendingBufferOps.end()));
        }
        return 0;
    }

    void clear() {
        pendingDSOps.clear();
        pendingBufferOps.clear();
        activeDSTokens.clear();
    }

    /// Append @p inst to the DS queue (and collect its tokens). Returns true iff appended.
    bool recordDSOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isDSMemoryOp(*inst)) {
            return false;
        }

        pendingDSOps.push_back(inst);
        if (auto* memTokenData = inst->getModifier<MemTokenData>()) {
            for (int token : memTokenData->tokens) {
                activeDSTokens.emplace(token);
            }
        }
        return true;
    }

    /// Append @p inst to the buffer queue. Returns true iff appended.
    bool recordBufferOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isBufferMemoryOp(*inst)) {
            return false;
        }

        pendingBufferOps.push_back(inst);
        return true;
    }

    /// Trim @p queue so only the last @p lastWait ops remain. No-op when there is no
    /// prior wait recorded (lastWait < 0) or it was 0 (the wait-0 emission paths
    /// already clear the queue themselves).
    void trimQueueToLastWait(std::deque<StinkyInstruction*>& queue, int lastWait) {
        if (lastWait <= 0) {
            return;
        }
        if (lastWait >= static_cast<int>(queue.size())) {
            return;
        }
        queue.erase(queue.begin(), queue.end() - lastWait);
    }
};

// ----------------------------------------------------------------------------
// Pass class
// ----------------------------------------------------------------------------

class StinkyWaitCntInsertionPass : public StinkyInstPass {
   public:
    /// List of (anchor instruction, wait spec) pairs in emission order.
    using WaitInsertionList = std::vector<std::pair<StinkyInstruction*, WaitCountSpec>>;

    StinkyWaitCntInsertionPass() = default;

    static char ID;

    const char* getName() const override {
        return "StinkyWaitCntInsertionPass";
    }

    PassID getPassID() const override {
        return &StinkyWaitCntInsertionPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, true);
        const auto& rpo = AM.getResult<BBIndexAnalysis>(func).rpo;

        buildBlockExitStates(func, passCtx);

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }
            WaitInsertionList waits = computeRequiredWaits(*bb);
            emitWaitInstructions(*bb, arch, waits);
        }

        reinsertTensorWaitsHeuristic(arch, passCtx, rpo);

        removePHIs(passCtx, rpo);
        return preserveCFGAnalyses();
    }

   private:
    /// Per-counter bookkeeping during a single block walk. Tracks the last emitted wait
    /// value and how many new ops have been issued since, enabling redundancy elision.
    struct CounterWaitState {
        int lastEmittedWait = WaitCountSpec::kUnused;
        int opsSinceLastWait = 0;

        void recordNewOp() {
            ++opsSinceLastWait;
        }

        bool needsNewWait(int required) const {
            return lastEmittedWait == WaitCountSpec::kUnused ||
                   lastEmittedWait + opsSinceLastWait > required;
        }

        void recordEmittedWait(int value) {
            lastEmittedWait = value;
            opsSinceLastWait = 0;
        }
    };

    /// Exit-state queues for each processed block (full block walk, program order).
    /// Pre-populated by buildBlockExitStates and refined per block by computeRequiredWaits.
    std::unordered_map<BasicBlock*, PendingMemOpTracker> blockExitMemState;

    /// Phase 1: pre-scan all processed blocks to record in-flight memory ops at exit.
    void buildBlockExitStates(Function& func, PassContext& passCtx) {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) {
                blockExitMemState.emplace(&bb, PendingMemOpTracker());
                scanBlockMemOps(bb);
            }
        }
    }

    /// Helper for buildBlockExitStates: record DS/buffer ops in @p bb in program order.
    void scanBlockMemOps(BasicBlock& bb) {
        PendingMemOpTracker& state = blockExitMemState[&bb];
        state.clear();
        for (IRBase& ir : bb) {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) {
                continue;
            }

            state.recordDSOperation(inst);
            state.recordBufferOperation(inst);
        }
    }

    /// Compute the minimum wait-count value for a single counter (DS or buffer) given
    /// the consumer's memory-op dependencies. The @p getCountFrom callback delegates to
    /// either pendingDSCountFrom or pendingBufferCountFrom so the same algorithm covers
    /// both counters. Returns (requiredWait, needsWait).
    ///
    /// See "Wait Value Computation: computeWaitValueForCounter" in the user docs.
    std::pair<int, bool> computeWaitValueForCounter(
        int initialPendingCount, const PendingMemOpTracker& localState,
        const std::unordered_set<StinkyInstruction*>& memOpDependencies,
        const CounterWaitState& counterState,
        const std::function<int(const PendingMemOpTracker&, StinkyInstruction*)>& getCountFrom) {
        int requiredWait = initialPendingCount;
        bool needsWait = false;
        std::vector<int> predecessorValues;

        for (StinkyInstruction* src : memOpDependencies) {
            int count = getCountFrom(localState, src);
            if (count > 0) {
                requiredWait = std::min(requiredWait, count - 1);
                needsWait = true;
            } else if (counterState.lastEmittedWait != 0) {
                // Unprocessed predecessor blocks insert an empty state via operator[].
                count = getCountFrom(blockExitMemState[src->getParent()], src);
                if (count > 0) {
                    predecessorValues.push_back(count - 1);
                    needsWait = true;
                }
            }
        }

        if (!predecessorValues.empty()) {
            requiredWait += *std::min_element(predecessorValues.begin(), predecessorValues.end());
        }

        return {requiredWait, needsWait};
    }

    /// Collect prior DS reads/atomics whose MemTokenData tokens overlap @p writer's.
    /// Scans @p localState (current block) and each predecessor's exit state to
    /// surface the WAR-on-LDS hazard that the SSA def-use chain does not encode.
    ///
    /// Per-pair pipeline filter: if a candidate reader is on the same hardware
    /// memory pipeline as @p writer (see isOnSameDSPipeline), it is skipped --
    /// the hardware FIFO-retires the pair on the shared counter, so no
    /// synthetic wait is required. In practice this excludes all candidates
    /// when @p writer is a ds_write (both writer and readers live on dlcnt),
    /// while leaving tensor_load_to_lds writers (on tlcnt) fully covered.
    std::unordered_set<StinkyInstruction*> collectLdsWarDependencies(
        StinkyInstruction* writer, BasicBlock& bb, const PendingMemOpTracker& localState) {
        std::unordered_set<StinkyInstruction*> warDeps;
        const MemTokenData* writerTokens = writer->getModifier<MemTokenData>();
        if (writerTokens == nullptr) {
            return warDeps;
        }

        auto isConflictingRead = [&](StinkyInstruction* op) {
            if (op == nullptr || op == writer) {
                return false;
            }
            if (!isDSRead(*op) && !isDSAtomic(*op)) {
                return false;
            }
            // Same-pipeline pairs are FIFO-ordered by hardware, so no synthetic
            // WAR wait is needed. Today this matches ds_write writer paired
            // with ds_read / ds_atomic readers on the DS (dlcnt) pipeline.
            if (isOnSameDSPipeline(*writer, *op)) {
                return false;
            }
            const MemTokenData* mt = op->getModifier<MemTokenData>();
            return mt != nullptr && hasTokenOverlap(mt->tokens, writerTokens->tokens);
        };

        for (StinkyInstruction* op : localState.pendingDSOps) {
            if (isConflictingRead(op)) {
                warDeps.insert(op);
            }
        }

        for (BasicBlock* pred : bb.getPredecessors()) {
            auto it = blockExitMemState.find(pred);
            if (it == blockExitMemState.end()) {
                continue;
            }
            for (StinkyInstruction* op : it->second.pendingDSOps) {
                if (isConflictingRead(op)) {
                    warDeps.insert(op);
                }
            }
        }

        return warDeps;
    }

    /// Phase 2: walk @p bb in program order; return (anchor, WaitCountSpec) pairs.
    WaitInsertionList computeRequiredWaits(BasicBlock& bb) {
        PendingMemOpTracker localState;
        WaitInsertionList waits;
        CounterWaitState dsState;
        CounterWaitState bufferState;

        for (auto it = bb.begin(); it != bb.end(); ++it) {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr || inst->getUnifiedOpcode() == GFX::PHI) {
                continue;
            }

            // Barrier with overlapping MemTokenData forces s_wait_dscnt 0 (see doc: section
            // "Barrier token conflict handling"). Barriers are never DS / buffer ops,
            // so the tail recording at the bottom of this loop body is a no-op for
            // them; the explicit continue here keeps the existing semantics.
            if (isBarrier(*inst)) {
                if (auto* memTokenData = inst->getModifier<MemTokenData>()) {
                    if (hasTokenOverlap(localState.activeDSTokens, memTokenData->tokens)) {
                        waits.emplace_back(inst, WaitCountSpec(0, WaitCountSpec::kUnused));
                        dsState.recordEmittedWait(0);
                        localState.activeDSTokens.clear();
                        localState.pendingDSOps.clear();
                        continue;
                    }
                }
            }

            auto memOpDependencies = collectSources(inst, [](StinkyInstruction* src) {
                return isDSMemoryOp(*src) || isBufferMemoryOp(*src);
            });

            // WAR-on-LDS: an LDS writer (tensor_load_to_lds / ds_write) must wait
            // for prior token-overlapping DS reads to drain. Run this before the
            // tail recording so that the writer is not yet in pendingDSOps, which
            // makes the count-based wait value match hardware semantics: the wait
            // executes before the writer issues, so the queue at wait-time excludes
            // the writer.
            if (isLdsWriterWithTokens(*inst)) {
                auto warDeps = collectLdsWarDependencies(inst, bb, localState);
                memOpDependencies.insert(warDeps.begin(), warDeps.end());
            }

            if (!memOpDependencies.empty()) {
                // Compute waits against the pre-consumer queue. For DS-op / buffer-op
                // consumers, this gives the correct dlcnt / vlcnt because the wait
                // executes before the consumer is issued.
                auto dsResult = computeWaitValueForCounter(
                    localState.pendingDSCount(), localState, memOpDependencies, dsState,
                    [](const PendingMemOpTracker& tracker, StinkyInstruction* src) {
                        return tracker.pendingDSCountFrom(src);
                    });
                if (dsResult.second && dsState.needsNewWait(dsResult.first)) {
                    waits.emplace_back(inst, WaitCountSpec(dsResult.first, WaitCountSpec::kUnused));
                    dsState.recordEmittedWait(dsResult.first);
                }

                auto bufferResult = computeWaitValueForCounter(
                    localState.pendingBufferCount(), localState, memOpDependencies, bufferState,
                    [](const PendingMemOpTracker& tracker, StinkyInstruction* src) {
                        return tracker.pendingBufferCountFrom(src);
                    });
                if (bufferResult.second && bufferState.needsNewWait(bufferResult.first)) {
                    waits.emplace_back(inst,
                                       WaitCountSpec(WaitCountSpec::kUnused, bufferResult.first));
                    bufferState.recordEmittedWait(bufferResult.first);
                }
            }

            // Tail recording: after any wait for this instruction has been emitted,
            // push inst onto the trackers so future iterations and the block exit
            // state see it as in-flight.
            if (localState.recordDSOperation(inst)) {
                dsState.recordNewOp();
            }
            if (localState.recordBufferOperation(inst)) {
                bufferState.recordNewOp();
            }
        }

        // Trim exit-state queues using the last emitted wait so successor blocks see
        // only ops still genuinely in flight.
        localState.trimQueueToLastWait(localState.pendingDSOps, dsState.lastEmittedWait);
        localState.trimQueueToLastWait(localState.pendingBufferOps, bufferState.lastEmittedWait);

        blockExitMemState[&bb] = localState;
        return waits;
    }

    /// Phase 3: insert wait IR nodes immediately before each anchor. The caller
    /// (computeRequiredWaits) is responsible for redundancy elision via
    /// CounterWaitState::needsNewWait; this method emits exactly what it is given.
    void emitWaitInstructions(BasicBlock& bb, GfxArchID arch, const WaitInsertionList& waits) {
        auto irBuilder = AsmIRBuilder(bb, arch);

        for (const auto& entry : waits) {
            StinkyInstruction* inst = entry.first;
            const WaitCountSpec& waitSpec = entry.second;

            if (waitSpec.dsCount != WaitCountSpec::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitSpec.dsCount));

                SWaitCntData waitCntData;
                waitCntData.dlcnt = waitSpec.dsCount;
                waitInst->addModifier<SWaitCntData>(waitCntData);
            }

            if (waitSpec.bufferCount != WaitCountSpec::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_loadcnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitSpec.bufferCount));

                SWaitCntData waitCntData;
                waitCntData.vlcnt = waitSpec.bufferCount;
                waitInst->addModifier<SWaitCntData>(waitCntData);
            }

            if (waitSpec.tensorCount != WaitCountSpec::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitSpec.tensorCount));

                SWaitTensorCntData waitCntData;
                waitCntData.tlcnt = waitSpec.tensorCount;
                waitInst->addModifier<SWaitTensorCntData>(waitCntData);
            }
        }
    }

    /// Phase 4: tensor wait heuristic.
    ///   1. Remove existing tensor waits in label_LoopBeginL / label_LoopEndL blocks.
    ///   2. Reinsert tensor waits before barriers using token matching against a
    ///      cross-block deque of tensor loads.
    void reinsertTensorWaitsHeuristic(GfxArchID arch, PassContext& passCtx,
                                      const std::vector<BasicBlock*>& rpo) {
        const std::unordered_set<std::string> processedLabels = {"label_LoopBeginL",
                                                                 "label_LoopEndL"};

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            if (processedLabels.find(bb->getLabel()) == processedLabels.end()) {
                continue;
            }

            for (auto it = bb->begin(); it != bb->end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && inst->is(InstFlag::IF_WaitTensorCnt)) {
                    it = bb->eraseIR(it);
                } else {
                    ++it;
                }
            }
        }

        // Cross-block deque: tensor loads accumulated in RPO order across all processed
        // blocks. At each barrier, we pop loads whose tokens overlap with the barrier's
        // tokens and emit a wait for the remaining count.
        std::deque<StinkyInstruction*> tensorLoads;
        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            WaitInsertionList tensorWaits;
            for (auto it = bb->begin(); it != bb->end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) {
                    continue;
                }

                if (isTensorLoad(*inst) && inst->getModifier<MemTokenData>() != nullptr) {
                    tensorLoads.push_back(inst);
                }

                if (isBarrier(*inst) && !tensorLoads.empty()) {
                    auto* memTokenData = inst->getModifier<MemTokenData>();
                    if (memTokenData == nullptr) {
                        continue;
                    }

                    const auto& barrierTokens = memTokenData->tokens;
                    int lastDependentIndex = -1;
                    for (int i = static_cast<int>(tensorLoads.size()) - 1; i >= 0; --i) {
                        auto* tlMemTokenData = tensorLoads[i]->getModifier<MemTokenData>();
                        if (tlMemTokenData == nullptr) {
                            continue;
                        }
                        if (hasTokenOverlap(tlMemTokenData->tokens, barrierTokens)) {
                            lastDependentIndex = i;
                            break;
                        }
                    }

                    if (lastDependentIndex >= 0) {
                        int remainingTensorLoads =
                            static_cast<int>(tensorLoads.size()) - 1 - lastDependentIndex;
                        tensorWaits.emplace_back(
                            inst, WaitCountSpec(WaitCountSpec::kUnused, WaitCountSpec::kUnused,
                                                remainingTensorLoads));

                        // Loads up to lastDependentIndex are guaranteed complete after wait.
                        tensorLoads.erase(tensorLoads.begin(),
                                          tensorLoads.begin() + lastDependentIndex + 1);
                    }
                }
            }

            emitWaitInstructions(*bb, arch, tensorWaits);
        }
    }

    /// Phase 5: strip PHI pseudo-instructions from processed blocks.
    void removePHIs(PassContext& passCtx, const std::vector<BasicBlock*>& rpo) {
        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            for (auto it = bb->begin(); it != bb->end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && inst->getUnifiedOpcode() == GFX::PHI) {
                    it = bb->eraseIR(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

char StinkyWaitCntInsertionPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyWaitCntInsertionPass() {
    return std::make_unique<StinkyWaitCntInsertionPass>();
}
}  // namespace stinkytofu
