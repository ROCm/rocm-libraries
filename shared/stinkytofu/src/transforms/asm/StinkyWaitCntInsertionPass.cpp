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

/// Collect all register sources of @p inst, flattening PHI operands recursively. Non-PHI
/// operands are listed in order; each PHI is replaced by its flattened incoming values.
/// @p seenPhi avoids infinite recursion on cyclic PHI webs (defensive).
static void collectSourcesRec(StinkyInstruction* inst,
                              std::unordered_set<StinkyInstruction*>& seenPhi,
                              std::unordered_set<StinkyInstruction*>& out,
                              std::function<bool(StinkyInstruction*)> filter = nullptr) {
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

/// Collect all sources of the instruction. If a source is a PHI, collect all its sources
/// recursively.
std::unordered_set<StinkyInstruction*> collectSources(
    StinkyInstruction* inst, std::function<bool(StinkyInstruction*)> filter = nullptr) {
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

bool isTensorMemoryOp(const StinkyInstruction& inst) {
    return isTensorLoad(inst);
}

bool isBarrierOp(const StinkyInstruction& inst) {
    return isBarrier(inst);
}

struct MemoryOperationState {
    /// Queue of DS read/write instructions in program order.
    std::deque<StinkyInstruction*> dsQueue;
    std::unordered_set<int> dsMemTokens;

    /// Queue of buffer read/write instructions in program order.
    std::deque<StinkyInstruction*> bufferQueue;

    int getDSCountToComplete() const {
        return static_cast<int>(dsQueue.size());
    }

    int getBufferCountToComplete() const {
        return static_cast<int>(bufferQueue.size());
    }

    void clear() {
        dsQueue.clear();
        bufferQueue.clear();
    }

    /// If @p inst is a DS op, append to @c dsQueue.
    bool recordDSOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isDSMemoryOp(*inst)) {
            return false;
        }

        dsQueue.push_back(inst);
        if (auto* memTokenData = inst->getModifier<MemTokenData>()) {
            for (int token : memTokenData->tokens) {
                dsMemTokens.emplace(token);
            }
        }
        return true;
    }

    /// If @p inst is a buffer op, append to @c bufferQueue.
    bool recordBufferOperation(StinkyInstruction* inst) {
        if (inst == nullptr || !isBufferMemoryOp(*inst)) {
            return false;
        }

        bufferQueue.push_back(inst);
        return true;
    }

    /// If @p inst is a barrier, clear the DS queue (modeled semantics).
    bool applyBarrierIfPresent(StinkyInstruction* inst) {
        if (inst == nullptr || !isBarrierOp(*inst)) {
            return false;
        }

        // dsQueue.clear();
        return true;
    }

    int getDSCountToCompleteIncluding(StinkyInstruction* inst) const {
        auto it = std::find(dsQueue.begin(), dsQueue.end(), inst);
        if (it != dsQueue.end()) {
            return static_cast<int>(std::distance(it, dsQueue.end()));
        }
        return 0;
    }

    int getBufferCountToCompleteIncluding(StinkyInstruction* inst) const {
        auto it = std::find(bufferQueue.begin(), bufferQueue.end(), inst);
        if (it != bufferQueue.end()) {
            return static_cast<int>(std::distance(it, bufferQueue.end()));
        }
        return 0;
    }
};

struct WaitCntInstruction {
    /// Sentinel: this counter is not used for this wait record.
    static constexpr int kUnused = -1;

    /// Immediate for @c s_wait_dscnt (dlcnt); @c kUnused if omitted.
    int dsCount;
    /// Immediate for @c s_wait_loadcnt (vlcnt); @c kUnused if omitted.
    int bufferCount;
    /// Immediate for @c s_wait_tensorcnt (tlcnt); @c kUnused if omitted.
    int tensorCount;
    WaitCntInstruction(int dsCount = kUnused, int bufferCount = kUnused, int tensorCount = kUnused)
        : dsCount(dsCount), bufferCount(bufferCount), tensorCount(tensorCount) {}

    bool isValid() const {
        return dsCount != kUnused || bufferCount != kUnused || tensorCount != kUnused;
    }
};

class StinkyWaitCntInsertionPass : public StinkyInstPass {
   public:
    /// List of instructions to insert waitcnts for and the corresponding waitcnt type
    using WaitInsertionList = std::vector<std::pair<StinkyInstruction*, WaitCntInstruction>>;

   public:
    StinkyWaitCntInsertionPass(bool insertTensorWaitCnt)
        : insertTensorWaitCnt(insertTensorWaitCnt) {}

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
        rebuildExitMemoryStateForProcessedBlocks(func, passCtx);

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }
            WaitInsertionList waits = collectWaitsForBlock(*bb);
            insertWaitsForBlock(*bb, arch, waits);
        }

        // Handle tensor waits for DS reads.
        if (insertTensorWaitCnt) {
            // handleTensorWaits(func, arch, passCtx, rpo);
            reinsertTensorWaitsHeuristic(func, arch, passCtx, rpo);
        }

        removePHIs(func, passCtx, rpo);
        return preserveCFGAnalyses();
    }

   private:
    bool insertTensorWaitCnt;

    /// Exit-state queues for each processed block (full block walk, program order).
    std::unordered_map<BasicBlock*, MemoryOperationState> blockStates;

    void rebuildExitMemoryStateForProcessedBlocks(Function& func, PassContext& passCtx) {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) {
                blockStates.emplace(&bb, MemoryOperationState());
                trackMemoryOperationsInBlock(bb);
            }
        }
    }

    void trackMemoryOperationsInBlock(BasicBlock& bb) {
        MemoryOperationState& state = blockStates[&bb];
        state.clear();
        for (IRBase& ir : bb) {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) {
                continue;
            }

            state.recordDSOperation(inst);
            state.recordBufferOperation(inst);
            state.applyBarrierIfPresent(inst);
        }
    }

    /// Simulate the block in order; return (anchor instruction, wait spec) pairs in order.
    WaitInsertionList collectWaitsForBlock(BasicBlock& bb) {
        MemoryOperationState currentBlockState;
        WaitInsertionList instructionsNeedWait;

        int lastDSWaitCount = WaitCntInstruction::kUnused;
        int incrementDSCountAfterLastWait = 0;
        int lastBufferWaitCount = WaitCntInstruction::kUnused;
        int incrementBufferCountAfterLastWait = 0;
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            StinkyInstruction* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (inst == nullptr || inst->getUnifiedOpcode() == GFX::PHI) {
                continue;
            }

            if (currentBlockState.recordDSOperation(inst)) {
                incrementDSCountAfterLastWait += 1;
            }

            if (currentBlockState.recordBufferOperation(inst)) {
                incrementBufferCountAfterLastWait += 1;
            }

            // if barrier has memTokens, check if currentBlockState has any token in barrier
            // memTokens if yes, insert ds wait 0 and clear currentBlockState.dsMemTokens and
            // currentBlockState.dsQueue if no, continue
            if (isBarrier(*inst)) {
                if (auto* memTokenData = inst->getModifier<MemTokenData>()) {
                    const auto& barrierMemTokens = memTokenData->tokens;
                    // check if currentBlockState has any token in barrier memTokens
                    if (std::any_of(
                            currentBlockState.dsMemTokens.begin(),
                            currentBlockState.dsMemTokens.end(), [barrierMemTokens](int token) {
                                return std::find(barrierMemTokens.begin(), barrierMemTokens.end(),
                                                 token) != barrierMemTokens.end();
                            })) {
                        // insert ds wait 0
                        instructionsNeedWait.emplace_back(
                            inst, WaitCntInstruction(0, WaitCntInstruction::kUnused));
                        lastDSWaitCount = 0;
                        incrementDSCountAfterLastWait = 0;
                        currentBlockState.dsMemTokens.clear();
                        currentBlockState.dsQueue.clear();
                        continue;
                    }
                }
            }

            auto srcs = collectSources(inst, [](StinkyInstruction* src) {
                return isDSMemoryOp(*src) || isBufferMemoryOp(*src);
            });
            if (srcs.empty()) {
                continue;
            }

            int dsCountBeforeInst = currentBlockState.getDSCountToComplete();
            int bufferCountBeforeInst = currentBlockState.getBufferCountToComplete();
            std::vector<int> dsCountFromPredecessor;
            std::vector<int> bufferCountFromPredecessor;
            bool needDSWait = false;
            bool needBufferWait = false;

            for (StinkyInstruction* src : srcs) {
                int dsCount = currentBlockState.getDSCountToCompleteIncluding(src);
                if (dsCount > 0) {
                    dsCountBeforeInst = std::min(dsCountBeforeInst, dsCount - 1);
                    needDSWait = true;
                } else if (lastDSWaitCount != 0) {
                    // Unprocessed predecessor blocks insert an empty state via operator[].
                    dsCount = blockStates[src->getParent()].getDSCountToCompleteIncluding(src);
                    if (dsCount > 0) {
                        dsCountFromPredecessor.push_back(dsCount - 1);
                        needDSWait = true;
                    }
                }

                int bufferCount = currentBlockState.getBufferCountToCompleteIncluding(src);
                if (bufferCount > 0) {
                    bufferCountBeforeInst = std::min(bufferCountBeforeInst, bufferCount - 1);
                    needBufferWait = true;
                } else if (lastBufferWaitCount != 0) {
                    bufferCount =
                        blockStates[src->getParent()].getBufferCountToCompleteIncluding(src);
                    if (bufferCount > 0) {
                        bufferCountFromPredecessor.push_back(bufferCount - 1);
                        needBufferWait = true;
                    }
                }
            }

            if (needDSWait) {
                if (!dsCountFromPredecessor.empty()) {
                    dsCountBeforeInst += *std::min_element(dsCountFromPredecessor.begin(),
                                                           dsCountFromPredecessor.end());
                }

                if (lastDSWaitCount == WaitCntInstruction::kUnused ||
                    lastDSWaitCount + incrementDSCountAfterLastWait > dsCountBeforeInst) {
                    instructionsNeedWait.emplace_back(
                        inst, WaitCntInstruction(dsCountBeforeInst, WaitCntInstruction::kUnused));
                    lastDSWaitCount = dsCountBeforeInst;
                    incrementDSCountAfterLastWait = 0;
                }
            }

            if (needBufferWait) {
                if (!bufferCountFromPredecessor.empty()) {
                    bufferCountBeforeInst += *std::min_element(bufferCountFromPredecessor.begin(),
                                                               bufferCountFromPredecessor.end());
                }

                if (lastBufferWaitCount == WaitCntInstruction::kUnused ||
                    lastBufferWaitCount + incrementBufferCountAfterLastWait >
                        bufferCountBeforeInst) {
                    instructionsNeedWait.emplace_back(
                        inst,
                        WaitCntInstruction(WaitCntInstruction::kUnused, bufferCountBeforeInst));
                    lastBufferWaitCount = bufferCountBeforeInst;
                    incrementBufferCountAfterLastWait = 0;
                }
            }
        }

        if (lastDSWaitCount != WaitCntInstruction::kUnused && lastDSWaitCount != 0 &&
            lastDSWaitCount < currentBlockState.dsQueue.size()) {
            currentBlockState.dsQueue.erase(currentBlockState.dsQueue.begin(),
                                            currentBlockState.dsQueue.end() - lastDSWaitCount);
        }

        if (lastBufferWaitCount != WaitCntInstruction::kUnused && lastBufferWaitCount != 0 &&
            lastBufferWaitCount < currentBlockState.bufferQueue.size()) {
            currentBlockState.bufferQueue.erase(
                currentBlockState.bufferQueue.begin(),
                currentBlockState.bufferQueue.end() - lastBufferWaitCount);
        }

        // assign currentBlockState back to blockStates[&bb]
        blockStates[&bb] = currentBlockState;

        return instructionsNeedWait;
    }

    /// Insert waits before each anchor. Consecutive identical immediates for the same
    /// counter are skipped (same insertion order as @p waits).
    void insertWaitsForBlock(BasicBlock& bb, GfxArchID arch, const WaitInsertionList& waits) {
        auto irBuilder = AsmIRBuilder(bb, arch);

        for (const auto& entry : waits) {
            StinkyInstruction* inst = entry.first;
            const WaitCntInstruction& waitCntInstruction = entry.second;

            if (waitCntInstruction.dsCount != WaitCntInstruction::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitCntInstruction.dsCount));

                SWaitCntData waitCntData;
                waitCntData.dlcnt = waitCntInstruction.dsCount;
                waitInst->addModifier<SWaitCntData>(waitCntData);
            }

            if (waitCntInstruction.bufferCount != WaitCntInstruction::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_loadcnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitCntInstruction.bufferCount));

                SWaitCntData waitCntData;
                waitCntData.vlcnt = waitCntInstruction.bufferCount;
                waitInst->addModifier<SWaitCntData>(waitCntData);
            }

            if (waitCntInstruction.tensorCount != WaitCntInstruction::kUnused) {
                StinkyInstruction* waitInst =
                    irBuilder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), inst);
                waitInst->addSrcReg(StinkyRegister(waitCntInstruction.tensorCount));

                SWaitTensorCntData waitCntData;
                waitCntData.tlcnt = waitCntInstruction.tensorCount;
                waitInst->addModifier<SWaitTensorCntData>(waitCntData);

                // // insert s_barrier_signal and s_barrier_wait
                // StinkyInstruction* barrierSignalInst
                //     = irBuilder.create(getMCIDByUOp(GFX::s_barrier_signal, arch), inst);
                // barrierSignalInst->addSrcReg(StinkyRegister(-1));
                // StinkyInstruction* barrierWaitInst
                //     = irBuilder.create(getMCIDByUOp(GFX::s_barrier_wait, arch), inst);
                // barrierWaitInst->addSrcReg(StinkyRegister(-1));
            }
        }
    }

    /// heuristic reinsert tensor waitcnts
    /// 1. remove tensor waitcnts in LoopBeginL and LoopEndL basic blocks
    /// 2. insert tensor waitcnts for tensor_load that has memTokens in barrier
    void reinsertTensorWaitsHeuristic(Function& func, GfxArchID arch, PassContext& passCtx,
                                      const std::vector<BasicBlock*>& rpo) {
        // handle only for label_LoopBeginL and label_LoopEndL basic block
        std::unordered_set<std::string> processedLabels = {"label_LoopBeginL", "label_LoopEndL"};

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

        std::deque<StinkyInstruction*> tensorLoads;
        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            WaitInsertionList tensorWaits;
            for (auto it = bb->begin(); it != bb->end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && isTensorLoad(*inst) && inst->getModifier<MemTokenData>() != nullptr) {
                    tensorLoads.push_back(inst);
                }

                if (inst && isBarrier(*inst)) {
                    if (tensorLoads.empty()) {
                        continue;
                    }

                    if (auto* memTokenData = inst->getModifier<MemTokenData>()) {
                        auto& barrierMemTokens = memTokenData->tokens;
                        int lastDependentIndex = -1;
                        for (int i = static_cast<int>(tensorLoads.size()) - 1; i >= 0; --i) {
                            auto* tensorLoad = tensorLoads[i];
                            auto* tensorMemTokenData = tensorLoad->getModifier<MemTokenData>();
                            if (tensorMemTokenData == nullptr) {
                                continue;
                            }

                            auto& tensorLoadMemTokens = tensorMemTokenData->tokens;
                            if (std::any_of(tensorLoadMemTokens.begin(), tensorLoadMemTokens.end(),
                                            [barrierMemTokens](int token) {
                                                return std::find(barrierMemTokens.begin(),
                                                                 barrierMemTokens.end(),
                                                                 token) != barrierMemTokens.end();
                                            })) {
                                lastDependentIndex = i;
                                break;
                            }
                        }

                        if (lastDependentIndex >= 0) {
                            int remainingTensorLoads =
                                static_cast<int>(tensorLoads.size()) - 1 - lastDependentIndex;
                            tensorWaits.emplace_back(inst,
                                                     WaitCntInstruction(WaitCntInstruction::kUnused,
                                                                        WaitCntInstruction::kUnused,
                                                                        remainingTensorLoads));

                            // Loads up to lastDependentIndex are guaranteed complete after wait.
                            tensorLoads.erase(tensorLoads.begin(),
                                              tensorLoads.begin() + lastDependentIndex + 1);
                        }
                    }
                }
            }

            insertWaitsForBlock(*bb, arch, tensorWaits);
        }
    }

    /// Handle tensor waits for DS reads.
    void handleTensorWaits(Function& func, GfxArchID arch, PassContext& passCtx,
                           const std::vector<BasicBlock*>& rpo) {
        struct TensorLoadBlockState {
            std::unordered_map<StinkyInstruction*, int> dsReadSources;
            std::deque<StinkyInstruction*> tensorLoadQueue;
        };

        std::unordered_map<BasicBlock*, TensorLoadBlockState> tensorLoadStates;

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            if (tensorLoadStates.find(bb) == tensorLoadStates.end()) {
                tensorLoadStates.emplace(bb, TensorLoadBlockState());
            }

            auto& tensorLoadState = tensorLoadStates[bb];
            for (auto it = bb->begin(); it != bb->end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) {
                    continue;
                }

                if (isDSRead(*inst)) {
                    auto sources = collectSources(inst);
                    if (!sources.empty()) {
                        for (StinkyInstruction* src : sources) {
                            if (tensorLoadState.dsReadSources.find(src) ==
                                tensorLoadState.dsReadSources.end()) {
                                tensorLoadState.dsReadSources.emplace(src, -1);
                            }
                        }
                    }
                }
            }

            // propagate to this block from predecessors
            for (BasicBlock* pred : bb->getPredecessors()) {
                if (pred == bb) {
                    continue;
                }
                auto& predTensorLoadState = tensorLoadStates[pred];
                for (auto& [inst, count] : predTensorLoadState.dsReadSources) {
                    tensorLoadState.dsReadSources[inst] = count;
                }
            }
        }

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) {
                continue;
            }

            auto& currentTensorLoadState = tensorLoadStates[bb];
            auto& currentDsReadSources = currentTensorLoadState.dsReadSources;
            for (auto& [inst, _] : currentDsReadSources) {
                std::vector<int> counts;
                for (BasicBlock* pred : bb->getPredecessors()) {
                    if (pred == bb) {
                        continue;
                    }
                    auto& predTensorLoadState = tensorLoadStates[pred];
                    if (predTensorLoadState.dsReadSources.find(inst) !=
                        predTensorLoadState.dsReadSources.end()) {
                        counts.push_back(predTensorLoadState.dsReadSources[inst]);
                    }
                }

                if (!counts.empty()) {
                    currentDsReadSources[inst] = *std::min_element(counts.begin(), counts.end());
                }
            }

            // propagate to this block from predecessors
            auto& tensorLoadQueue = currentTensorLoadState.tensorLoadQueue;
            for (BasicBlock* pred : bb->getPredecessors()) {
                if (pred == bb) {
                    continue;
                }
                auto& predTensorLoadQueue = tensorLoadStates[pred].tensorLoadQueue;
                for (auto* inst : predTensorLoadQueue) {
                    auto it = std::find(tensorLoadQueue.begin(), tensorLoadQueue.end(), inst);
                    if (it == tensorLoadQueue.end()) {
                        tensorLoadQueue.push_back(inst);
                    }
                }
            }

            WaitInsertionList tensorWaits;
            std::unordered_set<StinkyInstruction*> visited;
            for (auto it = bb->begin(); it != bb->end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr || inst->getUnifiedOpcode() == GFX::PHI) {
                    continue;
                }

                visited.emplace(inst);

                if (isTensorLoad(*inst)) {
                    tensorLoadQueue.push_back(inst);
                }

                if (isDSRead(*inst)) {
                    auto sources = collectSources(inst);
                    if (!sources.empty()) {
                        for (StinkyInstruction* src : sources) {
                            // The sources is delay defined in current block, skip it.
                            if (src->getParent() == bb && visited.find(src) == visited.end()) {
                                continue;
                            }

                            if (currentDsReadSources[src] == -1) {
                                currentDsReadSources[src] += 1;
                                if (!tensorLoadQueue.empty()) {
                                    tensorLoadQueue.clear();
                                    // insert tensor waitcnt
                                    tensorWaits.emplace_back(
                                        inst, WaitCntInstruction(WaitCntInstruction::kUnused,
                                                                 WaitCntInstruction::kUnused, 0));
                                }
                            }
                        }
                    }
                }
            }

            insertWaitsForBlock(*bb, arch, tensorWaits);
        }
    }

    /// Remove consecutive PHIs at block entry.
    void removePHIs(Function& func, PassContext& passCtx, const std::vector<BasicBlock*>& rpo) {
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
std::unique_ptr<Pass> createStinkyWaitCntInsertionPass(bool insertTensorWaitCnt) {
    return std::make_unique<StinkyWaitCntInsertionPass>(insertTensorWaitCnt);
}
}  // namespace stinkytofu
