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
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
    using namespace stinkytofu;

    /// Collect all register sources of @p inst, flattening PHI operands recursively. Non-PHI
    /// operands are listed in order; each PHI is replaced by its flattened incoming values.
    /// @p seenPhi avoids infinite recursion on cyclic PHI webs (defensive).
    static void collectSourcesRec(StinkyInstruction*                      inst,
                                  std::unordered_set<StinkyInstruction*>& seenPhi,
                                  std::vector<StinkyInstruction*>&        out)
    {
        if(inst == nullptr)
        {
            return;
        }

        if(inst->getUnifiedOpcode() == GFX::PHI)
        {
            if(!seenPhi.insert(inst).second)
            {
                return;
            }
        }

        for(StinkyInstruction* src : inst->getSources())
        {
            if(src == nullptr)
            {
                continue;
            }
            if(src->getUnifiedOpcode() == GFX::PHI)
            {
                collectSourcesRec(src, seenPhi, out);
            }
            else
            {
                out.push_back(src);
            }
        }
    }

    /// Collect all sources of the instruction. If a source is a PHI, collect all its sources recursively.
    std::vector<StinkyInstruction*> collectSources(StinkyInstruction* inst)
    {
        std::vector<StinkyInstruction*>        sources;
        std::unordered_set<StinkyInstruction*> seenPhi;
        if(inst != nullptr)
        {
            collectSourcesRec(inst, seenPhi, sources);
        }
        return sources;
    }

    bool isDSMemoryOp(const StinkyInstruction& inst)
    {
        return isDSRead(inst) || isDSWrite(inst);
    }

    bool isBufferMemoryOp(const StinkyInstruction& inst)
    {
        return isGlobalMemLoad(inst) || isGlobalMemStore(inst);
    }

    bool isTensorMemoryOp(const StinkyInstruction& inst)
    {
        return isTensorLoad(inst);
    }

    bool isBarrierOp(const StinkyInstruction& inst)
    {
        return isBarrier(inst);
    }

    struct MemoryOperationState
    {
        /// Queue of DS read/write instructions in program order.
        std::deque<StinkyInstruction*> dsQueue;

        /// Queue of buffer read/write instructions in program order.
        std::deque<StinkyInstruction*> bufferQueue;

        int getDSCountToComplete() const
        {
            return static_cast<int>(dsQueue.size());
        }

        int getBufferCountToComplete() const
        {
            return static_cast<int>(bufferQueue.size());
        }

        void clear()
        {
            dsQueue.clear();
            bufferQueue.clear();
        }

        /// If @p inst is a DS op, append to @c dsQueue.
        bool recordDSOperation(StinkyInstruction* inst)
        {
            if(inst == nullptr || !isDSMemoryOp(*inst))
            {
                return false;
            }

            dsQueue.push_back(inst);
            return true;
        }

        /// If @p inst is a buffer op, append to @c bufferQueue.
        bool recordBufferOperation(StinkyInstruction* inst)
        {
            if(inst == nullptr || !isBufferMemoryOp(*inst))
            {
                return false;
            }

            bufferQueue.push_back(inst);
            return true;
        }

        /// If @p inst is a barrier, clear the DS queue (modeled semantics).
        bool applyBarrierIfPresent(StinkyInstruction* inst)
        {
            if(inst == nullptr || !isBarrierOp(*inst))
            {
                return false;
            }

            dsQueue.clear();
            return true;
        }

        int getDSCountToCompleteIncluding(StinkyInstruction* inst) const
        {
            auto it = std::find(dsQueue.begin(), dsQueue.end(), inst);
            if(it != dsQueue.end())
            {
                return static_cast<int>(std::distance(it, dsQueue.end()));
            }
            return 0;
        }

        int getBufferCountToCompleteIncluding(StinkyInstruction* inst) const
        {
            auto it = std::find(bufferQueue.begin(), bufferQueue.end(), inst);
            if(it != bufferQueue.end())
            {
                return static_cast<int>(std::distance(it, bufferQueue.end()));
            }
            return 0;
        }
    };

    struct WaitCntInstruction
    {
        /// Sentinel: this counter is not used for this wait record.
        static constexpr int kUnused = -1;

        /// Immediate for @c s_wait_dscnt (dlcnt); @c kUnused if omitted.
        int dsCount;
        /// Immediate for @c s_wait_loadcnt (vlcnt); @c kUnused if omitted.
        int bufferCount;
        /// Immediate for @c s_wait_tensorcnt (tlcnt); @c kUnused if omitted.
        int tensorCount;
        WaitCntInstruction(int dsCount     = kUnused,
                           int bufferCount = kUnused,
                           int tensorCount = kUnused)
            : dsCount(dsCount)
            , bufferCount(bufferCount)
            , tensorCount(tensorCount)
        {
        }

        bool isValid() const
        {
            return dsCount != kUnused || bufferCount != kUnused || tensorCount != kUnused;
        }
    };

    std::unordered_set<StinkyInstruction*> getMemoryOperationSources(StinkyInstruction* inst)
    {
        std::unordered_set<StinkyInstruction*> srcs;
        if(inst == nullptr)
        {
            return srcs;
        }

        for(StinkyInstruction* src : inst->getSources())
        {
            if(src == nullptr)
            {
                continue;
            }

            if(src->getUnifiedOpcode() == GFX::PHI)
            {
                for(StinkyInstruction* srcPhi : src->getSources())
                {
                    if(srcPhi == nullptr)
                    {
                        continue;
                    }
                    if(isDSMemoryOp(*srcPhi) || isBufferMemoryOp(*srcPhi))
                    {
                        srcs.emplace(srcPhi);
                    }
                }
            }
            else
            {
                if(isDSMemoryOp(*src) || isBufferMemoryOp(*src))
                {
                    srcs.emplace(src);
                }
            }
        }

        return srcs;
    }

    class StinkyWaitCntInsertionPass : public StinkyInstPass
    {
    public:
        /// List of instructions to insert waitcnts for and the corresponding waitcnt type
        using WaitInsertionList = std::vector<std::pair<StinkyInstruction*, WaitCntInstruction>>;

    public:
        static char ID;

        const char* getName() const override
        {
            return "StinkyWaitCntInsertionPass";
        }

        PassID getPassID() const override
        {
            return &StinkyWaitCntInsertionPass::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);

            buildUseDefChain(func, false);
            rebuildExitMemoryStateForProcessedBlocks(func, passCtx);

            traverseCFGInRPO(func, [&](BasicBlock* bb) {
                if(!passCtx.shouldProcessBasicBlock(*bb))
                {
                    return;
                }
                WaitInsertionList waits = collectWaitsForBlock(*bb);
                insertWaitsForBlock(*bb, arch, waits);
            });

            // Handle tensor waits for DS reads.
            handleTensorWaits(func, arch, passCtx);

            removePHIs(func, passCtx);
        }

    private:
        /// Exit-state queues for each processed block (full block walk, program order).
        std::unordered_map<BasicBlock*, MemoryOperationState> blockStates;

        void rebuildExitMemoryStateForProcessedBlocks(Function& func, PassContext& passCtx)
        {
            for(BasicBlock& bb : func)
            {
                if(passCtx.shouldProcessBasicBlock(bb))
                {
                    blockStates.emplace(&bb, MemoryOperationState());
                    trackMemoryOperationsInBlock(bb);
                }
            }
        }

        void trackMemoryOperationsInBlock(BasicBlock& bb)
        {
            MemoryOperationState& state = blockStates[&bb];
            state.clear();
            for(IRBase& ir : bb)
            {
                StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir);
                if(inst == nullptr)
                {
                    continue;
                }

                state.recordDSOperation(inst);
                state.recordBufferOperation(inst);
                state.applyBarrierIfPresent(inst);
            }
        }

        /// Simulate the block in order; return (anchor instruction, wait spec) pairs in order.
        WaitInsertionList collectWaitsForBlock(BasicBlock& bb)
        {
            MemoryOperationState currentBlockState;
            WaitInsertionList    instructionsNeedWait;

            for(auto it = bb.begin(); it != bb.end(); ++it)
            {
                StinkyInstruction* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if(inst == nullptr || inst->getUnifiedOpcode() == GFX::PHI)
                {
                    continue;
                }

                currentBlockState.recordDSOperation(inst);
                currentBlockState.recordBufferOperation(inst);

                if(currentBlockState.getDSCountToComplete() > 0
                   && currentBlockState.applyBarrierIfPresent(inst))
                {
                    instructionsNeedWait.emplace_back(
                        inst, WaitCntInstruction(0, WaitCntInstruction::kUnused));
                    continue;
                }

                auto srcs = getMemoryOperationSources(inst);
                if(srcs.empty())
                {
                    continue;
                }

                int dsCountBeforeInst     = currentBlockState.getDSCountToComplete();
                int bufferCountBeforeInst = currentBlockState.getBufferCountToComplete();
                std::vector<int> dsCountFromPredecessor;
                std::vector<int> bufferCountFromPredecessor;
                bool             needDSWait     = false;
                bool             needBufferWait = false;

                for(StinkyInstruction* src : srcs)
                {
                    int dsCount = currentBlockState.getDSCountToCompleteIncluding(src);
                    if(dsCount > 0)
                    {
                        dsCountBeforeInst = std::min(dsCountBeforeInst, dsCount - 1);
                        needDSWait        = true;
                    }
                    else
                    {
                        // Unprocessed predecessor blocks insert an empty state via operator[].
                        dsCount = blockStates[src->getParent()].getDSCountToCompleteIncluding(src);
                        if(dsCount > 0)
                        {
                            dsCountFromPredecessor.push_back(dsCount - 1);
                            needDSWait = true;
                        }
                    }

                    int bufferCount = currentBlockState.getBufferCountToCompleteIncluding(src);
                    if(bufferCount > 0)
                    {
                        bufferCountBeforeInst = std::min(bufferCountBeforeInst, bufferCount - 1);
                        needBufferWait        = true;
                    }
                    else
                    {
                        bufferCount
                            = blockStates[src->getParent()].getBufferCountToCompleteIncluding(src);
                        if(bufferCount > 0)
                        {
                            bufferCountFromPredecessor.push_back(bufferCount - 1);
                            needBufferWait = true;
                        }
                    }
                }

                if(needDSWait)
                {
                    if(!dsCountFromPredecessor.empty())
                    {
                        dsCountBeforeInst += *std::min_element(dsCountFromPredecessor.begin(),
                                                               dsCountFromPredecessor.end());
                    }
                    instructionsNeedWait.emplace_back(
                        inst, WaitCntInstruction(dsCountBeforeInst, WaitCntInstruction::kUnused));
                }

                if(needBufferWait)
                {
                    if(!bufferCountFromPredecessor.empty())
                    {
                        bufferCountBeforeInst += *std::min_element(
                            bufferCountFromPredecessor.begin(), bufferCountFromPredecessor.end());
                    }
                    instructionsNeedWait.emplace_back(
                        inst,
                        WaitCntInstruction(WaitCntInstruction::kUnused, bufferCountBeforeInst));
                }
            }

            return instructionsNeedWait;
        }

        /// Insert waits before each anchor. Consecutive identical immediates for the same
        /// counter are skipped (same insertion order as @p waits).
        void insertWaitsForBlock(BasicBlock& bb, GfxArchID arch, const WaitInsertionList& waits)
        {
            auto irBuilder = AsmIRBuilder(bb, arch);

            const WaitCntInstruction* prevDSWaitCntInstruction     = nullptr;
            const WaitCntInstruction* prevBufferWaitCntInstruction = nullptr;

            for(const auto& entry : waits)
            {
                StinkyInstruction*        inst               = entry.first;
                const WaitCntInstruction& waitCntInstruction = entry.second;

                if(waitCntInstruction.dsCount != WaitCntInstruction::kUnused)
                {
                    if(prevDSWaitCntInstruction != nullptr
                       && prevDSWaitCntInstruction->dsCount == waitCntInstruction.dsCount)
                    {
                        continue;
                    }
                    StinkyInstruction* waitInst
                        = irBuilder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch), inst);
                    waitInst->addSrcReg(StinkyRegister(waitCntInstruction.dsCount));

                    SWaitCntData waitCntData;
                    waitCntData.dlcnt = waitCntInstruction.dsCount;
                    waitInst->addModifier<SWaitCntData>(waitCntData);
                    prevDSWaitCntInstruction = &waitCntInstruction;
                }

                if(waitCntInstruction.bufferCount != WaitCntInstruction::kUnused)
                {
                    if(prevBufferWaitCntInstruction != nullptr
                       && prevBufferWaitCntInstruction->bufferCount
                              == waitCntInstruction.bufferCount)
                    {
                        continue;
                    }
                    StinkyInstruction* waitInst
                        = irBuilder.create(getMCIDByUOp(GFX::s_wait_loadcnt, arch), inst);
                    waitInst->addSrcReg(StinkyRegister(waitCntInstruction.bufferCount));

                    SWaitCntData waitCntData;
                    waitCntData.vlcnt = waitCntInstruction.bufferCount;
                    waitInst->addModifier<SWaitCntData>(waitCntData);
                    prevBufferWaitCntInstruction = &waitCntInstruction;
                }

                if(waitCntInstruction.tensorCount != WaitCntInstruction::kUnused)
                {
                    StinkyInstruction* waitInst
                        = irBuilder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, arch), inst);
                    waitInst->addSrcReg(StinkyRegister(waitCntInstruction.tensorCount));

                    SWaitTensorCntData waitCntData;
                    waitCntData.tlcnt = waitCntInstruction.tensorCount;
                    waitInst->addModifier<SWaitTensorCntData>(waitCntData);
                }
            }
        }

        /// Handle tensor waits for DS reads.
        void handleTensorWaits(Function& func, GfxArchID arch, PassContext& passCtx)
        {
            std::unordered_map<BasicBlock*, std::unordered_map<StinkyInstruction*, int>>
                dsReadSourcesByBlock;
            traverseCFGInRPO(func, [&](BasicBlock* bb) {
                if(!passCtx.shouldProcessBasicBlock(*bb))
                {
                    return;
                }

                if(dsReadSourcesByBlock.find(bb) == dsReadSourcesByBlock.end())
                {
                    dsReadSourcesByBlock.emplace(bb, std::unordered_map<StinkyInstruction*, int>());
                }

                auto& dsReadSources = dsReadSourcesByBlock[bb];
                for(auto it = bb->begin(); it != bb->end(); ++it)
                {
                    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if(inst == nullptr)
                    {
                        continue;
                    }

                    if(isDSRead(*inst))
                    {
                        auto sources = collectSources(inst);
                        if(!sources.empty())
                        {
                            for(StinkyInstruction* src : sources)
                            {
                                if(dsReadSources.find(src) == dsReadSources.end())
                                {
                                    dsReadSources.emplace(src, -1);
                                }
                            }
                        }
                    }
                }

                // propagate to this block from predecessors
                for(BasicBlock* pred : bb->getPredecessors())
                {
                    if(pred == bb)
                    {
                        continue;
                    }
                    auto& predDsReadSources = dsReadSourcesByBlock[pred];
                    for(auto& [inst, count] : predDsReadSources)
                    {
                        dsReadSources[inst] = count;
                    }
                }
            });

            std::unordered_map<BasicBlock*, std::deque<StinkyInstruction*>> tensorLoadBlockQueue;
            traverseCFGInRPO(func, [&](BasicBlock* bb) {
                if(!passCtx.shouldProcessBasicBlock(*bb))
                {
                    return;
                }

                auto& currentDsReadSources = dsReadSourcesByBlock[bb];
                for(auto& [inst, _] : currentDsReadSources)
                {
                    std::vector<int> counts;
                    for(BasicBlock* pred : bb->getPredecessors())
                    {
                        if(pred == bb)
                        {
                            continue;
                        }
                        auto& predDsReadSources = dsReadSourcesByBlock[pred];
                        if(predDsReadSources.find(inst) != predDsReadSources.end())
                        {
                            counts.push_back(predDsReadSources[inst]);
                        }
                    }

                    if(!counts.empty())
                    {
                        currentDsReadSources[inst]
                            = *std::min_element(counts.begin(), counts.end());
                    }
                }

                if(tensorLoadBlockQueue.find(bb) == tensorLoadBlockQueue.end())
                {
                    tensorLoadBlockQueue.emplace(bb, std::deque<StinkyInstruction*>());
                }

                // propagate to this block from predecessors
                auto& tensorLoadQueue = tensorLoadBlockQueue[bb];
                for(BasicBlock* pred : bb->getPredecessors())
                {
                    if(pred == bb)
                    {
                        continue;
                    }
                    auto& predTensorLoadQueue = tensorLoadBlockQueue[pred];
                    for(StinkyInstruction* inst : predTensorLoadQueue)
                    {
                        if(std::find(tensorLoadQueue.begin(), tensorLoadQueue.end(), inst)
                           == tensorLoadQueue.end())
                        {
                            tensorLoadQueue.push_back(inst);
                        }
                    }
                }

                WaitInsertionList                      tensorWaits;
                std::unordered_set<StinkyInstruction*> visited;

                for(auto it = bb->begin(); it != bb->end(); ++it)
                {
                    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if(inst == nullptr)
                    {
                        continue;
                    }

                    visited.emplace(inst);
                    if(isTensorLoad(*inst))
                    {
                        tensorLoadQueue.push_back(inst);
                    }

                    if(isDSRead(*inst))
                    {
                        auto sources = collectSources(inst);
                        if(!sources.empty())
                        {
                            for(StinkyInstruction* src : sources)
                            {
                                // The sources is delay defined in current block, skip it.
                                if(src->getParent() == bb && visited.find(src) == visited.end())
                                {
                                    continue;
                                }

                                if(currentDsReadSources[src] == -1)
                                {
                                    currentDsReadSources[src] += 1;
                                    tensorWaits.emplace_back(
                                        inst,
                                        WaitCntInstruction(WaitCntInstruction::kUnused,
                                                           WaitCntInstruction::kUnused,
                                                           0));
                                }
                            }
                        }
                    }
                }

                insertWaitsForBlock(*bb, arch, tensorWaits);
            });
        }

        /// Remove consecutive PHIs at block entry.
        void removePHIs(Function& func, PassContext& passCtx)
        {
            traverseCFGInRPO(func, [&](BasicBlock* bb) {
                if(!passCtx.shouldProcessBasicBlock(*bb))
                {
                    return;
                }
                while(!bb->empty())
                {
                    IRBase& ir = *bb->begin();
                    if(ir.getType() != IRBase::IRType::StinkyTofu)
                    {
                        break;
                    }
                    auto* inst = cast<StinkyInstruction>(&ir);
                    if(inst->getUnifiedOpcode() != GFX::PHI)
                    {
                        break;
                    }
                    ir.erase();
                }
            });
        }
    };

    char StinkyWaitCntInsertionPass::ID = 0;
} // namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createStinkyWaitCntInsertionPass()
    {
        return std::make_unique<StinkyWaitCntInsertionPass>();
    }
}
