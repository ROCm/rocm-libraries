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
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

// Before dag/CDNA*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "StinkyDAGSchedulerPass"

#include "dag/CDNA3.hpp"
#include "dag/CDNA5.hpp"

namespace
{
    using namespace stinkytofu;

    // Check if instruction is a movable side effect (like s_barrier)
    static bool isMovableSideEffect(const StinkyInstruction& inst)
    {
        // This is a barrier and has manually defined dependencies.
        return isBarrier(inst) && !inst.getDestRegs().empty();
    }

    // --- Region scheduler (does NOT move fences) ---
    //
    // Build a DAG within a region and perform a stable topological schedule.
    // Adds RAW/WAR/WAW deps for physical regs and also respects explicitPreds
    // (only when both endpoints are inside the region).
    static void scheduleRegionWithMovableSideEffects(
        IRList::iterator                                       regionStart,
        IRList::iterator                                       regionEnd,
        IRList::iterator                                       blockBegin,
        std::vector<StinkyInstruction*>&                        scheduled,
        ReadyQueue&                                             readyQueue,
        const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex)
    {
        if(regionStart == regionEnd)
        {
            return; // Empty region, nothing to schedule.
        }

        PASS_DEBUG(std::cerr << "Scheduling region with movable side effects:\n");
        PASS_DEBUG(for(IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            inst.dump(std::cerr);
        });
        PASS_DEBUG(std::cerr << "\n");

        unsigned regionSize = std::distance(regionStart, regionEnd);

        std::string regionBbLabel;
        if(regionStart != regionEnd)
        {
            if(BasicBlock* pbb = getStinkyInst(regionStart).getParent())
                regionBbLabel = pbb->getLabel();
        }

        // Map each instruction to an unique id [0..n-1]
        DAGNodeList dagNodes;
        dagNodes.reserve(regionSize);

        unsigned id = 0;
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            dagNodes.emplace_back(&getStinkyInst(it), id++);
        }

        // Graph
        std::vector<std::unordered_set<unsigned>> dagGraph(regionSize);

        // Track last read/write per physreg inside the region
        /* To ensure correct node dependency, lastRead should track all
         * previous read nodes until the register is overwritten. */
        std::map<StinkyRegister, std::unordered_set<DAGNode*>> lastRead;
        std::map<StinkyRegister, DAGNode*>                     lastWrite;

        // Build deps graph - same as before for register dependencies
        for(unsigned i = 0; i < dagNodes.size(); ++i)
        {
            DAGNode&           dagNode = dagNodes[i];
            StinkyInstruction& inst    = *dagNode.inst;

            // RAW deps:
            // For each source register, add an edge to the last writer of that register.
            for(const StinkyRegister& srcReg : inst.getSrcRegs())
            {
                if(!srcReg.isRegister())
                    continue;

                for(unsigned off = 0; off < srcReg.reg.num; ++off)
                {
                    StinkyRegister reg(srcReg.reg.type, srcReg.reg.idx + off, 1);
                    auto           itLastWrite = lastWrite.find(reg);
                    // Only add edge if the last writer is in the region.
                    if(itLastWrite != lastWrite.end())
                    {
                        DAGNode* lastWriter = itLastWrite->second;
                        addEdgeById(lastWriter, &dagNode, dagGraph);
                    }
                    // Add node to track the last read of this register
                    lastRead[reg].insert(&dagNode);
                }
            }

            // WAW/WAR deps for defs
            for(const StinkyRegister& dstReg : inst.getDestRegs())
            {
                if(!dstReg.isRegister())
                    continue;

                for(unsigned off = 0; off < dstReg.reg.num; ++off)
                {
                    StinkyRegister reg(dstReg.reg.type, dstReg.reg.idx + off, 1);

                    // WAW: previous writer of reg must come before this writer
                    auto itLastWrite = lastWrite.find(reg);

                    // Only add edge if the last writer is in the region.
                    if(itLastWrite != lastWrite.end())
                    {
                        DAGNode* lastWriter = itLastWrite->second;
                        addEdgeById(lastWriter, &dagNode, dagGraph);
                    }

                    // WAR: previous reader of r must come before this writer
                    auto itLastRead = lastRead.find(reg);

                    // Only add edge if the last reader is in the region.
                    if(itLastRead != lastRead.end())
                    {
                        for(DAGNode* lastReader : itLastRead->second)
                        {
                            addEdgeById(lastReader, &dagNode, dagGraph);
                        }
                        // Clear last read tracking for this register due to it's overwritten
                        lastRead.erase(reg);
                    }

                    // track the last write for this register
                    lastWrite[reg] = &dagNode;
                }
            }
        }

        // Annotate ds_read nodes with WMMA affinity using cross-BB def-use chains.
        // Walks through PHI nodes transitively to find the first real WMMA consumer.
        for(unsigned i = 0; i < regionSize; ++i)
        {
            if(isDSRead(*dagNodes[i].inst))
            {
                // BFS through users, skipping PHI nodes, to find WMMA consumers.
                std::vector<StinkyInstruction*> worklist(
                    dagNodes[i].inst->getUsers().begin(),
                    dagNodes[i].inst->getUsers().end());
                std::unordered_set<StinkyInstruction*> visited;
                while(!worklist.empty())
                {
                    StinkyInstruction* user = worklist.back();
                    worklist.pop_back();
                    if(!visited.insert(user).second)
                        continue;
                    if(user->getUnifiedOpcode() == GFX::PHI)
                    {
                        // Walk through PHI to its real users.
                        for(StinkyInstruction* phiUser : user->getUsers())
                            worklist.push_back(phiUser);
                        continue;
                    }
                    auto it = wmmaIndex.find(user);
                    if(it != wmmaIndex.end())
                        dagNodes[i].wmmaAffinity
                            = std::min(dagNodes[i].wmmaAffinity, it->second);
                }
            }
        }

        PASS_DEBUG(dumpDAGGraph(dagGraph, dagNodes));

        readyQueue.onInitRegion(regionStart, regionEnd, blockBegin);

        // Kahn's algorithm with stable pick (by original order)

        assert(readyQueue.empty() && "Ready queue must be empty before scheduling a region");

        // Initialize the ready queue with instructions that have in-degree 0.
        for(unsigned i = 0; i < regionSize; ++i)
        {
            if(dagNodes[i].inDegree == 0)
            {
                readyQueue.push(&dagNodes[i]);
            }
        }

        // Process the ready queue until it's empty.
        unsigned orderInRegion = 0;
        while(!readyQueue.empty())
        {
            // Pop the last instruction from the ready queue.
            DAGNode* currentNode = readyQueue.pickOne();
            ++orderInRegion;

            if(isBarrier(*currentNode->inst))
            {
                PASS_DEBUG(std::cerr << "[DAG schedule] bb=\"" << regionBbLabel
                                     << "\" orderInRegion=" << orderInRegion << " dagId=" << currentNode->id
                                     << " movable barrier (position in region schedule)\n";
                           currentNode->inst->dump(std::cerr);
                           std::cerr << "\n");
            }

            // Add the instruction to the scheduled list.
            scheduled.push_back(currentNode->inst);

            // Process all successors of the current node.
            for(unsigned succId : dagGraph[currentNode->id])
            {
                DAGNode& succNode = dagNodes[succId];
                succNode.inDegree--;

                // If the successor now has in-degree 0, add it to the ready queue.
                if(succNode.inDegree == 0)
                {
                    readyQueue.push(&succNode);
                }
            }
        }
    }

    static bool hasSideEffect(const StinkyInstruction& inst)
    {
        if(
            // TODO: provide a configurable way to ignore certain instructions,
            //       e.g. LocalWriteInstruction
            //
            // dynamic_cast<const LocalWriteInstruction*>(op) ||
            //
            isGlobalMemStore(inst) || isBranch(inst) || isBarrier(inst) || isWaitCnt(inst)
            || isHasSideEffect(inst))
        {
            return true;
        }
        return false;
    }

    // Schedule the instructions in the given IRList.
    // This will split the instructions into regions based on side-effect instructions
    // and schedule each region in a DAG.
    //
    // In the end, the instructions will be reordered in the block
    // to reflect the scheduling order.
    static void scheduleInDAG(BasicBlock&                                             bb,
                              ReadyQueue&                                             readyQueue,
                              const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex)
    {
        PASS_DEBUG(std::cerr << "*** Scheduling Instructions in DAG: ***\n");

        if(bb.empty())
            return;

        std::vector<StinkyInstruction*> scheduled;
        scheduled.reserve(bb.size());

        BasicBlock::iterator beginIt = bb.begin();
        BasicBlock::iterator endIt   = bb.end();

        readyQueue.onInit(beginIt, endIt);

        BasicBlock::iterator regionStart = beginIt;

        for(BasicBlock::iterator it = beginIt; it != endIt; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            // Only break regions on non-movable side effects
            if(hasSideEffect(inst) && !isMovableSideEffect(inst))
            {
                scheduleRegionWithMovableSideEffects(
                    regionStart, it, beginIt, scheduled, readyQueue, wmmaIndex);

                scheduled.push_back(&inst);

                PASS_DEBUG(std::cerr << "Scheduling non-movable side-effect instruction:\n";
                           inst.dump(std::cerr);
                           std::cerr << "\n");

                // Start a new region after the side-effect instruction.
                regionStart = std::next(it);
            }
        }
        // Flush the last region if it has not been flushed yet.
        scheduleRegionWithMovableSideEffects(
            regionStart, endIt, beginIt, scheduled, readyQueue, wmmaIndex);

        assert(scheduled.size() == bb.size()
               && "Scheduled instructions size must match original instructions size");

        // Now we have a scheduled list of instructions.
        // Reorder the block to reflect the scheduling (move each to end in order).
        for(StinkyInstruction* inst : scheduled)
        {
            bb.removeIR(inst);
            bb.appendIR(inst);
        }

        readyQueue.onFinishBB();
    }

    std::unique_ptr<ReadyQueue> chooseReadyQueue(const PassContext& passCtx)
    {
        if(passCtx.getGemmTileConfig().arch[0] == 12 && passCtx.getGemmTileConfig().arch[1] == 5)
        {
            PASS_DEBUG(std::cerr << "Using CDNA5ReadyQueue for scheduling\n");
            return std::make_unique<CDNA5ReadyQueue>(passCtx);
        }
        else if(passCtx.getGemmTileConfig().arch[0] >= 9)
        {
            PASS_DEBUG(std::cerr << "Using CDNA3ReadyQueue for scheduling\n");
            return std::make_unique<CDNA3ReadyQueue>(passCtx);
        }
        else
        {
            PASS_DEBUG(std::cerr << "Using Default ReadyQueue for scheduling\n");
            return std::make_unique<ReadyQueueByDAGid>(passCtx);
        }
    }

    class StinkyDAGSchedulerPass : public StinkyInstPass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "StinkyDAGSchedulerPass";
        }

        PassID getPassID() const override
        {
            return &StinkyDAGSchedulerPass::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            // Build def-use chains so we can look up cross-BB WMMA consumers
            // of ds_reads for wmmaAffinity annotation.
            buildUseDefChain(func, true);

            // Pre-assign a function-wide index to each WMMA/SWMMA so wmmaAffinity
            // values are comparable across scheduling regions.
            std::unordered_map<StinkyInstruction*, unsigned> wmmaIndex;
            {
                unsigned idx = 0;
                traverseCFGInRPO(func, [&](BasicBlock* bb) {
                    for(auto it = bb->begin(); it != bb->end(); ++it)
                    {
                        StinkyInstruction& inst = getStinkyInst(it);
                        if(isWMMA(inst) || isSWMMA(inst))
                            wmmaIndex[&inst] = idx++;
                    }
                });
            }

            auto loops = detectLoops(func);

            PASS_DEBUG(
                for(const Loop& loop : loops) {
                    std::cerr << "[LoopDetection] Loop: header="
                              << (loop.headerBB ? loop.headerBB->getLabel() : "?")
                              << " latch="
                              << (loop.latchBB ? loop.latchBB->getLabel() : "?") << "\n";
                    for(BasicBlock* bb : loop.bodyBBs)
                    {
                        std::cerr << "  body: " << bb->getLabel() << " ->";
                        for(BasicBlock* succ : bb->getSuccessors())
                            std::cerr << " " << succ->getLabel();
                        std::cerr << "\n";
                    }
                }
            );

            // Cross-BB scheduling state shared across all BBs.
            // Written by all BBs in onFinishBB, read only by loop body BBs in onInit.
            ScheduleAnalysisCache analysisCache;

            // Per-loop ReadyQueue: shared across loop body BBs for loop-specific
            // scheduling state (wmmaNodeCounters, evenly-split config).
            std::map<const Loop*, std::unique_ptr<ReadyQueue>> loopQueues;

            // Map only loop body BBs to their loop — shared queue for loop iterations.
            std::unordered_map<BasicBlock*, const Loop*> bbToLoop;
            for(const Loop& loop : loops)
            {
                for(BasicBlock* bb : loop.bodyBBs)
                    bbToLoop[bb] = &loop;
            }

            traverseCFGInRPO(func, [&](BasicBlock* bb) {
                if(!passCtx.shouldProcessBasicBlock(*bb))
                    return;

                auto it = bbToLoop.find(bb);
                if(it != bbToLoop.end())
                {
                    const Loop* loop = it->second;
                    auto&       rq   = loopQueues[loop];
                    if(!rq)
                    {
                        rq = chooseReadyQueue(passCtx);
                        rq->setLoopContext(loop);
                    }
                    rq->setAnalysisCache(&analysisCache);
                    scheduleInDAG(*bb, *rq, wmmaIndex);
                }
                else
                {
                    auto rq = chooseReadyQueue(passCtx);
                    rq->setAnalysisCache(&analysisCache);
                    scheduleInDAG(*bb, *rq, wmmaIndex);
                }
            });
        }
    };

    char StinkyDAGSchedulerPass::ID = 0;
}

namespace stinkytofu
{
    std::unique_ptr<Pass> createStinkyDAGSchedulerPass()
    {
        return std::make_unique<StinkyDAGSchedulerPass>();
    }
}
