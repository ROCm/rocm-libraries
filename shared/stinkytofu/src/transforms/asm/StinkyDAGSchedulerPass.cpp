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
#include "dag/CDNA3.hpp"
#include "dag/CDNA5.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"

#define DEBUG_TYPE "StinkyDAGSchedulerPass"

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
    static void scheduleRegionWithMovableSideEffects(IRList::iterator                 regionStart,
                                                     IRList::iterator                 regionEnd,
                                                     std::vector<StinkyInstruction*>& scheduled,
                                                     ReadyQueue&                      readyQueue)
    {
        if(regionStart == regionEnd)
        {
            return; // Empty region, nothing to schedule.
        }

        PASS_DEBUG(std::cerr << "Scheduling region with movable side effects:\n");
        PASS_DEBUG(for(IRList::iterator it = regionStart; it != regionEnd; ++it) {
            StinkyInstruction& inst = getStinkyInst(it);
            inst.dump(std::cerr, true, "        ");
        });
        PASS_DEBUG(std::cerr << "\n");

        unsigned regionSize = std::distance(regionStart, regionEnd);

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

        PASS_DEBUG(dumpDAGGraph(dagGraph, dagNodes));

        readyQueue.onInitRegion(regionStart, regionEnd);

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
        while(!readyQueue.empty())
        {
            // Pop the last instruction from the ready queue.
            DAGNode* currentNode = readyQueue.pickOne();

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
    static void scheduleInDAG(BasicBlock& bb, ReadyQueue& readyQueue)
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
                scheduleRegionWithMovableSideEffects(regionStart, it, scheduled, readyQueue);

                scheduled.push_back(&inst);

                PASS_DEBUG(std::cerr << "Scheduling non-movable side-effect instruction:\n";
                           inst.dump(std::cerr, true, "        ");
                           std::cerr << "\n");

                // Start a new region after the side-effect instruction.
                regionStart = std::next(it);
            }
        }
        // Flush the last region if it has not been flushed yet.
        scheduleRegionWithMovableSideEffects(regionStart, endIt, scheduled, readyQueue);

        assert(scheduled.size() == bb.size()
               && "Scheduled instructions size must match original instructions size");

        // Now we have a scheduled list of instructions.
        // Reorder the block to reflect the scheduling (move each to end in order).
        for(StinkyInstruction* inst : scheduled)
        {
            bb.removeIR(inst);
            bb.appendIR(inst);
        }
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

        void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx)
        {
            std::unique_ptr<ReadyQueue> readyQueue = chooseReadyQueue(passCtx);
            scheduleInDAG(bb, *readyQueue);
        }

        void run(Function& func, PassContext& passCtx) override
        {
            // Traverse BasicBlocks in control flow order (reverse post-order)
            traverseCFGInRPO(func, [&](BasicBlock* bb) {
                if(passCtx.shouldProcessBasicBlock(*bb))
                    runOnBasicBlock(*bb, passCtx);
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
