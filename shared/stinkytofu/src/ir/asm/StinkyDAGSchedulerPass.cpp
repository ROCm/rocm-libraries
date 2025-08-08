/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <cmath>
#include <iostream> // TODO: don't use iostream.
#include <map> // FIXME: Use unordered_map if StinkyRegister::regType is not std::string
#include <queue>

#include "ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "StinkyDAGSchedulerPass"

namespace
{
    using namespace stinkytofu;

    // Build a use-def chain for the instructions in the given IRList.
    //
    // This will link each instruction's sources to their most recent definitions
    // and each instruction's users to the instructions that use its results.
    //
    // It assumes the instructions are in top-down order.
    //
    // The use-def chain is built based on the source and destination registers of each instruction.
    // It also handles the case where multiple consecutive registers are used (e.g., regIdx 0, 1, 2, 3).
    //
    // The use-def chain is stored in the `sources` and `users` vectors of each StinkyInstruction.
    //   * `sources` contains the instructions that define the registers used by this instruction,
    //   * `users` contains the instructions that use the results of this instruction.
    static void buildUseDefChain(IRList& insts)
    {
        struct RegisterKey
        {
            std::string_view type;
            unsigned         regIdx;
            bool             operator==(const RegisterKey& o) const noexcept
            {
                return regIdx == o.regIdx && type == o.type;
            }
        };

        struct RegisterKeyHash
        {
            size_t operator()(const RegisterKey& k) const noexcept
            {
                size_t h1 = std::hash<std::string_view>{}(k.type);
                size_t h2 = std::hash<unsigned>{}(k.regIdx);
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };

        std::unordered_map<RegisterKey, StinkyInstruction*, RegisterKeyHash> lastDef;

        // Build use-def chains for each instruction in top-down order.
        for(IRBase& ir : insts)
        {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(ir);
            // Link uses (sources) to their most recent defs.
            if(inst.srcRegs.size() > 0)
            {
                std::unordered_set<StinkyInstruction*> added;
                for(StinkyRegister& reg : inst.srcRegs)
                {
                    if(!reg.isRegister())
                        continue;

                    const std::string_view t = reg.regType;

                    // TODO: Currently we assume regNum <= 4 (DWords) consecutive registers.
                    //       So it is acceptable to iterate over them.
                    //       If regNum > 4, maybe we want to use a different approach.
                    for(int off = 0; off < reg.regNum; ++off)
                    {
                        auto itDef = lastDef.find(RegisterKey{t, reg.regIdx + off});
                        if(itDef != lastDef.end())
                        {
                            StinkyInstruction* def = itDef->second;
                            if(added.insert(def).second)
                            {
                                def->users.push_back(&inst);
                                inst.sources.push_back(def);
                            }
                        }
                    }
                }
            }

            // Record current def (destination) as the latest writer for its lanes.
            for(StinkyRegister& reg : inst.destRegs)
            {
                const std::string_view t = reg.regType;
                for(int off = 0; off < reg.regNum; ++off)
                {
                    // Update the last definition for this register.
                    lastDef[RegisterKey{t, reg.regIdx + off}] = &inst;
                }
            }
        }
    }

    struct DAGNode
    {
        StinkyInstruction* inst;
        unsigned           inDegree;
        unsigned           id;

        DAGNode(StinkyInstruction* inst, unsigned id)
            : inst(inst)
            , inDegree(0)
            , id(id)
        {
        }
    };

    // comparator: return true if a should come *after* b.
    struct CompareByDAGid
    {
        bool operator()(const DAGNode* a, const DAGNode* b) const
        {
            return a->id > b->id; // smaller id has higher priority
        }
    };

    using DAGNodeList = std::vector<DAGNode>;

    static void dumpUseDefChain(const IRList& insts)
    {
        std::cerr << "*** Use-Def Chain Dump: ***\n";
        for(const IRBase& ir : insts)
        {
            const StinkyInstruction& inst = *cast<StinkyInstruction>(&ir);

            std::cerr << "Instruction:\n";
            inst.dump(std::cerr, true, "  ");

            for(const StinkyInstruction* src : inst.sources)
            {
                std::cerr << "    Source:\n";
                src->dump(std::cerr, true, "      ");
                std::cerr << "\n";
            }

            for(const StinkyInstruction* user : inst.users)
            {
                std::cerr << "    User:\n";
                user->dump(std::cerr, true, "      ");
                std::cerr << "\n";
            }
        }
        std::cerr << "\n\n";
    }

    static void dumpDAGGraph(const std::vector<std::unordered_set<unsigned>>& dagGraph,
                             const DAGNodeList&                               dagNodes)
    {
        std::cerr << "*** DAG Graph Dump: ***\n";
        for(unsigned i = 0; i < dagGraph.size(); ++i)
        {
            std::cerr << "Node " << i << ": ";
            dagNodes[i].inst->dump(std::cerr, false);
            std::cerr << "  successors: ";
            for(unsigned succId : dagGraph[i])
            {
                std::cerr << succId << " ";
            }
            std::cerr << "\n";
        }
        std::cerr << "\n\n";
    }

    static void
        addEdgeById(DAGNode* from, DAGNode* to, std::vector<std::unordered_set<unsigned>>& dagGraph)
    {
        // Don't add duplicate edges, or self-loops.
        if(from->id == to->id || dagGraph[from->id].count(to->id) > 0)
            return;

        // Add edge from 'from' to 'to'
        dagGraph[from->id].insert(to->id);
        to->inDegree++;
    }

    class ReadyQueue
    {
    public:
        explicit ReadyQueue(const PassContext& passCtx)
            : passCtx_(passCtx)
        {
        }

        const PassContext& getPassContext() const
        {
            return passCtx_;
        }

        virtual ~ReadyQueue() = default;

        // Pick one node from the ready queue based on some strategy.
        virtual DAGNode* pickOne() = 0;

        // Push a node into the ready queue which is ready to be scheduled
        // (i.e. all its deps are satisfied).
        virtual void push(DAGNode* node) = 0;

        virtual bool empty() const = 0;

        // Hook for derived classes to do something when the first group of instructions are ready to issue.
        virtual void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) {}

        // Hook for derived classes to do something when the first group of instructions are ready to issue.
        virtual void onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd) {}

    private:
        // reference to const PassContext (object content immutable)
        const PassContext& passCtx_;
    };

    using DAGidPriorityQueue = std::priority_queue<DAGNode*, std::vector<DAGNode*>, CompareByDAGid>;

    class ReadyQueueByDAGid : public ReadyQueue
    {
        DAGidPriorityQueue queue;

    public:
        explicit ReadyQueueByDAGid(const PassContext& passCtx)
            : ReadyQueue(passCtx)
        {
        }

        DAGNode* pickOne() override;

        void push(DAGNode* node) override
        {
            queue.push(node);
        }

        bool empty() const override
        {
            return queue.empty();
        }
    };

    DAGNode* ReadyQueueByDAGid::pickOne()
    {
        assert(!queue.empty() && "Ready queue must not be empty");
        DAGNode* node = queue.top();
        queue.pop();
        return node;
    }

    class CDNA3ReadyQueue : public ReadyQueue
    {
        struct MFMAIssueConfig
        {
            int latency               = 0; // original mfma latency
            int avgIssueInterval      = 0; // average issue interval for mfma
            int totalIssuedCycles     = 0; // total issued cycles in the region
            int totalMfmaIssuedCycles = 0; // total mfma issued cycles in the region
            int issuedCount           = 0; // total mfma issued count in the region
        };
        DAGidPriorityQueue mfmaQueue;
        DAGidPriorityQueue globalReadQueue;
        DAGidPriorityQueue otherQueue;
        DAGidPriorityQueue barrierQueue;

        bool isInit = false;

        int globalReadCounter = 0; // tracking global read count during MFMA
        int globalReadPerMFMA = 1; // global read issue count per MFMA

        int mfmaCounter = 0;
        // For mfma latency tracking per register
        std::map<int, int> mfmaRegisterLatencyCounters;

        MFMAIssueConfig mfmaIssueConfig;

        void     updateMFMALatencyCounters(DAGNode* node);
        bool     isMFMALatencyFree();
        DAGNode* pickOneFromMFMA();
        void     MFMAIssueUpdate(DAGNode* node);

    public:
        explicit CDNA3ReadyQueue(const PassContext& passCtx)
            : ReadyQueue(passCtx)
        {
        }

        DAGNode* pickOne() override;
        void     push(DAGNode* node) override;
        bool     empty() const override;

        void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) override;

        void onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd) override;
    };

    void CDNA3ReadyQueue::updateMFMALatencyCounters(DAGNode* node)
    {
        // decrement the latency counters
        for(auto& [reg, counter] : mfmaRegisterLatencyCounters)
        {
            if(counter > 0)
                counter -= node->inst->issueCycles;
        }
    }

    bool CDNA3ReadyQueue::isMFMALatencyFree()
    {
        DAGNode* node = mfmaQueue.top();
        for(const StinkyRegister& dstReg : node->inst->srcRegs)
        {
            if(!dstReg.isRegister())
                continue;

            for(unsigned off = 0; off < dstReg.regNum; ++off)
            {
                auto it = mfmaRegisterLatencyCounters.find(dstReg.regIdx + off);
                if(it != mfmaRegisterLatencyCounters.end() && it->second > 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    DAGNode* CDNA3ReadyQueue::pickOneFromMFMA()
    {
        assert(!mfmaQueue.empty() && "The MFMA queue must not be empty");
        DAGNode* node = mfmaQueue.top();
        mfmaQueue.pop();
        // Use the original latency to avoid mfma issued continuously
        auto rest
            = (int)((mfmaIssueConfig.totalIssuedCycles - mfmaIssueConfig.totalMfmaIssuedCycles)
                    / mfmaIssueConfig.issuedCount);
        if(mfmaIssueConfig.issuedCount <= 0
           || (mfmaIssueConfig.issuedCount > 0 && rest < node->inst->latencyCycles))
        {
            // interleave mfma with other instructions if possible
            // mfma inst1 mfma inst2 mfma inst3
            if(node->inst->latencyCycles >= mfmaIssueConfig.avgIssueInterval)
            {
                mfmaCounter = std::max(rest, 1);
            }
            else
            {
                mfmaCounter = node->inst->latencyCycles;
            }
        }
        else
        {
            mfmaCounter = mfmaIssueConfig.avgIssueInterval;
        }
        mfmaIssueConfig.issuedCount--;
        mfmaIssueConfig.totalMfmaIssuedCycles -= node->inst->issueCycles;
        MFMAIssueUpdate(node);
        updateMFMALatencyCounters(node);
        globalReadCounter = 0;
        return node;
    }

    void CDNA3ReadyQueue::MFMAIssueUpdate(DAGNode* node)
    {
        // Use issue for all instructions
        mfmaCounter -= node->inst->issueCycles;
        mfmaIssueConfig.totalIssuedCycles -= node->inst->issueCycles;
    }

    DAGNode* CDNA3ReadyQueue::pickOne()
    {
        // Priority 1: Try to schedule MFMA if counter allows
        // TODO: Need to check if ds_read completes are done for the MFMA
        // mfmaRegisterLatencyCounters
        if(!mfmaQueue.empty() && mfmaCounter <= 0 && isMFMALatencyFree())
        {
            return pickOneFromMFMA();
        }

        // Priority 2: Schedule other instructions
        if(!globalReadQueue.empty())
        {
            if(globalReadCounter < globalReadPerMFMA || otherQueue.empty())
            {
                DAGNode* globalRead = globalReadQueue.top();
                globalReadQueue.pop();
                MFMAIssueUpdate(globalRead);
                updateMFMALatencyCounters(globalRead);
                globalReadCounter++;
                return globalRead;
            }
        }

        if(!otherQueue.empty())
        {
            DAGNode* node = otherQueue.top();
            otherQueue.pop();
            MFMAIssueUpdate(node);
            // If ds read, add its latency to the counter to mfmaRegisterLatencyCounters
            // Ignore global read for now
            if(isDSRead(*node->inst))
            {
                for(const StinkyRegister& dstReg : node->inst->destRegs)
                {
                    if(!dstReg.isRegister())
                        continue;

                    for(unsigned off = 0; off < dstReg.regNum; ++off)
                    {
                        mfmaRegisterLatencyCounters[dstReg.regIdx + off]
                            = node->inst->latencyCycles;
                    }
                }
            }
            updateMFMALatencyCounters(node);

            return node;
        }

        // Priority 3: Schedule barriers when their dependencies are satisfied
        if(!barrierQueue.empty())
        {
            DAGNode* barrier = barrierQueue.top();
            barrierQueue.pop();
            MFMAIssueUpdate(barrier);
            updateMFMALatencyCounters(barrier);
            return barrier;
        }

        return pickOneFromMFMA();
    }

    void CDNA3ReadyQueue::push(DAGNode* node)
    {
        if(isMFMA(*node->inst))
        {
            mfmaQueue.push(node);
            return;
        }

        if(getPassContext().getOptInfo().distributeGlobalRead && isGlobalMemLoad(*node->inst))
        {
            globalReadQueue.push(node);
            return;
        }

        if(isBarrier(*node->inst))
        {
            barrierQueue.push(node);
            return;
        }

        otherQueue.push(node);
    }

    bool CDNA3ReadyQueue::empty() const
    {
        return mfmaQueue.empty() && globalReadQueue.empty() && otherQueue.empty()
               && barrierQueue.empty();
    }

    void CDNA3ReadyQueue::onInit(IRList::iterator regionStart, IRList::iterator regionEnd)
    {
        // For for loop only optimization
        if(getPassContext().getOptInfo().unrollGemm == false)
            return;

        mfmaIssueConfig.latency = 0;
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isMFMA(inst) || isSMFMA(inst))
            {
                mfmaIssueConfig.latency = inst.latencyCycles;
                break;
            }
        }

        isInit = false;
    }

    void CDNA3ReadyQueue::onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd)
    {
        // For for loop only optimization
        if(getPassContext().getOptInfo().unrollGemm == false)
            return;

        mfmaIssueConfig.totalIssuedCycles     = 0;
        mfmaIssueConfig.totalMfmaIssuedCycles = 0;
        mfmaIssueConfig.issuedCount           = 0;
        int totalDSLatency                    = 0;
        // Get total issued cycles and total ds latency in the region
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);

            mfmaIssueConfig.totalIssuedCycles += inst.issueCycles;

            if(isDSRead(inst))
            {
                totalDSLatency += inst.latencyCycles;
            }

            if(isMFMA(inst) || isSMFMA(inst))
            {
                mfmaIssueConfig.issuedCount += 1;
                mfmaIssueConfig.totalMfmaIssuedCycles += inst.issueCycles;
            }
        }
        // Get possible longest average latency
        // The total issued cycles and total ds latency are in parallel
        // So we take the larger one to calculate average latency
        auto totalAvgLatency
            = (int)std::ceil((float)std::max(mfmaIssueConfig.totalIssuedCycles, totalDSLatency)
                             / mfmaIssueConfig.issuedCount);
        // Use the larger one as the mfma average counter
        // If mfma original latency is larger, then we don't have
        // to split the non-mfma instructions between mfma instructions
        if(totalAvgLatency > mfmaIssueConfig.latency)
        {
            mfmaIssueConfig.avgIssueInterval = totalAvgLatency;
        }
        else
        {
            mfmaIssueConfig.avgIssueInterval = mfmaIssueConfig.latency;
        }

        // Only init in the beginning of the loop
        if(!isInit)
        {
            mfmaCounter = mfmaIssueConfig.avgIssueInterval;
            isInit      = true;
        }
    }

    // Check if instruction is a movable side effect (like s_barrier)
    static bool isMovableSideEffect(const StinkyInstruction& inst)
    {
        return isBarrier(inst) && !inst.destRegs.empty();
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
            for(const StinkyRegister& srcReg : inst.srcRegs)
            {
                if(!srcReg.isRegister())
                    continue;

                for(unsigned off = 0; off < srcReg.regNum; ++off)
                {
                    StinkyRegister reg(srcReg.regType, srcReg.regIdx + off, 1);
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
            for(const StinkyRegister& dstReg : inst.destRegs)
            {
                if(!dstReg.isRegister())
                    continue;

                for(unsigned off = 0; off < dstReg.regNum; ++off)
                {
                    StinkyRegister reg(dstReg.regType, dstReg.regIdx + off, 1);

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
            isGlobalMemStore(inst) || isBranch(inst) || isBarrier(inst) || isWaitCnt(inst))
        {
            return true;
        }
        return false;
    }

    // Schedule the instructions in the given IRList.
    // This will split the instructions into regions based on side-effect instructions
    // and schedule each region in a DAG.
    //
    // In the end, the instructions will be reordered in the IRList
    // to reflect the scheduling order.
    static void scheduleInDAG(IRList& insts, ReadyQueue& readyQueue)
    {
        PASS_DEBUG(std::cerr << "*** Scheduling Instructions in DAG: ***\n");

        if(insts.empty())
            return;

        std::vector<StinkyInstruction*> scheduled;
        scheduled.reserve(insts.size());

        readyQueue.onInit(insts.begin(), insts.end());

        IRList::iterator regionStart = insts.begin();

        for(IRList::iterator it = insts.begin(); it != insts.end(); ++it)
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
        scheduleRegionWithMovableSideEffects(regionStart, insts.end(), scheduled, readyQueue);

        assert(scheduled.size() == insts.size()
               && "Scheduled instructions size must match original instructions size");

        // Now we have a scheduled list of instructions.
        // Modify the original insts list to reflect the scheduling.
        for(StinkyInstruction* inst : scheduled)
        {
            insts.moveBefore(IRList::iterator(inst), insts.end());
        }
    }

    std::unique_ptr<ReadyQueue> chooseReadyQueue(const PassContext& passCtx)
    {
        if(passCtx.getKernelInfo().arch[0] >= 9)
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

        void run(IRList& irlist, PassContext& passCtx) override
        {
            buildUseDefChain(irlist);

            PASS_DEBUG(dumpUseDefChain(irlist));

            std::unique_ptr<ReadyQueue> readyQueue = chooseReadyQueue(passCtx);
            scheduleInDAG(irlist, *readyQueue);
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
