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
#include <map>

#include "ReadyQueue.hpp"

namespace
{
    using namespace stinkytofu;

    // Temporarily name MI450 as CDNA5. Will rename it later.
    class CDNA5ReadyQueue : public ReadyQueue
    {
        DAGidPriorityQueue wmmaQueue;
        DAGidPriorityQueue globalReadQueue;
        DAGidPriorityQueue localReadQueue;
        DAGidPriorityQueue barrierQueue;
        DAGidPriorityQueue otherQueue;

        bool isInit = false;

        int globalReadCounter = 0; // tracking global read count during WMMA
        int globalReadPerWMMA = 1; // global read issue count per WMMA

        // Per-WMMA issue interval: cycles until this WMMA node can be issued (0 = ready).
        std::map<DAGNode*, int> wmmaNodeCounters;
        // ds_read latency per destination VGPR; when counter reaches 0 the VGPR is ready for WMMA.
        std::map<int, int> wmmaRegisterLatencyCounters;

        WMMAIssueConfig wmmaIssueConfig;

        // Force interleave: do not issue WMMA back-to-back when other instructions are available.
        bool lastPickedWasWMMA = false;

        void     updateWMMAStatus(DAGNode* node);
        bool     isWMMALatencyFree(DAGNode* node);
        DAGNode* pickOneFromWMMA();

    public:
        explicit CDNA5ReadyQueue(const PassContext& passCtx)
            : ReadyQueue(passCtx)
        {
        }

        DAGNode* pickOne() override;
        void     push(DAGNode* node) override;
        bool     empty() const override;

        void onInit(IRList::iterator regionStart, IRList::iterator regionEnd) override;

        void onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd) override;
    };

    void CDNA5ReadyQueue::updateWMMAStatus(DAGNode* node)
    {
        const int cycles = node->inst->issueCycles;

        wmmaIssueConfig.totalIssuedCycles -= cycles;

        // Decrement per-WMMA issue interval counters.
        for(auto& [n, counter] : wmmaNodeCounters)
        {
            if(counter > 0)
                counter -= cycles;
        }

        // Decrement ds_read latency per VGPR; remove when latency reaches 0 (VGPR ready).
        for(auto it = wmmaRegisterLatencyCounters.begin(); it != wmmaRegisterLatencyCounters.end();)
        {
            if(it->second > 0)
            {
                it->second -= cycles;
                if(it->second <= 0)
                    it = wmmaRegisterLatencyCounters.erase(it);
                else
                    ++it;
            }
            else
                ++it;
        }
    }

    bool CDNA5ReadyQueue::isWMMALatencyFree(DAGNode* node)
    {
        for(const StinkyRegister& dstReg : node->inst->getSrcRegs())
        {
            if(!dstReg.isRegister())
                continue;

            for(unsigned off = 0; off < dstReg.reg.num; ++off)
            {
                auto it = wmmaRegisterLatencyCounters.find(dstReg.reg.idx + off);
                if(it != wmmaRegisterLatencyCounters.end() && it->second > 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    DAGNode* CDNA5ReadyQueue::pickOneFromWMMA()
    {
        assert(!wmmaQueue.empty() && "The WMMA queue must not be empty");
        DAGNode* node = wmmaQueue.top();
        wmmaQueue.pop();

        // Compute issue interval for this WMMA; apply to other WMMA nodes still in queue.
        int  delay;
        auto rest
            = (int)((wmmaIssueConfig.totalIssuedCycles - wmmaIssueConfig.totalWmmaIssuedCycles)
                    / wmmaIssueConfig.issuedCount);
        if(wmmaIssueConfig.issuedCount <= 0
           || (wmmaIssueConfig.issuedCount > 0 && rest < node->inst->latencyCycles))
        {
            if(node->inst->latencyCycles >= wmmaIssueConfig.avgIssueInterval)
                delay = std::max(rest, 1);
            else
                delay = node->inst->latencyCycles;
        }
        else
            delay = wmmaIssueConfig.avgIssueInterval;

        for(auto& [n, counter] : wmmaNodeCounters)
        {
            if(n != node)
                counter = std::max(counter, delay);
        }
        wmmaNodeCounters.erase(node);

        wmmaIssueConfig.issuedCount--;
        wmmaIssueConfig.totalWmmaIssuedCycles -= node->inst->issueCycles;
        updateWMMAStatus(node);
        globalReadCounter = 0;
        return node;
    }

    DAGNode* CDNA5ReadyQueue::pickOne()
    {
        // Priority 1: Try to schedule WMMA if this node's counter allows and ds_read latencies are satisfied.
        // Force interleave: if we just issued a WMMA and there is work in other queues, skip WMMA and fall
        // through so priorities 2-5 can issue it (avoids back-to-back WMMAs when ds_reads etc. are still pending).
        bool otherQueuesHaveWork
            = !globalReadQueue.empty() || !localReadQueue.empty() || !otherQueue.empty();
        if(!wmmaQueue.empty())
        {
            DAGNode* top = wmmaQueue.top();
            int      c   = wmmaNodeCounters.count(top) ? wmmaNodeCounters.at(top) : 0;
            if(c <= 0 && isWMMALatencyFree(top) && (!lastPickedWasWMMA || !otherQueuesHaveWork))
            {
                DAGNode* node     = pickOneFromWMMA();
                lastPickedWasWMMA = true;
                return node;
            }
        }

        // Priority 2: Schedule global reads first to hide latency
        if(!globalReadQueue.empty())
        {
            if(globalReadCounter < globalReadPerWMMA || otherQueue.empty())
            {
                DAGNode* globalRead = globalReadQueue.top();
                globalReadQueue.pop();
                updateWMMAStatus(globalRead);
                globalReadCounter++;
                lastPickedWasWMMA = false;
                return globalRead;
            }
        }

        // Priority 3: Schedule local reads next to hide latency
        if(!localReadQueue.empty())
        {
            DAGNode* localRead = localReadQueue.top();
            localReadQueue.pop();
            updateWMMAStatus(localRead);
            lastPickedWasWMMA = false;
            return localRead;
        }

        // Priority 4: Schedule other instructions
        if(!otherQueue.empty())
        {
            DAGNode* node = otherQueue.top();
            otherQueue.pop();
            if(isDSRead(*node->inst))
            {
                for(const StinkyRegister& dstReg : node->inst->getDestRegs())
                {
                    if(!dstReg.isRegister())
                        continue;
                    for(unsigned off = 0; off < dstReg.reg.num; ++off)
                        wmmaRegisterLatencyCounters[dstReg.reg.idx + off]
                            = node->inst->latencyCycles;
                }
            }
            updateWMMAStatus(node);
            lastPickedWasWMMA = false;
            return node;
        }

        // Priority 5: Schedule barriers when their dependencies are satisfied
        if(!barrierQueue.empty())
        {
            DAGNode* barrier = barrierQueue.top();
            barrierQueue.pop();
            updateWMMAStatus(barrier);
            lastPickedWasWMMA = false;
            return barrier;
        }

        DAGNode* node     = pickOneFromWMMA();
        lastPickedWasWMMA = true;
        return node;
    }

    void CDNA5ReadyQueue::push(DAGNode* node)
    {
        if(isWMMA(*node->inst))
        {
            wmmaNodeCounters[node] = 0; // per-WMMA counter: ready to issue when <= 0
            wmmaQueue.push(node);
            return;
        }

        if(getPassContext().getPassFeatureConfig().dagFeatures.distributeGlobalRead
           && isTensorLoad(*node->inst))
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

    bool CDNA5ReadyQueue::empty() const
    {
        return wmmaQueue.empty() && globalReadQueue.empty() && otherQueue.empty()
               && barrierQueue.empty() && localReadQueue.empty();
    }

    void CDNA5ReadyQueue::onInit(IRList::iterator regionStart, IRList::iterator regionEnd)
    {
        // For for loop only optimization
        if(getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false)
            return;

        wmmaIssueConfig.latency = 0;
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);
            if(isWMMA(inst) || isSWMMA(inst))
            {
                wmmaIssueConfig.latency = inst.latencyCycles;
                break;
            }
        }

        isInit = false;
    }

    void CDNA5ReadyQueue::onInitRegion(IRList::iterator regionStart, IRList::iterator regionEnd)
    {
        lastPickedWasWMMA = false;
        // For for loop only optimization
        if(getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false)
            return;

        wmmaIssueConfig.totalIssuedCycles     = 0;
        wmmaIssueConfig.totalWmmaIssuedCycles = 0;
        wmmaIssueConfig.issuedCount           = 0;
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);

            wmmaIssueConfig.totalIssuedCycles += inst.issueCycles;

            if(isWMMA(inst) || isSWMMA(inst))
            {
                wmmaIssueConfig.issuedCount += 1;
                wmmaIssueConfig.totalWmmaIssuedCycles += inst.issueCycles;
            }
        }
        auto totalAvgLatency = (int)std::ceil((float)wmmaIssueConfig.totalIssuedCycles
                                              / wmmaIssueConfig.issuedCount);
        // Use the larger one as the wmma average counter
        // If wmma original latency is larger, then we don't have
        // to split the non-wmma instructions between wmma instructions
        if(totalAvgLatency > wmmaIssueConfig.latency)
        {
            wmmaIssueConfig.avgIssueInterval = totalAvgLatency;
        }
        else
        {
            wmmaIssueConfig.avgIssueInterval = wmmaIssueConfig.latency;
        }

        if(!isInit)
            isInit = true;
    }
}
