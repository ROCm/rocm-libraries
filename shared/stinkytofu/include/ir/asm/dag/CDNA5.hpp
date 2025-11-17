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

#include "ir/asm/dag/ReadyQueue.hpp"

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

        int wmmaCounter = 0;
        // For wmma latency tracking per register
        std::map<int, int> wmmaRegisterLatencyCounters;

        WMMAIssueConfig wmmaIssueConfig;

        void     updateWMMALatencyCounters(DAGNode* node);
        bool     isWMMALatencyFree();
        DAGNode* pickOneFromWMMA();
        void     WMMAIssueUpdate(DAGNode* node);

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

    void CDNA5ReadyQueue::updateWMMALatencyCounters(DAGNode* node)
    {
        // decrement the latency counters
        for(auto& [reg, counter] : wmmaRegisterLatencyCounters)
        {
            if(counter > 0)
                counter -= node->inst->issueCycles;
        }
    }

    bool CDNA5ReadyQueue::isWMMALatencyFree()
    {
        DAGNode* node = wmmaQueue.top();
        for(const StinkyRegister& dstReg : node->inst->srcRegs)
        {
            if(!dstReg.isRegister())
                continue;

            for(unsigned off = 0; off < dstReg.regNum; ++off)
            {
                auto it = wmmaRegisterLatencyCounters.find(dstReg.regIdx + off);
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
        // Use the original latency to avoid wmma issued continuously
        auto rest
            = (int)((wmmaIssueConfig.totalIssuedCycles - wmmaIssueConfig.totalWmmaIssuedCycles)
                    / wmmaIssueConfig.issuedCount);
        if(wmmaIssueConfig.issuedCount <= 0
           || (wmmaIssueConfig.issuedCount > 0 && rest < node->inst->latencyCycles))
        {
            // interleave wmma with other instructions if possible
            // wmma inst1 wmma inst2 wmma inst3
            if(node->inst->latencyCycles >= wmmaIssueConfig.avgIssueInterval)
            {
                wmmaCounter = std::max(rest, 1);
            }
            else
            {
                wmmaCounter = node->inst->latencyCycles;
            }
        }
        else
        {
            wmmaCounter = wmmaIssueConfig.avgIssueInterval;
        }
        wmmaIssueConfig.issuedCount--;
        wmmaIssueConfig.totalWmmaIssuedCycles -= node->inst->issueCycles;
        WMMAIssueUpdate(node);
        updateWMMALatencyCounters(node);
        globalReadCounter = 0;
        return node;
    }

    void CDNA5ReadyQueue::WMMAIssueUpdate(DAGNode* node)
    {
        // Use issue for all instructions
        wmmaCounter -= node->inst->issueCycles;
        wmmaIssueConfig.totalIssuedCycles -= node->inst->issueCycles;
    }

    DAGNode* CDNA5ReadyQueue::pickOne()
    {
        // Priority 1: Try to schedule WMMA if counter allows
        // TODO: Need to check if ds_read completes are done for the WMMA
        // wmmaRegisterLatencyCounters
        if(!wmmaQueue.empty() && wmmaCounter <= 0 && isWMMALatencyFree())
        {
            return pickOneFromWMMA();
        }

        // Priority 2: Schedule global reads first to hide latency
        if(!globalReadQueue.empty())
        {
            if(globalReadCounter < globalReadPerWMMA || otherQueue.empty())
            {
                DAGNode* globalRead = globalReadQueue.top();
                globalReadQueue.pop();
                WMMAIssueUpdate(globalRead);
                updateWMMALatencyCounters(globalRead);
                globalReadCounter++;
                return globalRead;
            }
        }

        // Priority 3: Schedule local reads next to hide latency
        if(!localReadQueue.empty())
        {
            DAGNode* localRead = localReadQueue.top();
            localReadQueue.pop();
            WMMAIssueUpdate(localRead);
            updateWMMALatencyCounters(localRead);
            return localRead;
        }

        // Priority 4: Schedule other instructions
        if(!otherQueue.empty())
        {
            DAGNode* node = otherQueue.top();
            otherQueue.pop();
            WMMAIssueUpdate(node);
            // If ds read, add its latency to the counter to wmmaRegisterLatencyCounters
            // Ignore global read for now
            if(isDSRead(*node->inst))
            {
                for(const StinkyRegister& dstReg : node->inst->destRegs)
                {
                    if(!dstReg.isRegister())
                        continue;

                    for(unsigned off = 0; off < dstReg.regNum; ++off)
                    {
                        wmmaRegisterLatencyCounters[dstReg.regIdx + off]
                            = node->inst->latencyCycles;
                    }
                }
            }
            updateWMMALatencyCounters(node);

            return node;
        }

        // Priority 5: Schedule barriers when their dependencies are satisfied
        if(!barrierQueue.empty())
        {
            DAGNode* barrier = barrierQueue.top();
            barrierQueue.pop();
            WMMAIssueUpdate(barrier);
            updateWMMALatencyCounters(barrier);
            return barrier;
        }

        return pickOneFromWMMA();
    }

    void CDNA5ReadyQueue::push(DAGNode* node)
    {
        if(isWMMA(*node->inst))
        {
            wmmaQueue.push(node);
            return;
        }

        if(getPassContext().getOptInfo().distributeGlobalRead && isTensorLoad(*node->inst))
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
        if(getPassContext().getOptInfo().unrollGemm == false)
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
        // For for loop only optimization
        if(getPassContext().getOptInfo().unrollGemm == false)
            return;

        wmmaIssueConfig.totalIssuedCycles     = 0;
        wmmaIssueConfig.totalWmmaIssuedCycles = 0;
        wmmaIssueConfig.issuedCount           = 0;
        int totalDSLatency                    = 0;
        // Get total issued cycles and total ds latency in the region
        for(IRList::iterator it = regionStart; it != regionEnd; ++it)
        {
            StinkyInstruction& inst = getStinkyInst(it);

            wmmaIssueConfig.totalIssuedCycles += inst.issueCycles;

            if(isDSRead(inst))
            {
                totalDSLatency += inst.latencyCycles;
            }

            if(isWMMA(inst) || isSWMMA(inst))
            {
                wmmaIssueConfig.issuedCount += 1;
                wmmaIssueConfig.totalWmmaIssuedCycles += inst.issueCycles;
            }
        }
        // Get possible longest average latency
        // The total issued cycles and total ds latency are in parallel
        // So we take the larger one to calculate average latency
        auto totalAvgLatency
            = (int)std::ceil((float)std::max(wmmaIssueConfig.totalIssuedCycles, totalDSLatency)
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

        // Only init in the beginning of the loop
        if(!isInit)
        {
            wmmaCounter = wmmaIssueConfig.avgIssueInterval;
            isInit      = true;
        }
    }
}
