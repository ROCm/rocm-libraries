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

    class CDNA3ReadyQueue : public ReadyQueue
    {
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
        for(const StinkyRegister& dstReg : node->inst->getSrcRegs())
        {
            if(!dstReg.isRegister())
                continue;

            for(unsigned off = 0; off < dstReg.reg.num; ++off)
            {
                auto it = mfmaRegisterLatencyCounters.find(dstReg.reg.idx + off);
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
                for(const StinkyRegister& dstReg : node->inst->getDestRegs())
                {
                    if(!dstReg.isRegister())
                        continue;

                    for(unsigned off = 0; off < dstReg.reg.num; ++off)
                    {
                        mfmaRegisterLatencyCounters[dstReg.reg.idx + off]
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

        if(getPassContext().getPassFeatureConfig().dagFeatures.distributeGlobalRead
           && isGlobalMemLoad(*node->inst))
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
        if(getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false)
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
        if(getPassContext().getPassFeatureConfig().loopConfig.unrollGemm == false)
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
}
