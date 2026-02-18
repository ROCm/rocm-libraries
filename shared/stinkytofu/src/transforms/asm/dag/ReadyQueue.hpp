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
#pragma once

#include <cmath>
#include <iostream> // TODO: don't use iostream.
#include <queue>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/DefUseChain.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace
{
    using namespace stinkytofu;

    // REMOVED: Local buildUseDefChain() has been replaced by stinkytofu::buildUseDefChain()
    // from DefUseChain.hpp. All callers now use the shared implementation.

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

    static void dumpUseDefChain(const BasicBlock& bb)
    {
        std::cerr << "*** Use-Def Chain Dump: ***\n";
        for(const IRBase& ir : bb)
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

    struct MFMAIssueConfig
    {
        int latency               = 0; // original mfma latency
        int avgIssueInterval      = 0; // average issue interval for mfma
        int totalIssuedCycles     = 0; // total issued cycles in the region
        int totalMfmaIssuedCycles = 0; // total mfma issued cycles in the region
        int issuedCount           = 0; // total mfma issued count in the region
    };

    struct WMMAIssueConfig
    {
        int latency               = 0; // original mfma latency
        int avgIssueInterval      = 0; // average issue interval for mfma
        int totalIssuedCycles     = 0; // total issued cycles in the region
        int totalWmmaIssuedCycles = 0; // total wmma issued cycles in the region
        int issuedCount           = 0; // total wmma issued count in the region
    };
}
