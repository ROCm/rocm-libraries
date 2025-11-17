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
#pragma once

#include <cmath>
#include <iostream> // TODO: don't use iostream.
#include <map> // FIXME: Use unordered_map if StinkyRegister::regType is not std::string
#include <queue>

#include "ir/asm/StinkyAsmIR.hpp"

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