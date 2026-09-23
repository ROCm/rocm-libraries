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

#include "RegionDAG.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <ostream>

namespace stinkytofu {
namespace dag {

namespace {

using namespace stinkytofu;

RegionDAG buildRegisterDependencyDAGImpl(const std::vector<StinkyInstruction*>& instructions) {
    RegionDAG result;
    const unsigned n = static_cast<unsigned>(instructions.size());
    if (n == 0) return result;

    result.nodes.reserve(n);
    result.graph.resize(n);
    result.instToId.reserve(n);

    for (unsigned i = 0; i < n; ++i) {
        result.nodes.emplace_back(instructions[i], i);
        result.instToId[instructions[i]] = i;
    }

    std::map<StinkyRegister, std::unordered_set<DAGNode*>> lastRead;
    std::map<StinkyRegister, DAGNode*> lastWrite;

    for (unsigned i = 0; i < n; ++i) {
        DAGNode& dagNode = result.nodes[i];
        StinkyInstruction& inst = *dagNode.inst;

        for (const StinkyRegister& srcReg : inst.getSrcRegs()) {
            if (!srcReg.isRegister()) continue;
            for (unsigned off = 0; off < srcReg.reg.num; ++off) {
                StinkyRegister reg(srcReg.reg.type, srcReg.reg.idx + off, 1);
                auto itLastWrite = lastWrite.find(reg);
                if (itLastWrite != lastWrite.end())
                    addEdgeById(itLastWrite->second, &dagNode, result.graph);
                lastRead[reg].insert(&dagNode);
            }
        }

        for (const StinkyRegister& dstReg : inst.getDestRegs()) {
            if (!dstReg.isRegister()) continue;
            for (unsigned off = 0; off < dstReg.reg.num; ++off) {
                StinkyRegister reg(dstReg.reg.type, dstReg.reg.idx + off, 1);
                auto itLastWrite = lastWrite.find(reg);
                if (itLastWrite != lastWrite.end())
                    addEdgeById(itLastWrite->second, &dagNode, result.graph);
                auto itLastRead = lastRead.find(reg);
                if (itLastRead != lastRead.end()) {
                    for (DAGNode* lastReader : itLastRead->second)
                        addEdgeById(lastReader, &dagNode, result.graph);
                    lastRead.erase(reg);
                }
                lastWrite[reg] = &dagNode;
            }
        }
    }

    return result;
}

}  // namespace

RegionDAG buildRegisterDependencyDAG(const std::vector<StinkyInstruction*>& instructions) {
    return buildRegisterDependencyDAGImpl(instructions);
}

RegionDAG buildRegisterDependencyDAG(IRList::iterator regionStart, IRList::iterator regionEnd) {
    std::vector<StinkyInstruction*> instructions;
    instructions.reserve(static_cast<size_t>(std::distance(regionStart, regionEnd)));
    for (IRList::iterator it = regionStart; it != regionEnd; ++it)
        instructions.push_back(&getStinkyInst(it));
    return buildRegisterDependencyDAGImpl(instructions);
}

void dumpDAGGraph(const RegionDAG& dag, std::ostream& os,
                  const std::set<std::pair<unsigned, unsigned>>& hardConstraintEdges) {
    os << "DAG nodes:\n";
    for (const DAGNode& node : dag.nodes) {
        os << node.id << ": ";
        node.inst->dump(os);
    }

    os << "DAG edges:\n";
    for (unsigned fromId = 0; fromId < dag.graph.size(); ++fromId) {
        std::vector<unsigned> successors(dag.graph[fromId].begin(), dag.graph[fromId].end());
        std::sort(successors.begin(), successors.end());
        for (unsigned toId : successors) {
            os << fromId << " -> " << toId;
            if (hardConstraintEdges.contains({fromId, toId})) os << "  (hard constraint)";
            os << '\n';
        }
    }
    os << '\n';
}

}  // namespace dag
}  // namespace stinkytofu
