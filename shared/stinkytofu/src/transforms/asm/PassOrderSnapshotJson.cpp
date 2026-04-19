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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/support/PassOrderSnapshotJson.hpp"

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dag/ReadyQueue.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/DAGScheduleJsonWriter.hpp"

namespace {
using namespace stinkytofu;

static void linearizeProgramOrderStinky(Function& func, const PassContext& passCtx,
                                        std::vector<StinkyInstruction*>& out) {
    out.clear();
    for (BasicBlock& bb : func) {
        if (!passCtx.shouldProcessBasicBlock(bb) || bb.empty()) continue;
        for (BasicBlock::iterator it = bb.begin(); it != bb.end(); ++it) {
            out.push_back(&getStinkyInst(it));
        }
    }
}

static std::string snapshotNodeLabel(const StinkyInstruction& inst) {
    const BasicBlock* bb = inst.getParent();
    const std::string bbTag = (bb && !bb->getLabel().empty()) ? bb->getLabel() : std::string{"?"};

    std::string s;
    if (inst.getUnifiedOpcode() == GFX::LABEL) {
        auto ld = inst.getModifier<LabelData>();
        if (ld && !ld->label.empty()) s = "@" + ld->label + " ";
    }
    s += "[" + bbTag + "] ";
    s += instructionJsonLabel(inst);
    return s;
}

static void buildRegisterDepDagForInstList(const std::vector<StinkyInstruction*>& insts,
                                           DAGNodeList& dagNodes,
                                           std::vector<std::unordered_set<unsigned>>& dagGraph) {
    const unsigned n = static_cast<unsigned>(insts.size());
    dagNodes.clear();
    dagGraph.assign(n, {});
    dagNodes.reserve(n);
    for (unsigned i = 0; i < n; ++i) {
        dagNodes.emplace_back(insts[i], i);
    }

    std::map<StinkyRegister, std::unordered_set<DAGNode*>> lastRead;
    std::map<StinkyRegister, DAGNode*> lastWrite;

    for (unsigned i = 0; i < n; ++i) {
        DAGNode& dagNode = dagNodes[i];
        StinkyInstruction& inst = *dagNode.inst;

        for (const StinkyRegister& srcReg : inst.getSrcRegs()) {
            if (!srcReg.isRegister()) continue;

            for (unsigned off = 0; off < srcReg.reg.num; ++off) {
                StinkyRegister reg(srcReg.reg.type, srcReg.reg.idx + off, 1);
                auto itLastWrite = lastWrite.find(reg);
                if (itLastWrite != lastWrite.end()) {
                    addEdgeById(itLastWrite->second, &dagNode, dagGraph);
                }
                lastRead[reg].insert(&dagNode);
            }
        }

        for (const StinkyRegister& dstReg : inst.getDestRegs()) {
            if (!dstReg.isRegister()) continue;

            for (unsigned off = 0; off < dstReg.reg.num; ++off) {
                StinkyRegister reg(dstReg.reg.type, dstReg.reg.idx + off, 1);

                auto itLastWrite = lastWrite.find(reg);
                if (itLastWrite != lastWrite.end()) {
                    addEdgeById(itLastWrite->second, &dagNode, dagGraph);
                }

                auto itLastRead = lastRead.find(reg);
                if (itLastRead != lastRead.end()) {
                    for (DAGNode* lastReader : itLastRead->second) {
                        addEdgeById(lastReader, &dagNode, dagGraph);
                    }
                    lastRead.erase(reg);
                }

                lastWrite[reg] = &dagNode;
            }
        }
    }
}

static std::vector<std::pair<unsigned, unsigned>> edgesFromDagGraph(
    const std::vector<std::unordered_set<unsigned>>& dagGraph) {
    std::vector<std::pair<unsigned, unsigned>> edges;
    for (unsigned u = 0; u < dagGraph.size(); ++u) {
        for (unsigned v : dagGraph[u]) {
            edges.emplace_back(u, v);
        }
    }
    return edges;
}

static void emitFunctionWideRegion(const std::vector<StinkyInstruction*>& programOrderBefore,
                                   Function& func, const PassContext& passCtx,
                                   const std::string& passNameJustRan,
                                   DAGScheduleJsonCollector& jsonCollector) {
    std::vector<StinkyInstruction*> afterOrder;
    linearizeProgramOrderStinky(func, passCtx, afterOrder);
    if (programOrderBefore.size() != afterOrder.size()) return;

    std::unordered_map<StinkyInstruction*, unsigned> idOfInst;
    idOfInst.reserve(programOrderBefore.size());
    for (unsigned i = 0; i < programOrderBefore.size(); ++i) {
        idOfInst[programOrderBefore[i]] = i;
    }

    std::vector<unsigned> scheduledOrderIds;
    scheduledOrderIds.reserve(afterOrder.size());
    for (StinkyInstruction* inst : afterOrder) {
        auto it = idOfInst.find(inst);
        if (it == idOfInst.end()) return;
        scheduledOrderIds.push_back(it->second);
    }

    DAGNodeList dumpNodes;
    std::vector<std::unordered_set<unsigned>> dumpGraph;
    buildRegisterDepDagForInstList(programOrderBefore, dumpNodes, dumpGraph);

    std::vector<std::pair<unsigned, std::string>> nodeEntries;
    nodeEntries.reserve(dumpNodes.size());
    for (const DAGNode& dn : dumpNodes) {
        nodeEntries.emplace_back(dn.id, snapshotNodeLabel(*dn.inst));
    }
    std::vector<unsigned> programOrderIds;
    programOrderIds.reserve(programOrderBefore.size());
    for (unsigned i = 0; i < programOrderBefore.size(); ++i) {
        programOrderIds.push_back(i);
    }

    const std::string& titlePfx = passCtx.getPassFeatureConfig().passOrderSnapshot.titlePrefix;
    std::string title = titlePfx.empty() ? "whole function" : (titlePfx + " · whole function");
    title += " · after " + passNameJustRan;
    std::vector<std::pair<unsigned, unsigned>> edges = edgesFromDagGraph(dumpGraph);
    jsonCollector.addRegion(title, nodeEntries, edges, programOrderIds, scheduledOrderIds);
}
}  // namespace

namespace stinkytofu {
bool shouldEmitPassOrderSnapshotAfterPass(const PassFeatureConfig& cfg,
                                          const std::string& passName) {
    if (cfg.passOrderSnapshot.jsonPath.empty()) return false;
    if (passName == "StinkyBuildImplicitDependencyPass") return false;
    const std::vector<std::string>& names = cfg.passOrderSnapshot.dumpAfterPasses;
    if (names.empty()) return passName == "StinkyDAGSchedulerPass";
    for (const std::string& n : names) {
        if (n == passName) return true;
    }
    return false;
}

void snapshotProgramOrderStinkyLinear(Function& func, const PassContext& passCtx,
                                      std::vector<StinkyInstruction*>& outOrder) {
    linearizeProgramOrderStinky(func, passCtx, outOrder);
}

void appendPassOrderSnapshotJsonAfterPass(Function& func,
                                          const std::vector<StinkyInstruction*>& beforeOrder,
                                          const PassContext& passCtx,
                                          const std::string& passNameJustRan,
                                          DAGScheduleJsonCollector& collector) {
    if (beforeOrder.empty()) return;
    emitFunctionWideRegion(beforeOrder, func, passCtx, passNameJustRan, collector);
}

//----------------------------------------------------------------------
// PassOrderSnapshotInstrumentation
//----------------------------------------------------------------------
PassOrderSnapshotInstrumentation::PassOrderSnapshotInstrumentation(
    std::shared_ptr<DAGScheduleJsonCollector> collector)
    : collector(std::move(collector)) {}

void PassOrderSnapshotInstrumentation::beforePass(const std::string& passName, Function& F,
                                                  PassContext& ctx) {
    beforeOrder.clear();
    if (collector && shouldEmitPassOrderSnapshotAfterPass(ctx.getPassFeatureConfig(), passName)) {
        snapshotProgramOrderStinkyLinear(F, ctx, beforeOrder);
    }
}

void PassOrderSnapshotInstrumentation::afterPass(const std::string& passName, Function& F,
                                                 PassContext& ctx) {
    if (!beforeOrder.empty()) {
        appendPassOrderSnapshotJsonAfterPass(F, beforeOrder, ctx, passName, *collector);
        beforeOrder.clear();
    }
}

}  // namespace stinkytofu
