// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MoveChecker.hpp"

#include <algorithm>

#include "../dag/RegionDAG.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu::coissue {

struct MoveChecker::BlockGraph {
    dag::RegionDAG dag;
};

namespace {

bool writesExec(const StinkyInstruction& inst) {
    for (const StinkyRegister& r : inst.getDestRegs())
        if (r.isRegister() && (r.reg.type == RegType::EXEC || r.reg.type == RegType::EXEC_LO ||
                               r.reg.type == RegType::EXEC_HI))
            return true;
    return false;
}

bool isPinnedOpcode(int op) {
    return op == GFX::s_nop || op == GFX::s_setprio || op == GFX::s_sleep ||
           op == GFX::s_wait_alu || op == GFX::s_set_vgpr_msb || op == GFX::s_delay_alu;
}

}  // namespace

MoveChecker::MoveChecker(const std::vector<StinkyInstruction*>& order,
                         const std::vector<int>& blockOf, const std::vector<int>& slotOf,
                         TimedInstCache& timed)
    : blockOf_(blockOf), slotOf_(slotOf), timed_(timed), memoryDefs_(kNumRegSlots, false) {
    const int blocks = order.empty() ? 0 : *std::max_element(blockOf.begin(), blockOf.end()) + 1;
    graphs_.resize(blocks);
    for (int b = 0; b < blocks; ++b) update(order, b);
    for (const StinkyInstruction* inst : order) {
        const TimedInst& t = timed_.get(*inst);
        if (t.isLabel || (t.kind != IssueClass::LdsLoad && t.kind != IssueClass::Memory)) continue;
        for (uint16_t slot : t.defs) memoryDefs_[slot] = true;
    }
}

MoveChecker::~MoveChecker() = default;

void MoveChecker::update(const std::vector<StinkyInstruction*>& order, int block) {
    std::vector<StinkyInstruction*> insts;
    for (size_t k = 0; k < order.size(); ++k)
        if (blockOf_[k] == block) insts.push_back(order[k]);
    auto graph = std::make_unique<BlockGraph>();
    graph->dag = dag::buildRegisterDependencyDAG(insts);
    graphs_[block] = std::move(graph);
}

bool MoveChecker::movable(const StinkyInstruction& inst) const {
    if (pinned_.count(&inst) != 0) return false;
    const TimedInst& t = timed_.get(inst);
    if (t.isLabel || (t.kind != IssueClass::Salu && t.kind != IssueClass::Valu)) return false;
    if (t.isBranch || isBarrier(inst) || isCall(inst) || hasSideEffect(inst) ||
        isExecMaskGroup(inst) || isPinnedOpcode(inst.getUnifiedOpcode()))
        return false;
    return !writesExec(inst);
}

bool MoveChecker::crossable(const StinkyInstruction& y) const {
    if (y.getUnifiedOpcode() == GFX::LABEL) return false;
    return !(isBranch(y) || isConditionalBranch(y) || isBarrier(y) || isCall(y));
}

bool MoveChecker::legal(const std::vector<StinkyInstruction*>& order, size_t i, size_t j) const {
    if (j == i || j == i + 1 || i >= order.size() || j > order.size()) return false;
    const StinkyInstruction& x = *order[i];
    if (!movable(x)) return false;
    // The position x ends up at; block and IR-node slots are fixed per position.
    const size_t p = j < i ? j : j - 1;
    if (blockOf_[p] != blockOf_[i] || slotOf_[p] != slotOf_[i]) return false;

    const TimedInst& tx = timed_.get(x);
    bool touchesMemory = false;
    for (uint16_t slot : tx.defs) touchesMemory |= memoryDefs_[slot];
    for (uint16_t slot : tx.uses) touchesMemory |= memoryDefs_[slot];

    const dag::RegionDAG& dag = graphs_[blockOf_[i]]->dag;
    const auto xIt = dag.instToId.find(order[i]);
    const size_t lo = j < i ? j : i + 1;
    const size_t hi = j < i ? i : j;
    for (size_t k = lo; k < hi; ++k) {
        const StinkyInstruction& y = *order[k];
        if (blockOf_[k] != blockOf_[i] || !crossable(y)) return false;
        if (touchesMemory && timed_.get(y).kind == IssueClass::MemWait) return false;
        const auto yIt = dag.instToId.find(order[k]);
        if (xIt == dag.instToId.end() || yIt == dag.instToId.end()) return false;
        if (dag.graph[xIt->second].count(yIt->second) != 0 ||
            dag.graph[yIt->second].count(xIt->second) != 0)
            return false;
    }
    return true;
}

}  // namespace stinkytofu::coissue
