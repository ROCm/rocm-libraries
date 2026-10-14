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
 * ************************************************************************ */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "../RegionDAG.hpp"

namespace stinkytofu {
namespace dag {

namespace detail {

// Sorted, de-duplicated PSEUDO source tokens of a ds_read (any pseudo register type, the
// same test the barrier threshold scans use to match a ds_read to a barrier).
inline std::vector<uint32_t> dsReadOrderKey(const StinkyInstruction& inst) {
    std::vector<uint32_t> tokens;
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (isPseudoReg(src)) tokens.push_back(src.reg.idx);
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

}  // namespace detail

// ModuleOptions::LockDsReadOrder. Chain the ds_reads that share a memory token in
// (dsReadPriority, DAG id) order and return the (predecessor, successor) pairs; a later load
// stays unready until every earlier one on that token has issued. The caller decides whether
// to request them.
//
// Only loads on the same token are chained: a cross-token priority edge can cycle with a real
// dependence when the latch's next-iteration WMMA index wraps back to the header. A load with
// no PSEUDO token is left unordered. An edge that still contradicts a real dependence is
// dropped as a cycle by the scheduler, one pair at a time. These edges are not part of the
// Layer 2 barrier overlap contract.
inline std::vector<HardSchedulingConstraint> planDsReadOrder(IRList::iterator regionStart,
                                                             IRList::iterator regionEnd,
                                                             const RegionDAG& dag) {
    constexpr unsigned kUnknown = std::numeric_limits<unsigned>::max();
    auto idOf = [&](StinkyInstruction* inst) {
        auto it = dag.instToId.find(inst);
        return it == dag.instToId.end() ? kUnknown : it->second;
    };
    auto priorityOf = [&](StinkyInstruction* inst) {
        const unsigned id = idOf(inst);
        return id == kUnknown ? kUnknown : dag.nodes[id].dsReadPriority;
    };

    std::map<std::vector<uint32_t>, std::vector<StinkyInstruction*>> loadsByKey;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || !isDSRead(*inst)) continue;
        std::vector<uint32_t> key = detail::dsReadOrderKey(*inst);
        if (key.empty()) continue;
        loadsByKey[std::move(key)].push_back(inst);
    }

    std::vector<HardSchedulingConstraint> edges;
    for (auto& entry : loadsByKey) {
        auto& loads = entry.second;
        std::stable_sort(loads.begin(), loads.end(),
                         [&](StinkyInstruction* a, StinkyInstruction* b) {
                             const unsigned priA = priorityOf(a);
                             const unsigned priB = priorityOf(b);
                             if (priA != priB) return priA < priB;
                             return idOf(a) < idOf(b);
                         });
        for (size_t i = 1; i < loads.size(); ++i) edges.emplace_back(loads[i - 1], loads[i]);
    }
    return edges;
}

}  // namespace dag
}  // namespace stinkytofu
