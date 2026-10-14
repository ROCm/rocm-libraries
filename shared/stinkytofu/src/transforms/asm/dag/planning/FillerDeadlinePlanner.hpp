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
#include <climits>
#include <unordered_map>
#include <vector>

#include "../RegionDAG.hpp"

namespace stinkytofu {
namespace dag {

// Per DAG node, the WMMA window by which a filler (VALU, transcendental, SALU) must issue so
// the prefetch, tensor_load or ds_read it feeds is not delayed: the planned window of the
// nearest such consumer, minus one per filler hop still to go. INT_MAX means no deadline.
// Prefetches use their planned earliest window (0 when unplanned), tensor_loads the window of
// the barrier they follow, and ds_reads are spread evenly over the region's WMMAs.
inline std::vector<int> planFillerDeadlines(
    const RegionDAG& dag,
    const std::unordered_map<const StinkyInstruction*, int>& prefetchEarliestWmma,
    const std::unordered_map<const StinkyInstruction*, int>& loadPlannedWindow, int dsTotal,
    int wmmaTotal) {
    const unsigned n = static_cast<unsigned>(dag.nodes.size());
    std::vector<int> needWindow(n, INT_MAX);
    std::vector<int> planned(n, INT_MAX);
    int dsOrdinal = 0;
    for (unsigned id = 0; id < n; ++id) {
        const StinkyInstruction* inst = dag.nodes[id].inst;
        if (isGlobalPrefetch(*inst)) {
            auto it = prefetchEarliestWmma.find(inst);
            planned[id] = it != prefetchEarliestWmma.end() ? it->second : 0;
        } else if (isTensorLoad(*inst)) {
            auto it = loadPlannedWindow.find(inst);
            if (it != loadPlannedWindow.end()) planned[id] = it->second;
        } else if (isDSRead(*inst) && dsTotal > 0) {
            planned[id] = dsOrdinal++ * wmmaTotal / dsTotal;
        }
    }

    auto isFiller = [](const StinkyInstruction& i) {
        return isVectorALU(i) || isTranscendental(i) || isScalarALU(i);
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (unsigned id = n; id-- > 0;) {
            if (!isFiller(*dag.nodes[id].inst)) continue;
            int need = needWindow[id];
            for (unsigned succ : dag.graph[id]) {
                const int via = planned[succ] != INT_MAX      ? planned[succ]
                                : needWindow[succ] != INT_MAX ? needWindow[succ] - 1
                                                              : INT_MAX;
                need = std::min(need, via);
            }
            if (need < needWindow[id]) {
                needWindow[id] = need;
                changed = true;
            }
        }
    }
    return needWindow;
}

}  // namespace dag
}  // namespace stinkytofu
