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
#include "stinkytofu/analysis/controlflow/Dominance.hpp"

#include <algorithm>
#include <cassert>

#include "stinkytofu/support/CFGTraversal.hpp"

namespace {
using namespace stinkytofu;
using BlockIndex = std::unordered_map<const BasicBlock*, unsigned>;

std::vector<BasicBlock*> computeRPO(Function& func) {
    std::vector<BasicBlock*> rpo;
    traverseCFGInRPO(func, [&](BasicBlock* bb) { rpo.push_back(bb); });
    return rpo;
}

// Cooper-Harvey-Kennedy iterative dominator algorithm.
// Time: O(N*E) worst case; near-linear for reducible CFGs.
std::vector<unsigned> computeIDom(const std::vector<BasicBlock*>& rpo, const BlockIndex& blkIdx) {
    const unsigned N = rpo.size();
    std::vector<unsigned> idom(N, DominanceInfo::kUndef);
    idom[0] = 0;

    auto intersect = [&](unsigned a, unsigned b) {
        while (a != b) {
            while (a > b) a = idom[a];
            while (b > a) b = idom[b];
        }
        return a;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned i = 1; i < N; ++i) {
            unsigned newIdom = DominanceInfo::kUndef;
            for (BasicBlock* p : rpo[i]->getPredecessors()) {
                auto it = blkIdx.find(p);
                // RPO is entry-reachable only; skip unreachable predecessors.
                if (it == blkIdx.end()) continue;

                unsigned pi = it->second;
                if (idom[pi] == DominanceInfo::kUndef) continue;
                newIdom = (newIdom == DominanceInfo::kUndef) ? pi : intersect(pi, newIdom);
            }
            if (newIdom != DominanceInfo::kUndef && idom[i] != newIdom) {
                idom[i] = newIdom;
                changed = true;
            }
        }
    }

    return idom;
}

// DF[b] = { j : b dominates a predecessor of j but does not
//               strictly dominate j }
// Time: O(N + E + F*log F), F = Sigma|DF[i]|.
std::vector<std::vector<unsigned>> computeDF(const std::vector<BasicBlock*>& rpo,
                                             const BlockIndex& blkIdx,
                                             const std::vector<unsigned>& idom) {
    const unsigned N = rpo.size();
    std::vector<std::vector<unsigned>> df(N);

    for (unsigned j = 0; j < N; ++j) {
        const auto& preds = rpo[j]->getPredecessors();
        if (preds.size() < 2) continue;

        for (BasicBlock* p : preds) {
            auto it = blkIdx.find(p);
            // DF is computed for entry-reachable blocks only.
            if (it == blkIdx.end()) continue;

            unsigned runner = it->second;
            while (runner != idom[j]) {
                df[runner].push_back(j);
                runner = idom[runner];
            }
        }
    }

    for (auto& v : df) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    return df;
}

}  // anonymous namespace

namespace stinkytofu {
DominanceInfo computeDominanceInfo(Function& func) {
    DominanceInfo info;

    info.rpo = computeRPO(func);
    const unsigned N = info.rpo.size();
    if (N == 0) return info;

    info.rpoIndex.reserve(N);
    for (unsigned i = 0; i < N; ++i) info.rpoIndex[info.rpo[i]] = i;

    info.idom = computeIDom(info.rpo, info.rpoIndex);
    info.df = computeDF(info.rpo, info.rpoIndex, info.idom);

    return info;
}

}  // namespace stinkytofu
