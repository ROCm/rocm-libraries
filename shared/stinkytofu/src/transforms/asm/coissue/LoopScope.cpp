// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "LoopScope.hpp"

#include <map>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu::coissue {

std::vector<LoopScope> innermostLoops(Function& func, const std::vector<Loop>& loops) {
    // Merge the back edges of one header, keeping headers in layout order.
    std::map<BasicBlock*, std::unordered_set<BasicBlock*>> bodies;
    for (const Loop& loop : loops) bodies[loop.headerBB].insert(loop.bodyBBs.begin(), loop.bodyBBs.end());

    std::vector<LoopScope> out;
    for (BasicBlock& bb : func) {
        auto it = bodies.find(&bb);
        if (it == bodies.end()) continue;
        const auto& body = it->second;
        bool innermost = true;
        for (const auto& [header, other] : bodies) {
            (void)other;
            if (header != &bb && body.count(header) != 0) innermost = false;
        }
        if (!innermost) continue;
        LoopScope scope;
        scope.header = &bb;
        for (BasicBlock& member : func)
            if (body.count(&member) != 0) scope.blocks.push_back(&member);
        out.push_back(std::move(scope));
    }
    return out;
}

std::vector<StinkyInstruction*> loopInstructions(const LoopScope& scope) {
    std::vector<StinkyInstruction*> out;
    for (BasicBlock* bb : scope.blocks)
        for (IRBase& node : *bb)
            if (auto* inst = dyn_cast<StinkyInstruction>(&node)) out.push_back(inst);
    return out;
}

}  // namespace stinkytofu::coissue
