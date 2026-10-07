// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "stinkytofu/support/LoopDetection.hpp"

namespace stinkytofu {
class BasicBlock;
class Function;
struct StinkyInstruction;
}  // namespace stinkytofu

namespace stinkytofu::coissue {

/// An innermost loop: its blocks in layout order, header first.
struct LoopScope {
    BasicBlock* header = nullptr;
    std::vector<BasicBlock*> blocks;
};

/// The innermost loops of `func`. Back edges to the same header form one loop.
std::vector<LoopScope> innermostLoops(Function& func, const std::vector<Loop>& loops);

/// Every instruction of the loop in layout order, labels included.
std::vector<StinkyInstruction*> loopInstructions(const LoopScope& scope);

}  // namespace stinkytofu::coissue
