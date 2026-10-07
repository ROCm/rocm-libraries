// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_set>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class BasicBlock;
struct StinkyInstruction;

/// One block of a function, with the instruction order to plan for.
struct BlockOrder {
    const BasicBlock* bb = nullptr;
    const std::vector<const StinkyInstruction*>* insts = nullptr;
};

/// The global prefetches PrefetchBridgeSubstitutionPass rewrites to flat_prefetch.
/// `layout` is every block of the function in layout order; `required` is
/// HWModel::waitHide.vmVsrcBridge. Only reads the instructions, so it can plan an order
/// that is not in the IR.
STINKYTOFU_EXPORT std::unordered_set<const StinkyInstruction*> planPrefetchBridge(
    const std::vector<BlockOrder>& layout, int required);

}  // namespace stinkytofu
