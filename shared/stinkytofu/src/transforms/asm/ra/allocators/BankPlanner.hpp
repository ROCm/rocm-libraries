// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Choosing a VGPR bank for each block before placement (quadrant=plan).
//
// Internal to allocators/, like GreedyPlacement.hpp: the engine calls it between
// measuring its blocks and placing them.

#include <cstdint>
#include <vector>

#include "GreedyPlacement.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"

namespace stinkytofu {

/// Give each block the plan may move a VGPR bank, chosen against
/// context.bankCost, and return how many got one.
///
/// Planned: unpinned, uncapped V blocks no wider than a bank, with a member the
/// cost model weights as used in a loop. Fixed: pinned blocks in the banks
/// their hints imply, and capped blocks in bank 0, where every ceiling lies,
/// since one comes from an operand field with no bank selector; a block capped
/// higher would be undecided. Every other block is undecided, which the model
/// prices as kUnknownBank.
///
/// Each bank's budget is its allocatable registers, held against the live
/// width of its fixed and planned members at every slot, so a plan never asks
/// a bank for more registers than it has at once. Alignment and fragmentation
/// are not seen; placement goes on to the rest of the file when a planned bank
/// has no room left.
///
/// Deterministic: fixed orders and no randomness, so one input gives one plan.
uint32_t planBanks(std::vector<Block>& blocks, const AllocationContext& context);

}  // namespace stinkytofu
