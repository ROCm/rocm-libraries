// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// How many s_set_vgpr_msb a colouring will cost, before it is applied.
//
// InsertVgprMsbPass runs after allocation and emits a switch whenever the next
// instruction needs a different VGPR bank word. Where the allocator puts a value
// therefore decides how many switches a loop pays, but the count only becomes
// visible in the final asm. This replays the pass's decision for a colouring that
// is still an AllocationResult, so the allocation report can say what a colouring
// costs and compare it with the producer's.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

namespace stinkytofu {

/// Predicted s_set_vgpr_msb for one loop.
struct LoopVgprMsbSwitches {
    /// Label of the loop's header block.
    std::string header;
    /// Switches InsertVgprMsbPass would emit in the loop's blocks, or nullopt when
    /// the colouring puts an operand there outside the encodable file, where no
    /// bank word exists to count.
    std::optional<uint32_t> switches;
};

/// One entry per loop in \p loops, in the same order, counting the switches
/// InsertVgprMsbPass would emit in that loop's blocks if \p colouring were applied.
///
/// A lifted operand is read at the register \p colouring gives its first unit;
/// anything else keeps the register it names. Blocks belonging to several nested
/// loops count towards each of them. \p vgprIndexCount is how many VGPRs an
/// operand can encode, past which the loop reports nullopt. \p policy is the
/// pass's: which words serve an instruction (AsmCapsConfig).
STINKYTOFU_EXPORT std::vector<LoopVgprMsbSwitches> predictLoopVgprMsbSwitches(
    const Function& function, const AllocationResult& colouring, const std::vector<Loop>& loops,
    uint32_t vgprIndexCount, VgprMsbPolicy policy = VgprMsbPolicy::ExactWord);

}  // namespace stinkytofu
