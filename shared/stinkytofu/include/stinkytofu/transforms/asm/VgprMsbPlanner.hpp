// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
struct StinkyInstruction;

/// One s_set_vgpr_msb InsertVgprMsbPass emits.
struct VgprMsbInsertion {
    /// Index, in the planned sequence, of the instruction it goes in front of.
    size_t before = 0;
    /// The s_set_vgpr_msb operand; Msb16 packs the previous state into bits [15:8].
    int immediate = 0;
    /// The new state: src0/src1/src2/dst MSBs in the low byte.
    int requiredMsb = 0;
    /// The first switch after a label: an s_nop 0 goes in front of it.
    bool nopFirst = false;
};

/// The bank switches InsertVgprMsbPass puts into one basic block, in order. `insts` is
/// the block's instructions in order, labels and other pseudo instructions included; the
/// state starts unknown, as at every block start. Only reads the instructions, so it can
/// plan an order that is not in the IR.
STINKYTOFU_EXPORT std::vector<VgprMsbInsertion> planVgprMsb(
    const std::vector<const StinkyInstruction*>& insts, VgprMsbMode mode);

}  // namespace stinkytofu
