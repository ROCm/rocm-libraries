// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class BasicBlock;
struct HWModel;
struct StinkyInstruction;

/// v_nops InsertCoexecHazardPass puts in front of one consumer.
struct CoexecNopInsertion {
    const StinkyInstruction* before = nullptr;
    int count = 0;
};

using BlockSequences = std::unordered_map<const BasicBlock*, std::vector<const StinkyInstruction*>>;

/// Plan InsertCoexecHazardPass's v_nops for `blocks`, in that order. A consumer's backward
/// scan runs through its block's sequence in `seqs` and on into every predecessor's. Each
/// planned v_nop is written into `seqs` as `vnop`, so later scans count it as a filler, as
/// they count the pass's own v_nops. Every block a scan can reach needs a sequence.
/// Returns the insertions of each block of `blocks`.
STINKYTOFU_EXPORT std::vector<std::vector<CoexecNopInsertion>> planCoexecNops(
    const HWModel& hw, const std::vector<const BasicBlock*>& blocks, BlockSequences& seqs,
    const StinkyInstruction& vnop);

}  // namespace stinkytofu
