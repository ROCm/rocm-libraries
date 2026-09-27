// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// What a kernel must declare as its register count once operands have been
// rewritten.
//
// Reallocating registers invalidates `.amdhsa_next_free_sgpr` and the `.sgpr_count`
// metadata beside it: compaction lowers the highest index a kernel touches, and
// leaving the declaration at the producer's number is safe but throws away the
// occupancy the compaction was for. Everything else in the kernel descriptor is
// an ABI statement about what the hardware does before entry and must not move;
// see docs/developer/register-allocation.md.

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {

class Function;

/// Highest index of \p regClass named anywhere in \p function, plus one, or 0
/// when the class is unused. Counts every operand rather than only allocated
/// values, so a register the allocator never saw still counts.
STINKYTOFU_EXPORT uint32_t highestRegisterCount(const Function& function, RegType regClass);

/// First even index after every register of \p regClass already named in
/// \p function. A 64-bit SGPR pair has to start even, so this is where a
/// later pass can sit a new pair without overlapping compact's colouring.
STINKYTOFU_EXPORT uint32_t nextEvenRegisterBase(const Function& function, RegType regClass);

/// Even base of a \p width-wide SGPR block a pass inserting at KERNEL ENTRY may
/// write, from inside the range the kernel already uses so the declared count
/// does not grow: with s57 the highest and \p width 3, s[54:56]. \p limit caps it
/// from above, so two passes writing at entry can stack rather than overlap.
///
/// Sound only at entry, where nothing has run yet, so no SGPR above the
/// dispatch-filled line holds a value -- a pass emitting mid-kernel must not use
/// this. Nothing when no block fits, or when the function does not publish that
/// line, understating which is wrong code rather than a slower kernel; callers
/// fall back to nextEvenRegisterBase above.
STINKYTOFU_EXPORT std::optional<uint32_t> reusableEvenSgprBase(
    const Function& function, uint32_t width,
    uint32_t limit = std::numeric_limits<uint32_t>::max());

/// At least how many SGPRs the hardware writes before the first instruction:
/// \p numSgprPreload preloaded kernargs plus the two for the kernarg segment
/// pointer, then one per enabled entry of \p workgroupIds.
///
/// A floor only: with no preloaded kernargs the descriptor omits
/// .amdhsa_user_sgpr_count and those two go uncounted. Fine for the declared
/// count below, which maxes it against real usage.
STINKYTOFU_EXPORT uint32_t dispatchFilledSgprCount(int numSgprPreload,
                                                   const std::array<int, 3>& workgroupIds);

/// The same count where the descriptor settles it, and nothing where it does
/// not: with no preloaded kernargs it is unsettled whether the kernarg segment
/// pointer takes s[0:1] ahead of the workgroup ids.
///
/// The line between a register that arrives holding something and one merely
/// named early, which is what AllocationConstraints pins live-ins against.
/// Separate from the floor above because understating the line unpins a register
/// the dispatch wrote -- wrong code, not a missed optimisation.
STINKYTOFU_EXPORT std::optional<uint32_t> settledDispatchFilledSgprCount(
    int numSgprPreload, const std::array<int, 3>& workgroupIds);

/// SGPR count \p function must declare.
///
/// The maximum of what the kernel uses and what the dispatch fills. The floor is
/// the part worth having a function for -- a preloaded argument the kernel never
/// reads appears in no operand, so a count taken only from usage can declare
/// fewer registers than the dispatch will write.
STINKYTOFU_EXPORT uint32_t requiredSgprCount(const Function& function, int numSgprPreload,
                                             const std::array<int, 3>& workgroupIds);

/// VGPRs the dispatch writes before the first instruction, which is the
/// workitem id and nothing else.
///
/// \p vgprWorkItem is the descriptor's .amdhsa_system_vgpr_workitem_id, which
/// counts *extra* dimensions, so 0 means x alone. How many registers those
/// dimensions occupy is the architecture's business, not the field's:
/// \p packedWorkitemId targets deliver x, y and z inside v0 and so fill one
/// register whatever the field says, while unpacked targets take one register
/// per enabled dimension from v0 upward.
///
/// Takes the packing as a bool rather than an architecture because its caller
/// is the descriptor refresher, which tolerates a target this build does not
/// know. False there means the unpacked count, which over-declares by a
/// register or two instead of under-declaring.
STINKYTOFU_EXPORT uint32_t dispatchFilledVgprCount(int vgprWorkItem, bool packedWorkitemId);

/// The VGPRs the dispatch fills where the architecture settles it, and nothing
/// where it does not -- the vector counterpart of
/// settledDispatchFilledSgprCount, and read for the same purpose.
///
/// A packed target fills v0 alone, which is settled by \p arch with no help
/// from the descriptor. On an unpacked target the count needs
/// .amdhsa_system_vgpr_workitem_id, which the allocator cannot reach, so the
/// answer is nothing and every vector live-in stays pinned. Understating the
/// line unpins a register the dispatch wrote, which is wrong code.
STINKYTOFU_EXPORT std::optional<uint32_t> settledDispatchFilledVgprCount(GfxArchID arch);

/// VGPR count \p function must declare: the larger of what it uses and what
/// the dispatch fills. A kernel that never reads its workitem id names it in
/// no operand, so a count taken from usage alone can declare fewer registers
/// than the dispatch writes.
STINKYTOFU_EXPORT uint32_t requiredVgprCount(const Function& function, int vgprWorkItem,
                                             bool packedWorkitemId);

}  // namespace stinkytofu
