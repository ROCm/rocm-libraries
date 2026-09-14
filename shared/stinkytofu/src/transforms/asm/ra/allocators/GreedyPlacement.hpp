/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#pragma once

// What the greedy engine places, and the order it places them in.
//
// Internal to allocators/. The engine lives in GreedyAllocator.cpp and is not
// shared; what is shared is the block it works on and the policy it consults,
// so that a policy can be written in its own translation unit.

#include <cstdint>
#include <optional>
#include <vector>

#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"

namespace stinkytofu {

struct Member {
    SSAValueID value = kInvalidSSAValueID;
    uint32_t offset = 0;
};

/// One block of values placed as a unit, each at its own fixed offset.
struct Block {
    RegType regClass = RegType::UNKNOWN;
    std::vector<Member> members;
    uint32_t width = 1;
    double weight = 0.0;
    /// Smallest member ID, so ordering is stable when weights tie.
    SSAValueID leader = kInvalidSSAValueID;
    /// Base the original registers imply, tried before any first-fit candidate.
    std::optional<uint32_t> hintBase;
    /// Must land on hintBase rather than prefer it. Null when free to move;
    /// otherwise the reason, for the diagnostic if that register is unusable.
    const char* pinReason = nullptr;
    /// Tightest index any member may occupy, when an operand field limits one.
    std::optional<uint32_t> maxIndex;
    /// Bases this block could legally take, ignoring who currently holds them.
    ///
    /// Width, the placement rules and any index ceiling all shrink it, so it is
    /// one number for how constrained a block is. A single unconstrained
    /// register has the whole file; an 8-wide range on even bases has about half
    /// of it; a 2-wide range capped to the first bank has an eighth.
    uint32_t placementFreedom = 0;
    /// Soft pairings naming one of this block's members, by index into
    /// AllocationConstraints::preferences(). Empty for almost every block,
    /// which is what keeps placement on its first-fit path.
    std::vector<size_t> preferences;
    uint32_t evictions = 0;
    bool placed = false;
    uint32_t base = 0;
};

/// Which block the engine places next, and whom it may displace to do it.
///
/// Function pointers rather than virtuals: a policy is three small decisions
/// with no state, and this way each one lives entirely in its own file while
/// the engine branches on none of it.
///
/// Invariants that are not a matter of policy stay in the engine. It decides
/// that a pinned block never moves and that a block cannot be evicted past
/// kMaxEvictionsPerBlock; the policy only answers which of two blocks has the
/// stronger claim on a register.
struct PlacementPolicy {
    const char* name = "";

    /// Placed before the general queue, whatever order that queue is in.
    bool (*placesEarly)(const Block& block) = nullptr;

    /// Ordering within the general queue: true when \p lhs goes first.
    bool (*placesBefore)(const Block& lhs, const Block& rhs) = nullptr;

    /// Whether \p evictor has a stronger claim than \p occupant on a register
    /// they both want.
    bool (*mayEvict)(const Block& evictor, const Block& occupant) = nullptr;
};

/// Heaviest first, with capped blocks taken in a phase ahead of the queue.
/// Defined in GreedyAllocator.cpp.
const PlacementPolicy& weightPolicy();

/// Fewest legal bases first, which folds the capped phase into the order.
/// Defined in FreedomOrderedGreedyAllocator.cpp.
const PlacementPolicy& freedomPolicy();

/// Colour \p context, with \p policy deciding what is placed when.
///
/// The one way in to the engine, which stays private to GreedyAllocator.cpp.
/// \p followHints decides whether a block prefers the register it was lifted
/// from; without hints placement packs from the bottom, which is the only way
/// the high-water mark can come down.
Expected<AllocationResult> runGreedyPlacement(const AllocationContext& context, bool followHints,
                                              const PlacementPolicy& policy);

}  // namespace stinkytofu
