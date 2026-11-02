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

// Placement ordered by how constrained a block is, rather than by how hot it is.
//
// The engine is shared with the weight-ordered policy and lives in
// GreedyAllocator.cpp; everything specific to this order is here.

#include "GreedyAllocator.hpp"
#include "GreedyPlacement.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"

namespace stinkytofu {
namespace {

/// Nothing goes ahead of the queue, because the order already says it.
///
/// Weight ordering needs a phase for capped blocks, since a ceiling makes a
/// block compete for one bank against the whole file and weight cannot express
/// that. Here a ceiling is simply a small placementFreedom, so the phase would
/// be a second way of saying what the comparator already says.
bool freedomPlacesEarly(const Block&) {
    return false;
}

/// Fewest legal bases first, then the weight order among equals.
///
/// Freedom counts the bases a block could take before occupancy is considered,
/// so it ranks by how little room a block has to be wrong about. Weight still
/// decides between blocks equally free, which is most of them: an unconstrained
/// register has the whole file, so singles tie and sort exactly as before.
bool freedomPlacesBefore(const Block& lhs, const Block& rhs) {
    if (lhs.placementFreedom != rhs.placementFreedom)
        return lhs.placementFreedom < rhs.placementFreedom;
    if (lhs.weight != rhs.weight) return lhs.weight > rhs.weight;
    return lhs.leader < rhs.leader;
}

/// A block with fewer places to go displaces one with more.
///
/// The weight policy refuses to move a hotter occupant, which is why a hot
/// single sitting inside an otherwise-free run can keep a wide range out of it
/// for good. Freedom answers the question the situation actually poses: which
/// of the two has somewhere else to be.
///
/// Termination needs no depth limit. Freedom strictly decreases along an
/// eviction chain, so the relation is acyclic, and the engine still caps how
/// often any one block may be evicted.
bool freedomMayEvict(const Block& evictor, const Block& occupant) {
    if (occupant.placementFreedom != evictor.placementFreedom)
        return occupant.placementFreedom > evictor.placementFreedom;
    return occupant.weight < evictor.weight;
}

struct FreedomRegistrar {
    FreedomRegistrar() {
        AllocatorRegistry::registerAllocator("greedy-compact-freedom", [] {
            return std::make_unique<FreedomOrderedGreedyAllocator>();
        });
    }
};
static FreedomRegistrar s_freedomRegistrar;

}  // namespace

const PlacementPolicy& freedomPolicy() {
    static const PlacementPolicy policy{"freedom", freedomPlacesEarly, freedomPlacesBefore,
                                        freedomMayEvict};
    return policy;
}

const char* FreedomOrderedGreedyAllocator::name() const {
    return "greedy-compact-freedom";
}

AllocatorCapabilities FreedomOrderedGreedyAllocator::capabilities() const {
    // Ordering changes which register a value gets, not what lowering has to do
    // about it, so this claims exactly what the other greedy policies claim.
    return {};
}

Expected<AllocationResult> FreedomOrderedGreedyAllocator::allocate(
    const AllocationContext& context) {
    return runGreedyPlacement(context, /*followHints=*/false, freedomPolicy());
}

// Without this the linker drops the translation unit from the static library,
// the registrar above never runs, and the name simply does not exist.
void anchorFreedomOrderedGreedyAllocator() {}  // NOLINT(misc-use-internal-linkage)

}  // namespace stinkytofu
