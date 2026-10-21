// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Placement in program order, largest block first, and loop-banked.
//
// More policies for the shared engine in GreedyAllocator.cpp. Each keeps the
// weight policy's early phase, so capped blocks still claim the one bank they
// can reach before anything that could have gone anywhere, and borrows an
// existing policy's answer to who may evict whom. The first two only change
// the queue order; loopbank keeps weight order and banks the loop values.

#include <cstdint>
#include <memory>

#include "GreedyAllocator.hpp"
#include "GreedyPlacement.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"

namespace stinkytofu {
namespace {

/// The block that becomes live first goes first; the wider one among equal
/// starts, so a tuple is laid down before the singles that begin beside it.
bool programPlacesBefore(const Block& lhs, const Block& rhs) {
    if (lhs.start != rhs.start) return lhs.start < rhs.start;
    if (lhs.width != rhs.width) return lhs.width > rhs.width;
    return lhs.leader < rhs.leader;
}

/// The block that needs the most register-time goes first.
bool largestPlacesBefore(const Block& lhs, const Block& rhs) {
    const uint64_t lhsSize = static_cast<uint64_t>(lhs.width) * lhs.span;
    const uint64_t rhsSize = static_cast<uint64_t>(rhs.width) * rhs.span;
    if (lhsSize != rhsSize) return lhsSize > rhsSize;
    return lhs.leader < rhs.leader;
}

struct OrderedRegistrar {
    OrderedRegistrar() {
        AllocatorRegistry::registerAllocator("greedy-compact-program", [] {
            return std::make_unique<ProgramOrderedGreedyAllocator>();
        });
        AllocatorRegistry::registerAllocator("greedy-compact-largest", [] {
            return std::make_unique<LargestFirstGreedyAllocator>();
        });
        AllocatorRegistry::registerAllocator(
            "greedy-compact-loopbank", [] { return std::make_unique<LoopBankGreedyAllocator>(); });
    }
};
static OrderedRegistrar s_orderedRegistrar;

}  // namespace

/// Evicts as the freedom order does: a block with fewer places to go displaces
/// one with more. A block that starts late cannot otherwise reclaim a register
/// an earlier, more mobile block took first.
const PlacementPolicy& programPolicy() {
    static const PlacementPolicy policy{"program", weightPolicy().placesEarly, programPlacesBefore,
                                        freedomPolicy().mayEvict};
    return policy;
}

/// Evicts by weight, as LLVM's greedy allocator evicts by spill weight.
const PlacementPolicy& largestPolicy() {
    static const PlacementPolicy policy{"largest", weightPolicy().placesEarly, largestPlacesBefore,
                                        weightPolicy().mayEvict};
    return policy;
}

/// Weight order for whatever the banked phase leaves to the queue, and its
/// eviction rule throughout.
const PlacementPolicy& loopBankPolicy() {
    static const PlacementPolicy policy{"loopbank", weightPolicy().placesEarly,
                                        weightPolicy().placesBefore, weightPolicy().mayEvict,
                                        /*banksLoopValues=*/true};
    return policy;
}

const char* ProgramOrderedGreedyAllocator::name() const {
    return "greedy-compact-program";
}

AllocatorCapabilities ProgramOrderedGreedyAllocator::capabilities() const {
    return {};
}

Expected<AllocationResult> ProgramOrderedGreedyAllocator::allocate(
    const AllocationContext& context) {
    return runGreedyPlacement(context, /*followHints=*/false, programPolicy());
}

const char* LargestFirstGreedyAllocator::name() const {
    return "greedy-compact-largest";
}

AllocatorCapabilities LargestFirstGreedyAllocator::capabilities() const {
    return {};
}

Expected<AllocationResult> LargestFirstGreedyAllocator::allocate(const AllocationContext& context) {
    return runGreedyPlacement(context, /*followHints=*/false, largestPolicy());
}

const char* LoopBankGreedyAllocator::name() const {
    return "greedy-compact-loopbank";
}

AllocatorCapabilities LoopBankGreedyAllocator::capabilities() const {
    return {};
}

Expected<AllocationResult> LoopBankGreedyAllocator::allocate(const AllocationContext& context) {
    return runGreedyPlacement(context, /*followHints=*/false, loopBankPolicy());
}

// Without this the linker drops the translation unit from the static library,
// the registrar above never runs, and the names simply do not exist.
void anchorOrderedGreedyAllocators() {}  // NOLINT(misc-use-internal-linkage)

}  // namespace stinkytofu
