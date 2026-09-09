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

// Weighted first-fit colouring with eviction, registered as "greedy".
//
// Heavier ranges are placed first and may evict lighter ones, which is what
// makes this greedy rather than linear assignment. It never splits a range and
// never spills: a value with no candidate left fails the function, and the
// driver then leaves the IR exactly as the lifter left it.
//
// Private to the library: callers select a policy by name through
// AllocatorRegistry, so the concrete type is visible only here and to the
// white-box unit tests that exercise the policy directly.

#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"

namespace stinkytofu {

class GreedyAllocator : public RegisterAllocator {
   public:
    const char* name() const override;
    AllocatorCapabilities capabilities() const override;
    Expected<AllocationResult> allocate(const AllocationContext& context) override;
};

/// The same policy with hints disabled, registered as "greedy-compact".
///
/// Placement packs from the bottom instead of preferring the register a value was
/// lifted from, so this is the only variant that can lower the high-water mark -
/// the number occupancy is computed from. It exists to measure how much a kernel's
/// numbering could tighten; it is not the default, because renumbering everything
/// makes a shadow diff against the producer unreadable and hands the post-RA
/// hazard passes a denser schedule to repair.
class CompactingGreedyAllocator : public RegisterAllocator {
   public:
    const char* name() const override;
    AllocatorCapabilities capabilities() const override;
    Expected<AllocationResult> allocate(const AllocationContext& context) override;
};

}  // namespace stinkytofu
