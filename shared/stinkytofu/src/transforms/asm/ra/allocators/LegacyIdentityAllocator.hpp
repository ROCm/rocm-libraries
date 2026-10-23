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

// Identity colouring behind the allocator interface.
//
// Wraps createLegacyColoring(). All capabilities are false. This is the first
// implementation so the seam is exercised before a real policy exists, and it
// stays the report baseline: lift, this allocator, apply must leave the physical
// program exactly as it was written.
//
// Private to the library: callers select a policy by name through
// AllocatorRegistry, so the concrete type is visible only here and to the
// white-box unit tests that exercise the policy directly.

#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"

namespace stinkytofu {

class LegacyIdentityAllocator : public RegisterAllocator {
   public:
    const char* name() const override;
    AllocatorCapabilities capabilities() const override;
    Expected<AllocationResult> allocate(const AllocationContext& context) override;
};

}  // namespace stinkytofu
