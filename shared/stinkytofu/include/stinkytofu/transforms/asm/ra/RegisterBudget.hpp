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

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {

class Function;

/// Highest index of \p regClass named anywhere in \p function, plus one, or 0
/// when the class is unused. Counts every operand rather than only allocated
/// values, so a register the allocator never saw still counts.
STINKYTOFU_EXPORT uint32_t highestRegisterCount(const Function& function, RegType regClass);

/// SGPR count \p function must declare.
///
/// The maximum of what the kernel uses and what the hardware writes before the
/// first instruction: \p numSgprPreload preloaded kernargs plus the two for the
/// kernarg segment pointer, then one per enabled entry of \p workgroupIds. The
/// floor is the part worth having a function for -- a preloaded argument the
/// kernel never reads appears in no operand, so a count taken only from usage
/// can declare fewer registers than the dispatch will fill.
STINKYTOFU_EXPORT uint32_t requiredSgprCount(const Function& function, int numSgprPreload,
                                             const std::array<int, 3>& workgroupIds);

}  // namespace stinkytofu
