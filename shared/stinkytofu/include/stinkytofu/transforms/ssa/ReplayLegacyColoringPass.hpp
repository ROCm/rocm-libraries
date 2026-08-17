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

// The identity colouring, and the round trip it exists to prove.
//
// Legacy replay assigns every SSA value the register it was lifted from, so
// lowering it must reproduce the original program exactly. That is the gate in
// front of enabling any real allocation: until it holds, a difference in
// emitted code cannot be attributed to allocation policy.

#include <memory>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/transforms/ssa/SSADestruction.hpp"

namespace stinkytofu {
class Function;
class Pass;

/// Rewrite \p function using each value's PhysicalBinding, undoing the lift
/// exactly.
STINKYTOFU_EXPORT SSADestructionResult replayLegacyColoring(Function& function);

/// Creates a pass that lowers attached SSA back to the registers it was lifted
/// from.
///
/// After a successful rewrite it clears attached SSA, which is what discards
/// value identity that described the pre-rewrite operands.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createReplayLegacyColoringPass();

}  // namespace stinkytofu
