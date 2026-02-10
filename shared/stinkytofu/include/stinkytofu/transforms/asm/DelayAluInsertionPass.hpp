/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#include <memory>

namespace stinkytofu
{
    struct Pass;

    /// Creates a DelayAluInsertionPass for RDNA3 (gfx11xx) architectures
    ///
    /// This pass inserts s_delay_alu instructions to handle ALU instruction
    /// dependencies on RDNA3 (gfx11xx) architectures where ALU pipelines
    /// are non-blocking.
    ///
    /// Example:
    ///   v_mul_f32 v0, v1, v2
    ///   s_delay_alu instid0(VALU_DEP_1)  <-- inserted by this pass
    ///   v_add_f32 v3, v0, v4             # uses v0, needs delay
    ///
    /// The pass:
    /// - Classifies instructions as VALU, SALU, or TRANS
    /// - Tracks register def-use chains (up to 5 instructions back)
    /// - Computes minimal wait counts based on dependency distance
    /// - Inserts s_delay_alu with proper encoding:
    ///   * VALU: max 4 instructions lookback
    ///   * SALU: max 1 instruction lookback
    ///   * TRANS: max 3 instructions lookback
    ///
    /// Architecture gating: Only runs for gfx11xx (RDNA3: gfx1100, gfx1150, gfx1151, etc.)
    std::unique_ptr<Pass> createDelayAluInsertionPass();

} // namespace stinkytofu
