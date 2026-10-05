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

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Repair matrix co-execution after wait insertion.
///
/// Later passes insert instructions into a schedule the DAG scheduler built
/// against a hardware co-execution model: final waits in front of their matrix
/// ops, and later still the s_set_vgpr_msb bank switches. Where that leaves more
/// work in front of a matrix op than its predecessor's window holds, the excess
/// delays it. This pass keeps the scheduler's order and moves only that excess,
/// past the matrix op into the next window, plus VALU work into the window of a
/// dependent pair, where it replaces spacers.
///
/// The wait contract is preserved exactly. Waits are kept out of the DAG and
/// re-emitted immediately before their original anchors with their immediates
/// untouched, and no instruction crosses a segment boundary. No hazard gap of
/// kCdna5HazardRules ends up shorter than both its input and the rule's
/// distance; a segment that would shorten one keeps its input order.
///
/// See docs/developer/repair-matrix-coexec-pass.md.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRepairMatrixCoexecPass();

}  // namespace stinkytofu
