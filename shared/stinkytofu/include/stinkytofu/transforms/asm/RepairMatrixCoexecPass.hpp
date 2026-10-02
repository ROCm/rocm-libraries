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

/// Restore matrix co-issue after wait insertion, using the scheduler's own rules.
///
/// Later passes insert instructions into a schedule the DAG scheduler built
/// against a hardware co-issue model, and final waits in particular leave matrix
/// ops with nothing to issue in their latency shadow. This pass replays each
/// segment through the same architecture ready queue the scheduler uses, so the
/// rules live in one place: a new rule, or a new architecture, needs no change
/// here.
///
/// The wait contract is preserved exactly. Waits are kept out of the DAG and
/// re-emitted immediately before their original anchors with their immediates
/// untouched, and no instruction crosses a segment boundary.
///
/// Replaced the earlier repair pass, which targeted a slot count measured
/// against its own input rather than the hardware's capacity; see
/// docs/developer/repair-matrix-coexec-pass.md.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRepairMatrixCoexecPass();

}  // namespace stinkytofu
