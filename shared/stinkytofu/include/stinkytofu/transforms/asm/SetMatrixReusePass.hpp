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
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class Function;

/// Sets matrix_a_reuse / matrix_b_reuse on matrix (WMMA/MFMA) instructions from
/// consecutive MMA operand equality in final program order (post-scheduler).
///
/// Reuse is a per-function microarchitectural promise: the chain never spans a
/// call site (the callee may clobber the operand-reuse buffer) and never spans
/// a function boundary.
///
/// When \p functions is non-empty the pass walks that whole-kernel list (entry
/// function plus callees), processing each function in isolation. When empty it
/// processes only the single Function the pipeline runs it on.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSetMatrixReusePass(
    std::vector<Function*> functions = {});

}  // namespace stinkytofu
