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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"

namespace stinkytofu {
class Function;

struct SymbolOrigin {
    std::string name;
    int64_t producerIndex = 0;
    std::vector<int64_t> allocatedIndices;
    std::string note;
};

struct SymbolSyncReport {
    size_t stable = 0;
    size_t movedUniquely = 0;
    size_t split = 0;
    size_t unresolvable = 0;
    size_t namesCleared = 0;
    size_t setsRewritten = 0;
    /// Operands whose name never agreed with their index (placeholder references).
    std::vector<std::string> suspectOperands;
    /// Producer name -> allocation outcome, for the register-map comment.
    std::vector<SymbolOrigin> origins;
};

struct SymbolSyncOptions {
    /// Insert a register-map TEXTBLOCK at the entry block (see
    /// docs/developer/register-allocation.md §11.3).
    bool emitRegisterMap = false;
    /// Append per-instruction breadcrumbs when a name was stripped.
    bool emitBreadcrumbs = false;
};

/// Restore the emit invariant after `destroyAttachedSSA` rewrote operands.
///
/// Operands absent from \p rewritten are left exactly as the producer wrote them.
/// No-op when \p rewritten is empty.
void syncRegisterSymbols(Function& function, const std::vector<RewrittenOperand>& rewritten,
                         SymbolSyncOptions options = {}, SymbolSyncReport* report = nullptr);

}  // namespace stinkytofu
