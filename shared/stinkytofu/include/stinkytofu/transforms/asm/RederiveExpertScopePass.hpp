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
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// Re-derive the \p scope group's instruction range from the live IR before a
/// downstream single-region ScopeAdaptor consumes it.
///
/// An earlier multi-region adaptor (e.g. {loopWithPrefetch, noLoadLoopBody}) can
/// mutate the single flat BB — its passes (DAG scheduler / RemoveNop) delete and
/// reorder instructions. Its splice-back only refreshes the ranges of the groups
/// it owns; a later, overlapping group (e.g. expertScheduleMode2, which shares
/// the noLoadLoopBody end boundary) keeps stale [first, last] iterators that can
/// dangle (point at deleted nodes). ScopeAdaptor::moveIRToBlock then walks off
/// the list -> SIGSEGV.
///
/// This pass recomputes \p scope's range from durable landmarks in the current
/// IR: \p startLabel for the begin (it precedes the mutated region and is never
/// deleted), and the refreshed last instruction of \p endGroup for the end
/// (which \p scope shares). If either landmark is missing the range is cleared so
/// findGroupRange() returns nullopt and the downstream adaptor safely no-ops
/// instead of dereferencing a dangling range.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRederiveExpertScopePass(StinkyAsmModule& module,
                                                                      std::string scope,
                                                                      std::string startLabel,
                                                                      std::string endGroup);

}  // namespace stinkytofu
