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

/// Re-merge every callee back into the entry function at its original in-stream
/// position (a single flat function again), by moving each callee's body to its
/// CALLEE_BODY marker. \p functions is the whole-kernel list (entry + callees,
/// e.g. StinkyAsmModule::getFunctions); callee bodies are resolved by name. The
/// emptied callee Functions are left in place.
///
/// WARNING: This is a temporary workaround that destroys the multi-function
/// structure. Do NOT build on it. A pass that truly cares about the final asm
/// stream order across callees must instead honor the CALLEE_BODY markers
/// directly — walk each function and, at each marker, account for the named
/// callee body at that point — rather than relying on this flatten. Remove this
/// pass once SwPrefetchInsertionPass (and any other byte-layout-sensitive pass)
/// does that.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createFlattenCalleesPass(
    std::vector<Function*> functions = {});

}  // namespace stinkytofu
