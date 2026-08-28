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

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that collapses a function's basic blocks back into the single
/// flat entry block, undoing CFGBuilderPass.
///
/// This is the inverse of a CFG build, not a simplification: every IR node is
/// moved, none is deleted or rewritten, and blocks are concatenated in list
/// order, which CFGBuilderPass wrote in program order. Label instructions are
/// therefore still present in the flat stream and a later CFGBuilderPass splits
/// on them again.
///
/// It exists because a ScopeAdaptor extracts its regions by walking one
/// instruction list between a group's stored first/last nodes (see
/// ScopeAdaptor's flat-BB invariant). Once a CFG has split the kernel, those two
/// nodes sit in different blocks and the walk cannot reach the end. Any CFG
/// built before a region adaptor must be flattened before the adaptor runs.
///
/// Whole-function, like every pass that restructures blocks: it refuses to run
/// when basic-block filtering excludes a block, because a region adaptor's
/// temporary Function has a different entry.
///
/// Merging blocks invalidates block arguments and slot indexes, so attached SSA
/// is cleared. A shadow-mode register allocation upstream leaves its arena
/// attached, and that arena describes the block structure being dissolved here.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createFlattenCFGPass();

}  // namespace stinkytofu
