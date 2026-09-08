// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
