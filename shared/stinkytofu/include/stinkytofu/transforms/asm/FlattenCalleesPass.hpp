// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class Function;

/// Re-merge every callable function back into the entry function at its ASM
/// placement marker (a single flat function again), by moving each function body
/// to its FUNCTION_ASM_PLACEMENT_MARKER. \p functions is the whole-kernel list
/// (entry + callable functions, e.g. StinkyAsmModule::getFunctions); function
/// bodies are resolved by name. The emptied callable Functions are left in place.
///
/// WARNING: This is a temporary workaround that destroys the multi-function
/// structure. Do NOT build on it. A pass that truly cares about the final asm
/// stream order across callable functions must instead honor the
/// FUNCTION_ASM_PLACEMENT_MARKER markers directly — walk each function and, at
/// each marker, account for the named function body at that point — rather than
/// relying on this flatten. Remove this pass once SwInstructionPrefetchRelStaticPass (and
/// any other byte-layout-sensitive pass) does that.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createFlattenCalleesPass(
    std::vector<Function*> functions = {});

}  // namespace stinkytofu
