// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
class StinkyAsmModule;

/// Dumps a REDUCED view of the IR centered on memory-token structure: only block
/// labels + successor edges (branch structure), the memtoken-candidate instructions
/// (tensor_load_to_lds, s_wait_tensorcnt, ds read/write/atomic, barriers), and their
/// trailing comments. All other IR is elided, so token/wait-group flow across the CFG
/// is legible without wading through full assembly.
struct DumpMemTokenIRStructurePassConfig {
    /// File to write the reduced dump to. If empty, nothing is written.
    std::string path;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createDumpMemTokenIRStructurePass(
    const StinkyAsmModule& module, DumpMemTokenIRStructurePassConfig config);
/// Per-function variant (no module): dumps only the function it is run on. Used by
/// the opt tool for standalone testing.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createDumpMemTokenIRStructurePass(
    DumpMemTokenIRStructurePassConfig config);
}  // namespace stinkytofu
