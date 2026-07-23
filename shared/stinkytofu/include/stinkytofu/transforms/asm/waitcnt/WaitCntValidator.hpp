// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Read-only validator that checks the s_wait_* instructions ALREADY present in
// a Function actually satisfy every register (VGPR/SGPR/acc) and LDS (memtoken)
// data dependency. It is the C++ counterpart of the reference tool
// shared/stinkytofu/tools/waitcnt-check/waitcnt_check.py.
//
// Unlike WaitDataflow (which COMPUTES and INSERTS waits), this validator never
// mutates the wait plan: it replays the CFG with a per-pred in-flight queue per
// counter, drains those queues only when it encounters a real s_wait_*
// instruction, and flags any consumer that still reads an in-flight async
// producer. It reuses WaitDataflow's per-pred queue model and the shared
// required-wait computation in WaitCntDataflowUtils.

#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace stinkytofu {
class BasicBlock;
class Function;
struct StinkyInstruction;

namespace waitcnt {

/// A single "missing wait" violation: `consumer` reads `producer` (an async
/// memory op tracked on `counter`) that is still in flight because no adequate
/// s_wait_* drains it beforehand.
struct WaitValidationViolation {
    CounterKind counter = CK_Count;
    StinkyInstruction* consumer = nullptr;
    StinkyInstruction* producer = nullptr;
    std::string message;
};

/// Verifies that the existing s_wait_* instructions satisfy every register /
/// LDS data dependency in a Function.
///
/// Preconditions: CFG built and buildUseDefChain(includePseudo=true) run, so
/// register RAW and memtoken (LDS pseudo-reg) dependencies both appear as SSA
/// def-use edges.
class STINKYTOFU_EXPORT WaitCntValidator {
   public:
    /// @param numWaves GemmTileConfig NumWaves (0 == multi-wave: the tensor
    ///                 counter only drains at barriers).
    explicit WaitCntValidator(unsigned numWaves = 0) : numWaves(numWaves) {}

    /// Validate `func` over `rpo` (reverse post-order block list). Returns at
    /// most one violation per counter kind (empty == all deps satisfied).
    std::vector<WaitValidationViolation> validate(Function& func,
                                                  const std::vector<BasicBlock*>& rpo);

   private:
    unsigned numWaves;
};

}  // namespace waitcnt
}  // namespace stinkytofu
