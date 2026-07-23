// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a read-only pass that validates the s_wait_* instructions present in
/// the Function actually satisfy every register / LDS data dependency. On a
/// missing wait the pass prints a diagnostic and aborts via report_fatal_error
/// (mirroring MemTokenConsistencyCheckPass). It does not insert or remove any
/// wait; buildUseDefChain(includePseudo=true) is run internally so register and
/// memtoken dependencies are visible as SSA edges.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createWaitCntValidationCheckPass();

}  // namespace stinkytofu
