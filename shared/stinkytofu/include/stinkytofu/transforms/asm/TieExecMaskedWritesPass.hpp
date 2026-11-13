// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Appends each vector destination written under a narrow exec mask to its own
/// instruction's srcRegs, so the lanes that write preserves are an operand the
/// allocator can tie the destination to. Leaves an operand already naming that
/// register alone, so the pass is idempotent.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createTieExecMaskedWritesPass();

}  // namespace stinkytofu
