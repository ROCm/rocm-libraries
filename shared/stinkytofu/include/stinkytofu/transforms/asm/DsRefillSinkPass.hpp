// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Reports, for every ds_load InsertWaitAluPass would give a va_vdst wait, the refill window it
/// has: from the WMMA count that hides the wait to the WMMA that reads its data, within its
/// stage. The report goes to the "DsRefillSinkPass" debug stream; the IR is not changed. Runs
/// after the DAG scheduler, under expert scheduling mode 2. The policy is in
/// docs/developer/ds-refill-placement.md.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createDsRefillSinkPass();

}  // namespace stinkytofu
