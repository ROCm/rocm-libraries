// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Sinks a ds_load that overwrites registers a recent WMMA read until InsertWaitAluPass needs no
/// va_vdst wait for it, without moving any WMMA. A refill moves only later, inside its window:
/// after the WMMA count that hides the wait, before the WMMA that reads its data needs it, and
/// within its stage. A move is kept only when the block owes fewer WMMAs and no vm_vsrc wait
/// gets stricter. Runs after the DAG scheduler and before wait-count insertion, under expert
/// scheduling mode 2. The policy is in docs/developer/ds-refill-placement.md.
struct DsRefillSinkOptions {
    /// Report every waiting ds_load, its window and the slot the policy picks, to the
    /// "DsRefillSinkPass" debug stream, and change nothing.
    bool reportOnly = false;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createDsRefillSinkPass(DsRefillSinkOptions options = {});

}  // namespace stinkytofu
