// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "stinkytofu/Export.hpp"

namespace stinkytofu {

class PassManager;
class StinkyAsmModule;

/// Add the passes that lift a kernel to SSA and allocate its registers, as its
/// ModuleOptions ask: the mode from RegisterAllocation, every other setting
/// from RegisterAllocationConfig (see RegisterAllocationConfig.hpp), and dumps
/// under RegisterAllocationDump. Adds nothing in mode off.
///
/// Shared across backends; the architecture's allocation rules are looked up
/// from the function's triple. A backend adds these where its instruction order
/// is final and before any pass that reads physical register numbers
/// (docs/developer/register-allocation.md, "Pipeline placement").
///
/// The producer reads mode 3 as "let an over-budget kernel reach allocation and
/// re-judge afterwards", a policy that changes nothing here, so 3 adds what 2
/// adds. A configuration that does not parse adds no allocation, so the kernel
/// keeps the producer's registers, and a missed remark says why.
STINKYTOFU_EXPORT void addRegisterAllocationPasses(PassManager& pm, const StinkyAsmModule& module);

}  // namespace stinkytofu
