// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include "Context.hpp"

namespace hip_kernel_provider::core
{

/// What a built execution context runs, as far as the provider can name it.
struct PlanDescription
{
    /// The kernel config the plan launches: the kernel descriptor's name (its id when it has
    /// none), or `<slot>=<name>` per stage for a slotted engine, comma-separated. Empty for an
    /// engine without named configs, for no plan, and while autotune has not chosen yet.
    std::string configName;
    /// The plan benchmarks its candidates at the first execution and has not run yet.
    bool autotunePending = false;
};

/// Describes the plan @p context holds. After an autotuned plan's first execution, names the
/// config benchmarking chose.
PlanDescription describePlan(const Context& context);

} // namespace hip_kernel_provider::core
