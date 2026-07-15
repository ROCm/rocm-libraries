// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "dispatcher/AotInstance.hpp"

// Generic, HIP-free and op-agnostic kernel-launch ABI helpers: symbolic grid
// evaluation, flat argument-buffer packing, and signature binding. Kept in their
// own translation unit so both the plan (execution) and CPU unit tests exercise
// the same logic without a GPU. An op adapter produces a dispatcher::LaunchBindings
// (kernel-arg values keyed by ABI name) and a grid symbol table; these helpers
// consume them without any op-specific knowledge.
namespace rocke_client::launch
{

using dispatcher::ScalarValue;

// Bind `signature` to concrete launch values from `bindings`, resolving each
// pointer argument's tensor uid to a device address through `devicePtrs`. Every
// op-specific name lives in `bindings`; this is op-agnostic. Throws
// HipdnnPluginException on a signature argument absent from `bindings`, or a
// missing/null device buffer for a bound pointer.
std::unordered_map<std::string, ScalarValue>
    bindArgs(const std::vector<dispatcher::KernelArgument>& signature,
             const dispatcher::LaunchBindings& bindings,
             const std::unordered_map<std::int64_t, void*>& devicePtrs);

// Pack the kernel arguments described by `signature` into a flat launch buffer,
// inserting natural-alignment padding before each argument. Argument sizes come
// from the parsed ABI (dispatcher::argSizeBytes); the manifest loader has
// already validated kind/dtype, so packing trusts them. Throws
// HipdnnPluginException on a value missing for a signature argument.
std::vector<std::byte> packArgs(const std::vector<dispatcher::KernelArgument>& signature,
                                const std::unordered_map<std::string, ScalarValue>& values);

// Evaluate the symbolic grid formula against `symbols` (grid symbol name ->
// value). Throws HipdnnPluginException on an unknown symbol or a non-positive
// ceil_div denominator.
std::array<unsigned int, 3> evalGrid(const dispatcher::GridFormula& formula,
                                     const std::unordered_map<std::string, std::int64_t>& symbols);

} // namespace rocke_client::launch
