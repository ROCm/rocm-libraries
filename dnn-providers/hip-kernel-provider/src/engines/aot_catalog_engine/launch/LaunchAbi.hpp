// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Op-agnostic launch ABI: bind an args_signature to concrete values, pack them
// into a natural-alignment kernarg buffer, and evaluate the symbolic grid.
// HIP-free by design (the actual hipModuleLaunchKernel lives in CatalogPlan).
// Forked near-verbatim from PR #9207's plans/LaunchAbi.{hpp,cpp}
// (rocke_client::launch), retargeted onto the generic catalog:: types.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "catalog/CatalogTypes.hpp"

namespace aot_catalog_engine::launch
{

using catalog::GridFormula;
using catalog::KernelArgument;
using catalog::LaunchBindings;
using catalog::ScalarValue;
using catalog::WorkspaceExpr;

// Grid symbol table: symbol name (e.g. "M", "BN") -> integer value, built by
// the op adapter from the decoded problem.
using SymbolTable = std::unordered_map<std::string, int64_t>;

// Resolves a device-buffer uid to its raw device pointer. Supplied by execute()
// from the runtime device-buffer map.
using PointerResolver = std::function<uint64_t(int64_t uid)>;

// Bind each args_signature entry, in order, to a concrete value:
//   POINTER -> uint64_t   (from pointerValues, else pointerUids via resolver)
//   SCALAR  -> float (F32) or int64_t (I32/I64), from the scalars map
// Fails closed (throws HipdnnPluginException) if any argument is unresolved.
std::vector<ScalarValue> bindArgs(const std::vector<KernelArgument>& signature,
                                  const LaunchBindings& bindings,
                                  const PointerResolver& resolvePointer);

// Pack bound values into a flat kernarg buffer using natural alignment: each
// argument is preceded by `(size - (offset % size)) % size` padding bytes,
// where size is the argument width (ptr/i64=8, i32/f32=4). `bound` must be the
// output of bindArgs for the same signature (same length/order).
std::vector<std::byte> packArgs(const std::vector<KernelArgument>& signature,
                                const std::vector<ScalarValue>& bound);

// Evaluated 3D launch grid (block counts).
struct Grid
{
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

// Evaluate each grid axis over `symbols`. Throws if a referenced symbol is
// missing or an axis evaluates negative.
Grid evalGrid(const GridFormula& formula, const SymbolTable& symbols);

// Evaluate a workspace-size expression over `symbols` (the kernel's grid
// symbols plus, when the dtype is known, `elem_size`). Throws (fails closed) if
// a referenced symbol is missing, a divisor/alignment is zero, or the result is
// negative. Returns the byte count of scratch the kernel needs for this problem.
int64_t evalWorkspace(const WorkspaceExpr& expr, const SymbolTable& symbols);

} // namespace aot_catalog_engine::launch
