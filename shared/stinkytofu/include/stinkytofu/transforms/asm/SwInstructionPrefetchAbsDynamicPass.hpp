/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsStaticPass.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// SW instruction prefetch — absolute address, dynamic-kernel policy.
///
/// **Regime:** `totalLayoutBytes > 65536` (kernel larger than the design
/// ~64 KiB I-cache; software prefetch competes with streaming execution, so
/// sites must be replacement-aware — per-k targets at `align128(P(k))`,
/// CFG-aware sites in loop preheaders, capped `MaxAheadBytes`). See §16.4
/// Pass B of SwPrefetchAbsInsertionPass-Design.md.
///
/// **Status (Phase P2): STUB.** This pass is registered alongside the static
/// pass so the single user knob `EnableSwInstructionPrefetchAbs` covers both,
/// but the dynamic policy is **not yet implemented**. It no-ops for every
/// kernel size and, when `totalLayoutBytes > 65536`, emits a debug log noting
/// that the dynamic pass is not implemented. No `s_prefetch_inst` / getpc is
/// inserted. The real per-k + CFG implementation lands in a later PR (P2/P3).
///
/// No-op cases:
///   - `totalLayoutBytes <= 32640`        : CP preload covers everything.
///   - `32640 < totalLayoutBytes <= 65536`: static regime — handled by
///                                          `SwInstructionPrefetchAbsStaticPass`.
///   - `totalLayoutBytes > 65536`         : dynamic regime — not implemented yet.
///   - `baseSgpr < 0`                     : no reserved SGPR pair.
///
/// Mutually exclusive with the PC-rel passes — do not run together.
/// Debug output: `sw_prefetch_abs_dynamic_pass.txt`.

/// \p baseSgpr  Low index of the reserved 64-bit SGPR pair. Pass -1 to no-op.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    int baseSgpr, const std::string& debugOutputPath = {});

/// Overload that reads base SGPR and debug path from \p module options:
/// `SwInstructionPrefetchAbsBaseSgpr` and `StinkyTofuCostOutputDir`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    StinkyAsmModule& module);

}  // namespace stinkytofu
