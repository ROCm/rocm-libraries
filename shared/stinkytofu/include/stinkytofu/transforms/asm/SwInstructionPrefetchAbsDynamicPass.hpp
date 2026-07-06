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
/// **Regime:** the **post-CP region** — any kernel with `totalLayoutBytes > P(0)=32640`
/// (CP preload covers only `[0, P(0))`). This is **not** gated on the 64 KiB I-cache size;
/// the prefetch targets are chosen by the global-write branch logic (GSU / beta dispatch),
/// so the policy applies whenever post-CP code exists, regardless of total kernel size. See
/// SwInstructionPrefetchAbsDynamic-CfgTarget-Design.md §10–§12.
///
/// **Status (D1): DETECTOR + Variant-1 EMISSION (enabled).** Always runs the §10.3 CFG-target
/// detector (read-only debug dump) for the post-CP region. When `totalLayoutBytes > 65536` and a
/// reserved `baseSgpr` is available, it also EMITS the predicated GSU→beta prefetch ladder
/// (getpc + `s_add_i32 label,4` + `s_prefetch_inst`) immediately after `label_MultiGemmEnd`.
/// Basic 3-case model (GSU/beta) only; it bails (no emission) for Stream-K, GSU0 (undefined
/// `sgprGSU`), and no-beta kernels — MBSK-reduction / f64 / activation remain D2+ (flagged).
///
/// No-op cases (no emission):
///   - `totalLayoutBytes <= 32640`        : whole kernel fits the CP window; no post-CP region.
///   - `32640 < totalLayoutBytes <= 65536`: static regime (abs static emits); detector still dumps.
///   - `baseSgpr < 0`                     : no reserved SGPR pair → detector-only.
///   - Stream-K / GSU0 / no-beta          : unsupported dispatch → detector-only.
///
/// Mutually exclusive with the PC-rel passes — do not run together. Regime split with the abs
/// static pass: static owns `(32640, 65536]`, this pass owns `> 65536`.
/// Debug output: `sw_prefetch_abs_dynamic_pass.txt`.

/// \p baseSgpr  Low index of the reserved 64-bit SGPR pair. Pass -1 to no-op.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    int baseSgpr, const std::string& debugOutputPath = {});

/// Overload that reads base SGPR and debug path from \p module options:
/// `SwInstructionPrefetchAbsBaseSgpr` and `StinkyTofuCostOutputDir`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    StinkyAsmModule& module);

}  // namespace stinkytofu
