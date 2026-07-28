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

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// CFG-aware PC-rel SW prefetch: Phase 1 accumulate + Phase 2 CFG-gated insert.
/// Gated at P(0) = 32640. Shares enable with static pass
/// (`EnableSwInstructionPrefetchRelStatic`).
///
/// \p usePerBbAnchorPrefetchGrid When true (default), Phase 2 uses per-BB anchor grid
/// (`insertSwPrefetchLabelsDynamicPerBbAnchor`, §15). When false, uses global `32640 + k×4096`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchRelDynamicPass(
    const std::string& debugOutputPath, bool usePerBbAnchorPrefetchGrid = true);

/// Debug output: `<outputDir>/<kernel_basename>/sw_inst_prefetch_rel_dynamic_pass.txt`
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchRelDynamicPass(
    StinkyAsmModule& module, bool usePerBbAnchorPrefetchGrid = true);

}  // namespace stinkytofu
