// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
/// (`insertSwPrefetchLabelsDynamicPerBbAnchor`). When false, uses global `32640 + k×4096`.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchRelDynamicPass(
    const std::string& debugOutputPath, bool usePerBbAnchorPrefetchGrid = true);

/// Debug output: `<outputDir>/<kernel_basename>/sw_inst_prefetch_rel_dynamic_pass.txt`
STINKYTOFU_EXPORT std::unique_ptr<Pass> createSwInstructionPrefetchRelDynamicPass(
    StinkyAsmModule& module, bool usePerBbAnchorPrefetchGrid = true);

}  // namespace stinkytofu
