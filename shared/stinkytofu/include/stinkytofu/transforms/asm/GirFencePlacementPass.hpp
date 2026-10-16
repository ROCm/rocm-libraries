/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <memory>
#include <set>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"

namespace stinkytofu {
class Pass;
class Function;
struct StinkyInstruction;

/// FrameMap-derived mandatory RAW publication points before physical GIR fences exist.
///
/// `markers` contains every counter kind. `tensorMarkers` is the subset whose producer runs on
/// the tensor counter and therefore becomes an `s_wait_tensorcnt` when it survives the complete
/// fence/wait plan. The scheduler uses only this frame-derived identity; instruction proximity and
/// LDS token names are not substitutes.
struct GirRawPlacementPlan {
    std::set<const StinkyInstruction*> markers;
    std::set<const StinkyInstruction*> tensorMarkers;
};

STINKYTOFU_EXPORT GirRawPlacementPlan
computeGirRawPlacementPlan(Function& function, const GirFrameAnalysis::Result& frames,
                           const GirFrameHazardAnalysis::Result& hazards);

/// Moves each GIR fence inside its admissible window to the slot with the deepest counter
/// residual, so the wait the next pass anchors on it drains as little as possible.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGirFencePlacementPass();

}  // namespace stinkytofu
