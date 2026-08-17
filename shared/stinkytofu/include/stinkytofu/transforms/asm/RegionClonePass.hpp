// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/pipeline/CloneSpec.hpp"

namespace stinkytofu {
class Pass;

/// Generic region-clone pass.
///
/// Each entry in `cloneList` declares one region-clone job:
///   - startLabel : BB label naming the region's first BB
///   - name       : recipe key dispatched to a per-kind post-clone transform
///                  inside the pass (e.g. "InitCIterWmma" -> zero acc2 on
///                  chain-head MFMAs in the clone)
///
/// The region end is computed in-pass as the last chain head (the last MFMA
/// whose src C acc first appears while scanning forward from startBB).
///
/// For each region the pass: splits the boundary BB after the end inst, clones
/// the [startBB .. boundary] region physically before startBB, rewrites
/// intra-region branches to cloned labels, applies the per-kind transform,
/// appends a branch from the cloned tail to the post-boundary BB, and reroutes
/// the pre-region forward entries to the cloned region. Returns
/// PreservedAnalyses::none(); a CFGBuilder pass should run after.
///
/// Empty cloneList -> no-op.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRegionClonePass(std::vector<CloneSpec> cloneList);

}  // namespace stinkytofu
