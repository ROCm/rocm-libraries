/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that merges nearby barrier groups within loop bodies.
///
/// Intended to run immediately after StinkyDAGSchedulerPass. After scheduling,
/// two (or more) barrier groups (an s_barrier_signal / s_barrier_wait pair that
/// share a memory-token set) can end up separated by only a handful of cycles.
/// When the modeled cycle-distance between two consecutive groups is smaller
/// than the merge threshold (PassFeatureConfig::DagFeatures::mergeBarrierThreshold,
/// or the CDNA5 default when unset), the two groups are fused into a single
/// group whose barriers carry the union of both token sets, removing the
/// redundant second signal/wait pair.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyMergeBarrierPass();

}  // namespace stinkytofu
