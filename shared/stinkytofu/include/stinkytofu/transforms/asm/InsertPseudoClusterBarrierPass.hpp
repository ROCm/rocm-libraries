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

/// Creates a pass that plants PSEUDO_CLUSTER_BARRIER placeholders before the DAG
/// scheduler (Step B of the cluster-barrier reimplementation; see
/// docs/developer/pseudo-cluster-barrier-plan.md).
///
/// Anchor detection is basic-block based and mirrors the legacy
/// InsertClusterBarrierPass Rule 4: for each `tensor_load_to_lds` the nearest
/// preceding `s_barrier_wait -1` in the SAME basic block is the anchor, and a
/// single placeholder is inserted immediately AFTER that wait. Anchors are
/// deduplicated by identity so multiple loads sharing one wait yield one
/// placeholder, and the backward scan stops at the BB boundary so it never
/// crosses a CFG edge. Must run after CFGBuilderPass (needs materialized BBs)
/// and before the DAG scheduler.
///
/// The placeholder carries an SCC destination operand: its post-DAG expansion
/// emits a WaveIdx-gated compare that clobbers SCC, so modeling it as an SCC
/// writer makes the scheduler keep it out of any live SCC def->use range (it
/// cannot be reordered between an SCC producer and its consumer).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertPseudoClusterBarrierPass();

}  // namespace stinkytofu
