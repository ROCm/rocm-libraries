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

#include <cstdint>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class BasicBlock;
class AsmIRBuilder;

/// Collapse each narrow-exec-write..full-mask-reset span into a single opaque
/// ExecMaskGroup pseudo-instruction so the DAG scheduler cannot reorder into or
/// out of the span. Call expandExecMaskedGroups() after scheduling to restore.
STINKYTOFU_EXPORT void collapseExecMaskedRegions(BasicBlock& bb, AsmIRBuilder& builder,
                                                 uint32_t wavefrontSize);

/// Tag each PSEUDO_CLUSTER_BARRIER placeholder together with its adjacent workgroup
/// barrier as a two-member StickChainData chain, WITHOUT wrapping them in an
/// ExecMaskGroup. The CDNA5 scheduler's StickChain promotion rule then keeps the pair
/// back-to-back (nothing scheduled between them) while each instruction keeps its own
/// tokens/latency/forced-barrier threshold — so, unlike the EXEC_GROUP fusion, the
/// workgroup signal/wait keeps its own threshold. Chain order matches program order:
///   - SignalOnly: [placeholder (member 0), s_barrier_signal -1 (member 1)]
///   - WaitOnly / SignalWait: [s_barrier_wait -1 (member 0), placeholder (member 1)]
/// so member 0 always precedes member 1 in the IR (the DAG chain edge stays acyclic).
/// The scheduler triggers the chain from the workgroup barrier's forced-barrier threshold
/// and issues member 0 first, keeping the placeholder glued adjacent to its barrier in
/// the required order. Idempotent: a placeholder already carrying StickChainData is skipped.
STINKYTOFU_EXPORT void tagClusterBarrierChains(BasicBlock& bb);

/// Inverse of collapseExecMaskedRegions (also restores collapseClusterBarrierPairs).
STINKYTOFU_EXPORT void expandExecMaskedGroups(BasicBlock& bb);

}  // namespace stinkytofu
