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

#include <map>
#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {

class BasicBlock;
class Pass;
struct StinkyInstruction;

// ─────────────────────────────────────────────────────────────────────────────
// Shared data types
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous VGPR group identified by its base index and size.
struct RegGroup {
    uint32_t base;
    uint16_t size;
    bool operator==(const RegGroup& o) const {
        return base == o.base && size == o.size;
    }
    bool operator<(const RegGroup& o) const {
        return base < o.base || (base == o.base && size < o.size);
    }
};

/// Live interval expressed as wmma-sequence positions (or full-instruction
/// positions for richer liveness backends). Consumers treat values as opaque
/// ordinals and only compare them for overlap.
struct RegInterval {
    unsigned first = 0;
    unsigned last = 0;
    bool overlaps(const RegInterval& o) const {
        return first <= o.last && o.first <= last;
    }
};

/// A single per-operand rewrite entry. The downstream register-renaming pass
/// applies this list mechanically — no further analysis is required.
struct RegReplacement {
    StinkyInstruction* inst;  ///< instruction whose operand must change
    unsigned operandIdx;      ///< index into srcRegs (isSrc=true) or destRegs
    bool isSrc;
    StinkyRegister oldReg;
    StinkyRegister newReg;
};

/// Stable output contract between this pass and all downstream consumers.
/// Never changes regardless of which liveness backend or algorithm is active.
struct WmmaReorderAnalysisResult {
    /// True when the optimization applies to the analyzed basic block.
    bool applicable = false;

    /// Desired wmma ordering: a permutation of the original wmma instruction
    /// pointers. A downstream reorder pass applies this directly to the BB.
    std::vector<StinkyInstruction*> desiredWmmaOrder;

    /// Flat per-operand replacement map. Each entry describes exactly one
    /// operand that must be rewritten to eliminate the aliasable register range.
    std::vector<RegReplacement> replacements;

    /// Total VGPRs freed when all replacements are applied.
    unsigned totalVgprSaved = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// ABI 1 — Liveness backend (swappable)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal wmma descriptor used by both ABIs.
struct WmmaNode {
    StinkyInstruction* inst;
    RegGroup aGroup;  ///< src0 (A matrix tile)
    RegGroup bGroup;  ///< src1 (B matrix tile)
    RegGroup cGroup;  ///< dest (C accumulator tile)
};

/// Compute live intervals for A/B register groups referenced by wmma.
/// Implementations may use wmma-sequence positions (fast) or full-instruction
/// positions (precise). Callers treat interval values as opaque ordinals.
class STINKYTOFU_EXPORT IRegLivenessAnalysis {
   public:
    virtual ~IRegLivenessAnalysis() = default;

    /// Return the live interval for every RegGroup that appears as an A or B
    /// source in @p wmmaSeq. The @p bb argument is provided so full-instruction
    /// backends can inspect all instruction types; wmma-only backends may ignore it.
    virtual std::map<RegGroup, RegInterval> computeLiveness(
        const BasicBlock& bb, const std::vector<WmmaNode>& wmmaSeq) const = 0;
};

/// Fast wmma-only liveness: interval = [first wmma index, last wmma index]
/// that reads the group. Sufficient for the double-buffered GEMM pattern.
class STINKYTOFU_EXPORT WmmaIntervalLiveness : public IRegLivenessAnalysis {
   public:
    std::map<RegGroup, RegInterval> computeLiveness(
        const BasicBlock& bb, const std::vector<WmmaNode>& wmmaSeq) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// ABI 2 — Reorder algorithm (swappable)
// ─────────────────────────────────────────────────────────────────────────────

struct AliasCandidate {
    RegGroup canonical;  ///< register group to keep (e.g. A_X0[i])
    RegGroup aliasable;  ///< register group to remap onto canonical (e.g. A_X1[i])
    unsigned vgprSaved;  ///< == canonical.size
};

/// Given the wmma sequence and precomputed live intervals, decide the optimal
/// wmma ordering and which register groups can be aliased. The algorithm does
/// not know how liveness was computed.
class STINKYTOFU_EXPORT IWmmaReorderAlgorithm {
   public:
    virtual ~IWmmaReorderAlgorithm() = default;

    struct Result {
        std::vector<WmmaNode> desiredOrder;  ///< permutation of wmmaSeq
        std::vector<AliasCandidate> aliases;
    };

    virtual Result solve(const std::vector<std::vector<WmmaNode>>& pools,
                         const std::map<RegGroup, RegInterval>& liveness) const = 0;
};

/// Reorders each pool so its pool-varying operand (A after detectABIndices relabeling)
/// is contiguous, tightening its liveness and enabling cross-pool register aliasing.
/// O(n_wmma). Covers standard N-buffered GEMM kernels regardless of which hardware
/// src (A or B) carries the pool-varying registers.
class STINKYTOFU_EXPORT PoolVaryingReorderAlgorithm : public IWmmaReorderAlgorithm {
   public:
    Result solve(const std::vector<std::vector<WmmaNode>>& pools,
                 const std::map<RegGroup, RegInterval>& liveness) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass factory
// ─────────────────────────────────────────────────────────────────────────────

/// Analysis-only pass that determines the optimal wmma reordering to reduce
/// VGPR usage by aliasing A_X1 register groups onto A_X0.
///
/// Runs before StinkyDAGSchedulerPass. Does not mutate any instruction or
/// register operand — all output is in WmmaReorderAnalysisResult.
///
/// The liveness backend and reorder algorithm are injected at construction so
/// either axis can be swapped independently without touching consumers.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyWmmaVgprReorderPass(
    std::unique_ptr<IRegLivenessAnalysis> liveness = nullptr,
    std::unique_ptr<IWmmaReorderAlgorithm> algorithm = nullptr);

/// Retrieve the analysis result produced for @p bb by the most recent pass run.
/// Returns nullptr if the pass has not run or the BB was not processed.
STINKYTOFU_EXPORT const WmmaReorderAnalysisResult* getWmmaReorderResult(const BasicBlock& bb);

}  // namespace stinkytofu
