// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {

class BasicBlock;
class Pass;
struct StinkyInstruction;

// ─────────────────────────────────────────────────────────────────────────────
// Loop region
// ─────────────────────────────────────────────────────────────────────────────

/// One rewrite target: an unrolled-loop body and the preheader that holds the
/// iteration-0 ds_load prefetches feeding it.
struct WmmaLoopRegion {
    BasicBlock* preheader = nullptr;
    BasicBlock* body = nullptr;

    /// Every wmma in @ref body, in current program order. This is the sequence
    /// an IWmmaOrderProvider permutes.
    std::vector<StinkyInstruction*> wmma;
};

// ─────────────────────────────────────────────────────────────────────────────
// ABI — reorder mode (swappable)
// ─────────────────────────────────────────────────────────────────────────────

/// Supplies the desired wmma issue order for one loop body. Each reordering
/// method (VGPR-pressure analysis, an externally tuned permutation, a future
/// cost model) is one implementation; the rewrite machinery below is shared.
class STINKYTOFU_EXPORT IWmmaOrderProvider {
   public:
    virtual ~IWmmaOrderProvider() = default;

    /// Short mode name, used in remarks and debug output.
    virtual const char* name() const = 0;

    /// Return a permutation of @p region.wmma, or an empty vector to leave the
    /// loop untouched. Anything that is not a permutation of the input is
    /// rejected by the pass and treated as "leave untouched".
    virtual std::vector<StinkyInstruction*> desiredOrder(const WmmaLoopRegion& region) const = 0;
};

/// Mode: minimize VGPR pressure. Runs the wmma reorder analysis itself, so it
/// needs no other pass in the pipeline. Yields no order for a block the analysis
/// does not apply to — e.g. one whose wmma carry no WmmaPoolData.
class STINKYTOFU_EXPORT VgprAnalysisOrderProvider : public IWmmaOrderProvider {
   public:
    const char* name() const override {
        return "vgpr-analysis";
    }
    std::vector<StinkyInstruction*> desiredOrder(const WmmaLoopRegion& region) const override;
};

/// Mode: caller-supplied permutation of body positions — `perm[i]` is the
/// original body index of the wmma that should issue i-th. Used by tuning
/// sweeps that search wmma patterns, and by tests.
class STINKYTOFU_EXPORT ExplicitOrderProvider : public IWmmaOrderProvider {
   public:
    explicit ExplicitOrderProvider(std::vector<unsigned> perm) : perm_(std::move(perm)) {}
    const char* name() const override {
        return "explicit";
    }
    std::vector<StinkyInstruction*> desiredOrder(const WmmaLoopRegion& region) const override;

   private:
    std::vector<unsigned> perm_;
};

/// Mode: reverse the body's wmma order. Deliberately crude — it exists so the
/// rewrite machinery can be exercised on any kernel without a tuner attached.
class STINKYTOFU_EXPORT ReverseOrderProvider : public IWmmaOrderProvider {
   public:
    const char* name() const override {
        return "reverse";
    }
    std::vector<StinkyInstruction*> desiredOrder(const WmmaLoopRegion& region) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
// Options and result
// ─────────────────────────────────────────────────────────────────────────────

struct WmmaReorderOptions {
    /// Label of the unrolled-loop body block (Tensile's openLoop label).
    std::string loopLabel = "label_LoopBeginL";

    /// How many wmma slots a ds_load must lead its first consumer by.
    /// Negative means "derive from the ds_load latency and the wmma issue rate".
    int prefetchDistance = -1;

    /// Allow ds_loads to migrate across the loop back-edge — cloning into the
    /// preheader when a load becomes cross-iteration, and deleting the
    /// preheader copy when it stops being cross-iteration. With this off the
    /// body is permuted in place and the preheader is never touched.
    bool allowCrossIteration = true;
};

/// Per-loop outcome, for remarks and tests.
struct WmmaReorderOutcome {
    bool applied = false;

    unsigned wmmaMoved = 0;        ///< wmma whose body position changed
    unsigned dsLoadMoved = 0;      ///< ds_load whose body position changed
    unsigned prefetchAdded = 0;    ///< ds_loads newly cloned into the preheader
    unsigned prefetchRemoved = 0;  ///< preheader ds_loads deleted as no longer needed

    /// LDS byte distance between one iteration's ds_load and the next, derived
    /// from the preheader/body ds_load pairs the kernel already has.
    int iterOffsetDelta = 0;

    unsigned prefetchDistance = 0;

    /// Empty when applied; otherwise why the loop was left alone.
    std::string skipReason;
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass factory
// ─────────────────────────────────────────────────────────────────────────────

/// Rewrites each unrolled-loop body to a new wmma order and re-places the
/// ds_loads that feed it, moving loads across the loop back-edge as the new
/// order demands.
///
/// The loop body is only ever permuted — no instruction is created or destroyed
/// there. Instructions are added to and removed from the preheader, which is
/// where the software pipeline's iteration-0 prefetches live.
///
/// Runs before wait-count insertion and before the DAG scheduler.
///
/// @param mode  reordering method; defaults to VgprAnalysisOrderProvider.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyWmmaReorderPass(
    std::unique_ptr<IWmmaOrderProvider> mode = nullptr, WmmaReorderOptions options = {});

/// Retrieve the result produced for loop body @p bb by the most recent run.
/// Returns nullptr if the pass has not run or @p bb was not a loop body.
STINKYTOFU_EXPORT const WmmaReorderOutcome* getWmmaReorderOutcome(const BasicBlock& bb);

}  // namespace stinkytofu
