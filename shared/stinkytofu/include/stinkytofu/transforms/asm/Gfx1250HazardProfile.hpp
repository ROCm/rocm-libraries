// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_set>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

namespace stinkytofu {

enum class XcntDrainReason {
    AtomicRule4a,
    SmemRule3,
    FlatRule2,
    ForeverSleep,
    ScalarPrefetch,
    VgprMsb,
    Count,
};

/// Reported name of each reason, in enum order.
constexpr std::array kXcntDrainReasonNames = {"atomic",       "smem",           "flat",
                                              "foreverSleep", "scalarPrefetch", "vgprMsb"};
static_assert(kXcntDrainReasonNames.size() == static_cast<size_t>(XcntDrainReason::Count),
              "every XcntDrainReason needs a reported name");

/// Hooks Gfx1250HazardPass calls while it inserts s_wait_xcnt drains.
/// makeXcntDrainProfile() picks the implementation: XcntDrainProfile to count
/// and report, EmptyXcntDrainProfile to do nothing.
class XcntDrainProfileBase {
   public:
    virtual ~XcntDrainProfileBase() = default;

    /// Called before the pass walks \p func, so per-function facts can be collected.
    virtual void beginFunction(Function& func) = 0;
    virtual void noteTensorLoad() = 0;
    virtual void record(XcntDrainReason reason, const IRBase* anchor) = 0;
    virtual void print() const = 0;
};

/// Everything reported for one scope. Adding two scopes gives the whole-kernel
/// numbers, so the pass never has to count the same drain twice.
struct XcntDrainCounts {
    uint32_t total = 0;
    std::array<uint32_t, kXcntDrainReasonNames.size()> reasons{};
    // One per reported site: loop+matrix, loop, matrix, other.
    uint32_t loopMatrixSites = 0;
    uint32_t loopSites = 0;
    uint32_t matrixSites = 0;
    uint32_t otherSites = 0;
    bool usesTensorLoad = false;

    XcntDrainCounts& operator+=(const XcntDrainCounts& other) {
        total += other.total;
        for (size_t i = 0; i < reasons.size(); ++i) reasons[i] += other.reasons[i];
        loopMatrixSites += other.loopMatrixSites;
        loopSites += other.loopSites;
        matrixSites += other.matrixSites;
        otherSites += other.otherSites;
        usesTensorLoad |= other.usesTensorLoad;
        return *this;
    }
};

/// Counts inserted drains by rule and by where they landed, over the whole
/// kernel: the counts accumulate across every function the pass walks, and are
/// additionally split into the kernel body (the entry function) and the helper
/// functions it calls (activation functions and friends), which are usually
/// worth judging separately.
///
/// Placement is described by two independent facts about the block holding the
/// drain's anchor: whether the block belongs to a loop (the drain is paid on
/// every iteration) and whether it holds a matrix instruction (the drain sits in
/// the MAC pipeline). Both come from the final CFG at the moment the pass runs,
/// so unlike recorded region ranges they cannot go stale.
class XcntDrainProfile final : public XcntDrainProfileBase {
   public:
    /// Only the per-function CFG facts are rebuilt here; the counts keep adding
    /// up. Drain insertion only adds instructions to existing blocks, so neither
    /// the loop structure nor the per-block matrix flag changes during the walk.
    void beginFunction(Function& func) override {
        if (func.getIsCallable()) hasHelpers = true;
        current = func.getIsCallable() ? &helpers : &kernelBody;

        loopBlocks.clear();
        matrixBlocks.clear();

        for (const Loop& loop : detectLoops(func))
            loopBlocks.insert(loop.bodyBBs.begin(), loop.bodyBBs.end());

        for (BasicBlock& bb : func)
            if (holdsMatrixInstruction(bb)) matrixBlocks.insert(&bb);
    }

    void noteTensorLoad() override {
        current->usesTensorLoad = true;
    }

    void record(XcntDrainReason reason, const IRBase* anchor) override {
        ++current->total;
        ++current->reasons[static_cast<size_t>(reason)];
        ++siteCounter(anchor);
    }

    void print() const override {
        XcntDrainCounts wholeKernel = kernelBody;
        wholeKernel += helpers;
        printCounts("whole kernel ", wholeKernel);

        // With no helper function the split would just repeat the totals.
        if (!hasHelpers) return;
        printCounts("kernel body ", kernelBody);
        printCounts("helper functions ", helpers);
    }

   private:
    static void printCounts(const char* scope, const XcntDrainCounts& counts) {
        constexpr const char* tag = "[Gfx1250HazardPass] ";

        std::cerr << tag << scope << "xcnt drains: total=" << counts.total
                  << ", loop+matrix=" << counts.loopMatrixSites << ", loop=" << counts.loopSites
                  << ", matrix=" << counts.matrixSites << ", other=" << counts.otherSites << "\n";
        std::cerr << tag << scope << "xcnt drain rules:";
        const char* separator = " ";
        for (size_t i = 0; i < counts.reasons.size(); ++i) {
            std::cerr << separator << kXcntDrainReasonNames[i] << "=" << counts.reasons[i];
            separator = ", ";
        }
        std::cerr << "\n";
        std::cerr << tag << scope
                  << "tensor_load_to_lds: " << (counts.usesTensorLoad ? "used" : "not used")
                  << "\n";
    }

    static bool holdsMatrixInstruction(BasicBlock& bb) {
        for (IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && isMatrixInstruction(*inst)) return true;
        }
        return false;
    }

    /// The two facts are independent, so each combination keeps its own count:
    /// a loop block without matrix instructions (the loop-close block, say)
    /// still pays its drain on every iteration.
    uint32_t& siteCounter(const IRBase* anchor) {
        const BasicBlock* bb = anchor != nullptr ? anchor->getParent() : nullptr;
        const bool inLoop = loopBlocks.contains(bb);
        const bool hasMatrix = matrixBlocks.contains(bb);

        if (inLoop) return hasMatrix ? current->loopMatrixSites : current->loopSites;
        return hasMatrix ? current->matrixSites : current->otherSites;
    }

    // Facts about the function currently being walked.
    std::unordered_set<const BasicBlock*> loopBlocks;
    std::unordered_set<const BasicBlock*> matrixBlocks;

    XcntDrainCounts kernelBody;
    XcntDrainCounts helpers;
    // Scope of the function currently being walked; the pass always calls
    // beginFunction() first, so this only guards a stray record().
    XcntDrainCounts* current = &kernelBody;
    bool hasHelpers = false;
};

/// Profiling off: no counters, no per-function analysis, no output.
class EmptyXcntDrainProfile final : public XcntDrainProfileBase {
   public:
    void beginFunction(Function&) override {}
    void noteTensorLoad() override {}
    void record(XcntDrainReason, const IRBase*) override {}
    void print() const override {}
};

inline std::unique_ptr<XcntDrainProfileBase> makeXcntDrainProfile(bool enabled) {
    if (enabled) return std::make_unique<XcntDrainProfile>();
    return std::make_unique<EmptyXcntDrainProfile>();
}

}  // namespace stinkytofu
