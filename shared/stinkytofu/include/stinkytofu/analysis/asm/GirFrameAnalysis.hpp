/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {
struct StinkyInstruction;

inline constexpr const char* kGirFrameContractKey = "gir.frame_contract";

//: Identifies the blob; NOT a version. The producer (Tensile's `gir/frame_contract.py`) and this
//: parser ship together, so a mismatch is a build error rather than something to negotiate.
inline constexpr const char* kGirFrameContractMarker = "gir-frame-contract";

enum class GirHazardKind : uint8_t { RAW, WAR, WAW };

struct GirGenerationSpec {
    int id = -1;
    int ring = 1;
    int entry = 0;
};

struct GirAccessSpec {
    uint64_t actionId = 0;
    bool isWrite = false;
    int genId = -1;
    int ring = 1;
    int gdelta = 0;
    int absoluteGeneration = -1;
    bool crossAgent = false;
    // The storage this touch names: one id per (operand, region) pair, assigned by the producer.
    int operand = -1;
};

struct GirFrameIncomingSpec {
    uint64_t destinationAction = 0;
    uint64_t sourceAction = 0;
    int genId = -1;
    int value = 0;
    bool relative = false;
};

struct GirActionSpec {
    uint64_t id = 0;
    uint64_t anchorAction = 0;
    GirActionKind kind = GirActionKind::Other;
    std::vector<size_t> accesses;
};

/// An edge is takeable only when `genId` carries one of `values`.  The mirror of
/// `GirFrameIncomingSpec`: incoming ASSIGNS a phase on an edge, this CONSTRAINS one, so a guard
/// generation can refuse the arms that cannot reach a successor.  Absence constrains nothing.
struct GirFrameRequiresSpec {
    uint64_t destinationAction = 0;
    uint64_t sourceAction = 0;
    int genId = -1;
    std::vector<int> values;
};

struct GirFrameContract {
    //: Whether a contract was supplied at all, as opposed to one that supplied no facts.
    bool loaded = false;
    std::map<int, GirGenerationSpec> generations;
    std::map<uint64_t, GirActionSpec> actions;
    std::vector<GirAccessSpec> accesses;
    std::vector<GirFrameIncomingSpec> incomings;
    std::vector<GirFrameRequiresSpec> requires_;
};

/// A `gir.frame_contract` metadata blob, for hand-written `.stir` tests. `text` is the body
/// without its braces. Null on a malformed or unknown record rather than a contract that silently
/// says less than its author wrote.
STINKYTOFU_EXPORT std::shared_ptr<const GirFrameContract> parseGirFrameContract(
    std::string_view text, std::string* error = nullptr);

struct GirFrame {
    std::vector<std::pair<int, int>> phases;

    int phaseOf(int genId) const;
    void setPhase(int genId, int phase);

    bool operator==(const GirFrame&) const = default;
    bool operator<(const GirFrame& other) const {
        return phases < other.phases;
    }
};

struct GirFrameHash {
    size_t operator()(const GirFrame& frame) const;
};

struct GirFrameNode {
    BasicBlock* block = nullptr;
    GirFrame frame;
    uint64_t incomingAction = 0;

    bool operator==(const GirFrameNode&) const = default;
};

struct GirFrameNodeHash {
    size_t operator()(const GirFrameNode& node) const;
};

/// Frame-graph successors, iterated in INSERTION order; the fill walks blocks in program order.
/// Hashing on the block address let the heap decide it, and with it every tie downstream.
class GirFrameEdges {
   public:
    using Entry = std::pair<GirFrameNode, std::vector<GirFrameNode>>;
    using const_iterator = std::vector<Entry>::const_iterator;

    std::vector<GirFrameNode>& operator[](const GirFrameNode& node) {
        auto [slot, fresh] = index.try_emplace(node, entries.size());
        if (fresh) entries.emplace_back(node, std::vector<GirFrameNode>{});
        return entries[slot->second].second;
    }

    const_iterator find(const GirFrameNode& node) const {
        auto slot = index.find(node);
        return slot == index.end() ? entries.end() : entries.begin() + slot->second;
    }

    const_iterator begin() const {
        return entries.begin();
    }
    const_iterator end() const {
        return entries.end();
    }
    size_t size() const {
        return entries.size();
    }

   private:
    std::vector<Entry> entries;
    std::unordered_map<GirFrameNode, size_t, GirFrameNodeHash> index;
};

struct GirAccessOccurrence {
    StinkyInstruction* inst = nullptr;
    BasicBlock* block = nullptr;
    size_t instructionIndex = 0;
    size_t accessIndex = 0;
    GirFrame frame;
    int storage = 0;
};

struct STINKYTOFU_EXPORT GirFrameAnalysis {
    STINKYTOFU_ANALYSIS_KEY("GirFrameAnalysis")

    class Result {
       public:
        GirFrameContract contract;
        std::unordered_map<uint64_t, std::vector<StinkyInstruction*>> actionInstructions;
        std::unordered_map<BasicBlock*, std::vector<GirFrame>> blockFrames;
        GirFrameEdges edges;
        std::vector<GirAccessOccurrence> occurrences;
        std::unordered_set<uint64_t> backEdges;

        bool empty() const {
            return !contract.loaded;
        }
        bool isBackEdge(const BasicBlock* from, const BasicBlock* to) const;
        const std::vector<GirFrame>& frames(const BasicBlock* block) const;
    };

    static Result run(Function& function, AnalysisManager& AM);
};

struct GirFrameHazard {
    GirHazardKind kind = GirHazardKind::RAW;
    StinkyInstruction* producer = nullptr;
    StinkyInstruction* consumer = nullptr;
    BasicBlock* producerBlock = nullptr;
    BasicBlock* consumerBlock = nullptr;
    size_t producerIndex = 0;
    size_t consumerIndex = 0;
    GirFrame producerFrame;
    GirFrame consumerFrame;
    // Frame-graph NODES traversed between the two occurrences, not generations: 0 means one
    // execution of the pair in one node, >=1 means another node was entered -- which is a later
    // trip only when both ends sit in the same block.
    int gap = 0;
    bool crossAgent = false;
    uint64_t producerAction = 0;
    uint64_t consumerAction = 0;
    // The concrete LDS slot both ends touch (`GirAccessOccurrence::storage`).
    int storage = -1;
};

struct GirHazardFenceOwner {
    StinkyInstruction* anchor = nullptr;
    GirFrameNode producerNode;
    GirFrameNode anchorNode;
    GirFrameNode consumerNode;
    int producerToAnchorGap = 0;
};

struct GirHazardFenceCover {
    std::vector<GirHazardFenceOwner> owners;
    bool hasHazardPath = false;
    bool hasUnfencedPath = false;
};

inline constexpr const char* kGirFenceCoverageKey = "gir.fence_coverage";

using GirFrameHazards = std::set<const GirFrameHazard*>;

/// A fence distinguishes hazards by kind and slot, so one covered hazard answers for its class.
inline bool coversHazard(const GirFrameHazards& coverage, const GirFrameHazard& hazard) {
    return std::any_of(coverage.begin(), coverage.end(), [&](const GirFrameHazard* covered) {
        return covered->kind == hazard.kind && covered->storage == hazard.storage;
    });
}

/// Which hazards each materialized GIR fence covers, and on which frame. Placement is the only
/// thing that knows which hazards it placed a fence for, so it records them here and consumers
/// query instead of deriving ownership again -- the kinds give the fence stamp, the storages give
/// the memory tokens. Merging two fences unions their entries frame by frame.
struct STINKYTOFU_EXPORT GirFenceCoverage {
    /// Hazards are held BY VALUE: this outlives the hazard analysis that produced them, which
    /// placement invalidates before any consumer reads it.
    using Covered = std::vector<GirFrameHazard>;
    std::map<const StinkyInstruction*, std::map<GirFrame, Covered>> fences;

    void add(const StinkyInstruction* fence, const GirFrame& frame, const GirFrameHazard& hazard) {
        Covered& covered = fences[fence][frame];
        for (const GirFrameHazard& seen : covered)
            if (seen.kind == hazard.kind && seen.storage == hazard.storage) return;
        covered.push_back(hazard);
    }

    /// Hazards \p fence covers on \p frame, or nullptr when it covers none there.
    const Covered* on(const StinkyInstruction* fence, const GirFrame& frame) const {
        auto entry = fences.find(fence);
        if (entry == fences.end()) return nullptr;
        auto covered = entry->second.find(frame);
        return covered == entry->second.end() ? nullptr : &covered->second;
    }

    /// Fold \p from into \p into, as a barrier merge does to the fences it collapses.
    void merge(const StinkyInstruction* into, const StinkyInstruction* from) {
        auto source = fences.find(from);
        if (source == fences.end()) return;
        for (const auto& [frame, covered] : source->second)
            for (const GirFrameHazard& hazard : covered) add(into, frame, hazard);
        fences.erase(from);
    }
};

struct STINKYTOFU_EXPORT GirFrameHazardAnalysis {
    STINKYTOFU_ANALYSIS_KEY("GirFrameHazardAnalysis")

    struct Result {
        std::vector<GirFrameHazard> hazards;

        bool empty() const {
            return hazards.empty();
        }
    };

    static Result run(Function& function, AnalysisManager& AM);
};

/// Fence coverage of every cross-agent hazard under one marker plan: for hazard `h`,
/// `byHazard[h]` names the fences covering it on each exact dynamic path, the frame each stands
/// on, and whether any path is left unfenced. Built in one walk so every consumer reads the same
/// answer instead of deriving ownership for itself.
struct STINKYTOFU_EXPORT GirPlanFenceCoverage {
    std::vector<GirHazardFenceCover> byHazard;
};

STINKYTOFU_EXPORT GirPlanFenceCoverage buildGirPlanFenceCoverage(
    const GirFrameAnalysis::Result& frames, const GirFrameHazardAnalysis::Result& hazards,
    const std::function<bool(const StinkyInstruction&)>& alsoFences = {});

}  // namespace stinkytofu
