/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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

/// Last fence on each exact dynamic path of `hazard`.
STINKYTOFU_EXPORT GirHazardFenceCover
enclosingFences(const GirFrameAnalysis::Result& frames, const GirFrameHazard& hazard,
                const std::function<bool(const StinkyInstruction&)>& alsoFences = {});

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

}  // namespace stinkytofu
