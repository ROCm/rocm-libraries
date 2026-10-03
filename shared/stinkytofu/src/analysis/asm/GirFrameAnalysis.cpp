/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"

#include <algorithm>
#include <charconv>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {

namespace {

uint64_t edgeKey(const BasicBlock* from, const BasicBlock* to) {
    const auto a = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(from));
    const auto b = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(to));
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
}

//: One record's `key=value` words. A bare word is a flag, stored with an empty value.

const Loop* containingLoop(const std::vector<Loop>& loops, const BasicBlock* block) {
    const Loop* best = nullptr;
    for (const Loop& loop : loops) {
        if (!loop.contains(block)) continue;
        if (best == nullptr || loop.bodyBBs.size() < best->bodyBBs.size()) best = &loop;
    }
    return best;
}

bool instructionRealizes(GirActionKind kind, const StinkyInstruction& inst) {
    if (kind == GirActionKind::Read) return isDSRead(inst);
    if (kind == GirActionKind::Copy) return isTensorLoad(inst);
    return false;
}

GirFrame advanceFrame(const GirFrame& input, BasicBlock* from, BasicBlock* to,
                      const std::vector<Loop>& loops,
                      const std::unordered_map<int, const Loop*>& genLoops,
                      const GirFrameContract& contract) {
    GirFrame result = input;
    for (const auto& [genId, loop] : genLoops) {
        const int ring = contract.generations.at(genId).ring;
        // A back edge runs from inside the loop INTO the header.  Treating every edge out of the
        // latch as one rotated the loop's EXIT too, so a single-block loop handed its drain a
        // phase one trip ahead and a buffer aliased onto the wrong generation.
        // The back edge rotates by its relative `incoming`, applied in `advanceNode`. Keep the
        // setPhase here: a frame's identity is which generations appear in it, not only their
        // values, so dropping the call changes the node even when the phase would not.
        const bool backEdge = to == loop->headerBB && loop->contains(from);
        if (backEdge)
            result.setPhase(genId, result.phaseOf(genId) % ring);
        else if (to == loop->headerBB)
            result.setPhase(genId, contract.generations.at(genId).entry % ring);
    }
    return result;
}

// The operand id already names one (operand, region) pair, so two touches meet exactly when
// they carry the same id.
bool disjointStorage(const GirAccessSpec& a, const GirAccessSpec& b) {
    return a.operand != b.operand;
}

bool wawOrderedByRead(const GirAccessSpec& producer, const GirAccessSpec& consumer,
                      const GirFrameContract& contract) {
    if (producer.ring <= 1) return false;
    for (const GirAccessSpec& access : contract.accesses)
        if (!access.isWrite && !disjointStorage(producer, access) &&
            !disjointStorage(consumer, access))
            return true;
    return false;
}

using StorageKey = std::pair<int, int>;

// (operand, generation) names exactly one slot, so a touch has exactly one storage id.
int concreteStorage(const GirAccessSpec& access, int generation,
                    std::map<StorageKey, int>& storageIds) {
    StorageKey key{access.operand, generation};
    return storageIds.emplace(std::move(key), static_cast<int>(storageIds.size())).first->second;
}

struct StaticTouchKey {
    StinkyInstruction* inst = nullptr;
    size_t accessIndex = 0;

    bool operator==(const StaticTouchKey&) const = default;
    bool operator<(const StaticTouchKey& other) const {
        if (inst != other.inst) return std::less<StinkyInstruction*>{}(inst, other.inst);
        return accessIndex < other.accessIndex;
    }
};

struct LiveTouchKey {
    GirFrame frame;
    StaticTouchKey touch;

    bool operator==(const LiveTouchKey&) const = default;
    bool operator<(const LiveTouchKey& other) const {
        if (!(frame == other.frame)) return frame < other.frame;
        return touch < other.touch;
    }
};

struct SlotLive {
    std::map<LiveTouchKey, int> writes;
    std::map<LiveTouchKey, int> reads;
};

using SlotState = std::map<int, SlotLive>;
using HazardState = std::map<GirFrame, SlotState>;
using OccurrenceRegistry = std::map<LiveTouchKey, const GirAccessOccurrence*>;
using OccurrencesByNode =
    std::unordered_map<GirFrameNode, std::vector<const GirAccessOccurrence*>, GirFrameNodeHash>;

using HazardEmitter =
    std::function<void(const GirAccessOccurrence&, const GirAccessOccurrence&, int)>;

HazardState walkHazards(const GirFrameNode& node,
                        const std::vector<const GirAccessOccurrence*>& touches, HazardState state,
                        const GirFrameContract& contract, const OccurrenceRegistry& registry,
                        const HazardEmitter* emit = nullptr) {
    for (const GirAccessOccurrence* touch : touches) {
        const GirAccessSpec& spec = contract.accesses.at(touch->accessIndex);
        const LiveTouchKey me{node.frame, {touch->inst, touch->accessIndex}};
        const int storage = touch->storage;
        if (emit) {
            for (const auto& [storedFrame, slots] : state) {
                (void)storedFrame;
                auto slot = slots.find(storage);
                if (slot == slots.end()) continue;
                if (spec.isWrite)
                    for (const auto& [prior, distance] : slot->second.reads) {
                        auto occurrence = registry.find(prior);
                        if (occurrence == registry.end())
                            report_fatal_error(
                                "GirFrameHazardAnalysis lost a live reader occurrence");
                        (*emit)(*occurrence->second, *touch, distance);
                    }
                for (const auto& [prior, distance] : slot->second.writes) {
                    auto occurrence = registry.find(prior);
                    if (occurrence == registry.end())
                        report_fatal_error("GirFrameHazardAnalysis lost a live writer occurrence");
                    (*emit)(*occurrence->second, *touch, distance);
                }
            }
        }

        if (spec.isWrite) {
            // A write kills this absolute storage slot in every frame picture.
            for (auto& [storedFrame, slots] : state) {
                (void)storedFrame;
                slots.erase(storage);
            }
            SlotLive replacement;
            replacement.writes.emplace(me, 0);
            state[node.frame][storage] = std::move(replacement);
        } else {
            bool filed = false;
            for (auto& [storedFrame, slots] : state) {
                (void)storedFrame;
                auto slot = slots.find(storage);
                if (slot == slots.end()) continue;
                slot->second.reads[me] = 0;
                filed = true;
            }
            if (!filed) state[node.frame][storage].reads[me] = 0;
        }
    }
    return state;
}

HazardState stepHazards(HazardState state) {
    for (auto& [frame, slots] : state) {
        (void)frame;
        for (auto& [storage, live] : slots) {
            (void)storage;
            for (auto& [touch, distance] : live.writes) {
                (void)touch;
                ++distance;
            }
            for (auto& [touch, distance] : live.reads) {
                (void)touch;
                ++distance;
            }
        }
    }
    return state;
}

template <typename Map>
bool mergeNearest(Map& destination, const Map& source) {
    bool changed = false;
    for (const auto& [key, distance] : source) {
        auto found = destination.find(key);
        if (found == destination.end() || distance < found->second) {
            destination[key] = distance;
            changed = true;
        }
    }
    return changed;
}

bool mergeHazardState(HazardState& destination, const HazardState& source) {
    bool changed = false;
    for (const auto& [frame, slots] : source)
        for (const auto& [storage, live] : slots) {
            SlotLive& target = destination[frame][storage];
            changed |= mergeNearest(target.writes, live.writes);
            changed |= mergeNearest(target.reads, live.reads);
        }
    return changed;
}

std::unordered_map<GirFrameNode, HazardState, GirFrameNodeHash> reachingHazards(
    const GirFrameAnalysis::Result& frames, const OccurrencesByNode& occurrences,
    const OccurrenceRegistry& registry) {
    std::unordered_map<GirFrameNode, HazardState, GirFrameNodeHash> incoming;
    std::deque<GirFrameNode> work;
    for (const auto& [node, successors] : frames.edges) {
        (void)successors;
        incoming.emplace(node, HazardState{});
        work.push_back(node);
    }

    while (!work.empty()) {
        GirFrameNode node = std::move(work.back());
        work.pop_back();
        auto touches = occurrences.find(node);
        static const std::vector<const GirAccessOccurrence*> empty;
        HazardState aged =
            stepHazards(walkHazards(node, touches == occurrences.end() ? empty : touches->second,
                                    incoming[node], frames.contract, registry));
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            auto target = incoming.find(successor);
            if (target == incoming.end()) continue;
            if (mergeHazardState(target->second, aged)) work.push_back(successor);
        }
    }
    return incoming;
}

uint64_t actionOf(const StinkyInstruction* inst) {
    const GirActionData* data = inst ? inst->getModifier<GirActionData>() : nullptr;
    return data ? data->actionId : 0;
}

}  // namespace

namespace {

std::string_view trimRecord(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

/// `tag [positional] key=value ...` split into its words. A bare word after the tag is a flag.
struct Record {
    std::string_view tag;
    std::vector<std::string_view> positional;
    std::map<std::string_view, std::string_view, std::less<>> fields;
};

Record splitRecord(std::string_view line) {
    Record out;
    size_t pos = 0;
    while (pos < line.size()) {
        const size_t begin = line.find_first_not_of(" \t", pos);
        if (begin == std::string_view::npos) break;
        const size_t end = line.find_first_of(" \t", begin);
        const std::string_view word =
            line.substr(begin, end == std::string_view::npos ? end : end - begin);
        if (out.tag.empty())
            out.tag = word;
        else if (const size_t eq = word.find('='); eq != std::string_view::npos)
            out.fields.emplace(word.substr(0, eq), word.substr(eq + 1));
        else
            out.positional.push_back(word);
        if (end == std::string_view::npos) break;
        pos = end;
    }
    return out;
}

bool readInt(std::string_view text, int& out) {
    return std::from_chars(text.data(), text.data() + text.size(), out).ec == std::errc{};
}

bool readField(const Record& record, std::string_view key, int& out) {
    auto found = record.fields.find(key);
    return found != record.fields.end() && readInt(found->second, out);
}

bool readAction(const Record& record, std::string_view key, uint64_t& out) {
    auto found = record.fields.find(key);
    if (found == record.fields.end()) return false;
    const std::string_view text = found->second;
    return std::from_chars(text.data(), text.data() + text.size(), out).ec == std::errc{};
}

}  // namespace

std::shared_ptr<const GirFrameContract> parseGirFrameContract(std::string_view text,
                                                              std::string* error) {
    auto contract = std::make_shared<GirFrameContract>();
    bool marked = false;
    unsigned lineNumber = 0;
    const auto fail = [&](const std::string& what) {
        if (error) *error = "gir.frame_contract line " + std::to_string(lineNumber) + ": " + what;
        return nullptr;
    };

    std::istringstream stream{std::string(text)};
    for (std::string raw; std::getline(stream, raw);) {
        ++lineNumber;
        const std::string_view line = trimRecord(raw);
        if (line.empty() || line.front() == '#') continue;
        if (line == kGirFrameContractMarker) {
            marked = true;
            continue;
        }
        if (!marked) return fail("a record precedes the marker");

        const Record record = splitRecord(line);
        if (record.tag == "gen") {
            GirGenerationSpec gen;
            if (record.positional.size() != 1 || !readInt(record.positional.front(), gen.id))
                return fail("gen needs one positional id");
            if (!readField(record, "ring", gen.ring) || !readField(record, "entry", gen.entry))
                return fail("gen needs ring= and entry=");
            if (gen.ring < 1) return fail("gen ring must be positive");
            contract->generations[gen.id] = gen;
        } else if (record.tag == "incoming") {
            GirFrameIncomingSpec incoming;
            int relative = 0;
            readField(record, "relative", relative);
            incoming.relative = relative != 0;
            if (!readAction(record, "dst", incoming.destinationAction) ||
                !readAction(record, "src", incoming.sourceAction) ||
                !readField(record, "gen", incoming.genId) ||
                !readField(record, "value", incoming.value))
                return fail("incoming needs dst=, src=, gen= and value=");
            contract->incomings.push_back(incoming);
        } else if (record.tag == "requires") {
            GirFrameRequiresSpec need;
            if (!readAction(record, "dst", need.destinationAction) ||
                !readAction(record, "src", need.sourceAction) ||
                !readField(record, "gen", need.genId))
                return fail("requires needs dst=, src= and gen=");
            auto values = record.fields.find("values");
            if (values == record.fields.end()) return fail("requires needs values=");
            std::string_view rest = values->second;
            while (!rest.empty()) {
                const size_t comma = rest.find(',');
                int value = 0;
                if (!readInt(rest.substr(0, comma), value))
                    return fail("requires values= is not a list of integers");
                need.values.push_back(value);
                if (comma == std::string_view::npos) break;
                rest = rest.substr(comma + 1);
            }
            contract->requires_.push_back(need);
        } else {
            return fail("unknown record '" + std::string(record.tag) + "'");
        }
    }

    if (!marked) return fail("missing the marker line");
    contract->loaded = true;
    return contract;
}

int GirFrame::phaseOf(int genId) const {
    auto it = std::lower_bound(phases.begin(), phases.end(), genId,
                               [](const auto& item, int id) { return item.first < id; });
    return it != phases.end() && it->first == genId ? it->second : 0;
}

void GirFrame::setPhase(int genId, int phase) {
    auto it = std::lower_bound(phases.begin(), phases.end(), genId,
                               [](const auto& item, int id) { return item.first < id; });
    if (it != phases.end() && it->first == genId)
        it->second = phase;
    else
        phases.insert(it, {genId, phase});
}

size_t GirFrameHash::operator()(const GirFrame& frame) const {
    size_t result = 0;
    for (const auto& [gen, phase] : frame.phases)
        result ^= std::hash<int>{}(gen) ^
                  (std::hash<int>{}(phase) + 0x9e3779b9U + (result << 6U) + (result >> 2U));
    return result;
}

size_t GirFrameNodeHash::operator()(const GirFrameNode& node) const {
    size_t result = std::hash<const BasicBlock*>{}(node.block);
    result ^= GirFrameHash{}(node.frame) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
    return result ^ (std::hash<uint64_t>{}(node.incomingAction) + 0x9e3779b9U + (result << 6U) +
                     (result >> 2U));
}

bool GirFrameAnalysis::Result::isBackEdge(const BasicBlock* from, const BasicBlock* to) const {
    return backEdges.contains(edgeKey(from, to));
}

const std::vector<GirFrame>& GirFrameAnalysis::Result::frames(const BasicBlock* block) const {
    static const std::vector<GirFrame> empty;
    auto it = blockFrames.find(const_cast<BasicBlock*>(block));
    return it == blockFrames.end() ? empty : it->second;
}

GirFrameAnalysis::Result GirFrameAnalysis::run(Function& function, AnalysisManager& AM) {
    Result result;
    const auto* encoded = function.getStructMetaData<GirFrameContract>(kGirFrameContractKey);
    std::shared_ptr<const GirFrameContract> parsed;
    if (!encoded) {
        // A hand-written contract arrives as text; the struct is what a producer installs.
        const auto text = function.getStringMetaData(kGirFrameContractKey);
        if (!text || text->empty()) return result;
        std::string problem;
        parsed = parseGirFrameContract(*text, &problem);
        if (!parsed) report_fatal_error("GirFrameAnalysis: " + problem);
        encoded = parsed.get();
    }
    if (!encoded->loaded) return result;
    result.contract = *encoded;

    std::unordered_map<StinkyInstruction*, size_t> instructionIndex;
    std::unordered_map<StinkyInstruction*, BasicBlock*> instructionBlock;
    std::unordered_map<BasicBlock*, uint64_t> blockAnchor;
    std::unordered_map<BasicBlock*, uint64_t> blockAnchorOrder;
    std::unordered_map<BasicBlock*, std::unordered_set<uint64_t>> blockActions;
    std::unordered_map<uint64_t, std::unordered_set<BasicBlock*>> anchorBlocks;
    std::unordered_set<uint64_t> realizedAnchors;
    for (BasicBlock& block : function) {
        size_t index = 0;
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            instructionIndex[inst] = index++;
            instructionBlock[inst] = &block;
            if (const auto* action = inst->getModifier<GirActionData>()) {
                result.actionInstructions[action->actionId].push_back(inst);
                blockActions[&block].insert(action->actionId);
                // The action and what it touches are facts ABOUT THIS INSTRUCTION, so they come
                // off the modifier.  Legalization can split one rocisa instruction into several
                // carrying the same id, so only the first occurrence contributes its accesses.
                auto [spec, fresh] = result.contract.actions.try_emplace(
                    action->actionId,
                    GirActionSpec{action->actionId, action->anchorAction, action->kind, {}});
                if (fresh) {
                    for (const GirAccessData& touch : action->accesses) {
                        if (touch.ring < 1)
                            report_fatal_error("GirFrameAnalysis: access ring must be positive");
                        spec->second.accesses.push_back(result.contract.accesses.size());
                        result.contract.accesses.push_back(GirAccessSpec{
                            action->actionId, touch.isWrite, touch.genId, touch.ring, touch.gdelta,
                            touch.absolute, touch.crossAgent, touch.operand});
                    }
                }
                const uint64_t canonical = spec->second.anchorAction;
                realizedAnchors.insert(canonical);
                anchorBlocks[canonical].insert(&block);
                auto [order, inserted] = blockAnchorOrder.emplace(&block, action->actionId);
                if (inserted) {
                    blockAnchor[&block] = canonical;
                } else if (action->actionId > order->second) {
                    order->second = action->actionId;
                    blockAnchor[&block] = canonical;
                }
            }
        }
    }
    if (!result.contract.actions.empty() && result.actionInstructions.empty())
        report_fatal_error(
            "GirFrameAnalysis: frame contract imported but no GirActionData survived lowering");

    struct IncomingValue {
        int value = 0;
        bool relative = false;

        bool operator==(const IncomingValue&) const = default;
    };
    using IncomingValues = std::map<int, IncomingValue>;
    std::unordered_map<BasicBlock*, std::map<uint64_t, IncomingValues>> incomingValues;
    // `{destination block: {source action: {guard gen: values that may take this edge}}}` -- the
    // mirror of incomingValues: that one assigns a phase on an edge, this refuses one.
    std::unordered_map<BasicBlock*, std::map<uint64_t, std::map<int, std::set<int>>>>
        requiredValues;
    std::unordered_map<uint64_t, std::vector<BasicBlock*>> anchorEntries;
    for (const auto& [anchor, blocks] : anchorBlocks) {
        std::vector<BasicBlock*>& entries = anchorEntries[anchor];
        for (BasicBlock* block : blocks) {
            const auto& predecessors = block->getPredecessors();
            const bool entersGroup =
                predecessors.empty() ||
                std::any_of(predecessors.begin(), predecessors.end(),
                            [&](BasicBlock* pred) { return !blocks.contains(pred); });
            if (entersGroup) entries.push_back(block);
        }
        // A self-contained loop can have no predecessor outside its own logical action group.
        if (entries.empty()) entries.assign(blocks.begin(), blocks.end());
    }
    // An ANCHOR is a logical identity: a block names one whether or not an instruction carries
    // that exact id, so an edge endpoint is valid if it is a realized action OR a named anchor.
    std::unordered_set<uint64_t> namedActions;
    for (const auto& [actionId, action] : result.contract.actions) {
        namedActions.insert(actionId);
        namedActions.insert(action.anchorAction);
    }
    for (const GirFrameRequiresSpec& need : result.contract.requires_) {
        auto destinations = anchorEntries.find(need.destinationAction);
        if (destinations == anchorEntries.end()) continue;
        for (BasicBlock* block : destinations->second)
            requiredValues[block][need.sourceAction][need.genId].insert(need.values.begin(),
                                                                        need.values.end());
    }

    for (const GirFrameIncomingSpec& incoming : result.contract.incomings) {
        if (!result.contract.generations.contains(incoming.genId))
            report_fatal_error("GirFrameAnalysis: INCOMING names unknown generation");
        auto destinations = anchorEntries.find(incoming.destinationAction);
        const bool source = realizedAnchors.contains(incoming.sourceAction);
        const bool destination = destinations != anchorEntries.end();
        // The contract is MODULE scope and this is one function of it, so an edge between two
        // actions neither of which lands here belongs to another function. One end landing and the
        // other not is a dangling edge, which is still an error.
        if (!source && !destination) continue;
        if (!namedActions.contains(incoming.sourceAction) ||
            !namedActions.contains(incoming.destinationAction))
            report_fatal_error("GirFrameAnalysis: INCOMING names unknown action");
        if (!source || !destination)
            report_fatal_error(
                "GirFrameAnalysis: INCOMING action anchor has no physical realization");
        const int ring = result.contract.generations.at(incoming.genId).ring;
        int value = incoming.value % ring;
        if (value < 0) value += ring;
        for (BasicBlock* block : destinations->second) {
            auto [stored, inserted] = incomingValues[block][incoming.sourceAction].emplace(
                incoming.genId, IncomingValue{value, incoming.relative});
            if (!inserted && stored->second != IncomingValue{value, incoming.relative})
                report_fatal_error(
                    "GirFrameAnalysis: conflicting INCOMING values for physical block");
        }
    }

    const auto& loops = AM.getResult<LoopAnalysis>(function);
    std::unordered_map<int, const Loop*> genLoops;
    for (const auto& [genId, _gen] : result.contract.generations) {
        std::unordered_map<const Loop*, size_t> votes;
        bool realizedHere = false;
        for (const GirAccessSpec& access : result.contract.accesses) {
            if (access.genId != genId) continue;
            auto actionIt = result.actionInstructions.find(access.actionId);
            if (actionIt == result.actionInstructions.end()) continue;
            for (StinkyInstruction* inst : actionIt->second) {
                realizedHere = true;
                const Loop* loop = containingLoop(loops, instructionBlock.at(inst));
                if (loop) ++votes[loop];
            }
        }
        // The contract is MODULE scope; a generation no access here realizes is another
        // function's and has no loop of ours to map to.
        if (!realizedHere) continue;
        const Loop* chosen = nullptr;
        size_t best = 0;
        bool tied = false;
        for (const Loop& candidate : loops) {
            auto vote = votes.find(&candidate);
            if (vote == votes.end()) continue;
            if (vote->second > best) {
                chosen = &candidate;
                best = vote->second;
                tied = false;
            } else if (vote->second == best) {
                tied = true;
            }
        }
        // An INCOMING names its destination block, so it says which loop rotates this generation
        // exactly; the vote only counts where accesses happen to sit, and an unequal split across
        // two loops is a silent majority rather than an error.
        std::set<const Loop*> byIncoming;
        for (const GirFrameIncomingSpec& in : result.contract.incomings) {
            if (in.genId != genId) continue;
            auto destinations = anchorEntries.find(in.destinationAction);
            if (destinations == anchorEntries.end()) continue;
            for (BasicBlock* block : destinations->second)
                if (const Loop* loop = containingLoop(loops, block)) byIncoming.insert(loop);
        }
        if (byIncoming.size() == 1) {
            genLoops[genId] = *byIncoming.begin();
        } else if (chosen && !tied) {
            genLoops[genId] = chosen;
        } else if (!votes.empty()) {
            report_fatal_error("GirFrameAnalysis: generation maps ambiguously to ST loops");
        } else if (std::any_of(result.contract.incomings.begin(), result.contract.incomings.end(),
                               [genId](const GirFrameIncomingSpec& in) {
                                   return in.genId == genId && in.relative;
                               })) {
            // Dropping it silently leaves its phase at 0 for the whole function, so every access
            // collapses onto `gdelta % ring` and distinct buffers alias onto one storage id.
            // A generation rotates exactly when a relative incoming advances it; a guard has none.
            report_fatal_error("GirFrameAnalysis: rotating generation maps to no ST loop");
        }
    }

    for (const Loop& loop : loops) result.backEdges.insert(edgeKey(loop.latchBB, loop.headerBB));

    GirFrame base;
    for (const auto& [genId, _] : result.contract.generations) base.setPhase(genId, 0);
    BasicBlock* entry = function.getEntryBlock();
    if (!entry) return result;

    using FrameState = std::pair<GirFrame, uint64_t>;
    std::unordered_map<BasicBlock*, std::set<FrameState>> frameSets;
    std::deque<GirFrameNode> work;
    frameSets[entry].insert({base, 0});
    work.push_back({entry, base, 0});
    size_t stateCount = 1;
    constexpr size_t kMaxFrameStates = 1U << 20U;
    // Can this frame take this edge?  A guard generation records which arm ran, so an edge the
    // arm cannot reach is refused -- an undecided guard satisfies every constraint, which is what
    // keeps a kernel with no correlated branches behaving exactly as before.
    auto edgeFeasible = [&](const GirFrameNode& node, BasicBlock* successor) {
        auto blockNeeds = requiredValues.find(successor);
        if (blockNeeds == requiredValues.end()) return true;
        uint64_t outgoingAction = node.incomingAction;
        auto anchor = blockAnchor.find(node.block);
        if (anchor != blockAnchor.end()) outgoingAction = anchor->second;
        auto source = blockNeeds->second.find(outgoingAction);
        if (source == blockNeeds->second.end()) return true;
        for (const auto& [genId, allowed] : source->second)
            if (!allowed.contains(node.frame.phaseOf(genId))) return false;
        return true;
    };

    auto advanceNode = [&](const GirFrameNode& node, BasicBlock* successor) {
        GirFrame next =
            advanceFrame(node.frame, node.block, successor, loops, genLoops, result.contract);
        uint64_t outgoingAction = node.incomingAction;
        auto anchor = blockAnchor.find(node.block);
        if (anchor != blockAnchor.end()) outgoingAction = anchor->second;
        auto blockIncoming = incomingValues.find(successor);
        if (blockIncoming != incomingValues.end()) {
            auto source = blockIncoming->second.find(outgoingAction);
            if (source != blockIncoming->second.end())
                for (const auto& [genId, incoming] : source->second) {
                    int value = incoming.value;
                    if (incoming.relative) {
                        const int ring = result.contract.generations.at(genId).ring;
                        value = (node.frame.phaseOf(genId) + value) % ring;
                    }
                    next.setPhase(genId, value);
                }
        }
        const bool successorHasActions = blockActions.find(successor) != blockActions.end() &&
                                         !blockActions.at(successor).empty();
        return GirFrameNode{successor, std::move(next), successorHasActions ? 0 : outgoingAction};
    };
    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        for (BasicBlock* succ : node.block->getSuccessors()) {
            if (!edgeFeasible(node, succ)) continue;
            GirFrameNode next = advanceNode(node, succ);
            if (frameSets[succ].insert({next.frame, next.incomingAction}).second) {
                if (++stateCount > kMaxFrameStates)
                    report_fatal_error("GirFrameAnalysis: finite frame graph exceeds bound");
                work.push_back(std::move(next));
            }
        }
    }

    // Program order, not `frameSets` order: a pointer-keyed hash map iterates differently
    // between runs, and the edge map keeps what it is given.
    for (BasicBlock& block : function) {
        auto states = frameSets.find(&block);
        if (states == frameSets.end()) continue;
        std::set<GirFrame> frames;
        for (const auto& [frame, _incomingAction] : states->second) frames.insert(frame);
        result.blockFrames[&block] = std::vector<GirFrame>(frames.begin(), frames.end());
        for (const auto& [frame, incomingAction] : states->second) {
            GirFrameNode node{&block, frame, incomingAction};
            auto& successors = result.edges[node];
            for (BasicBlock* succ : block.getSuccessors())
                if (edgeFeasible(node, succ)) successors.push_back(advanceNode(node, succ));
        }
    }

    std::map<StorageKey, int> storageIds;
    for (size_t accessIndex = 0; accessIndex < result.contract.accesses.size(); ++accessIndex) {
        const GirAccessSpec& access = result.contract.accesses[accessIndex];
        auto actionSpec = result.contract.actions.find(access.actionId);
        auto instructions = result.actionInstructions.find(access.actionId);
        if (actionSpec == result.contract.actions.end() ||
            instructions == result.actionInstructions.end())
            continue;  // A folded GIR action can intentionally have no physical realization.
        for (StinkyInstruction* inst : instructions->second) {
            if (!instructionRealizes(actionSpec->second.kind, *inst)) continue;
            BasicBlock* block = instructionBlock.at(inst);
            std::set<GirFrame> frames;
            for (const auto& [frame, _incomingAction] : frameSets.at(block)) frames.insert(frame);
            for (const GirFrame& frame : frames) {
                int generation = access.absoluteGeneration;
                if (generation < 0) {
                    if (access.genId < 0)
                        report_fatal_error(
                            "GirFrameAnalysis: relative shared access has no generation");
                    generation = frame.phaseOf(access.genId) + access.gdelta;
                }
                generation %= access.ring;
                if (generation < 0) generation += access.ring;
                result.occurrences.push_back({inst, block, instructionIndex.at(inst), accessIndex,
                                              frame,
                                              concreteStorage(access, generation, storageIds)});
            }
        }
    }

    return result;
}

GirHazardFenceCover enclosingFences(
    const GirFrameAnalysis::Result& frames, const GirFrameHazard& hazard,
    const std::function<bool(const StinkyInstruction&)>& alsoFences) {
    GirHazardFenceCover cover;

    std::vector<GirFrameNode> producerNodes;
    const auto rememberProducer = [&](const GirFrameNode& node) {
        if (node.block != hazard.producerBlock || !(node.frame == hazard.producerFrame)) return;
        if (std::find(producerNodes.begin(), producerNodes.end(), node) == producerNodes.end())
            producerNodes.push_back(node);
    };
    for (const auto& [node, successors] : frames.edges) {
        rememberProducer(node);
        for (const GirFrameNode& successor : successors) rememberProducer(successor);
    }

    struct Step {
        GirFrameNode producerNode;
        GirFrameNode node;
        size_t begin = 0;
        int gap = 0;
        StinkyInstruction* anchor = nullptr;
        GirFrameNode anchorNode;
        int anchorGap = 0;
    };

    const auto opensPair = [](const StinkyInstruction& inst) {
        return isBarrierSignal(inst) && isSplitBarrierAllWave(inst);
    };
    const auto closesPair = [](const StinkyInstruction& inst) {
        return isBarrierWait(inst) && isSplitBarrierAllWave(inst);
    };
    const auto scan = [&](Step& step, size_t end, bool includeVirtualAtEnd) {
        size_t index = 0;
        for (IRBase& ir : *step.node.block) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (!inst) continue;
            const size_t here = index++;
            if (here < step.begin) continue;
            if (here > end || (here == end && !includeVirtualAtEnd)) break;
            if (here == end) {
                if (alsoFences && alsoFences(*inst)) {
                    step.anchor = inst;
                    step.anchorNode = step.node;
                    step.anchorGap = step.gap;
                }
                break;
            }
            if (!(isFence(*inst) || isBarrier(*inst) || (alsoFences && alsoFences(*inst))))
                continue;
            if (closesPair(*inst) && step.anchor && step.anchorNode == step.node &&
                opensPair(*step.anchor))
                continue;
            step.anchor = inst;
            step.anchorNode = step.node;
            step.anchorGap = step.gap;
        }
    };

    std::deque<Step> work;
    for (const GirFrameNode& producerNode : producerNodes)
        work.push_back(
            {producerNode, producerNode, hazard.producerIndex + 1, 0, nullptr, GirFrameNode{}, 0});

    using StateKey =
        std::tuple<BasicBlock*, GirFrame, uint64_t, BasicBlock*, GirFrame, uint64_t, size_t, int,
                   StinkyInstruction*, BasicBlock*, GirFrame, uint64_t, int>;
    std::set<StateKey> seen;
    const auto keyOf = [](const Step& step) {
        return StateKey{step.producerNode.block,
                        step.producerNode.frame,
                        step.producerNode.incomingAction,
                        step.node.block,
                        step.node.frame,
                        step.node.incomingAction,
                        step.begin,
                        step.gap,
                        step.anchor,
                        step.anchorNode.block,
                        step.anchorNode.frame,
                        step.anchorNode.incomingAction,
                        step.anchorGap};
    };
    for (const Step& step : work) seen.insert(keyOf(step));

    while (!work.empty()) {
        Step step = std::move(work.front());
        work.pop_front();
        const bool atConsumer =
            step.node.block == hazard.consumerBlock && step.node.frame == hazard.consumerFrame;

        if (step.gap == hazard.gap) {
            if (!atConsumer || hazard.consumerIndex < step.begin) continue;
            scan(step, hazard.consumerIndex, true);
            cover.hasHazardPath = true;
            if (!step.anchor) {
                cover.hasUnfencedPath = true;
                continue;
            }
            GirHazardFenceOwner owner{step.anchor, step.producerNode, step.anchorNode, step.node,
                                      step.anchorGap};
            const bool duplicate = std::any_of(
                cover.owners.begin(), cover.owners.end(), [&](const GirHazardFenceOwner& prior) {
                    return prior.anchor == owner.anchor &&
                           prior.producerNode == owner.producerNode &&
                           prior.anchorNode == owner.anchorNode &&
                           prior.consumerNode == owner.consumerNode &&
                           prior.producerToAnchorGap == owner.producerToAnchorGap;
                });
            if (!duplicate) cover.owners.push_back(std::move(owner));
            continue;
        }

        if (atConsumer && hazard.consumerIndex >= step.begin) continue;

        scan(step, std::numeric_limits<size_t>::max(), false);
        auto successors = frames.edges.find(step.node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            Step next{step.producerNode, successor,     0, step.gap + 1, step.anchor,
                      step.anchorNode,   step.anchorGap};
            if (seen.insert(keyOf(next)).second) work.push_back(std::move(next));
        }
    }

    return cover;
}

GirFrameHazardAnalysis::Result GirFrameHazardAnalysis::run(Function& function,
                                                           AnalysisManager& AM) {
    Result result;
    const auto& frames = AM.getResult<GirFrameAnalysis>(function);
    if (frames.empty()) return result;

    OccurrenceRegistry registry;
    OccurrencesByNode occurrences;
    for (const GirAccessOccurrence& occurrence : frames.occurrences) {
        LiveTouchKey key{occurrence.frame, {occurrence.inst, occurrence.accessIndex}};
        if (!registry.emplace(key, &occurrence).second)
            report_fatal_error("GirFrameHazardAnalysis found a duplicate dynamic touch");
        occurrences[{occurrence.block, occurrence.frame}].push_back(&occurrence);
    }
    for (auto& [node, touches] : occurrences) {
        (void)node;
        std::sort(touches.begin(), touches.end(), [](const auto* a, const auto* b) {
            if (a->instructionIndex != b->instructionIndex)
                return a->instructionIndex < b->instructionIndex;
            if (a->inst != b->inst) return std::less<StinkyInstruction*>{}(a->inst, b->inst);
            return a->accessIndex < b->accessIndex;
        });
    }

    auto incoming = reachingHazards(frames, occurrences, registry);
    std::set<std::tuple<int, LiveTouchKey, LiveTouchKey, int>> emitted;
    const HazardEmitter emit = [&](const GirAccessOccurrence& producer,
                                   const GirAccessOccurrence& consumer, int gap) {
        const GirAccessSpec& producerSpec = frames.contract.accesses[producer.accessIndex];
        const GirAccessSpec& consumerSpec = frames.contract.accesses[consumer.accessIndex];
        GirHazardKind kind = GirHazardKind::WAR;
        if (producerSpec.isWrite)
            kind = consumerSpec.isWrite ? GirHazardKind::WAW : GirHazardKind::RAW;
        if (!producerSpec.isWrite && !consumerSpec.isWrite) return;
        if (kind == GirHazardKind::WAW &&
            wawOrderedByRead(producerSpec, consumerSpec, frames.contract))
            return;

        const bool crossAgent = function.getGemmTileConfig().NumWaves > 1 &&
                                (producerSpec.crossAgent || consumerSpec.crossAgent);
        // The walk counts frame-graph nodes, which reduces mod the ring: a loop-carried pair
        // exactly one ring period apart lands back on the SAME node and reports 0. The unreduced
        // gdeltas recover that span only when program order proves the edge wraps around the
        // block. If the producer already precedes the consumer, 0 is the real same-trip gap for
        // both same- and cross-agent hazards; a barrier does not turn logical generation distance
        // into execution distance or retire the producer's completion counter.
        if (producer.block == consumer.block && producerSpec.genId == consumerSpec.genId &&
            producer.instructionIndex >= consumer.instructionIndex &&
            producerSpec.absoluteGeneration < 0 && consumerSpec.absoluteGeneration < 0) {
            const int span = consumerSpec.gdelta - producerSpec.gdelta;
            if (span > gap) gap = span;
        }

        LiveTouchKey producerKey{producer.frame, {producer.inst, producer.accessIndex}};
        LiveTouchKey consumerKey{consumer.frame, {consumer.inst, consumer.accessIndex}};
        if (!emitted.emplace(static_cast<int>(kind), producerKey, consumerKey, gap).second) return;
        result.hazards.push_back({kind, producer.inst, consumer.inst, producer.block,
                                  consumer.block, producer.instructionIndex,
                                  consumer.instructionIndex, producer.frame, consumer.frame, gap,
                                  crossAgent, actionOf(producer.inst), actionOf(consumer.inst)});
    };

    // Emit in program order, by POSITION: a label carries a per-run suffix, so ordering on it is
    // as unstable as the address it replaced.
    std::unordered_map<const BasicBlock*, size_t> position;
    for (BasicBlock& block : function) position.emplace(&block, position.size());
    const auto positionOf = [&position](const BasicBlock* block) {
        auto found = position.find(block);
        return found == position.end() ? std::numeric_limits<size_t>::max() : found->second;
    };
    std::vector<const GirFrameNode*> order;
    order.reserve(incoming.size());
    for (const auto& [node, _state] : incoming) order.push_back(&node);
    std::sort(order.begin(), order.end(), [&](const GirFrameNode* lhs, const GirFrameNode* rhs) {
        const size_t left = positionOf(lhs->block), right = positionOf(rhs->block);
        if (left != right) return left < right;
        if (!(lhs->frame == rhs->frame)) return lhs->frame < rhs->frame;
        return lhs->incomingAction < rhs->incomingAction;
    });
    for (const GirFrameNode* node : order) {
        auto touches = occurrences.find(*node);
        static const std::vector<const GirAccessOccurrence*> empty;
        (void)walkHazards(*node, touches == occurrences.end() ? empty : touches->second,
                          incoming.at(*node), frames.contract, registry, &emit);
    }

    return result;
}

}  // namespace stinkytofu
