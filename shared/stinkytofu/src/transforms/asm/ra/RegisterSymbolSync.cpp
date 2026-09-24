/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/ra/RegisterSymbolSync.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/SymbolicRegName.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

struct OperandRef {
    StinkyInstruction* instruction = nullptr;
    bool isDestination = false;
    size_t operand = 0;

    bool operator==(const OperandRef& other) const {
        return instruction == other.instruction && isDestination == other.isDestination &&
               operand == other.operand;
    }
};

struct OperandRefHash {
    size_t operator()(const OperandRef& ref) const {
        const size_t h1 = std::hash<StinkyInstruction*>{}(ref.instruction);
        const size_t h2 = std::hash<size_t>{}(ref.operand);
        return h1 ^ (h2 << 1) ^ (ref.isDestination ? 0x9e3779b9u : 0u);
    }
};

struct NamedUse {
    OperandRef ref;
    std::string baseSymbol;
    RegType regType = RegType::UNKNOWN;
    uint32_t idx = 0;
    std::string fullName;
    /// The index this name claims: `.set` value plus the name's own offset terms.
    /// `sgprSrdD+1` against `.set sgprSrdD, 20` claims 21, not 20.
    std::optional<int64_t> claimedIdx;
    /// The allocator moved this operand, so its name is only safe to keep if it
    /// can be shown to still be accurate.
    bool rewritten = false;
    /// Rewritten *and* the name agreed with where the operand used to sit, so
    /// this use can take part in the per-symbol delta classification.
    bool eligible = false;

    /// How far the operand sits from what its name claims. Zero means the name
    /// is still accurate, whatever offset it carries.
    int64_t delta() const {
        return static_cast<int64_t>(idx) - claimedIdx.value_or(static_cast<int64_t>(idx));
    }
};

const StinkyRegister* operandAt(const StinkyInstruction* instruction, bool isDestination,
                                size_t operand) {
    if (instruction == nullptr) return nullptr;
    if (isDestination) {
        if (operand >= instruction->getNumDestRegs()) return nullptr;
        return &instruction->getDestRegs()[operand];
    }
    if (operand >= instruction->getNumSrcRegs()) return nullptr;
    return &instruction->getSrcRegs()[operand];
}

void clearSymbolicName(StinkyInstruction* instruction, bool isDestination, size_t operand) {
    const StinkyRegister* reg = operandAt(instruction, isDestination, operand);
    if (reg == nullptr || !reg->hasSymbolicName()) return;
    StinkyRegister updated = *reg;
    updated.setSymbolicName("");
    if (isDestination)
        instruction->setDestReg(operand, updated);
    else
        instruction->setSrcReg(operand, updated);
}

/// Why a name was removed. Worth saying, because the remedies are opposites: a
/// split is the allocator doing its job and the numeric operand is correct, while
/// an unresolved `.set` means sync could not see the binding at all — the
/// directive is outside the region being processed, or defined more than once.
enum class StripReason { Split, UnresolvedSet };

const char* stripReasonText(StripReason reason) {
    return reason == StripReason::Split ? "split" : "unresolved .set";
}

/// The operand as the emitter will now print it: `v20`, or `v[20:23]` for a range.
std::string physicalOperandText(RegType type, uint32_t idx, unsigned num) {
    const std::string prefix = regTypeToString(type);
    if (num <= 1) return prefix + std::to_string(idx);
    return prefix + "[" + std::to_string(idx) + ":" + std::to_string(idx + num - 1) + "]";
}

void appendBreadcrumb(StinkyInstruction* instruction, const std::string& note) {
    if (instruction == nullptr || note.empty()) return;
    if (CommentData* existing = instruction->getModifier<CommentData>())
        existing->comment += ", " + note;
    else
        instruction->addModifier<CommentData>(CommentData{note});
}

void collectNamedUses(Function& function, const std::unordered_map<std::string, int64_t>& setValues,
                      std::vector<NamedUse>& out) {
    out.clear();
    for (BasicBlock& bb : function) {
        for (IRBase& ir : bb) {
            auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;

            auto visit = [&](bool isDestination, size_t count) {
                for (size_t operand = 0; operand < count; ++operand) {
                    const StinkyRegister* reg = operandAt(instruction, isDestination, operand);
                    if (reg == nullptr || !reg->hasSymbolicName()) continue;
                    const std::optional<ParsedSymbolicRegName> parsed =
                        parseSymbolicRegName(reg->getSymbolicName());
                    if (!parsed.has_value()) continue;
                    NamedUse use;
                    use.ref = OperandRef{instruction, isDestination, operand};
                    use.baseSymbol = parsed->start.base;
                    use.regType = reg->reg.type;
                    use.idx = reg->reg.idx;
                    use.fullName = reg->getSymbolicName();
                    use.claimedIdx =
                        resolveNamedIndex(reg->getSymbolicName(), setValues, reg->reg.num);
                    out.push_back(std::move(use));
                }
            };
            visit(/*isDestination=*/true, instruction->getNumDestRegs());
            visit(/*isDestination=*/false, instruction->getNumSrcRegs());
        }
    }
}

void emitRegisterMap(Function& function, const std::vector<SymbolOrigin>& origins) {
    if (origins.empty()) return;
    BasicBlock* entry = function.getEntryBlock();
    if (entry == nullptr) return;

    std::ostringstream body;
    body << "// register-map: producer -> allocated\n";
    for (const SymbolOrigin& origin : origins) {
        body << "// " << origin.name << "  " << origin.producerIndex << " -> ";
        for (size_t i = 0; i < origin.allocatedIndices.size(); ++i) {
            if (i > 0) body << ", ";
            body << origin.allocatedIndices[i];
        }
        if (!origin.note.empty()) body << "  " << origin.note;
        body << "\n";
    }

    AsmDirective* directive = IRBase::createIR<AsmDirective>();
    directive->kind = AsmDirectiveKind::TEXTBLOCK;
    directive->value = body.str();
    entry->insertIR(entry->begin(), directive);
}

}  // namespace

void syncRegisterSymbols(Function& function, const std::vector<RewrittenOperand>& rewritten,
                         SymbolSyncOptions options, SymbolSyncReport* report) {
    if (rewritten.empty()) return;

    SymbolSyncReport localReport;
    SymbolSyncReport& out = report == nullptr ? localReport : *report;

    std::unordered_map<std::string, AsmSetSymbolInfo> setInfo;
    collectAsmSetSymbolInfo(function, setInfo);

    // Only symbols whose right-hand side actually resolved. An unresolved one
    // carries value 0, which would otherwise read as a genuine `.set FOO, 0` and
    // make every name that refers to it look verifiable.
    std::unordered_map<std::string, int64_t> setValues;
    for (const auto& kv : setInfo) {
        if (kv.second.resolved) setValues[kv.first] = kv.second.value;
    }

    std::vector<NamedUse> namedUses;
    collectNamedUses(function, setValues, namedUses);

    std::unordered_map<OperandRef, StripReason, OperandRefHash> clearNames;
    std::unordered_map<std::string, int64_t> rewriteSetTo;
    std::unordered_map<OperandRef, std::string, OperandRefHash> breadcrumbNotes;

    std::unordered_map<std::string, std::vector<const NamedUse*>> bySymbol;
    std::unordered_map<OperandRef, NamedUse*, OperandRefHash> useByRef;
    for (NamedUse& use : namedUses) {
        bySymbol[use.baseSymbol].push_back(&use);
        useByRef[use.ref] = &use;
    }

    for (const RewrittenOperand& entry : rewritten) {
        const StinkyRegister* reg =
            operandAt(entry.instruction, entry.isDestination, entry.operand);
        if (reg == nullptr || !reg->hasSymbolicName()) continue;

        const std::optional<int64_t> resolved =
            resolveNamedIndex(reg->getSymbolicName(), setValues, reg->reg.num);
        const OperandRef ref{entry.instruction, entry.isDestination, entry.operand};

        // Destruction records every operand it writes back, including ones it
        // put down exactly where they were. Those did not move, so whatever
        // their name meant it means still, and sync leaves them alone -- which
        // is what keeps a placeholder reference intact.
        const bool moved = entry.beforeIdx != entry.afterIdx || entry.beforeType != entry.afterType;

        const auto useIt = useByRef.find(ref);
        if (useIt != useByRef.end()) useIt->second->rewritten = moved;

        if (!resolved.has_value()) {
            clearNames.emplace(ref, StripReason::UnresolvedSet);
            continue;
        }

        if (*resolved != static_cast<int64_t>(entry.beforeIdx)) {
            // The name never described where this operand sat, so nothing about
            // where it went can be inferred from it. Only a hazard once it has
            // actually moved; standing still, it is the producer's own oddity.
            out.suspectOperands.push_back(
                "@" + function.getName() + " " + (entry.isDestination ? "dst" : "src") +
                std::to_string(entry.operand) + ": name '" + reg->getSymbolicName() +
                "' resolves to " + std::to_string(*resolved) + " but idx was " +
                std::to_string(entry.beforeIdx));
            if (moved) clearNames.emplace(ref, StripReason::UnresolvedSet);
            continue;
        }

        if (useIt != useByRef.end()) useIt->second->eligible = true;
    }

    std::unordered_set<std::string> processedSymbols;
    for (const NamedUse& use : namedUses) processedSymbols.insert(use.baseSymbol);

    for (const std::string& symbol : processedSymbols) {
        const auto infoIt = setInfo.find(symbol);
        const bool resolvedOnce = infoIt != setInfo.end() && infoIt->second.definitionCount == 1 &&
                                  setValues.contains(symbol);

        SymbolOrigin origin{symbol, resolvedOnce ? setValues.at(symbol) : 0, {}, ""};

        std::vector<const NamedUse*> eligible;
        for (const NamedUse* use : bySymbol[symbol])
            if (use->eligible) eligible.push_back(use);

        if (!resolvedOnce) {
            ++out.unresolvable;
            origin.note = "unresolvable, names cleared";
            // Every *rewritten* use, not just the ones whose name could be
            // verified. Eligibility requires resolution to have succeeded, so
            // keying on it here made this branch do nothing in exactly the case
            // it exists for: a symbol redefined or released with `UNDEF`, whose
            // uses the allocator moved. Untouched operands keep their names.
            for (const NamedUse* use : bySymbol[symbol]) {
                if (!use->rewritten) continue;
                clearNames.emplace(use->ref, StripReason::UnresolvedSet);
                ++out.namesCleared;
                origin.allocatedIndices.push_back(use->idx);
            }
            if (!origin.allocatedIndices.empty()) out.origins.push_back(std::move(origin));
            continue;
        }

        if (eligible.empty()) {
            ++out.stable;
            continue;
        }

        const int64_t oldValue = setValues.at(symbol);
        for (const NamedUse* use : eligible)
            origin.allocatedIndices.push_back(static_cast<int64_t>(use->idx));
        std::sort(origin.allocatedIndices.begin(), origin.allocatedIndices.end());
        origin.allocatedIndices.erase(
            std::unique(origin.allocatedIndices.begin(), origin.allocatedIndices.end()),
            origin.allocatedIndices.end());

        // Classify on the distance between where each operand sits and where its own
        // name claims it sits, not on the raw index. The offset terms in the name are
        // part of the claim, so `sgprSrdD+1` at s21 under `.set sgprSrdD, 20` has a
        // delta of zero and is just as stable as `sgprSrdD+0` at s20. Moving the whole
        // tuple shifts every member by the same delta, which is what makes a single
        // `.set` rewrite sufficient.
        std::unordered_set<int64_t> deltas;
        for (const NamedUse* use : eligible) deltas.insert(use->delta());

        if (deltas.size() == 1 && *deltas.begin() == 0) {
            ++out.stable;
            origin.note = "stable, name kept";
        } else if (deltas.size() == 1) {
            ++out.movedUniquely;
            rewriteSetTo[symbol] = oldValue + *deltas.begin();
            ++out.setsRewritten;
            origin.note = "moved, .set rewritten";
        } else {
            ++out.split;
            origin.note = "SPLIT, name kept where it still resolves";
            for (const NamedUse* use : eligible) {
                if (use->delta() == 0) continue;
                clearNames.emplace(use->ref, StripReason::Split);
                ++out.namesCleared;
            }
        }

        out.origins.push_back(std::move(origin));
    }

    for (const auto& kv : rewriteSetTo) {
        for (BasicBlock& bb : function) {
            for (IRBase& ir : bb) {
                auto* directive = dyn_cast<AsmDirective>(&ir);
                if (directive == nullptr || directive->kind != AsmDirectiveKind::SET) continue;
                if (directive->symbol != kv.first) continue;
                directive->value = std::to_string(kv.second);
            }
        }
    }

    for (const auto& [ref, reason] : clearNames) {
        const StinkyRegister* before = operandAt(ref.instruction, ref.isDestination, ref.operand);
        if (before == nullptr) continue;
        const std::string oldName = before->getSymbolicName();
        const std::string operandText =
            physicalOperandText(before->reg.type, before->reg.idx, before->reg.num);
        clearSymbolicName(ref.instruction, ref.isDestination, ref.operand);
        if (options.emitBreadcrumbs && !oldName.empty()) {
            std::string note = operandText;
            note += " was ";
            note += oldName;
            note += " (";
            note += stripReasonText(reason);
            note += ')';
            breadcrumbNotes.emplace(ref, std::move(note));
        }
    }

    // Walk the function rather than the hash set: an instruction that loses two
    // names must annotate them in operand order every run, not in whatever order
    // the set happens to iterate.
    if (!breadcrumbNotes.empty()) {
        for (BasicBlock& bb : function) {
            for (IRBase& ir : bb) {
                auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                if (instruction == nullptr) continue;
                // One note per distinct fact: a dest and a src that shared both a
                // register and a name would otherwise state it twice.
                std::vector<std::string> notes;
                auto visit = [&](bool isDestination, size_t count) {
                    for (size_t operand = 0; operand < count; ++operand) {
                        const auto it =
                            breadcrumbNotes.find(OperandRef{instruction, isDestination, operand});
                        if (it == breadcrumbNotes.end()) continue;
                        if (std::find(notes.begin(), notes.end(), it->second) == notes.end())
                            notes.push_back(it->second);
                    }
                };
                visit(/*isDestination=*/true, instruction->getNumDestRegs());
                visit(/*isDestination=*/false, instruction->getNumSrcRegs());
                for (const std::string& note : notes) appendBreadcrumb(instruction, note);
            }
        }
    }

    if (options.emitRegisterMap) emitRegisterMap(function, out.origins);
}

}  // namespace stinkytofu
