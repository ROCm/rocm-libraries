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
#include "stinkytofu/serialization/ssa/CanonicalSSAPrinter.hpp"

#include <algorithm>
#include <limits>
#include <ostream>
#include <sstream>
#include <tuple>
#include <utility>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

constexpr uint32_t kUnknownOrder = std::numeric_limits<uint32_t>::max();

/// Register class of a value, matching AsmPrinter's spelling.
std::string classText(RegType type) {
    if (type == RegType::AGPR) return "acc";
    if (!isValidRegType(type)) return "?";
    return regTypeToString(type);
}

const char* kindText(SSAValueKind kind) {
    switch (kind) {
        case SSAValueKind::LiveIn:
            return "livein";
        case SSAValueKind::Undef:
            return "undef";
        case SSAValueKind::InstructionDef:
            return "instruction_def";
        case SSAValueKind::Phi:
            return "phi_result";
    }
    return "unknown";
}

std::string join(const std::vector<std::string>& parts, const char* separator) {
    std::string text;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) text += separator;
        text += parts[i];
    }
    return text;
}

}  // namespace

CanonicalSSAPrinter::CanonicalSSAPrinter(std::ostream& os,
                                         const CanonicalSSAPrinterOptions& options)
    : os_(os), options_(options) {}

void CanonicalSSAPrinter::line(unsigned depth, const std::string& text) {
    os_ << std::string(static_cast<size_t>(depth) * options_.indent, ' ') << text << "\n";
}

void CanonicalSSAPrinter::buildNames(const Function& function) {
    blockNames_.clear();
    blockOrder_.clear();
    instructionOrder_.clear();

    // A label is only usable as an identity if it is present and unique.
    std::unordered_map<std::string, uint32_t> labelCounts;
    for (const BasicBlock& bb : function) {
        if (!bb.getLabel().empty()) ++labelCounts[bb.getLabel()];
    }

    uint32_t blockIndex = 0;
    uint32_t instructionIndex = 0;
    for (const BasicBlock& bb : function) {
        const std::string& label = bb.getLabel();
        const bool unique = !label.empty() && labelCounts[label] == 1;
        blockNames_.emplace(&bb, unique ? label : ("bb" + std::to_string(blockIndex)));
        blockOrder_.emplace(&bb, blockIndex);
        ++blockIndex;

        for (const IRBase& ir : bb) {
            if (const auto* inst = dyn_cast<StinkyInstruction>(&ir))
                instructionOrder_.emplace(inst, instructionIndex++);
        }
    }
}

std::string CanonicalSSAPrinter::valueRef(SSAValueID id) const {
    if (ssa_ == nullptr || !ssa_->containsValue(id))
        return "<invalid-ssa:%" + std::to_string(id) + ">";
    return "%" + std::to_string(id) + ":" + classText(ssa_->value(id).origin.type);
}

std::string CanonicalSSAPrinter::phiRef(SSAPhiID id) const {
    if (ssa_ == nullptr || !ssa_->containsPhi(id))
        return "<invalid-phi:phi#" + std::to_string(id) + ">";
    return "phi#" + std::to_string(id);
}

std::string CanonicalSSAPrinter::blockRef(const BasicBlock* block) const {
    if (block == nullptr) return "<null-block>";
    auto it = blockNames_.find(block);
    if (it == blockNames_.end()) return "<foreign-block>";
    return "^" + it->second;
}

std::string CanonicalSSAPrinter::instructionRef(const StinkyInstruction* instruction) const {
    if (instruction == nullptr) return "<null-instruction>";
    auto it = instructionOrder_.find(instruction);
    if (it == instructionOrder_.end()) return "<foreign-instruction>";
    return "#" + std::to_string(it->second);
}

std::string CanonicalSSAPrinter::bindingText(const SSAOperandBinding& binding) const {
    std::vector<std::string> units;
    units.reserve(binding.units.size());
    for (SSAValueID id : binding.units) units.push_back(valueRef(id));
    return "[" + join(units, ", ") + "]";
}

std::string CanonicalSSAPrinter::originListText(
    const std::vector<SSAOperandBinding>& bindings) const {
    std::vector<std::string> origins;
    for (const SSAOperandBinding& binding : bindings) {
        for (SSAValueID id : binding.units) {
            origins.push_back(ssa_ != nullptr && ssa_->containsValue(id)
                                  ? regKeyToString(ssa_->value(id).origin)
                                  : "?");
        }
    }
    if (origins.empty()) return "";
    return "[" + join(origins, ", ") + "]";
}

std::string CanonicalSSAPrinter::physicalText(const StinkyInstruction& instruction) const {
    std::vector<std::string> dests;
    for (const StinkyRegister& reg : instruction.getDestRegs()) dests.push_back(toString(reg));
    std::vector<std::string> srcs;
    for (const StinkyRegister& reg : instruction.getSrcRegs()) srcs.push_back(toString(reg));

    std::string text;
    if (!dests.empty()) text += join(dests, ", ") + " = ";
    text += "\"st." + std::string(instruction.getHwInstDesc()->mnemonic) + "\"";
    text += "(" + join(srcs, ", ") + ")";
    return text;
}

std::string CanonicalSSAPrinter::useListText(const SSAValue& value) const {
    // Sort by function position so the list does not depend on insertion order.
    using Key = std::tuple<int, uint32_t, uint32_t, uint32_t>;
    std::vector<std::pair<Key, std::string>> entries;
    entries.reserve(value.uses.size());

    for (const SSAUse& use : value.uses) {
        if (use.isPhiUse()) {
            uint32_t predOrder = kUnknownOrder;
            if (use.predecessor != nullptr) {
                auto it = blockOrder_.find(use.predecessor);
                if (it != blockOrder_.end()) predOrder = it->second;
            }
            entries.emplace_back(
                Key{1, use.phi, predOrder, 0},
                "{ " + phiRef(use.phi) + ", pred = " + blockRef(use.predecessor) + " }");
            continue;
        }

        uint32_t instOrder = kUnknownOrder;
        if (use.instruction != nullptr) {
            auto it = instructionOrder_.find(use.instruction);
            if (it != instructionOrder_.end()) instOrder = it->second;
        }
        entries.emplace_back(Key{0, instOrder, use.operand, use.unit},
                             "{ inst = " + instructionRef(use.instruction) +
                                 ", src = " + std::to_string(use.operand) +
                                 ", unit = " + std::to_string(use.unit) + " }");
    }

    std::stable_sort(entries.begin(), entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<std::string> texts;
    texts.reserve(entries.size());
    for (auto& entry : entries) texts.push_back(std::move(entry.second));
    return "[" + join(texts, ", ") + "]";
}

void CanonicalSSAPrinter::printMissing(const Function& function) {
    line(0, "ssa.func @" + function.getName() + " {");
    line(1, "<no canonical SSA attached>");
    line(0, "}");
}

void CanonicalSSAPrinter::print(const Function& function, const CanonicalSSA& ssa) {
    ssa_ = &ssa;
    buildNames(function);
    printedValues_.assign(ssa.valueCount() + 1, false);

    line(0, "ssa.func @" + function.getName() + " {");
    printInitialValues();
    for (const BasicBlock& bb : function) printBlock(bb);
    printUnprintedValues();
    line(0, "}");

    ssa_ = nullptr;
}

void CanonicalSSAPrinter::printInitialValues() {
    std::vector<const SSAValue*> initial;
    for (const SSAValue& value : ssa_->values()) {
        if (value.kind == SSAValueKind::LiveIn || value.kind == SSAValueKind::Undef)
            initial.push_back(&value);
    }
    if (initial.empty()) return;

    line(1, "initial_values:");
    for (const SSAValue* value : initial) {
        std::string text = valueRef(value->id) + " = " + kindText(value->kind);
        if (options_.printProvenance) text += " { origin = " + regKeyToString(value->origin) + " }";
        line(2, text);
        if (options_.printUses) line(3, "uses = " + useListText(*value));
        if (value->id < printedValues_.size()) printedValues_[value->id] = true;
    }
}

void CanonicalSSAPrinter::printBlock(const BasicBlock& block) {
    line(1, blockRef(&block) + ":");

    for (SSAPhiID id : ssa_->phisForBlock(block)) {
        if (!ssa_->containsPhi(id)) {
            line(2, phiRef(id));
            continue;
        }
        printPhi(ssa_->phi(id));
    }

    for (const IRBase& ir : block) {
        const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
        if (instruction == nullptr) continue;
        // Labels are block boundaries, not dataflow.
        if (instruction->getUnifiedOpcode() == GFX::LABEL) continue;
        printInstruction(*instruction);
    }
}

void CanonicalSSAPrinter::printPhi(const SSAPhi& phi) {
    std::vector<std::string> edges;
    edges.reserve(phi.incoming.size());
    for (const SSAPhiIncoming& incoming : phi.incoming)
        edges.push_back(blockRef(incoming.predecessor) + ": " + valueRef(incoming.value));

    std::string text = valueRef(phi.result) + " = phi(" + join(edges, ", ") + ")";
    if (options_.printProvenance) text += " { origin = " + regKeyToString(phi.origin) + " }";
    line(2, text);

    if (options_.printUses && ssa_->containsValue(phi.result))
        line(3, "uses = " + useListText(ssa_->value(phi.result)));
    if (phi.result < printedValues_.size()) printedValues_[phi.result] = true;
}

void CanonicalSSAPrinter::printInstruction(const StinkyInstruction& instruction) {
    const SSAInstructionInfo* info = ssa_->findInstructionInfo(instruction);

    std::vector<std::string> attributes{"inst = " + instructionRef(&instruction)};
    std::string text;

    if (info == nullptr) {
        attributes.emplace_back("unmapped");
        text = "\"st." + std::string(instruction.getHwInstDesc()->mnemonic) + "\"()";
    } else {
        // Destination operands: units within an operand are comma separated,
        // separate operands are split by '|' so grouping stays unambiguous.
        std::vector<std::string> destGroups;
        for (const SSAOperandBinding& binding : info->destinations) {
            if (binding.units.empty()) continue;
            std::vector<std::string> units;
            units.reserve(binding.units.size());
            for (SSAValueID id : binding.units) units.push_back(valueRef(id));
            destGroups.push_back(join(units, ", "));
        }
        if (!destGroups.empty()) text += join(destGroups, " | ") + " = ";

        text += "\"st." + std::string(instruction.getHwInstDesc()->mnemonic) + "\"(";
        std::vector<std::string> sources;
        sources.reserve(info->sources.size());
        for (size_t operand = 0; operand < info->sources.size(); ++operand) {
            sources.push_back("src" + std::to_string(operand) + " = " +
                              bindingText(info->sources[operand]));
        }
        text += join(sources, ", ") + ")";

        if (info->sources.size() != instruction.getSrcRegs().size() ||
            info->destinations.size() != instruction.getDestRegs().size()) {
            attributes.emplace_back("operand-count-mismatch");
        }
        if (options_.printProvenance) {
            const std::string origins = originListText(info->destinations);
            if (!origins.empty()) attributes.push_back("origin = " + origins);
        }
    }

    line(2, text + " { " + join(attributes, ", ") + " }");

    if (options_.printPhysicalInstruction) line(3, "// physical: " + physicalText(instruction));

    if (info == nullptr) return;
    for (const SSAOperandBinding& binding : info->destinations) {
        for (SSAValueID id : binding.units) {
            if (!ssa_->containsValue(id)) continue;
            if (options_.printUses)
                line(3, valueRef(id) + " uses = " + useListText(ssa_->value(id)));
            if (id < printedValues_.size()) printedValues_[id] = true;
        }
    }
}

void CanonicalSSAPrinter::printUnprintedValues() {
    std::vector<const SSAValue*> unprinted;
    for (const SSAValue& value : ssa_->values()) {
        if (value.id < printedValues_.size() && !printedValues_[value.id])
            unprinted.push_back(&value);
    }
    if (unprinted.empty()) return;

    // Values whose definition site is unreachable from the function; kept in the
    // dump so a malformed graph is fully visible.
    line(1, "unprinted_values:");
    for (const SSAValue* value : unprinted) {
        std::string text = valueRef(value->id) + " = " + kindText(value->kind);
        std::vector<std::string> attributes;
        if (options_.printProvenance)
            attributes.push_back("origin = " + regKeyToString(value->origin));
        if (value->kind == SSAValueKind::InstructionDef) {
            attributes.push_back("inst = " + instructionRef(value->definingInstruction));
            attributes.push_back("dst = " + std::to_string(value->definingOperand) +
                                 ", unit = " + std::to_string(value->definingUnit));
        }
        if (value->kind == SSAValueKind::Phi)
            attributes.push_back("phi = " + phiRef(value->definingPhi));
        if (!attributes.empty()) text += " { " + join(attributes, ", ") + " }";
        line(2, text);
        if (options_.printUses) line(3, "uses = " + useListText(*value));
    }
}

std::string canonicalSSAToString(const Function& function, const CanonicalSSA& ssa,
                                 const CanonicalSSAPrinterOptions& options) {
    std::ostringstream out;
    CanonicalSSAPrinter(out, options).print(function, ssa);
    return out.str();
}

}  // namespace stinkytofu
