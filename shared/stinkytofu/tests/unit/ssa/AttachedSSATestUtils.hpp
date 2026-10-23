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
#pragma once

// Read-only queries over attached SSA.
//
// AttachedSSA stores one slot per lifted DWORD in a flat list, so recovering
// "which values did operand N bind" means walking srcRegs/destRegs alongside it.
// Tests do that constantly, and getting the walk wrong shifts every later
// operand onto the wrong value, so it lives here once.
//
// Nothing here constructs SSA: lifting is the only builder.

#include <cstddef>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

namespace stinkytofu {
namespace test {

/// Register classes \p instruction's attached SSA was built for, so the operand
/// walk steps the same way lifting did.
inline RegClassSet liftedClassesOf(const StinkyInstruction& instruction) {
    const BasicBlock* block = instruction.getParentBlock();
    const Function* function = block == nullptr ? nullptr : block->getParentFunc();
    return function == nullptr ? RegClassSet::all() : function->ssaArena().liftedClasses();
}

/// Values bound to source operand \p operand, in DWORD order. Empty when the
/// operand was not lifted or the instruction carries no attached SSA.
inline std::vector<StinkySSAValue*> ssaSourceUnits(const StinkyInstruction& instruction,
                                                   size_t operand) {
    if (!instruction.hasAttachedSSA()) return {};
    const RegClassSet classes = liftedClassesOf(instruction);
    size_t cursor = 0;
    const std::vector<StinkyRegister>& srcRegs = instruction.getSrcRegs();
    for (size_t i = 0; i < srcRegs.size(); ++i) {
        const size_t units = liftedSSAUnits(srcRegs[i], classes);
        if (units == 0) {
            // A non-lifted operand still occupies one immediate slot.
            if (cursor < instruction.getNumSSAOperands()) ++cursor;
            if (i == operand) return {};
            continue;
        }
        if (cursor + units > instruction.getNumSSAOperands()) return {};
        if (i != operand) {
            cursor += units;
            continue;
        }
        std::vector<StinkySSAValue*> values;
        values.reserve(units);
        for (size_t unit = 0; unit < units; ++unit)
            values.push_back(instruction.getSSAOperandValue(cursor + unit));
        return values;
    }
    return {};
}

/// Values defined by destination operand \p operand, in DWORD order.
inline std::vector<StinkySSAValue*> ssaDestUnits(const StinkyInstruction& instruction,
                                                 size_t operand) {
    if (!instruction.hasAttachedSSA()) return {};
    const RegClassSet classes = liftedClassesOf(instruction);
    size_t cursor = 0;
    const std::vector<StinkyRegister>& destRegs = instruction.getDestRegs();
    for (size_t i = 0; i < destRegs.size(); ++i) {
        const size_t units = liftedSSAUnits(destRegs[i], classes);
        // A non-lifted destination defines nothing and occupies no slot.
        if (units == 0) {
            if (i == operand) return {};
            continue;
        }
        if (cursor + units > instruction.getNumSSAResults()) return {};
        if (i != operand) {
            cursor += units;
            continue;
        }
        std::vector<StinkySSAValue*> values;
        values.reserve(units);
        for (size_t unit = 0; unit < units; ++unit)
            values.push_back(instruction.getSSAResult(cursor + unit));
        return values;
    }
    return {};
}

/// First value bound to source operand \p operand, or null.
inline StinkySSAValue* ssaSourceValue(const StinkyInstruction& instruction, size_t operand) {
    const std::vector<StinkySSAValue*> units = ssaSourceUnits(instruction, operand);
    return units.empty() ? nullptr : units.front();
}

/// Unit \p unit of the instruction's first lifted destination, or null.
inline StinkySSAValue* ssaDefinedValue(const StinkyInstruction& instruction, size_t unit = 0) {
    const std::vector<StinkySSAValue*> units = ssaDestUnits(instruction, 0);
    return unit < units.size() ? units[unit] : nullptr;
}

/// Physical register a value was lifted from.
inline RegKey bindingKeyOf(const StinkySSAValue* value) {
    if (value == nullptr || !value->hasPhysicalBinding())
        return RegKey{RegType::UNKNOWN, 0, RegHalf::NONE};
    const StinkySSAValue::PhysicalBinding& binding = value->physical();
    return RegKey{binding.type, binding.idx, RegHalf::NONE};
}

/// Physical register indices behind \p values, for grouping checks.
inline std::vector<unsigned> bindingIndicesOf(const std::vector<StinkySSAValue*>& values) {
    std::vector<unsigned> indices;
    indices.reserve(values.size());
    for (const StinkySSAValue* value : values) indices.push_back(bindingKeyOf(value).idx);
    return indices;
}

/// Block argument bound to \p key, or null when the block takes none for it.
inline const SSABlockArgument* blockArgumentFor(const BasicBlock& block, const RegKey& key) {
    for (const SSABlockArgument& arg : block.ssaArguments()) {
        if (bindingKeyOf(arg.value) == key) return &arg;
    }
    return nullptr;
}

inline const SSABlockArgument* vgprArgumentFor(const BasicBlock& block, unsigned idx) {
    return blockArgumentFor(block, RegKey{RegType::V, idx, RegHalf::NONE});
}

/// Arguments that merge control-flow edges, as opposed to function live-ins.
inline size_t mergeArgumentCount(const BasicBlock& block) {
    size_t count = 0;
    for (const SSABlockArgument& arg : block.ssaArguments()) {
        if (!arg.incoming.empty()) ++count;
    }
    return count;
}

inline size_t mergeArgumentCount(const Function& function) {
    size_t count = 0;
    for (const BasicBlock& block : function) count += mergeArgumentCount(block);
    return count;
}

/// Values arriving at \p arg from \p predecessor. A block can be a predecessor
/// more than once, so this is a list rather than a single value.
///
/// Lift appends incoming as its dominator walk reaches each predecessor, so
/// looking edges up by predecessor is what keeps a test independent of that
/// walk order.
inline std::vector<StinkySSAValue*> incomingValuesFrom(const SSABlockArgument& arg,
                                                       const BasicBlock* predecessor) {
    std::vector<StinkySSAValue*> values;
    for (const SSABlockIncoming& incoming : arg.incoming) {
        if (incoming.predecessor != predecessor) continue;
        values.push_back(incoming.use == nullptr ? nullptr : incoming.use->value());
    }
    return values;
}

/// The single value arriving from \p predecessor, or null when that edge is
/// absent or duplicated.
inline StinkySSAValue* incomingValueFrom(const SSABlockArgument& arg,
                                         const BasicBlock* predecessor) {
    const std::vector<StinkySSAValue*> values = incomingValuesFrom(arg, predecessor);
    return values.size() == 1 ? values.front() : nullptr;
}

}  // namespace test
}  // namespace stinkytofu
