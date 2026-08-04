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

// Hand-construction helpers for canonical SSA graphs.
//
// These build graphs the way the lifting pass is specified to build them, so a
// test can produce a valid graph in a few lines and then deliberately break
// one invariant. They are not a lifting implementation: source values are
// always supplied explicitly by the caller.

#include <cstdint>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace test {

/// Per-operand source values; one inner list per physical source operand, and
/// one entry per DWORD of that operand (empty for non-allocatable operands).
using SSASourceBindings = std::vector<std::vector<SSAValueID>>;

inline RegKey vgprKey(unsigned idx) {
    return RegKey{RegType::V, idx, RegHalf::NONE};
}

/// Units an operand is expected to bind under the full-DWORD VGPR/SGPR rule.
inline size_t ssaUnitsOf(const StinkyRegister& reg) {
    if (!reg.isRegister() || reg.isVirtualReg()) return 0;
    if (!isAllocatableReg(reg.reg.type)) return 0;
    return reg.reg.num;
}

inline SSAValueID addInitialValue(CanonicalSSABuilder& builder, SSAValueKind kind, RegKey origin) {
    SSAValue value;
    value.kind = kind;
    value.origin = origin;
    return builder.addValue(std::move(value));
}

inline SSAValueID addLiveIn(CanonicalSSABuilder& builder, unsigned vgpr) {
    return addInitialValue(builder, SSAValueKind::LiveIn, vgprKey(vgpr));
}

inline SSAValueID addUndef(CanonicalSSABuilder& builder, unsigned vgpr) {
    return addInitialValue(builder, SSAValueKind::Undef, vgprKey(vgpr));
}

/// Bind one instruction: link \p sources and create one new value per
/// destination DWORD. Returns the destination values in operand/unit order.
inline std::vector<SSAValueID> bindInstruction(CanonicalSSABuilder& builder,
                                               StinkyInstruction& instruction,
                                               const SSASourceBindings& sources) {
    SSAInstructionInfo info;

    info.sources.resize(instruction.getSrcRegs().size());
    for (size_t operand = 0; operand < sources.size() && operand < info.sources.size(); ++operand) {
        info.sources[operand].units = sources[operand];
        for (size_t unit = 0; unit < sources[operand].size(); ++unit) {
            SSAUse use;
            use.instruction = &instruction;
            use.operand = static_cast<uint32_t>(operand);
            use.unit = static_cast<uint32_t>(unit);
            builder.value(sources[operand][unit]).uses.push_back(use);
        }
    }

    std::vector<SSAValueID> defined;
    info.destinations.resize(instruction.getDestRegs().size());
    for (size_t operand = 0; operand < info.destinations.size(); ++operand) {
        const StinkyRegister& reg = instruction.getDestRegs()[operand];
        const size_t units = ssaUnitsOf(reg);
        for (size_t unit = 0; unit < units; ++unit) {
            SSAValue value;
            value.kind = SSAValueKind::InstructionDef;
            value.origin = toRegKey(reg, static_cast<unsigned>(unit));
            value.definingInstruction = &instruction;
            value.definingOperand = static_cast<uint32_t>(operand);
            value.definingUnit = static_cast<uint32_t>(unit);
            const SSAValueID id = builder.addValue(std::move(value));
            info.destinations[operand].units.push_back(id);
            defined.push_back(id);
        }
    }

    builder.setInstructionInfo(instruction, std::move(info));
    return defined;
}

/// Add a canonical PHI at \p block. \p incoming is ordered like the block's
/// predecessors. Returns the PHI result value.
inline SSAValueID addPhi(CanonicalSSABuilder& builder, BasicBlock& block, RegKey origin,
                         const std::vector<SSAValueID>& incoming) {
    SSAValue result;
    result.kind = SSAValueKind::Phi;
    result.origin = origin;
    const SSAValueID resultID = builder.addValue(std::move(result));

    SSAPhi phi;
    phi.block = &block;
    phi.origin = origin;
    phi.result = resultID;
    const std::vector<BasicBlock*>& predecessors = block.getPredecessors();
    for (size_t edge = 0; edge < incoming.size(); ++edge) {
        BasicBlock* predecessor = edge < predecessors.size() ? predecessors[edge] : nullptr;
        phi.incoming.push_back(SSAPhiIncoming{predecessor, incoming[edge]});
    }

    const SSAPhiID phiID = builder.addPhi(std::move(phi));
    builder.value(resultID).definingPhi = phiID;
    for (const SSAPhiIncoming& edge : builder.phi(phiID).incoming) {
        SSAUse use;
        use.phi = phiID;
        use.predecessor = edge.predecessor;
        builder.value(edge.value).uses.push_back(use);
    }
    builder.addPhiToBlock(block, phiID);
    return resultID;
}

}  // namespace test
}  // namespace stinkytofu
