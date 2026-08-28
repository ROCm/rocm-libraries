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
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"

#include <cassert>
#include <iostream>  // TODO: don't use iostream.
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

#define DEBUG_TYPE "ExecMaskGrouping"

namespace stinkytofu {
namespace {

// EXEC (64-bit) aliases EXEC_LO and EXEC_HI; EXEC_LO and EXEC_HI do not alias
// each other. In wave32 (execReg = EXEC_LO), EXEC_HI writes are irrelevant and
// correctly return false here; EXEC writes (e.g. s_mov_b64 exec, X) return true
// because EXEC covers EXEC_LO.
bool overlapsExec(const StinkyRegister& reg, const StinkyRegister& execReg) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;
    RegType t = reg.reg.type;
    RegType et = execReg.reg.type;
    if (t == et) return true;
    if (t == RegType::EXEC || et == RegType::EXEC)
        return t == RegType::EXEC_LO || t == RegType::EXEC_HI || et == RegType::EXEC_LO ||
               et == RegType::EXEC_HI;
    return false;
}

bool writesExecReg(const StinkyInstruction& inst, const StinkyRegister& execReg) {
    for (const StinkyRegister& d : inst.getDestRegs())
        if (overlapsExec(d, execReg)) return true;
    return false;
}

// A full-mask reset: any exec write whose sole source is the literal -1.
// The single-source constraint is the line we draw — it covers every sane reset
// idiom (s_mov_b32 exec_lo, -1; s_mov_b64 exec, -1) without needing to evaluate
// multi-operand expressions whose result might also be -1.
bool isFullMaskReset(const StinkyInstruction& inst, const StinkyRegister& execReg) {
    if (!writesExecReg(inst, execReg)) return false;
    if (inst.getSrcRegs().size() != 1) return false;
    const StinkyRegister& src = inst.getSrcRegs()[0];
    return src.dataType == StinkyRegister::Type::LiteralInt && src.getLiteralInt() == -1;
}

// A narrow write: any exec write that isn't a detected full-mask reset.
// Opcode-agnostic by design — unlike LLVM where codegen is tightly controlled,
// StinkyTofu IR is written by hand, so we match the semantic intent (exec != -1)
// rather than an opcode allowlist that would need constant maintenance.
bool isExecNarrowWrite(const StinkyInstruction& inst, const StinkyRegister& execReg) {
    return writesExecReg(inst, execReg) && !isFullMaskReset(inst, execReg);
}

}  // namespace

void collapseExecMaskedRegions(BasicBlock& bb, AsmIRBuilder& builder, uint32_t wavefrontSize) {
    const StinkyRegister execReg = StinkyRegister::getEXECRegister(wavefrontSize);

    for (auto it = bb.begin(); it != bb.end();) {
        auto* beginInst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!beginInst || !isExecNarrowWrite(*beginInst, execReg)) {
            ++it;
            continue;
        }

        int depth = 1;
        auto spanEnd = std::next(it);
        for (; spanEnd != bb.end(); ++spanEnd) {
            auto* cur = dyn_cast<StinkyInstruction>(spanEnd.getNodePtr());
            if (!cur) continue;
            if (isFullMaskReset(*cur, execReg)) {
                if (--depth == 0) break;
            } else if (isExecNarrowWrite(*cur, execReg)) {
                ++depth;
            }
            // Other exec writes (s_or, s_and, saveexec, ...) are ignored as span
            // boundaries and fall naturally into the current span's children.
        }
        if (spanEnd == bb.end()) {
            PASS_DEBUG(std::cerr << "[collapseExecMaskedRegions] unmatched exec narrow write, "
                                    "leaving ungrouped\n");
            ++it;
            continue;
        }
        ++spanEnd;

        std::vector<StinkyInstruction*> children;
        std::vector<StinkyRegister> unionSrc, unionDest;
        int totalIssue = 0, totalLatency = 0;
        for (auto cIt = it; cIt != spanEnd; ++cIt) {
            auto* child = dyn_cast<StinkyInstruction>(cIt.getNodePtr());
            assert(child && "exec-masked span must contain only StinkyInstructions");
            children.push_back(child);
            unionSrc.insert(unionSrc.end(), child->getSrcRegs().begin(), child->getSrcRegs().end());
            unionDest.insert(unionDest.end(), child->getDestRegs().begin(),
                             child->getDestRegs().end());
            totalIssue += child->issueCycles;
            totalLatency += child->latencyCycles;
        }

        StinkyInstruction* group = builder.createExecMaskGroup(&*it);
        group->setSrcRegs(unionSrc);
        group->setDestRegs(unionDest);
        group->issueCycles = totalIssue;
        group->latencyCycles = totalLatency;
        group->addModifier<ExecGroupData>(ExecGroupData{children});

        auto groupIt = IRList::iterator(group);
        for (StinkyInstruction* child : children) bb.removeIR(child);

        it = std::next(groupIt);
    }
}

void expandExecMaskedGroups(BasicBlock& bb) {
    for (auto it = bb.begin(); it != bb.end();) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst || !isExecMaskGroup(*inst)) {
            ++it;
            continue;
        }

        auto* groupData = inst->getModifier<ExecGroupData>();
        assert(groupData && "ExecMaskGroup instruction missing ExecGroupData");

        for (StinkyInstruction* child : groupData->children) bb.insertIR(it, child);
        it = bb.eraseIR(it);
    }
}

}  // namespace stinkytofu
