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

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace test {
// -----------------------------------------------------------------
// Function configuration helpers
// -----------------------------------------------------------------

/// Sets gemmConfig.arch on \p func from a GfxArchID.
/// Must be called before constructing AsmIRBuilder on blocks in \p func.
inline void setFunctionArch(Function& func, GfxArchID archID) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archID);
    assert(archInfo && "Invalid GfxArchID");
    GemmTileConfig config = func.getGemmTileConfig();
    config.arch = {static_cast<int>(archInfo->major), static_cast<int>(archInfo->minor),
                   static_cast<int>(archInfo->stepping)};
    func.setGemmTileConfig(config);
}

// -----------------------------------------------------------------
// Instruction creation helpers
// -----------------------------------------------------------------

inline StinkyInstruction* createVAddInBlock(BasicBlock* bb, GfxArchID arch, int destReg,
                                            int src0Reg, int src1Reg) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));
    inst->addDestReg(StinkyRegister("v", destReg, 1));
    inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
    inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
    return inst;
}

/// Create ds_read_b128 / ds_load_b128 in \p bb: v[destReg:destReg+3] = mem[v[addrReg]].
/// Defines a 4-DWORD wide vector register.
/// Gfx1250 uses ds_load_b128.
inline StinkyInstruction* createDsReadB128InBlock(BasicBlock* bb, GfxArchID arch, int destReg,
                                                  int addrReg) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::ds_load_b128, arch));
    inst->addDestReg(StinkyRegister("v", destReg, 4));
    inst->addSrcReg(StinkyRegister("v", addrReg, 1));
    return inst;
}

/// Create ds_load_b128 / ds_read_b128 with optional memory tokens.
/// Internally delegates to createDsReadB128InBlock for the instruction,
/// then attaches a MemTokenData modifier when \p memTokens is non-empty.
inline StinkyInstruction* createDSLoadInBlock(BasicBlock* bb, GfxArchID arch, int destReg,
                                              int addrReg, std::vector<int> memTokens = {}) {
    StinkyInstruction* inst = createDsReadB128InBlock(bb, arch, destReg, addrReg);
    if (!memTokens.empty()) inst->addModifier<MemTokenData>(MemTokenData{memTokens});
    return inst;
}

/// Create a tensor_load_to_lds instruction.
/// \p src0Reg is a 4-SGPR base address; \p src1Reg is an 8-SGPR descriptor.
/// Optionally attach a MemTokenData modifier with \p memTokens.
inline StinkyInstruction* createTensorLoadInBlock(BasicBlock* bb, GfxArchID arch, int src0Reg,
                                                  int src1Reg, std::vector<int> memTokens = {}) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::tensor_load_to_lds, arch));
    inst->addSrcReg(StinkyRegister("s", src0Reg, 4));
    inst->addSrcReg(StinkyRegister("s", src1Reg, 8));
    if (!memTokens.empty()) inst->addModifier<MemTokenData>(MemTokenData{memTokens});
    return inst;
}

/// Create a ds_write_b64 instruction: mem[v[addrReg]] = v[dataReg:dataReg+1].
/// Optionally attach a MemTokenData modifier with \p memTokens.
inline StinkyInstruction* createDSWriteInBlock(BasicBlock* bb, GfxArchID arch, int addrReg,
                                               int dataReg, std::vector<int> memTokens = {}) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::ds_store_b64, arch));
    inst->addSrcReg(StinkyRegister("v", addrReg, 1));
    inst->addSrcReg(StinkyRegister("v", dataReg, 2));
    if (!memTokens.empty()) inst->addModifier<MemTokenData>(MemTokenData{memTokens});
    return inst;
}

// -----------------------------------------------------------------
// PHI counting
// -----------------------------------------------------------------

inline size_t countPhis(const BasicBlock& bb) {
    size_t count = 0;
    for (const IRBase& ir : bb) {
        if (ir.getType() == IRBase::IRType::StinkyTofu) {
            auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() == GFX::PHI) count++;
        }
    }
    return count;
}

inline size_t countPhisInFunction(const Function& func) {
    size_t count = 0;
    for (const BasicBlock& bb : func) count += countPhis(bb);
    return count;
}

/// Find a PHI in \p bb whose dest register matches (type, idx).
/// Returns nullptr if not found.
inline StinkyInstruction* findPhi(BasicBlock& bb, RegType type, unsigned idx) {
    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) break;
        auto* inst = cast<StinkyInstruction>(&ir);
        if (inst->getUnifiedOpcode() != GFX::PHI) break;
        if (inst->getDestReg(0).reg.type == type && inst->getDestReg(0).reg.idx == idx) return inst;
    }
    return nullptr;
}

// -----------------------------------------------------------------
// Chain-consistency verification (gtest assertions)
// -----------------------------------------------------------------

/// For each PHI operandDef, verify the PHI appears in that def's users list.
inline void verifyPhiChainConsistency(const Function& func) {
    for (const BasicBlock& bb : func) {
        for (const IRBase& ir : bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = cast<StinkyInstruction>(&ir);
            if (inst->getUnifiedOpcode() != GFX::PHI) continue;
            for (StinkyInstruction* def : inst->getSources()) {
                if (def == nullptr) continue;
                const auto& defUsers = def->getUsers();
                EXPECT_TRUE(std::find(defUsers.begin(), defUsers.end(), inst) != defUsers.end())
                    << "PHI must appear in its operand-def's users list";
            }
        }
    }
}

/// For each instruction's operandDef, verify the instruction appears in
/// that def's users list.  nullptr sources are allowed only for PHIs.
inline void verifyDefUseChainConsistency(const Function& func) {
    for (const BasicBlock& bb : func) {
        for (const IRBase& ir : bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* userInst = cast<StinkyInstruction>(&ir);
            for (StinkyInstruction* def : userInst->getSources()) {
                if (def == nullptr) {
                    EXPECT_EQ(userInst->getUnifiedOpcode(), GFX::PHI)
                        << "Only PHI instructions may have nullptr sources";
                    continue;
                }
                const auto& defUsers = def->getUsers();
                EXPECT_TRUE(std::find(defUsers.begin(), defUsers.end(), userInst) != defUsers.end())
                    << "Def-use invariant: user must be in def->getUsers()";
            }
        }
    }
}

/// Reverse check: for each def's user, the user's sources must contain def.
inline void verifyUsersSourcesConsistency(const Function& func) {
    for (const BasicBlock& bb : func) {
        for (const IRBase& ir : bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* defInst = cast<StinkyInstruction>(&ir);
            for (StinkyInstruction* user : defInst->getUsers()) {
                ASSERT_NE(user, nullptr);
                const auto& opDefs = user->getSources();
                EXPECT_TRUE(std::find(opDefs.begin(), opDefs.end(), defInst) != opDefs.end())
                    << "Def-use invariant: def must be in user->getSources()";
            }
        }
    }
}

// -----------------------------------------------------------------
// Original helpers
// -----------------------------------------------------------------

/**
 * @brief Add a vector of instructions to a BasicBlock
 *
 * Helper function for tests using StinkyTofu builder, which returns
 * std::vector<StinkyInstruction*> to support composite instructions.
 *
 * @param bb The BasicBlock to append instructions to
 * @param insts Vector of instructions from StinkyTofu builder
 * @return The first instruction, or nullptr if vector is empty
 *
 * Usage (prefer BasicBlock's begin/end for iteration):
 * @code
 *   StinkyTofu builder({12, 5, 0});
 *
 *   // Add single instruction
 *   auto* add = addToIRList(*bb, builder.VAddF32(vgpr(0), vgpr(1), vgpr(2)));
 *
 *   // Add composite instruction (might be multiple instructions)
 *   auto* pk = addToIRList(*bb, builder.VAddPKF32(...));
 *
 *   // Iterate instructions using BasicBlock's begin/end
 *   for (auto it = bb->begin(); it != bb->end(); ++it) { ... }
 * @endcode
 */
inline StinkyInstruction* addToIRList(BasicBlock& bb, std::vector<StinkyInstruction*>&& insts) {
    if (insts.empty()) return nullptr;

    for (auto* inst : insts) {
        bb.appendIR(inst);
    }
    return insts[0];
}

/**
 * @brief Convenience function to create a VGPR register
 */
inline StinkyRegister vgpr(int idx, int count = 1) {
    return StinkyRegister("v", idx, count);
}

/**
 * @brief Convenience function to create an SGPR register
 */
inline StinkyRegister sgpr(int idx, int count = 1) {
    return StinkyRegister("s", idx, count);
}

/**
 * @brief Convenience function to create an AGPR register
 */
inline StinkyRegister agpr(int idx, int count = 1) {
    return StinkyRegister("a", idx, count);
}

/**
 * @brief Convenience function to create a literal constant register
 */
inline StinkyRegister literal(double value) {
    StinkyRegister reg;
    reg.dataType = StinkyRegister::Type::LiteralDouble;
    reg.literalDouble = value;
    return reg;
}

}  // namespace test
}  // namespace stinkytofu
