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

#include "stinkytofu/transforms/logical/IntrinsicExpansionPass.hpp"

#include <cstring>
#include <iostream>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/ir/logical/IntrinsicCall.hpp"
#include "stinkytofu/ir/logical/IntrinsicRegistry.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalOpcode.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
char IntrinsicExpansionPass::ID = 0;

IntrinsicExpansionPass::IntrinsicExpansionPass() = default;

IntrinsicExpansionPass::~IntrinsicExpansionPass() = default;

PreservedAnalyses IntrinsicExpansionPass::run(Function& func, PassContext& passCtx,
                                              AnalysisManager& /*AM*/) {
    // Check if intrinsic registry is initialized
    auto& registry = IntrinsicRegistry::instance();
    if (!registry.isInitialized()) {
        std::cerr << "[IntrinsicExpansionPass] Warning: IntrinsicRegistry not initialized, "
                  << "skipping intrinsic expansion\n";
        return preserveCFGAnalyses();
    }

    // Process all basic blocks
    for (auto& bb : func) {
        // Skip filtered basic blocks
        if (!passCtx.shouldProcessBasicBlock(bb)) continue;

        expandIntrinsicsInBlock(bb);
    }
    return preserveCFGAnalyses();
}

void IntrinsicExpansionPass::expandIntrinsicsInBlock(BasicBlock& bb) {
    // Find all IntrinsicCall instructions first (to avoid iterator invalidation)
    std::vector<BasicBlock::iterator> intrinsicCalls;

    for (auto it = bb.begin(); it != bb.end(); ++it) {
        IRBase* irNode = &(*it);
        if (irNode->getType() == IRBase::IRType::LogicalIR) {
            LogicalInstruction* logicalInst = static_cast<LogicalInstruction*>(irNode);
            if (std::strcmp(logicalInst->getLogicalName(), "IntrinsicCall") == 0) {
                intrinsicCalls.push_back(it);
            }
        }
    }

    // Expand each IntrinsicCall
    for (auto it : intrinsicCalls) {
        LogicalInstruction* call = static_cast<LogicalInstruction*>(&(*it));
        auto expanded = expandIntrinsic(call);

        if (!expanded.empty()) {
            // Insert expanded instructions before the IntrinsicCall
            for (auto* inst : expanded) {
                bb.insertIR(it, inst);
            }
            call->safeErase();
        }
    }
}

std::vector<LogicalInstruction*> IntrinsicExpansionPass::expandIntrinsic(LogicalInstruction* inst) {
    // Cast to IntrinsicCall - safe because we checked getLogicalName() == "IntrinsicCall"
    IntrinsicCall* call = static_cast<IntrinsicCall*>(inst);

    const std::string& funcName = call->getFunctionName();

    // Look up intrinsic definition from registry
    auto& registry = IntrinsicRegistry::instance();
    const Pattern* pattern = registry.lookup(funcName);

    if (!pattern) {
        std::cerr << "[IntrinsicExpansionPass] Error: Unknown intrinsic '" << funcName << "'\n";
        std::cerr << "[IntrinsicExpansionPass] Available intrinsics: ";
        for (const auto& name : registry.getIntrinsicNames()) {
            std::cerr << name << " ";
        }
        std::cerr << "\n";
        return {};
    }

    // Validate argument count
    const auto& callArgs = call->dests;  // All registers stored in dests
    if (callArgs.size() != pattern->arguments.size()) {
        std::cerr << "[IntrinsicExpansionPass] Error: Intrinsic '" << funcName << "' expects "
                  << pattern->arguments.size() << " arguments, got " << callArgs.size() << "\n";
        return {};
    }

    // Build register map from intrinsic argument names to actual registers
    std::unordered_map<std::string, StinkyRegister> regMap;
    for (size_t i = 0; i < pattern->arguments.size(); ++i) {
        regMap[pattern->arguments[i].name] = callArgs[i];
    }

    // Create expanded instructions from intrinsic body
    std::vector<LogicalInstruction*> expanded;

    for (const auto& instDef : pattern->body) {
        LogicalInstruction* newInst = createInstructionFromIntrinsic(instDef, regMap);
        if (newInst) {
            expanded.push_back(newInst);
        } else {
            std::cerr << "[IntrinsicExpansionPass] Error: Failed to create instruction: "
                      << instDef.operation << "\n";
            // Clean up and return empty
            for (auto* i : expanded) {
                i->safeErase();
            }
            return {};
        }
    }

    return expanded;
}

LogicalInstruction* IntrinsicExpansionPass::createInstructionFromIntrinsic(
    const IntrinsicInstruction& instDef,
    const std::unordered_map<std::string, StinkyRegister>& regMap) {
    // Parse instruction name to opcode (pure enum design!)
    const std::string& opName = instDef.operation;
    logical::Opcode opcode = logical::parseOpcode(opName.c_str());

    if (opcode == logical::UNKNOWN) {
        std::cerr << "[IntrinsicExpansionPass] FATAL: Unknown instruction '" << opName
                  << "' in intrinsic definition.\n"
                  << "This indicates an error in the intrinsic definition or missing "
                  << "instruction in LogicalOpcode enum.\n";
        STINKY_UNREACHABLE("Unknown instruction in intrinsic body");
    }

    // Create instruction directly (no factory needed!)
    auto* inst = IRBase::createIR<LogicalInstruction>(opcode);

    // Resolve and set destination register
    StinkyRegister dest;
    if (!resolveOperand(instDef.destReg, regMap, dest)) {
        std::cerr << "[IntrinsicExpansionPass] Error: Destination must be a register: "
                  << instDef.destReg << "\n";
        inst->safeErase();
        return nullptr;
    }
    inst->dests.push_back(dest);

    // Resolve and set source registers and immediates
    for (const auto& srcDef : instDef.operands) {
        if (srcDef.type == IntrinsicOperand::Register) {
            StinkyRegister srcReg;
            if (!resolveOperand(srcDef.registerName, regMap, srcReg)) {
                std::cerr << "[IntrinsicExpansionPass] Error: Unknown register: "
                          << srcDef.registerName << "\n";
                inst->safeErase();
                return nullptr;
            }
            inst->srcs.push_back(srcReg);
        } else if (srcDef.type == IntrinsicOperand::IntLiteral) {
            // Create integer literal
            inst->srcs.push_back(StinkyRegister(static_cast<int>(srcDef.intValue)));
        } else if (srcDef.type == IntrinsicOperand::FloatLiteral) {
            // Create float literal
            inst->srcs.push_back(StinkyRegister(srcDef.floatValue));
        } else if (srcDef.type == IntrinsicOperand::HexLiteral) {
            // Hex literal treated as float bits
            inst->srcs.push_back(StinkyRegister(srcDef.floatValue));
        }
    }

    return inst;
}

bool IntrinsicExpansionPass::resolveOperand(
    const std::string& operand, const std::unordered_map<std::string, StinkyRegister>& regMap,
    StinkyRegister& outReg) {
    // Check if operand is a register name from the intrinsic definition
    auto it = regMap.find(operand);
    if (it != regMap.end()) {
        outReg = it->second;
        return true;
    }

    // Not a mapped register, could be an immediate or constant
    return false;
}

std::unique_ptr<Pass> createIntrinsicExpansionPass() {
    return std::make_unique<IntrinsicExpansionPass>();
}

}  // namespace stinkytofu
