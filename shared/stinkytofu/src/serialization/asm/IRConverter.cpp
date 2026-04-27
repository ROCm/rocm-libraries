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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/serialization/asm/IRConverter.hpp"

#include <climits>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "ModifierSerializer.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

namespace stinkytofu {
StinkyIRConverter::StinkyIRConverter() : arch({12, 5, 0}) {}

StinkyIRConverter::StinkyIRConverter(const std::array<int, 3>& targetArch) : arch(targetArch) {}

static void convertInstruction(AsmIRBuilder& irBuilder,
                               const std::unique_ptr<ParsedInstruction>& inst, GfxArchID arch) {
    if (inst->isLabel) {
        irBuilder.createLabel(inst->opcodeStr);
        return;
    }

    // "FENCE" is the mnemonic printed by AsmPrinter; "scheduling_fence" is the
    // rocisa instruction string. Both round-trip to a scheduling fence.
    // Fences carry no modifiers — they are hard region boundaries with no tokens.
    if (inst->opcodeStr == "FENCE" || inst->opcodeStr == "scheduling_fence") {
        irBuilder.createFence();
    }

    if (inst->opcodeStr == "asm_directive") {
        AsmDirective* d = irBuilder.createIR<AsmDirective>();
        if (!inst->srcRegs.empty() &&
            inst->srcRegs[0].dataType == StinkyRegister::Type::LiteralString) {
            d->name = inst->srcRegs[0].literalValue;
            if (d->name == ".set") d->kind = AsmDirectiveKind::SET;
        }
        if (inst->srcRegs.size() > 1 &&
            inst->srcRegs[1].dataType == StinkyRegister::Type::LiteralString) {
            d->symbol = inst->srcRegs[1].literalValue;
        }
        return;
    }

    auto opcode = getMnemonicToIsaOpcode(inst->opcodeStr, arch);
    if (opcode == GFX::INVALID) {
        std::cerr << "Error: No ISA opcode found for mnemonic " << inst->opcodeStr << " in arch gfx"
                  << static_cast<int>(arch) << "\n";
        return;
    }
    const HwInstDesc* hwInstDesc = getMCIDByIsaOp(static_cast<IsaOpcode>(opcode), arch);

    if (hwInstDesc == nullptr) {
        std::cerr << "Warning: No hardware instruction descriptor found for opcode " << opcode
                  << " in arch gfx" << static_cast<int>(arch) << "\n";
        return;
    }

    StinkyInstruction* stinkyInst = irBuilder.create(hwInstDesc);
    stinkyInst->setDestRegs(inst->destRegs);
    stinkyInst->setSrcRegs(inst->srcRegs);

    if (inst->issueCycles > 0) stinkyInst->issueCycles = inst->issueCycles;
    if (inst->latencyCycles > 0) stinkyInst->latencyCycles = inst->latencyCycles;

    if (!inst->modifiers.empty()) {
        ModifierSerializer::deserialize(stinkyInst, inst->modifiers);
    }
}

StinkyErrorCode StinkyIRConverter::populateFunctionFromString(const std::string& irText,
                                                              Function& func, PassContext& passCtx,
                                                              GfxArchID arch) {
    auto result = parseSourceStringWithDiagnostics(irText);

    if (result.hasErrors()) return StinkyErrorCode::PARSE_ERROR;

    if (!result.parsedFunction || result.parsedFunction->blocks.empty())
        return StinkyErrorCode::PARSE_ERROR;

    func.clear();
    func.setName(result.parsedFunction->funcName);

    std::unordered_map<std::string, BasicBlock*> blockMap;

    for (auto& parsedBlock : result.parsedFunction->blocks) {
        BasicBlock* bb = func.createBasicBlock(parsedBlock->blockId);
        blockMap[parsedBlock->blockId] = bb;

        AsmIRBuilder irBuilder(*bb, arch);
        for (auto& inst : parsedBlock->instructions) {
            convertInstruction(irBuilder, inst, arch);
        }
    }

    // Wire successors
    for (size_t i = 0; i < result.parsedFunction->blocks.size(); ++i) {
        ParsedBlock& parsedBlock = *result.parsedFunction->blocks[i];
        BasicBlock* bb = blockMap[parsedBlock.blockId];
        if (!bb) continue;

        for (const std::string& succId : parsedBlock.successorIds) {
            std::string key = succId;
            if (!key.empty() && key[0] == '^') key = key.substr(1);
            BasicBlock* succ = blockMap[key];
            if (succ) {
                bb->addSuccessor(succ);
                succ->addPredecessor(bb);
            }
        }
    }

    return StinkyErrorCode::SUCCESS;
}

StinkyErrorCode StinkyIRConverter::populateFunctionFromParsed(ParsedFunction& parsedFunc,
                                                              Function& func, GfxArchID arch) {
    if (parsedFunc.blocks.empty()) return StinkyErrorCode::PARSE_ERROR;

    func.clear();
    func.setName(parsedFunc.funcName);

    std::unordered_map<std::string, BasicBlock*> blockMap;

    for (auto& parsedBlock : parsedFunc.blocks) {
        BasicBlock* bb = func.createBasicBlock(parsedBlock->blockId);
        blockMap[parsedBlock->blockId] = bb;

        AsmIRBuilder irBuilder(*bb, arch);
        for (auto& inst : parsedBlock->instructions) {
            convertInstruction(irBuilder, inst, arch);
        }
    }

    for (size_t i = 0; i < parsedFunc.blocks.size(); ++i) {
        ParsedBlock& parsedBlock = *parsedFunc.blocks[i];
        BasicBlock* bb = blockMap[parsedBlock.blockId];
        if (!bb) continue;

        for (const std::string& succId : parsedBlock.successorIds) {
            std::string key = succId;
            if (!key.empty() && key[0] == '^') key = key.substr(1);
            BasicBlock* succ = blockMap[key];
            if (succ) {
                bb->addSuccessor(succ);
                succ->addPredecessor(bb);
            }
        }
    }

    return StinkyErrorCode::SUCCESS;
}

Function* StinkyIRConverter::convertToFunction(const std::string& rawInstructions) {
    function = std::make_unique<Function>("kernel");
    passCtx = std::make_unique<PassContext>();

    // Set up kernel configuration
    GemmTileConfig config;
    config.arch = arch;
    config.TileA0 = 0;
    config.TileB0 = 0;
    config.TileM0 = 0;
    config.NumGRA = 0;
    config.NumGRB = 0;
    config.NumGRM = 0;
    config.NumWaves = 0;
    passCtx->setGemmTileConfig(config);

    GfxArchID archID = getGfxArchID(arch[0], arch[1], arch[2]);
    StinkyErrorCode result =
        populateFunctionFromString(rawInstructions, *function, *passCtx, archID);
    if (result != StinkyErrorCode::SUCCESS) {
        function.reset();
        passCtx.reset();
        return nullptr;
    }

    return function.get();
}

PassContext* StinkyIRConverter::getPassContext() {
    return passCtx.get();
}

void StinkyIRConverter::cleanup() {
    function.reset();
    passCtx.reset();
}

StinkyIRConverter::~StinkyIRConverter() {
    cleanup();
}
}  // namespace stinkytofu
