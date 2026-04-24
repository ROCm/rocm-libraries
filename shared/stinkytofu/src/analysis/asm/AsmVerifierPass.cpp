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

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
char StinkyIRVerifierPass::ID = 0;

PreservedAnalyses StinkyIRVerifierPass::run(Function& func, PassContext&, AnalysisManager& /*AM*/) {
    std::string error = validateStinkyIR(func, config_);
    if (!error.empty()) {
        if (config_.abortOnError) {
            // Temporarily disable unreachable error due to rocisa
            // STINKY_UNREACHABLE(error.c_str());
        }
        std::cerr << "[StinkyIRVerifier] " << error;
    }
    return PreservedAnalyses::all();
}

// ===========================================================================
// Register width validation (ASM-only; uses HwInstDesc from hardware defs)
// ===========================================================================

static RegType fieldTypeToRegType(FieldType ft) {
    switch (ft) {
        case FieldType::vgpr:
        case FieldType::src_vgpr:
        case FieldType::src_vgpr_or_inline:
            return RegType::V;
        case FieldType::sreg:
        case FieldType::sreg_m0:
        case FieldType::sgpr:
        case FieldType::sdst:
        case FieldType::ssrc:
            return RegType::S;
        default:
            return RegType::UNKNOWN;
    }
}

// TODO: We should fix this by adding a new field to the HwInstDesc to indicate if the instruction
// can use less operands.
static bool canUseLessOperand(const StinkyInstruction* inst) {
    if (inst->getUnifiedOpcode() == GFX::tensor_load_to_lds) return true;
    return false;
}

static std::string checkRegisterWidths(const StinkyInstruction* inst,
                                       const AsmVerifierConfig& config) {
    const HwInstDesc* hwDesc = inst->getHwInstDesc();
    if (!hwDesc || !hwDesc->mnemonic) return "";

    if (hwDesc->operandFields.empty()) return "";

    std::stringstream errors;

    unsigned destIdx = 0;
    unsigned srcIdx = 0;
    for (const auto& field : hwDesc->operandFields) {
        bool isDest = field.isDest;
        unsigned operandIndex = isDest ? destIdx++ : srcIdx++;
        const auto& regs = isDest ? inst->getDestRegs() : inst->getSrcRegs();

        if (field.fieldSizeBits == 0) continue;

        unsigned expectedWidth = field.fieldSizeBits / 32;

        if (operandIndex >= regs.size() && !canUseLessOperand(inst)) {
            errors << "Instruction '" << hwDesc->mnemonic << "' missing operand "
                   << (isDest ? "dest[" : "src[") << operandIndex << "]\n";
            continue;
        }

        if (operandIndex >= regs.size()) {
            break;
        }

        const StinkyRegister& reg = regs[operandIndex];
        // Non-register operands (e.g. the "off" keyword used as MUBUF vaddr,
        // or integer/double literals) carry no register-level constraints.
        if (reg.dataType != StinkyRegister::Type::Register) continue;

        // Width check: only meaningful for operands wider than one DWORD.
        if (expectedWidth > 1) {
            // M64 operands are 64-bit lane masks that may be truncated to
            // 32 bits in wave32 mode, so width 1 is valid when expected is 2.
            bool m64Truncated = field.isM64 && expectedWidth == 2 && reg.reg.num == 1;

            if (reg.reg.num != expectedWidth && !m64Truncated) {
                errors << "Instruction '";
                inst->dump(errors);
                errors << "' operand " << (isDest ? "dest[" : "src[") << operandIndex << "] "
                       << "has register width " << reg.reg.num << ", expected " << expectedWidth
                       << "\n";
            }
        }

        // Type check: applies to all fields — a 32-bit VGPR field must still
        // receive a VGPR, not a SGPR (and vice versa).
        RegType expectedType = fieldTypeToRegType(field.fieldType);
        if (expectedType != RegType::UNKNOWN && reg.reg.type != expectedType) {
            errors << "Instruction '";
            inst->dump(errors);
            errors << "' operand " << (isDest ? "dest[" : "src[") << operandIndex << "] "
                   << "has register type '" << regTypeToString(reg.reg.type) << "', expected '"
                   << regTypeToString(expectedType) << "'\n";
        }
    }

    return errors.str();
}

// ===========================================================================
// Read-write operand validation
// ===========================================================================

static std::string checkReadWriteOperands(const StinkyInstruction* inst) {
    const HwInstDesc* hwDesc = inst->getHwInstDesc();
    if (!hwDesc || !hwDesc->mnemonic) return "";

    if (hwDesc->operandFields.empty()) return "";

    const auto& destRegs = inst->getDestRegs();
    const auto& srcRegs = inst->getSrcRegs();

    std::stringstream errors;
    unsigned destIdx = 0, srcIdx = 0;

    for (const auto& field : hwDesc->operandFields) {
        if (!field.isReadWrite) {
            if (field.isDest)
                destIdx++;
            else
                srcIdx++;
            continue;
        }

        const StinkyRegister* reg = nullptr;
        if (field.isDest && destIdx < destRegs.size()) {
            reg = &destRegs[destIdx];
            destIdx++;
        } else if (srcIdx < srcRegs.size()) {
            reg = &srcRegs[srcIdx];
            srcIdx++;
        }

        if (!reg || reg->dataType != StinkyRegister::Type::Register) continue;

        bool inDest = std::find(destRegs.begin(), destRegs.end(), *reg) != destRegs.end();
        bool inSrc = std::find(srcRegs.begin(), srcRegs.end(), *reg) != srcRegs.end();

        if (!inDest || !inSrc) {
            errors << "Instruction '" << hwDesc->mnemonic << "' has read-write field '"
                   << static_cast<int>(field.encodeField) << "' with register that is missing "
                   << "from " << (!inDest ? "destRegs" : "srcRegs") << "\n";
        }
    }

    return errors.str();
}

// ===========================================================================
// StinkyTofu Assembly IR validation (ASM pipeline only)
// ===========================================================================

std::string validateStinkyIR(Function& func, const AsmVerifierConfig& config) {
    if (config.verbose) {
        std::cout << "[StinkyIRVerifier] Verifying StinkyTofu Assembly IR...\n";
    }

    if (func.empty()) return "Function is empty (no basic blocks)";

    if (!func.getEntryBlock()) return "Function has no entry basic block";

    size_t logicalCount = 0;
    size_t stinkyCount = 0;
    size_t totalBlocks = 0;
    size_t invalidHwDesc = 0;
    std::stringstream widthErrors;
    std::stringstream rwErrors;

    for (BasicBlock& bb : func) {
        totalBlocks++;

        for (IRBase& ir : bb) {
            if (ir.getType() == IRBase::IRType::LogicalIR) {
                logicalCount++;
            } else if (ir.getType() == IRBase::IRType::StinkyTofu) {
                stinkyCount++;

                auto* stinkyInst = static_cast<StinkyInstruction*>(&ir);
                const HwInstDesc* hwDesc = stinkyInst->getHwInstDesc();
                if (!hwDesc || !hwDesc->mnemonic || hwDesc->mnemonic[0] == '\0') {
                    invalidHwDesc++;
                } else {
                    if (config.checkRegisterWidths) {
                        std::string widthError = checkRegisterWidths(stinkyInst, config);
                        if (!widthError.empty()) widthErrors << widthError;
                    }
                    if (config.checkReadWriteOperands) {
                        std::string rwError = checkReadWriteOperands(stinkyInst);
                        if (!rwError.empty()) rwErrors << rwError;
                    }
                }
            }
        }
    }

    if (logicalCount > 0) {
        std::stringstream ss;
        ss << "StinkyTofu Assembly IR contains " << logicalCount << " Logical instructions. "
           << "This suggests IR is not fully lowered or mixed.";
        return ss.str();
    }

    if (stinkyCount == 0) return "Function contains no StinkyTofu instructions (empty IR)";

    if (invalidHwDesc > 0) {
        std::stringstream ss;
        ss << "Found " << invalidHwDesc << " StinkyTofu instruction(s) with invalid or missing "
           << "hardware instruction descriptors";
        return ss.str();
    }

    std::string widthErrorStr = widthErrors.str();
    if (!widthErrorStr.empty()) {
        std::stringstream ss;
        ss << "Register width validation failed:\n" << widthErrorStr;
        return ss.str();
    }

    std::string rwErrorStr = rwErrors.str();
    if (!rwErrorStr.empty()) {
        std::stringstream ss;
        ss << "Read-write operand validation failed:\n" << rwErrorStr;
        return ss.str();
    }

    if (config.verbose) {
        std::cout << "[StinkyIRVerifier] OK: " << totalBlocks << " blocks, " << stinkyCount
                  << " stinky instructions\n";
    }

    return "";
}
}  // namespace stinkytofu
