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
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "stinkytofu/ir/asm/ReadWriteOperands.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/AttachedSSAVerifier.hpp"
#include "stinkytofu/support/Casting.hpp"

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

static bool isScalarRegType(RegType type) {
    switch (type) {
        case RegType::S:
        // FieldType::sreg_m0 means "an sreg or m0", and fieldTypeToRegType folds it
        // to RegType::S, so the actual m0 register has to be accepted here too.
        case RegType::M:
        case RegType::SCC:
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return true;
        default:
            return false;
    }
}

static bool isExpectedTypeMatch(FieldType fieldType, RegType expectedType, RegType actualType) {
    if (expectedType == RegType::UNKNOWN) return true;

    switch (fieldType) {
        case FieldType::sreg:
        case FieldType::sreg_m0:
        case FieldType::sgpr:
        case FieldType::sdst:
        case FieldType::ssrc:
            return isScalarRegType(actualType);
        default:
            return actualType == expectedType;
    }
}

// TODO: We should fix this by adding a new field to the HwInstDesc to indicate if the instruction
// can use less operands.
static bool canUseLessOperand(const StinkyInstruction* inst) {
    if (inst->getUnifiedOpcode() == GFX::tensor_load_to_lds) return true;
    if (inst->getUnifiedOpcode() == GFX::s_delay_alu) return true;
    if (inst->getUnifiedOpcode() == GFX::s_wait_alu) return true;
    return false;
}

// Workaround: v_wmma_f32_16x16x128_f8f6f4 and v_wmma_scale_f32_16x16x128_f8f6f4 have
// matrix-format-dependent packed source widths. For src0/src1, accept dynamic widths
// (256/384/512 bits) even though the static format metadata uses a fixed maximum width.
static bool allowDynamicWmmaScaleSrcWidth(const HwInstDesc* hwDesc, bool isDest,
                                          unsigned operandIndex, unsigned expectedWidth,
                                          unsigned actualWidth) {
    if (!hwDesc || !hwDesc->mnemonic || isDest) return false;
    if (operandIndex > 1) return false;  // only src0/src1 are dynamic
    if (expectedWidth != 16) return false;
    std::string_view mnemonic(hwDesc->mnemonic);
    if (mnemonic != "v_wmma_f32_16x16x128_f8f6f4" &&
        mnemonic != "v_wmma_scale_f32_16x16x128_f8f6f4")
        return false;
    return actualWidth == 8 || actualWidth == 12 || actualWidth == 16;
}

/// Where an instruction sits. The verifier runs after every pass under
/// --verify-each, so the location strings are built only for a report.
struct InstructionPlace {
    const Function& function;
    const BasicBlock& block;
    int index;

    IRLocation location() const {
        return {function.getName(), block.getLabel(), index};
    }
};

/// The presence, width, and register class of each operand \p inst has an
/// encoding field for.
static void checkRegisterWidths(const StinkyInstruction& inst, const InstructionPlace& at,
                                std::vector<Diagnostic>& out) {
    const HwInstDesc* hwDesc = inst.getHwInstDesc();
    if (!hwDesc || !hwDesc->mnemonic) return;

    if (hwDesc->operandFields.empty()) return;

    auto report = [&](DiagId id, const std::string& message) {
        out.emplace_back(Diagnostic::Level::Error, id,
                         std::string("'") + hwDesc->mnemonic + "' " + message, at.location());
    };

    unsigned destIdx = 0;
    unsigned srcIdx = 0;
    for (const auto& field : hwDesc->operandFields) {
        bool isDest = field.isDest;
        unsigned operandIndex = isDest ? destIdx++ : srcIdx++;
        const auto& regs = isDest ? inst.getDestRegs() : inst.getSrcRegs();
        auto operand = [&] {
            return std::string(isDest ? "dest[" : "src[") + std::to_string(operandIndex) + "]";
        };

        if (field.fieldSizeBits == 0) continue;

        unsigned expectedWidth = field.fieldSizeBits / 32;

        if (operandIndex >= regs.size() && !canUseLessOperand(&inst)) {
            report(DiagId::MissingOperand, "missing operand " + operand());
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
            bool dynamicWmmaWidth = allowDynamicWmmaScaleSrcWidth(hwDesc, isDest, operandIndex,
                                                                  expectedWidth, reg.reg.num);

            if (reg.reg.num != expectedWidth && !m64Truncated && !dynamicWmmaWidth) {
                report(DiagId::RegisterWidth, "operand " + operand() + " has register width " +
                                                  std::to_string(reg.reg.num) + ", expected " +
                                                  std::to_string(expectedWidth));
            }
        }

        // Type check: applies to all fields — a 32-bit VGPR field must still
        // receive a VGPR, not a SGPR (and vice versa).
        RegType expectedType = fieldTypeToRegType(field.fieldType);
        if (!isExpectedTypeMatch(field.fieldType, expectedType, reg.reg.type)) {
            // Check promoted encoding: e.g. VOP2 src1 is vgpr-only, but VOP3
            // promotion widens it to src (accepts SGPR).  The assembler will
            // promote automatically, so accept the promoted field type too.
            unsigned fieldIdx = static_cast<unsigned>(&field - hwDesc->operandFields.data());
            bool promotedOk = false;
            if (fieldIdx < hwDesc->promotedFields.size()) {
                auto promotedFT = hwDesc->promotedFields[fieldIdx].fieldType;
                auto promotedET = fieldTypeToRegType(promotedFT);
                promotedOk = isExpectedTypeMatch(promotedFT, promotedET, reg.reg.type);
            }
            if (!promotedOk) {
                report(DiagId::RegisterType, "operand " + operand() + " has register type '" +
                                                 regTypeToString(reg.reg.type) + "', expected '" +
                                                 regTypeToString(expectedType) + "'");
            }
        }
    }
}

// ===========================================================================
// Read-write operand validation
// ===========================================================================

static void checkReadWriteOperands(const StinkyInstruction& inst, const InstructionPlace& at,
                                   std::vector<Diagnostic>& out) {
    const HwInstDesc* hwDesc = inst.getHwInstDesc();
    if (!hwDesc || !hwDesc->mnemonic) return;

    if (hwDesc->operandFields.empty()) return;

    const auto& destRegs = inst.getDestRegs();
    const auto& srcRegs = inst.getSrcRegs();

    unsigned destIdx = 0, srcIdx = 0;

    for (const auto& field : hwDesc->operandFields) {
        // A destination is read-write when the opcode always reads it, and also
        // when this instance's write keeps part of the register.
        const bool isReadWrite = field.isDest ? readsDestination(inst, destIdx) : field.isReadWrite;
        if (!isReadWrite) {
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
            out.emplace_back(Diagnostic::Level::Error, DiagId::ReadWriteOperand,
                             std::string("'") + hwDesc->mnemonic + "' has read-write field '" +
                                 std::to_string(static_cast<int>(field.encodeField)) +
                                 "' with register that is missing from " +
                                 (!inDest ? "destRegs" : "srcRegs"),
                             at.location());
        }
    }
}

// ===========================================================================
// StinkyTofu Assembly IR validation (ASM pipeline only)
// ===========================================================================

/// The checks validateStinkyIR reports, in its order: an empty function, Logical
/// IR, a function without instructions, then each instruction's own problems.
static void checkFunction(const Function& func, const AsmVerifierConfig& config,
                          std::vector<Diagnostic>& out) {
    const IRLocation whole{func.getName()};
    if (func.empty()) {
        out.emplace_back(Diagnostic::Level::Error, DiagId::EmptyFunction,
                         "Function is empty (no basic blocks)", whole);
        return;
    }

    size_t logicalCount = 0;
    size_t stinkyCount = 0;
    std::vector<Diagnostic> perInstruction;
    for (const BasicBlock& bb : func) {
        int index = 0;
        for (const IRBase& ir : bb) {
            if (ir.getType() == IRBase::IRType::LogicalIR) {
                logicalCount++;
                continue;
            }
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            stinkyCount++;

            const auto& inst = static_cast<const StinkyInstruction&>(ir);
            const InstructionPlace at{func, bb, isIndexedInstruction(ir) ? index++ : -1};
            const HwInstDesc* hwDesc = inst.getHwInstDesc();
            if (!hwDesc || !hwDesc->mnemonic || hwDesc->mnemonic[0] == '\0') {
                perInstruction.emplace_back(Diagnostic::Level::Error, DiagId::NoHardwareDescriptor,
                                            "Instruction has no hardware descriptor",
                                            at.location());
                continue;
            }
            if (config.checkRegisterWidths) checkRegisterWidths(inst, at, perInstruction);
            if (config.checkReadWriteOperands) checkReadWriteOperands(inst, at, perInstruction);
        }
    }

    if (logicalCount > 0) {
        out.emplace_back(Diagnostic::Level::Error, DiagId::LogicalIR,
                         "StinkyTofu Assembly IR contains " + std::to_string(logicalCount) +
                             " Logical instructions. This suggests IR is not fully lowered or "
                             "mixed.",
                         whole);
    }
    if (stinkyCount == 0) {
        out.emplace_back(Diagnostic::Level::Error, DiagId::EmptyFunction,
                         "Function contains no StinkyTofu instructions (empty IR)", whole);
    }
    out.insert(out.end(), std::make_move_iterator(perInstruction.begin()),
               std::make_move_iterator(perInstruction.end()));
}

static void checkBranchTargets(const Function& func, std::vector<Diagnostic>& out) {
    std::unordered_set<std::string> defined;
    for (const BasicBlock& bb : func) {
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr || !isLabel(*inst)) continue;
            if (const auto* label = inst->getModifier<LabelData>()) defined.insert(label->label);
        }
    }
    for (const BasicBlock& bb : func) {
        int index = 0;
        for (const IRBase& ir : bb) {
            if (!isIndexedInstruction(ir)) continue;
            const auto& inst = *cast<StinkyInstruction>(&ir);
            for (const std::string& target : getBranchTargets(inst)) {
                if (defined.count(target) != 0) continue;
                const HwInstDesc* desc = inst.getHwInstDesc();
                const char* mnemonic = desc && desc->mnemonic ? desc->mnemonic : "branch";
                out.emplace_back(Diagnostic::Level::Error, DiagId::UndefinedLabel,
                                 std::string("'") + mnemonic + "' targets '" + target +
                                     "', which no label defines",
                                 IRLocation{func.getName(), bb.getLabel(), index});
            }
            ++index;
        }
    }
}

static void checkAttachedSSA(const Function& func, std::vector<Diagnostic>& out) {
    for (const std::string& error : verifyAttachedSSA(func).errors)
        out.emplace_back(Diagnostic::Level::Error, DiagId::InvalidSSA, error,
                         IRLocation{func.getName()});
}

std::vector<Diagnostic> verifyStinkyIR(const Function& func, const AsmVerifierConfig& config) {
    std::vector<Diagnostic> found;
    checkFunction(func, config, found);
    checkBranchTargets(func, found);
    checkAttachedSSA(func, found);
    return found;
}

std::string validateStinkyIR(Function& func, const AsmVerifierConfig& config) {
    if (config.verbose) {
        std::cout << "[StinkyIRVerifier] Verifying StinkyTofu Assembly IR...\n";
    }

    std::vector<Diagnostic> found;
    checkFunction(func, config, found);

    // Only the first kind of problem is reported, in checkFunction's order.
    size_t invalidHwDesc = 0;
    std::string widthErrors;
    std::string rwErrors;
    for (const Diagnostic& diagnostic : found) {
        switch (diagnostic.getId()) {
            case DiagId::EmptyFunction:
            case DiagId::LogicalIR:
                return diagnostic.getMessage();
            case DiagId::NoHardwareDescriptor:
                invalidHwDesc++;
                break;
            case DiagId::MissingOperand:
            case DiagId::RegisterWidth:
            case DiagId::RegisterType:
                widthErrors += diagnostic.format() + "\n";
                break;
            case DiagId::ReadWriteOperand:
                rwErrors += diagnostic.format() + "\n";
                break;
            default:
                break;
        }
    }

    if (invalidHwDesc > 0) {
        return "Found " + std::to_string(invalidHwDesc) +
               " StinkyTofu instruction(s) with invalid or missing hardware instruction "
               "descriptors";
    }
    if (!widthErrors.empty()) return "Register width validation failed:\n" + widthErrors;
    if (!rwErrors.empty()) return "Read-write operand validation failed:\n" + rwErrors;

    if (config.verbose) {
        size_t stinkyCount = 0;
        for (const BasicBlock& bb : func) {
            for (const IRBase& ir : bb)
                if (ir.getType() == IRBase::IRType::StinkyTofu) stinkyCount++;
        }
        std::cout << "[StinkyIRVerifier] OK: " << func.size() << " blocks, " << stinkyCount
                  << " stinky instructions\n";
    }

    return "";
}
}  // namespace stinkytofu
