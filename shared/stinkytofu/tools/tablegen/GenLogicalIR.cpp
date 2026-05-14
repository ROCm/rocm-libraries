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

/**
 * @file GenLogicalIR.cpp
 * @brief Generates high-level IR instruction class definitions and builder methods
 *
 * This generator creates:
 * 1. IR instruction class definitions (LogicalInstructions_generated.hpp)
 * 2. Builder method forward declarations (StinkyBuilder_decls_generated.inc)
 * 3. Builder method implementations (StinkyBuilder_impls_generated.inc)
 * (Logical IR -> ASM mnemonic is per-arch via LogicalToAsmMappings_generated.inc from
 * GenLogicalToAsmMapping.cpp)
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace stinkytofu {

// IR instruction definition (Logical IR has no mnemonics; ASM mnemonic is per-arch via
// LogicalToAsmMappings)
struct IRInstDef {
    std::string className;  // e.g., "VAddF32"
    std::string comment;    // e.g., "Vector add F32: dst = src0 + src1"
    int numSrcs;            // Total number of source operands (required + optional)
    int numRequiredSrcs;    // Number of required sources (rest are optional pointers)
    bool hasDest;           // Whether it has a destination operand
    std::string category;   // e.g., "Vector Arithmetic", "Scalar Bitwise"
    bool supportsDPP;       // Whether this instruction supports DPP modifiers
    bool supportsSDWA;      // Whether this instruction supports SDWA modifiers
    bool hasDS;             // Whether this instruction has DS modifiers (for LDS operations)
    std::string flags;      // Instruction flags (e.g., "IF_DSRead|IF_Commutative|IF_VALU")

    IRInstDef(const std::string& cls, const std::string& cmt, int srcs, bool dest = true,
              const std::string& cat = "", bool dpp = false, bool sdwa = false, bool ds = false,
              const std::string& flags = "",
              int reqSrcs = -1)  // -1 means all sources are required (at end for backward compat)
        : className(cls),
          comment(cmt),
          numSrcs(srcs),
          numRequiredSrcs(reqSrcs == -1 ? srcs : reqSrcs),
          hasDest(dest),
          category(cat),
          supportsDPP(dpp),
          supportsSDWA(sdwa),
          hasDS(ds),
          flags(flags) {}
};

// Derive parse name (snake_case) from logical class name for getOpcodeMnemonic/parseOpcode.
static std::string classNameToParseName(const std::string& className) {
    std::string out;
    out.reserve(className.size() + 4);
    for (size_t i = 0; i < className.size(); ++i) {
        char c = className[i];
        if (c >= 'A' && c <= 'Z') {
            if (i != 0) out += '_';
            out += char(c - 'A' + 'a');
        } else
            out += c;
    }
    return out;
}

// Define all high-level IR instructions
// This is the "single source of truth" replacing manual class definitions
// Instruction definitions are in a separate file for maintainability
static std::vector<IRInstDef> getIRInstructions() {
    return {
#include "../../src/ir/logical/LogicalInstructionDefs.inc"
    };
}

// Generate special instruction factory functions (pure enum design with specialData_)
bool genSpecialMFMAClasses(std::ofstream& out) {
    // Factory functions for special instructions (MFMA, Label, etc.)
    out << "    // "
           "========================================================================\n";
    out << "    // Special Instruction Factory Functions (Pure Enum Design)\n";
    out << "    // "
           "========================================================================\n\n";

    out << "    /**\n";
    out << "     * @brief MFMA (Matrix Fused Multiply-Add) factory function\n";
    out << "     * \n";
    out << "     * Registers stored in: dests[0]=acc, srcs[0]=a, srcs[1]=b, srcs[2]=acc2 "
           "(optional)\n";
    out << "     * Metadata stored in: specialData_ as MFMAData*\n";
    out << "     */\n";
    out << "    inline LogicalInstruction* MFMA(\n";
    out << "        const std::string& instType,\n";
    out << "        const std::string& accType,\n";
    out << "        int m,\n";
    out << "        int n,\n";
    out << "        int k,\n";
    out << "        int blocks,\n";
    out << "        bool mfma1k,\n";
    out << "        const StinkyRegister& acc,\n";
    out << "        const StinkyRegister& a,\n";
    out << "        const StinkyRegister& b,\n";
    out << "        const StinkyRegister* acc2 = nullptr,\n";
    out << "        bool neg = false,\n";
    out << "        const std::string& comment = \"\")\n";
    out << "    {\n";
    out << "        auto* inst = IRBase::createIR<LogicalInstruction>(logical::MFMA);\n";
    out << "        \n";
    out << "        // Populate registers\n";
    out << "        inst->dests.push_back(acc);\n";
    out << "        inst->srcs.push_back(a);\n";
    out << "        inst->srcs.push_back(b);\n";
    out << "        if (acc2) inst->srcs.push_back(*acc2);\n";
    out << "        \n";
    out << "        // Create and set special data\n";
    out << "        auto* data = new MFMAData(instType, accType, m, n, k, blocks, mfma1k, "
           "neg);\n";
    out << "        inst->setSpecialData(data);\n";
    out << "        inst->comment = comment;\n";
    out << "        \n";
    out << "        return inst;\n";
    out << "    }\n\n";

    // MXMFMA factory
    out << "    /**\n";
    out << "     * @brief MXMFMA (MX format MFMA with scale factors) factory function\n";
    out << "     * \n";
    out << "     * Registers: dests[0]=acc, srcs[0]=a, srcs[1]=b, srcs[2]=acc2, srcs[3]=mxsa, "
           "srcs[4]=mxsb\n";
    out << "     * Metadata stored in: specialData_ as MXMFMAData*\n";
    out << "     */\n";
    out << "    inline LogicalInstruction* MXMFMA(\n";
    out << "        const std::string& instType,\n";
    out << "        const std::string& accType,\n";
    out << "        const std::string& mxScaleATypeStr,\n";
    out << "        const std::string& mxScaleBTypeStr,\n";
    out << "        int m,\n";
    out << "        int n,\n";
    out << "        int k,\n";
    out << "        int block,\n";
    out << "        const StinkyRegister& acc,\n";
    out << "        const StinkyRegister& a,\n";
    out << "        const StinkyRegister& b,\n";
    out << "        const StinkyRegister& acc2,\n";
    out << "        const StinkyRegister& mxsa,\n";
    out << "        const StinkyRegister& mxsb,\n";
    out << "        bool reuseA = false,\n";
    out << "        bool reuseB = false,\n";
    out << "        const std::string& comment = \"\")\n";
    out << "    {\n";
    out << "        auto* inst = IRBase::createIR<LogicalInstruction>(logical::MXMFMA);\n";
    out << "        \n";
    out << "        // Populate registers\n";
    out << "        inst->dests.push_back(acc);\n";
    out << "        inst->srcs.push_back(a);\n";
    out << "        inst->srcs.push_back(b);\n";
    out << "        inst->srcs.push_back(acc2);\n";
    out << "        inst->srcs.push_back(mxsa);\n";
    out << "        inst->srcs.push_back(mxsb);\n";
    out << "        \n";
    out << "        // Create and set special data\n";
    out << "        auto* data = new MXMFMAData(instType, accType, mxScaleATypeStr, "
           "mxScaleBTypeStr,\n";
    out << "                                    m, n, k, block, reuseA, reuseB);\n";
    out << "        inst->setSpecialData(data);\n";
    out << "        inst->comment = comment;\n";
    out << "        \n";
    out << "        return inst;\n";
    out << "    }\n\n";

    // SMFMA factory
    out << "    /**\n";
    out << "     * @brief SMFMA (Sparse MFMA) factory function\n";
    out << "     * \n";
    out << "     * Registers: dests[0]=acc, srcs[0]=a, srcs[1]=b, srcs[2]=metadata\n";
    out << "     * Metadata stored in: specialData_ as SMFMAData*\n";
    out << "     */\n";
    out << "    inline LogicalInstruction* SMFMA(\n";
    out << "        const std::string& instType,\n";
    out << "        const std::string& accType,\n";
    out << "        int m,\n";
    out << "        int n,\n";
    out << "        int k,\n";
    out << "        int blocks,\n";
    out << "        bool mfma1k,\n";
    out << "        const StinkyRegister& acc,\n";
    out << "        const StinkyRegister& a,\n";
    out << "        const StinkyRegister& b,\n";
    out << "        const StinkyRegister& metadata,\n";
    out << "        bool neg = false,\n";
    out << "        const std::string& comment = \"\")\n";
    out << "    {\n";
    out << "        auto* inst = IRBase::createIR<LogicalInstruction>(logical::SMFMA);\n";
    out << "        \n";
    out << "        // Populate registers\n";
    out << "        inst->dests.push_back(acc);\n";
    out << "        inst->srcs.push_back(a);\n";
    out << "        inst->srcs.push_back(b);\n";
    out << "        inst->srcs.push_back(metadata);\n";
    out << "        \n";
    out << "        // Create and set special data\n";
    out << "        auto* data = new SMFMAData(instType, accType, m, n, k, blocks, mfma1k, "
           "neg);\n";
    out << "        inst->setSpecialData(data);\n";
    out << "        inst->comment = comment;\n";
    out << "        \n";
    out << "        return inst;\n";
    out << "    }\n\n";

    // Label factory
    out << "    // "
           "========================================================================\n";
    out << "    // Control Flow Instructions\n";
    out << "    // "
           "========================================================================\n\n";

    out << "    /**\n";
    out << "     * @brief Label factory function for control flow\n";
    out << "     * \n";
    out << "     * Defines a label that can be used as a branch target.\n";
    out << "     * Labels have no operands and do not produce output.\n";
    out << "     * Metadata stored in: specialData_ as LogicalLabelData*\n";
    out << "     */\n";
    out << "    inline LogicalInstruction* Label(const std::string& labelName)\n";
    out << "    {\n";
    out << "        auto* inst = IRBase::createIR<LogicalInstruction>(logical::Label);\n";
    out << "        \n";
    out << "        // Labels have no operands\n";
    out << "        \n";
    out << "        // Create and set special data\n";
    out << "        auto* data = new LogicalLabelData(labelName);\n";
    out << "        inst->setSpecialData(data);\n";
    out << "        \n";
    out << "        return inst;\n";
    out << "    }\n\n";

    return true;
}

// Generate IR instruction class definitions
// Generate opcode enum values
bool genOpcodeEnum(const std::string& outdir) {
    std::ofstream out(outdir + "/LogicalOpcodes_generated.inc");
    if (!out) {
        std::cerr << "Failed to open LogicalOpcodes_generated.inc for writing\n";
        return false;
    }

    out << "// Auto-generated by TableGen - DO NOT EDIT\n";
    out << "// High-level IR opcode enumeration\n\n";

    int count = 0;

    // Generate opcode for each instruction
    for (const auto& inst : getIRInstructions()) {
        out << "            " << inst.className << " = " << (count + 1) << ",\n";
        count++;
    }

    std::cout << "Generated " << count << " opcode enum values -> LogicalOpcodes_generated.inc\n";
    return true;
}

// Generate opcode to string mapping tables
bool genOpcodeMappings(const std::string& outdir) {
    std::ofstream out(outdir + "/stinkytofu/ir/logical/LogicalOpcode.cpp");
    if (!out) {
        std::cerr << "Failed to open LogicalOpcode.cpp for writing\n";
        return false;
    }

    // File header
    out << "/* ************************************************************************\n";
    out << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n";
    out << " * AUTO-GENERATED FILE - DO NOT EDIT\n";
    out << " * Generated by: tools/tablegen/GenLogicalIR.cpp\n";
    out << " * ************************************************************************ */\n\n";

    out << "#include \"stinkytofu/ir/logical/LogicalOpcode.hpp\"\n\n";
    out << "namespace stinkytofu\n";
    out << "{\n";
    out << "namespace logical\n";
    out << "{\n\n";

    // Generate getOpcodeName function
    out << "const char* getOpcodeName(Opcode opcode)\n";
    out << "{\n";
    out << "    switch (opcode)\n";
    out << "    {\n";
    out << "    case UNKNOWN:\n";
    out << "        return \"UNKNOWN\";\n";

    for (const auto& inst : getIRInstructions()) {
        out << "    case " << inst.className << ":\n";
        out << "        return \"" << inst.className << "\";\n";
    }

    // Special opcodes (not auto-generated)
    out << "    case MFMA:\n";
    out << "        return \"MFMA\";\n";
    out << "    case MXMFMA:\n";
    out << "        return \"MXMFMA\";\n";
    out << "    case SMFMA:\n";
    out << "        return \"SMFMA\";\n";
    out << "    case Label:\n";
    out << "        return \"Label\";\n";
    out << "    case IntrinsicCall:\n";
    out << "        return \"IntrinsicCall\";\n";

    out << "    default:\n";
    out << "        return \"INVALID\";\n";
    out << "    }\n";
    out << "}\n\n";

    // Generate getOpcodeMnemonic function
    out << "const char* getOpcodeMnemonic(Opcode opcode)\n";
    out << "{\n";
    out << "    switch (opcode)\n";
    out << "    {\n";
    out << "    case UNKNOWN:\n";
    out << "        return \"unknown\";\n";

    for (const auto& inst : getIRInstructions()) {
        out << "    case " << inst.className << ":\n";
        out << "        return \"" << classNameToParseName(inst.className) << "\";\n";
    }

    // Special opcodes (not auto-generated)
    out << "    case MFMA:\n";
    out << "        return \"mfma\";\n";
    out << "    case MXMFMA:\n";
    out << "        return \"mxmfma\";\n";
    out << "    case SMFMA:\n";
    out << "        return \"smfma\";\n";
    out << "    case Label:\n";
    out << "        return \"label\";\n";
    out << "    case IntrinsicCall:\n";
    out << "        return \"intrinsic_call\";\n";

    out << "    default:\n";
    out << "        return \"invalid\";\n";
    out << "    }\n";
    out << "}\n\n";

    out << "} // namespace logical\n";
    out << "} // namespace stinkytofu\n";

    std::cout << "Generated opcode mapping functions -> LogicalOpcode.cpp\n";
    return true;
}

bool genIRClasses(const std::string& outdir) {
    std::ofstream out(outdir + "/stinkytofu/ir/logical/LogicalInstructions_generated.hpp");
    if (!out) {
        std::cerr << "Failed to open LogicalInstructions_generated.hpp for writing\n";
        return false;
    }

    // File header with header guards
    out << "/* ************************************************************************\n";
    out << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n";
    out << " *\n";
    out << " * Permission is hereby granted, free of charge, to any person obtaining a copy\n";
    out << " * of this software and associated documentation files (the \"Software\"), to "
           "deal\n";
    out << " * in the Software without restriction, including without limitation the rights\n";
    out << " * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n";
    out << " * copies of the Software, and to permit persons to whom the Software is\n";
    out << " * furnished to do so, subject to the following conditions:\n";
    out << " *\n";
    out << " * The above copyright notice and this permission notice shall be included in\n";
    out << " * all copies or substantial portions of the Software.\n";
    out << " *\n";
    out << " * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n";
    out << " * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n";
    out << " * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n";
    out << " * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n";
    out << " * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n";
    out << " * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n";
    out << " * THE SOFTWARE.\n";
    out << " *\n";
    out << " * ************************************************************************ */\n\n";
    out << "// Auto-generated by TableGen - DO NOT EDIT\n";
    out << "// This file contains high-level IR instruction class definitions\n\n";
    out << "#pragma once\n\n";
    out << "#include \"stinkytofu/ir/asm/StinkyAsmIR.hpp\"\n";
    out << "#include \"stinkytofu/ir/asm/StinkyModifiers.hpp\"\n";
    out << "#include \"stinkytofu/core/stinkytofu.hpp\"\n";
    out << "#include <iostream>\n";
    out << "#include <optional>\n";
    out << "#include <string>\n";
    out << "\n// TODO: High-level IR should not depend on assembly-level IR "
           "(StinkyAsmIR.hpp).\n";
    out << "// Extract StinkyRegister and RegType into a separate header (e.g., "
           "stinkytofu/ir/StinkyRegister.hpp)\n";
    out << "// to fix the inverted dependency. StinkyRegister is a shared primitive used by "
           "both\n";
    out << "// high-level IR (LogicalInstruction) and assembly IR (StinkyInstruction).\n";
    out << "#include <vector>\n\n";
    out << "namespace stinkytofu\n";
    out << "{\n\n";
    out << "    // NOTE: LogicalInstruction base class must be defined before including this "
           "file\n\n";

    std::string currentCategory = "";
    for (const auto& inst : getIRInstructions()) {
        if (inst.category != currentCategory) {
            if (!currentCategory.empty()) {
                out << "\n";
            }
            out << "    // "
                   "========================================================================\n";
            out << "    // " << inst.category << "\n";
            out << "    // "
                   "========================================================================"
                   "\n\n";
            currentCategory = inst.category;
        }

        // Generate factory function instead of subclass
        out << "    /**\n";
        out << "     * @brief Factory function for " << inst.comment << "\n";
        out << "     */\n";
        out << "    inline LogicalInstruction* " << inst.className << "(";

        // Function parameters
        if (inst.hasDest) {
            out << "const StinkyRegister& dst";
            if (inst.numSrcs > 0) out << ",\n" << std::string(inst.className.length() + 12, ' ');
        }

        for (int i = 0; i < inst.numSrcs; ++i) {
            if (i > 0 || inst.hasDest) {
                // Optional sources use pointer with default nullptr
                if (i >= inst.numRequiredSrcs) {
                    out << "const StinkyRegister* src" << i << " = nullptr";
                } else {
                    out << "const StinkyRegister& src" << i;
                }

                if (i < inst.numSrcs - 1)
                    out << ",\n" << std::string(inst.className.length() + 12, ' ');
            } else {
                // First source (no dest case)
                if (i >= inst.numRequiredSrcs) {
                    out << "const StinkyRegister* src" << i << " = nullptr";
                } else {
                    out << "const StinkyRegister& src" << i;
                }

                if (i < inst.numSrcs - 1)
                    out << ",\n" << std::string(inst.className.length() + 12, ' ');
            }
        }

        if (inst.hasDest || inst.numSrcs > 0) {
            out << ",\n" << std::string(inst.className.length() + 12, ' ');
        }

        // Add modifier parameters
        if (inst.supportsDPP) {
            out << "std::optional<DPPModifiers> dpp = std::nullopt,\n"
                << std::string(inst.className.length() + 12, ' ');
        }
        if (inst.supportsSDWA) {
            out << "std::optional<SDWAModifiers> sdwa = std::nullopt,\n"
                << std::string(inst.className.length() + 12, ' ');
        }
        if (inst.hasDS) {
            out << "std::optional<DSModifiers> ds = std::nullopt,\n"
                << std::string(inst.className.length() + 12, ' ');
        }

        out << "const std::string& comment = \"\")\n";
        out << "    {\n";

        // Function body: create LogicalInstruction with opcode
        out << "        auto* inst = IRBase::createIR<LogicalInstruction>(logical::"
            << inst.className << ");\n";

        // Set destinations
        if (inst.hasDest) {
            out << "        inst->dests.push_back(dst);\n";
        }

        // Set sources (required and optional)
        for (int i = 0; i < inst.numSrcs; ++i) {
            if (i >= inst.numRequiredSrcs) {
                // Optional source: check if pointer is non-null before adding
                out << "        if(src" << i << ") inst->srcs.push_back(*src" << i << ");\n";
            } else {
                // Required source: always add
                out << "        inst->srcs.push_back(src" << i << ");\n";
            }
        }

        // Set modifiers
        if (inst.supportsDPP) {
            out << "        inst->dpp = dpp;\n";
        }
        if (inst.supportsSDWA) {
            out << "        inst->sdwa = sdwa;\n";
        }
        if (inst.hasDS) {
            out << "        inst->ds = ds;\n";
        }

        // Set comment
        out << "        inst->comment = comment;\n";

        // Set flags (parse from flags string)
        if (!inst.flags.empty()) {
            out << "        // Set instruction flags\n";
            out << "        InstFlagSet flags;\n";

            // Split flags by '|' and generate set() calls
            std::string flagsStr = inst.flags;
            size_t pos = 0;
            while ((pos = flagsStr.find('|')) != std::string::npos) {
                std::string flag = flagsStr.substr(0, pos);
                // Trim whitespace
                flag.erase(0, flag.find_first_not_of(" \t"));
                flag.erase(flag.find_last_not_of(" \t") + 1);
                if (!flag.empty()) {
                    out << "        flags.set(" << flag << ");\n";
                }
                flagsStr.erase(0, pos + 1);
            }
            // Handle last flag
            flagsStr.erase(0, flagsStr.find_first_not_of(" \t"));
            flagsStr.erase(flagsStr.find_last_not_of(" \t") + 1);
            if (!flagsStr.empty()) {
                out << "        flags.set(" << flagsStr << ");\n";
            }

            out << "        inst->setFlags(flags);\n";
        }

        out << "        return inst;\n";
        out << "    }\n\n";
    }

    // Generate special MFMA classes
    genSpecialMFMAClasses(out);

    // Generate helper implementations
    out << "\n// "
           "============================================================================\n";
    out << "// Helper implementations (auto-generated)\n";
    out << "// "
           "============================================================================\n\n";

    // Note: isCommutative() is now inline in LogicalInstructions.hpp using
    // flags_.test(IF_Commutative)

    // Generate isComposite() implementation
    // Note: Composite instructions are determined by CompositeInstructionLoweringPass
    out << "inline bool LogicalInstruction::isComposite() const\n";
    out << "{\n";
    out << "    switch (opcode_) {\n";
    out << "        case logical::VAddPKF32:\n";
    out << "        case logical::VMulPKF32:\n";
    out << "        case logical::VMovB64:\n";
    out << "        case logical::VLShiftLeftOrB32:\n";
    out << "            return true;\n";
    out << "        default:\n";
    out << "            return false;\n";
    out << "    }\n";
    out << "}\n\n";

    // Close namespace
    out << "} // namespace stinkytofu\n";

    std::cout << "Generated " << getIRInstructions().size()
              << " LogicalInstruction factory functions + 5 special instruction factories "
                 "(MFMA/MXMFMA/SMFMA/Label/IntrinsicCall) -> "
                 "LogicalInstructions_generated.hpp\n";
    return true;
}

// Generate builder method forward declarations
// Generate builder method implementations
// Generate C++ factory functions for all IR instructions
// Generate Python bindings for all IR instructions
bool genPythonBindings(const std::string& outdir) {
    std::ofstream out(outdir + "/PythonBindings_generated.inc");
    if (!out) {
        std::cerr << "Failed to open PythonBindings_generated.inc for writing\n";
        return false;
    }

    out << "// Auto-generated Python bindings for IR instructions\n";
    out << "// DO NOT EDIT - Generated by GenLogicalIR.cpp\n\n";

    int count = 0;

    // Generate bindings for all regular IR instructions
    for (const auto& inst : getIRInstructions()) {
        std::string className = inst.className;

        // Generate factory function instead of constructor
        out << "    m.def(\"" << className << "\", [](";

        // Constructor parameter types
        std::vector<std::string> paramTypes;
        std::vector<std::string> paramNames;
        std::vector<std::string> defaults;

        // Add destination register (if instruction has one)
        if (inst.hasDest) {
            paramTypes.push_back("const StinkyRegister&");
            paramNames.push_back("dest");
            defaults.push_back("");
        }

        // Add source operands (required and optional)
        for (int i = 0; i < inst.numSrcs; i++) {
            if (i >= inst.numRequiredSrcs) {
                // Optional source: use std::optional wrapper for Python
                paramTypes.push_back("std::optional<StinkyRegister>");
                paramNames.push_back("src" + std::to_string(i));
                defaults.push_back("std::nullopt");
            } else {
                // Required source: regular reference
                paramTypes.push_back("const StinkyRegister&");
                paramNames.push_back("src" + std::to_string(i));
                defaults.push_back("");
            }
        }

        // Add optional modifiers
        if (inst.supportsDPP && inst.supportsSDWA) {
            paramTypes.push_back("std::optional<DPPModifiers>");
            paramNames.push_back("dpp");
            defaults.push_back("std::nullopt");

            paramTypes.push_back("std::optional<SDWAModifiers>");
            paramNames.push_back("sdwa");
            defaults.push_back("std::nullopt");
        } else if (inst.hasDS) {
            paramTypes.push_back("std::optional<DSModifiers>");
            paramNames.push_back("ds");
            defaults.push_back("std::nullopt");
        }

        // Add comment parameter
        paramTypes.push_back("const std::string&");
        paramNames.push_back("comment");
        defaults.push_back("\"\"");

        // Output lambda parameter list
        for (size_t i = 0; i < paramTypes.size(); i++) {
            out << paramTypes[i] << " " << paramNames[i];
            if (i + 1 < paramTypes.size()) out << ", ";
        }
        out << ") {\n";

        // Return factory function wrapped in shared_ptr (Python-owned: set ownedExternally)
        out << "        return makeLogicalInstructionShared(" << className << "(";

        // Pass parameters, converting optional sources to pointers
        for (size_t i = 0; i < paramNames.size(); i++) {
            std::string paramName = paramNames[i];

            // Check if this is an optional source parameter
            bool isOptionalSrc = paramName.find("src") == 0 &&
                                 paramTypes[i].find("std::optional") != std::string::npos;

            if (isOptionalSrc) {
                // Convert std::optional<StinkyRegister> to const StinkyRegister*
                out << "(" << paramName << ".has_value() ? &" << paramName << ".value() : nullptr)";
            } else {
                // Regular parameter: pass as-is
                out << paramName;
            }

            if (i + 1 < paramNames.size()) out << ", ";
        }
        out << "));\n";
        out << "    },\n";

        // Output parameter names and defaults
        for (size_t i = 0; i < paramNames.size(); i++) {
            out << "    nb::arg(\"" << paramNames[i] << "\")";
            if (!defaults[i].empty()) out << " = " << defaults[i];
            if (i + 1 < paramNames.size()) out << ",\n";
        }
        out << ",\n";

        // Add docstring
        out << "    \"Create a " << classNameToParseName(inst.className) << " instruction\");\n\n";
        count++;
    }

    std::cout << "Generated Python bindings for " << count
              << " IR instructions -> PythonBindings_generated.inc\n";
    return true;
}

// Generate all high-level IR artifacts
bool genLogicalIR(const std::string& outdir) {
    bool success = true;

    std::cout << "\n=== Generating High-Level IR ===\n";

    success &= genOpcodeEnum(outdir);
    success &= genOpcodeMappings(outdir);
    success &= genIRClasses(outdir);
    // Obsolete: genBuilderDecls, genBuilderImpls, genFactoryFunctions
    // These are replaced by inline factory functions in LogicalInstructions_generated.hpp
    success &= genPythonBindings(outdir);

    if (success) {
        std::cout << "=== High-Level IR generation completed successfully ===\n\n";
    } else {
        std::cerr << "=== High-Level IR generation failed ===\n\n";
    }

    return success;
}

}  // namespace stinkytofu
