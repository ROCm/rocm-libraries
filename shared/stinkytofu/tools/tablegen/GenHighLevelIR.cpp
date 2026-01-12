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
 * @file GenHighLevelIR.cpp
 * @brief Generates high-level IR instruction class definitions and builder methods
 *
 * This generator creates:
 * 1. IR instruction class definitions (StinkyInstructions_generated.inc)
 * 2. Builder method forward declarations (StinkyBuilder_decls_generated.inc)
 * 3. Builder method implementations (StinkyBuilder_impls_generated.inc)
 * 4. Mnemonic-to-IR mappings for ToStinkyAsmPass (IRMnemonics_generated.inc)
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace stinkytofu
{

    // IR instruction definition
    struct IRInstDef
    {
        std::string className; // e.g., "VAddF32"
        std::string mnemonic; // e.g., "v_add_f32"
        std::string comment; // e.g., "Vector add F32: dst = src0 + src1"
        int         numSrcs; // Number of source operands
        bool        hasDest; // Whether it has a destination operand
        std::string category; // e.g., "Vector Arithmetic", "Scalar Bitwise"
        bool        supportsDPP; // Whether this instruction supports DPP modifiers
        bool        supportsSDWA; // Whether this instruction supports SDWA modifiers
        bool        hasDS; // Whether this instruction has DS modifiers (for LDS operations)
        bool        isCommutative; // Whether src0 and src1 can be swapped

        IRInstDef(const std::string& cls,
                  const std::string& mn,
                  const std::string& cmt,
                  int                srcs,
                  bool               dest        = true,
                  const std::string& cat         = "",
                  bool               dpp         = false,
                  bool               sdwa        = false,
                  bool               ds          = false,
                  bool               commutative = false)
            : className(cls)
            , mnemonic(mn)
            , comment(cmt)
            , numSrcs(srcs)
            , hasDest(dest)
            , category(cat)
            , supportsDPP(dpp)
            , supportsSDWA(sdwa)
            , hasDS(ds)
            , isCommutative(commutative)
        {
        }
    };

    // Define all high-level IR instructions
    // This is the "single source of truth" replacing manual class definitions
    // Instruction definitions are in a separate file for maintainability
    static std::vector<IRInstDef> getIRInstructions()
    {
        return {
#include "IRInstructionDefs.inc"
        };
    }

    // Generate special MFMA instruction classes (these have custom constructors)
    bool genSpecialMFMAClasses(std::ofstream& out)
    {
        // MFMA class
        out << "    // "
               "========================================================================\n";
        out << "    // Special Matrix Instructions\n";
        out << "    // "
               "========================================================================\n\n";

        out << "    /**\n";
        out << "     * @brief MFMA (Matrix Fused Multiply-Add) instruction\n";
        out << "     */\n";
        out << "    class MFMA : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string instType;      ///< Input data type (bf16, f16, i8, etc.)\n";
        out << "        std::string accType;       ///< Accumulator type (f32, i32)\n";
        out << "        int         m;             ///< Matrix M dimension\n";
        out << "        int         n;             ///< Matrix N dimension\n";
        out << "        int         k;             ///< Matrix K dimension\n";
        out << "        int         blocks;        ///< Number of blocks\n";
        out << "        bool        mfma1k;        ///< Whether this is a _1k variant\n";
        out << "        StinkyRegister acc;        ///< Accumulator destination\n";
        out << "        StinkyRegister a;          ///< Matrix A source\n";
        out << "        StinkyRegister b;          ///< Matrix B source\n";
        out << "        std::optional<StinkyRegister> acc2; ///< Optional accumulator source\n";
        out << "        bool        neg;           ///< Negate operands\n";
        out << "\n";
        out << "        MFMA(const std::string& instType,\n";
        out << "             const std::string& accType,\n";
        out << "             int m,\n";
        out << "             int n,\n";
        out << "             int k,\n";
        out << "             int blocks,\n";
        out << "             bool mfma1k,\n";
        out << "             const StinkyRegister& acc,\n";
        out << "             const StinkyRegister& a,\n";
        out << "             const StinkyRegister& b,\n";
        out << "             const StinkyRegister* acc2 = nullptr,\n";
        out << "             bool neg = false,\n";
        out << "             const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , instType(instType)\n";
        out << "            , accType(accType)\n";
        out << "            , m(m)\n";
        out << "            , n(n)\n";
        out << "            , k(k)\n";
        out << "            , blocks(blocks)\n";
        out << "            , mfma1k(mfma1k)\n";
        out << "            , acc(acc)\n";
        out << "            , a(a)\n";
        out << "            , b(b)\n";
        out << "            , acc2(acc2 ? std::optional<StinkyRegister>(*acc2) : std::nullopt)\n";
        out << "            , neg(neg)\n";
        out << "        {\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"MFMA\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"MFMA (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // MXMFMA class
        out << "    /**\n";
        out << "     * @brief MXMFMA (MX format MFMA with scale factors) instruction\n";
        out << "     */\n";
        out << "    class MXMFMA : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string instType;         ///< Input data type (f8, f4, bf8, etc.)\n";
        out << "        std::string accType;          ///< Accumulator type (f32)\n";
        out << "        std::string mxScaleATypeStr;  ///< Scale format for matrix A\n";
        out << "        std::string mxScaleBTypeStr;  ///< Scale format for matrix B\n";
        out << "        int         m;                ///< Matrix M dimension\n";
        out << "        int         n;                ///< Matrix N dimension\n";
        out << "        int         k;                ///< Matrix K dimension\n";
        out << "        int         block;            ///< Block size\n";
        out << "        StinkyRegister acc;           ///< Accumulator destination\n";
        out << "        StinkyRegister a;             ///< Matrix A source\n";
        out << "        StinkyRegister b;             ///< Matrix B source\n";
        out << "        StinkyRegister acc2;          ///< Accumulator source\n";
        out << "        StinkyRegister mxsa;          ///< Scale factor A register\n";
        out << "        StinkyRegister mxsb;          ///< Scale factor B register\n";
        out << "        bool        reuseA;           ///< Matrix A reuse flag\n";
        out << "        bool        reuseB;           ///< Matrix B reuse flag\n";
        out << "\n";
        out << "        MXMFMA(const std::string& instType,\n";
        out << "               const std::string& accType,\n";
        out << "               const std::string& mxScaleATypeStr,\n";
        out << "               const std::string& mxScaleBTypeStr,\n";
        out << "               int m,\n";
        out << "               int n,\n";
        out << "               int k,\n";
        out << "               int block,\n";
        out << "               const StinkyRegister& acc,\n";
        out << "               const StinkyRegister& a,\n";
        out << "               const StinkyRegister& b,\n";
        out << "               const StinkyRegister& acc2,\n";
        out << "               const StinkyRegister& mxsa,\n";
        out << "               const StinkyRegister& mxsb,\n";
        out << "               bool reuseA = false,\n";
        out << "               bool reuseB = false,\n";
        out << "               const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , instType(instType)\n";
        out << "            , accType(accType)\n";
        out << "            , mxScaleATypeStr(mxScaleATypeStr)\n";
        out << "            , mxScaleBTypeStr(mxScaleBTypeStr)\n";
        out << "            , m(m)\n";
        out << "            , n(n)\n";
        out << "            , k(k)\n";
        out << "            , block(block)\n";
        out << "            , acc(acc)\n";
        out << "            , a(a)\n";
        out << "            , b(b)\n";
        out << "            , acc2(acc2)\n";
        out << "            , mxsa(mxsa)\n";
        out << "            , mxsb(mxsb)\n";
        out << "            , reuseA(reuseA)\n";
        out << "            , reuseB(reuseB)\n";
        out << "        {\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"MXMFMA\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"MXMFMA (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // SMFMA class
        out << "    /**\n";
        out << "     * @brief SMFMA (Sparse MFMA) instruction\n";
        out << "     */\n";
        out << "    class SMFMA : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string instType;      ///< Input data type (bf16, f16, i8, etc.)\n";
        out << "        std::string accType;       ///< Accumulator type (f32, i32)\n";
        out << "        int         m;             ///< Matrix M dimension\n";
        out << "        int         n;             ///< Matrix N dimension\n";
        out << "        int         k;             ///< Matrix K dimension\n";
        out << "        int         blocks;        ///< Number of blocks\n";
        out << "        bool        mfma1k;        ///< Whether this is a _1k variant\n";
        out << "        StinkyRegister acc;        ///< Accumulator destination\n";
        out << "        StinkyRegister a;          ///< Matrix A source\n";
        out << "        StinkyRegister b;          ///< Matrix B source\n";
        out << "        StinkyRegister metadata;   ///< Sparsity metadata register\n";
        out << "        bool        neg;           ///< Negate operands\n";
        out << "\n";
        out << "        SMFMA(const std::string& instType,\n";
        out << "              const std::string& accType,\n";
        out << "              int m,\n";
        out << "              int n,\n";
        out << "              int k,\n";
        out << "              int blocks,\n";
        out << "              bool mfma1k,\n";
        out << "              const StinkyRegister& acc,\n";
        out << "              const StinkyRegister& a,\n";
        out << "              const StinkyRegister& b,\n";
        out << "              const StinkyRegister& metadata,\n";
        out << "              bool neg = false,\n";
        out << "              const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , instType(instType)\n";
        out << "            , accType(accType)\n";
        out << "            , m(m)\n";
        out << "            , n(n)\n";
        out << "            , k(k)\n";
        out << "            , blocks(blocks)\n";
        out << "            , mfma1k(mfma1k)\n";
        out << "            , acc(acc)\n";
        out << "            , a(a)\n";
        out << "            , b(b)\n";
        out << "            , metadata(metadata)\n";
        out << "            , neg(neg)\n";
        out << "        {\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"SMFMA\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"SMFMA (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // TensorLoadToLds class
        out << "    // "
               "========================================================================\n";
        out << "    // Tensor Memory Instructions\n";
        out << "    // "
               "========================================================================\n\n";

        out << "    /**\n";
        out << "     * @brief TensorLoadToLds instruction\n";
        out << "     * \n";
        out << "     * Loads tensor data to LDS (Local Data Share). Takes 2-4 SGPR groups.\n";
        out << "     * All groups must be scalar registers (SGPRs).\n";
        out << "     */\n";
        out << "    class TensorLoadToLds : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        TensorLoadToLds(const StinkyRegister& group0,\n";
        out << "                        const StinkyRegister& group1,\n";
        out << "                        const StinkyRegister* group2 = nullptr,\n";
        out << "                        const StinkyRegister* group3 = nullptr,\n";
        out << "                        const std::string& comment_ = \"\")\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "        {\n";
        out << "            // No destination register for this instruction\n";
        out << "            srcs.push_back(group0);\n";
        out << "            srcs.push_back(group1);\n";
        out << "            if (group2) srcs.push_back(*group2);\n";
        out << "            if (group3) srcs.push_back(*group3);\n";
        out << "            this->comment = comment_;\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return "
               "\"TensorLoadToLds\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << \"TensorLoadToLds (IR)\";\n";
        out << "            if(!comment.empty())\n";
        out << "                out << \"  // \" << comment;\n";
        out << "        }\n";
        out << "    };\n\n";

        // Label class
        out << "    // "
               "========================================================================\n";
        out << "    // Control Flow Instructions\n";
        out << "    // "
               "========================================================================\n\n";

        out << "    /**\n";
        out << "     * @brief Label instruction for control flow\n";
        out << "     * \n";
        out << "     * Defines a label that can be used as a branch target.\n";
        out << "     * Labels have no operands and do not produce output.\n";
        out << "     */\n";
        out << "    class Label : public IRInstruction\n";
        out << "    {\n";
        out << "    public:\n";
        out << "        std::string label_name;\n";
        out << "\n";
        out << "        explicit Label(const std::string& name)\n";
        out << "            : IRInstruction(IRType::StinkyTofu)\n";
        out << "            , label_name(name)\n";
        out << "        {\n";
        out << "            // Labels have no operands\n";
        out << "        }\n";
        out << "\n";
        out << "        const char* getLogicalName() const override { return \"Label\"; }\n";
        out << "\n";
        out << "        void dump(std::ostream& out) const override\n";
        out << "        {\n";
        out << "            out << label_name << \":\";\n";
        out << "        }\n";
        out << "    };\n\n";

        return true;
    }

    // Generate IR instruction class definitions
    // Generate opcode enum values
    bool genOpcodeEnum(const std::string& outdir)
    {
        std::ofstream out(outdir + "/IROpcodes_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open IROpcodes_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// High-level IR opcode enumeration\n\n";

        int count = 0;

        // Generate opcode for each instruction
        for(const auto& inst : getIRInstructions())
        {
            out << "            " << inst.className << " = " << (count + 1) << ",\n";
            count++;
        }

        std::cout << "Generated " << count << " opcode enum values -> IROpcodes_generated.inc\n";
        return true;
    }

    // Generate opcode to string mapping tables
    bool genOpcodeMappings(const std::string& outdir)
    {
        std::ofstream out(outdir + "/ir/IROpcode.cpp");
        if(!out)
        {
            std::cerr << "Failed to open IROpcode.cpp for writing\n";
            return false;
        }

        // File header
        out << "/* ************************************************************************\n";
        out << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n";
        out << " * AUTO-GENERATED FILE - DO NOT EDIT\n";
        out << " * Generated by: tools/tablegen/GenHighLevelIR.cpp\n";
        out << " * ************************************************************************ */\n\n";

        out << "#include \"ir/IROpcode.hpp\"\n\n";
        out << "namespace stinkytofu\n";
        out << "{\n";
        out << "namespace HLIR\n";
        out << "{\n\n";

        // Generate getOpcodeName function
        out << "const char* getOpcodeName(Opcode opcode)\n";
        out << "{\n";
        out << "    switch (opcode)\n";
        out << "    {\n";
        out << "    case UNKNOWN:\n";
        out << "        return \"UNKNOWN\";\n";

        for(const auto& inst : getIRInstructions())
        {
            out << "    case " << inst.className << ":\n";
            out << "        return \"" << inst.className << "\";\n";
        }

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

        for(const auto& inst : getIRInstructions())
        {
            out << "    case " << inst.className << ":\n";
            out << "        return \"" << inst.mnemonic << "\";\n";
        }

        out << "    default:\n";
        out << "        return \"invalid\";\n";
        out << "    }\n";
        out << "}\n\n";

        out << "} // namespace HLIR\n";
        out << "} // namespace stinkytofu\n";

        std::cout << "Generated opcode mapping functions -> IROpcode.cpp\n";
        return true;
    }

    bool genIRClasses(const std::string& outdir)
    {
        std::ofstream out(outdir + "/ir/StinkyInstructions_generated.hpp");
        if(!out)
        {
            std::cerr << "Failed to open StinkyInstructions_generated.hpp for writing\n";
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
        out << "#include \"ir/asm/StinkyAsmIR.hpp\"\n";
        out << "#include \"ir/asm/StinkyModifiers.hpp\"\n";
        out << "#include \"stinkytofu.hpp\"\n";
        out << "#include <iostream>\n";
        out << "#include <optional>\n";
        out << "#include <string>\n";
        out << "\n// TODO: High-level IR should not depend on assembly-level IR "
               "(StinkyAsmIR.hpp).\n";
        out << "// Extract StinkyRegister and RegType into a separate header (e.g., "
               "ir/StinkyRegister.hpp)\n";
        out << "// to fix the inverted dependency. StinkyRegister is a shared primitive used by "
               "both\n";
        out << "// high-level IR (IRInstruction) and assembly IR (StinkyInstruction).\n";
        out << "#include <vector>\n\n";
        out << "namespace stinkytofu\n";
        out << "{\n\n";
        out << "    // NOTE: IRInstruction base class must be defined before including this "
               "file\n\n";

        std::string currentCategory = "";
        for(const auto& inst : getIRInstructions())
        {
            if(inst.category != currentCategory)
            {
                if(!currentCategory.empty())
                {
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

            out << "    /**\n";
            out << "     * @brief " << inst.comment << "\n";
            out << "     */\n";
            out << "    class " << inst.className << " : public IRInstruction\n";
            out << "    {\n";
            out << "    public:\n";

            // Add modifier member variables based on instruction definition
            if(inst.supportsDPP || inst.supportsSDWA)
            {
                if(inst.supportsDPP)
                    out << "        std::optional<DPPModifiers>  dpp;  ///< Optional DPP "
                           "modifier\n";
                if(inst.supportsSDWA)
                    out << "        std::optional<SDWAModifiers> sdwa; ///< Optional SDWA "
                           "modifier\n";
                out << "\n";
            }
            if(inst.hasDS)
            {
                out << "        std::optional<DSModifiers> ds; ///< Optional DS modifier\n\n";
            }

            out << "        " << inst.className << "(";

            // Constructor parameters
            if(inst.hasDest)
            {
                out << "const StinkyRegister& dst";
                if(inst.numSrcs > 0)
                    out << ",\n" << std::string(inst.className.length() + 9, ' ');
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                if(i > 0 || inst.hasDest)
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 9, ' ');
                }
                else
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 9, ' ');
                }
            }

            if(inst.hasDest || inst.numSrcs > 0)
            {
                out << ",\n" << std::string(inst.className.length() + 9, ' ');
            }

            // Add modifier parameters based on instruction definition
            if(inst.supportsDPP)
            {
                out << "std::optional<DPPModifiers> dpp_ = std::nullopt,\n"
                    << std::string(inst.className.length() + 9, ' ');
            }
            if(inst.supportsSDWA)
            {
                out << "std::optional<SDWAModifiers> sdwa_ = std::nullopt,\n"
                    << std::string(inst.className.length() + 9, ' ');
            }
            if(inst.hasDS)
            {
                out << "std::optional<DSModifiers> ds_ = std::nullopt,\n"
                    << std::string(inst.className.length() + 9, ' ');
            }

            out << "const std::string& comment = \"\")\n";
            out << "            : IRInstruction(IRType::StinkyTofu)\n";

            // Add initializer list for modifiers
            if(inst.supportsDPP)
            {
                out << "            , dpp(dpp_)\n";
            }
            if(inst.supportsSDWA)
            {
                out << "            , sdwa(sdwa_)\n";
            }
            if(inst.hasDS)
            {
                out << "            , ds(ds_)\n";
            }

            out << "        {\n";

            // Constructor body
            if(inst.hasDest)
            {
                out << "            dests.push_back(dst);\n";
            }
            for(int i = 0; i < inst.numSrcs; ++i)
            {
                out << "            srcs.push_back(src" << i << ");\n";
            }
            out << "            this->comment = comment;\n";
            out << "        }\n\n";

            // getLogicalName method
            out << "        const char* getLogicalName() const override\n";
            out << "        {\n";
            out << "            return \"" << inst.className << "\";\n";
            out << "        }\n\n";

            // getOpcode method
            out << "        HLIR::Opcode getOpcode() const override\n";
            out << "        {\n";
            out << "            return HLIR::" << inst.className << ";\n";
            out << "        }\n\n";

            // isCommutative method (override if instruction is commutative)
            if(inst.isCommutative)
            {
                out << "        bool isCommutative() const override\n";
                out << "        {\n";
                out << "            return true;\n";
                out << "        }\n\n";
            }

            // Modifier accessor overrides (only if instruction has modifiers)
            if(inst.supportsDPP)
            {
                out << "        std::optional<DPPModifiers> getDPP() const override\n";
                out << "        {\n";
                out << "            return dpp;\n";
                out << "        }\n\n";
            }
            if(inst.supportsSDWA)
            {
                out << "        std::optional<SDWAModifiers> getSDWA() const override\n";
                out << "        {\n";
                out << "            return sdwa;\n";
                out << "        }\n\n";
            }
            if(inst.hasDS)
            {
                out << "        std::optional<DSModifiers> getDS() const override\n";
                out << "        {\n";
                out << "            return ds;\n";
                out << "        }\n\n";
            }

            // dump method
            out << "        void dump(std::ostream& out) const override\n";
            out << "        {\n";
            out << "            out << \"" << inst.className << " (IR)\";\n";
            out << "            if(!comment.empty())\n";
            out << "                out << \"  // \" << comment;\n";
            out << "        }\n";
            out << "    };\n\n";
        }

        // Generate special MFMA classes
        genSpecialMFMAClasses(out);

        // Close namespace
        out << "\n} // namespace stinkytofu\n";

        std::cout << "Generated " << getIRInstructions().size()
                  << " IR instruction classes + 5 special classes "
                     "(MFMA/MXMFMA/SMFMA/TensorLoadToLds/Label) -> "
                     "StinkyInstructions_generated.hpp\n";
        return true;
    }

    // Generate builder method forward declarations
    bool genBuilderDecls(const std::string& outdir)
    {
        std::ofstream out(outdir + "/StinkyBuilder_decls_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open StinkyBuilder_decls_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// Builder method declarations for IR instruction classes\n\n";

        std::string currentCategory = "";
        for(const auto& inst : getIRInstructions())
        {
            if(inst.category != currentCategory)
            {
                if(!currentCategory.empty())
                {
                    out << "\n";
                }
                out << "        // " << inst.category << "\n";
                currentCategory = inst.category;
            }

            // Generate method declaration with proper return type
            out << "        stinkytofu::" << inst.className << "* " << inst.className << "(";

            // Parameters
            bool hasParams = false;
            if(inst.hasDest)
            {
                out << "const StinkyRegister& dst";
                hasParams = true;
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                if(hasParams)
                    out << ",\n" << std::string(inst.className.length() + 10, ' ');
                out << "const StinkyRegister& src" << i;
                hasParams = true;
            }

            if(hasParams)
                out << ",\n" << std::string(inst.className.length() + 10, ' ');

            out << "const std::string& comment = \"\");\n\n";
        }

        std::cout << "Generated builder method declarations -> StinkyBuilder_decls_generated.inc\n";
        return true;
    }

    // Generate builder method implementations
    bool genBuilderImpls(const std::string& outdir)
    {
        std::ofstream out(outdir + "/StinkyBuilder_impls_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open StinkyBuilder_impls_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// Builder method implementations for IR instruction classes\n\n";

        std::string currentCategory = "";
        for(const auto& inst : getIRInstructions())
        {
            if(inst.category != currentCategory)
            {
                if(!currentCategory.empty())
                {
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

            // Method signature
            out << "    stinkytofu::" << inst.className << "* StinkyTofu::" << inst.className
                << "(";

            // Parameters
            if(inst.hasDest)
            {
                out << "const StinkyRegister& dst";
                if(inst.numSrcs > 0)
                    out << ",\n" << std::string(inst.className.length() + 26, ' ');
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                if(i > 0 || inst.hasDest)
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 26, ' ');
                }
                else
                {
                    out << "const StinkyRegister& src" << i;
                    if(i < inst.numSrcs - 1)
                        out << ",\n" << std::string(inst.className.length() + 26, ' ');
                }
            }

            if(inst.hasDest || inst.numSrcs > 0)
            {
                out << ",\n" << std::string(inst.className.length() + 26, ' ');
            }

            out << "const std::string& comment)\n";
            out << "    {\n";

            // Method body - construct and return IR instruction
            out << "        return new stinkytofu::" << inst.className << "(";

            // Arguments to IR constructor
            if(inst.hasDest)
            {
                out << "dst";
                if(inst.numSrcs > 0)
                    out << ", ";
            }

            for(int i = 0; i < inst.numSrcs; ++i)
            {
                out << "src" << i;
                if(i < inst.numSrcs - 1)
                    out << ", ";
            }

            // Add std::nullopt for modifiers
            if(inst.numSrcs > 0 || inst.hasDest)
                out << ", ";

            if(inst.supportsDPP)
            {
                out << "std::nullopt, ";
            }
            if(inst.supportsSDWA)
            {
                out << "std::nullopt, ";
            }
            if(inst.hasDS)
            {
                out << "std::nullopt, ";
            }

            out << "comment);\n";
            out << "    }\n\n";
        }

        std::cout << "Generated " << getIRInstructions().size()
                  << " builder method implementations -> StinkyBuilder_impls_generated.inc\n";
        return true;
    }

    // Generate mnemonic mappings for ToStinkyAsmPass
    bool genMnemonicMappings(const std::string& outdir)
    {
        std::ofstream out(outdir + "/ir/IRMnemonics_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open IRMnemonics_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated by TableGen - DO NOT EDIT\n";
        out << "// Logical IR name to assembly mnemonic mappings\n\n";

        int count = 0;

        // Regular instructions
        for(const auto& inst : getIRInstructions())
        {
            out << "            else if(std::string(logicalName) == \"" << inst.className
                << "\")\n";
            out << "            {\n";
            out << "                mnemonic = \"" << inst.mnemonic << "\";\n";
            out << "            }\n";
            count++;
        }

        // Note: Special instructions (MFMA/MXMFMA/SMFMA/TensorLoadToLds/Label)
        // generate their mnemonics dynamically, not through this mapping

        std::cout << "Generated " << count << " mnemonic mappings -> IRMnemonics_generated.inc\n";
        return true;
    }

    // Generate Python bindings for all IR instructions
    bool genPythonBindings(const std::string& outdir)
    {
        std::ofstream out(outdir + "/PythonBindings_generated.inc");
        if(!out)
        {
            std::cerr << "Failed to open PythonBindings_generated.inc for writing\n";
            return false;
        }

        out << "// Auto-generated Python bindings for IR instructions\n";
        out << "// DO NOT EDIT - Generated by GenHighLevelIR.cpp\n\n";

        int count = 0;

        // Generate bindings for all regular IR instructions
        for(const auto& inst : getIRInstructions())
        {
            std::string className = inst.className;

            // Generate factory function instead of constructor
            out << "    m.def(\"" << className << "\", [](";

            // Constructor parameter types
            std::vector<std::string> paramTypes;
            std::vector<std::string> paramNames;
            std::vector<std::string> defaults;

            // Add destination register (if instruction has one)
            if(inst.hasDest)
            {
                paramTypes.push_back("const StinkyRegister&");
                paramNames.push_back("dest");
                defaults.push_back("");
            }

            // Add source operands
            for(int i = 0; i < inst.numSrcs; i++)
            {
                paramTypes.push_back("const StinkyRegister&");
                paramNames.push_back("src" + std::to_string(i));
                defaults.push_back("");
            }

            // Add optional modifiers
            if(inst.supportsDPP && inst.supportsSDWA)
            {
                paramTypes.push_back("std::optional<DPPModifiers>");
                paramNames.push_back("dpp");
                defaults.push_back("std::nullopt");

                paramTypes.push_back("std::optional<SDWAModifiers>");
                paramNames.push_back("sdwa");
                defaults.push_back("std::nullopt");
            }
            else if(inst.hasDS)
            {
                paramTypes.push_back("std::optional<DSModifiers>");
                paramNames.push_back("ds");
                defaults.push_back("std::nullopt");
            }

            // Add comment parameter
            paramTypes.push_back("const std::string&");
            paramNames.push_back("comment");
            defaults.push_back("\"\"");

            // Output lambda parameter list
            for(size_t i = 0; i < paramTypes.size(); i++)
            {
                out << paramTypes[i] << " " << paramNames[i];
                if(i + 1 < paramTypes.size())
                    out << ", ";
            }
            out << ") {\n";

            // Return std::make_shared as base class pointer
            out << "        return std::shared_ptr<IRInstruction>(std::make_shared<" << className
                << ">(";
            for(size_t i = 0; i < paramNames.size(); i++)
            {
                out << paramNames[i];
                if(i + 1 < paramNames.size())
                    out << ", ";
            }
            out << "));\n";
            out << "    },\n";

            // Output parameter names and defaults
            for(size_t i = 0; i < paramNames.size(); i++)
            {
                out << "    nb::arg(\"" << paramNames[i] << "\")";
                if(!defaults[i].empty())
                    out << " = " << defaults[i];
                if(i + 1 < paramNames.size())
                    out << ",\n";
            }
            out << ",\n";

            // Add docstring
            out << "    \"Create a " << inst.mnemonic << " instruction\");\n\n";
            count++;
        }

        std::cout << "Generated Python bindings for " << count
                  << " IR instructions -> PythonBindings_generated.inc\n";
        return true;
    }

    // Generate all high-level IR artifacts
    bool genHighLevelIR(const std::string& outdir)
    {
        bool success = true;

        std::cout << "\n=== Generating High-Level IR ===\n";

        success &= genOpcodeEnum(outdir);
        success &= genOpcodeMappings(outdir);
        success &= genIRClasses(outdir);
        success &= genBuilderDecls(outdir);
        success &= genBuilderImpls(outdir);
        success &= genMnemonicMappings(outdir);
        success &= genPythonBindings(outdir);

        if(success)
        {
            std::cout << "=== High-Level IR generation completed successfully ===\n\n";
        }
        else
        {
            std::cerr << "=== High-Level IR generation failed ===\n\n";
        }

        return success;
    }

} // namespace stinkytofu
