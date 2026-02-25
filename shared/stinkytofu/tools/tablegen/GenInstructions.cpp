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
 * @file GenInstructions.cpp
 * @brief Generates instruction metadata from .def files
 *
 * This generator:
 * 1. Parses format definitions (DEF_FORMAT) from GfxXXXFormats.def
 * 2. Parses instruction definitions (DEF_T) from GfxXXXInstructions.def
 * 3. Applies format inheritance (instructions inherit format defaults)
 * 4. Generates .inc files:
 *    - *_costs.inc: Instruction cost table
 *    - *_operands.inc: Operand requirements
 *    - *_init.inc: DEF_T initialization calls
 *
 * Usage: tablegen --gen-instructions --arch=gfx1250 --input-dir=hardware/src/gfx --output-dir=<build>/hardware (per-arch .def in GfxXXX/ subdirs)
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <filesystem>

namespace stinkytofu
{

    //==========================================================================
    // DATA STRUCTURES
    //==========================================================================

    // Operand specification
    struct OperandSpec
    {
        bool                     isDest = false;
        int                      width  = 1; // Register width (number of consecutive registers)
        std::vector<std::string> valueTypes; // e.g., {"Register", "LiteralInt"}
        std::vector<std::string> regTypes; // e.g., {"VGPR", "SGPR"}

        // Empty vectors mean "any type"
        bool isAnyValueType() const
        {
            return valueTypes.empty();
        }
        bool isAnyRegType() const
        {
            return regTypes.empty();
        }
    };

    // Architecture metadata (from DEF_ARCH in Formats.def)
    struct ArchDef
    {
        std::string name;
        int         major          = 0;
        int         minor          = 0;
        int         stepping       = 0;
        int         wavefront      = 64;
        int         maxVGPR        = 256;
        int         maxSGPR        = 102;
        int         maxAGPR        = 0;
        int         defaultCycle   = 4;
        int         defaultLatency = 4;
    };

    // Format definition (provides defaults for instructions)
    // Hardware formats (VOP1, VOP3P, MUBUF, SMRD, etc.) are defined first; opcode-family
    // formats (e.g. MFMA on CDNA) can inherit from a hardware format via .parent.
    struct FormatDef
    {
        std::string              name;
        std::string              parent; // e.g. "VOP3P" for MFMA (inherits encoding, etc.)
        std::vector<std::string> modifiers;
        std::string              unit;
        std::vector<int>         encoding;
        int                      maxOperands = 0;
        std::string              pipeline;
        std::vector<std::string> flags;
    };

    // Operand width requirement: {operandIndex, width, isDest, regTypeChar ('S','V','A',...)}
    struct OperandWidthEntry
    {
        int  operandIndex = 0;
        int  width        = 1;
        bool isDest       = false;
        char regType      = 'S'; // S=SGPR, V=VGPR, A=AGPR, etc.
    };

    // Instruction definition
    struct InstructionDef
    {
        std::string instClass; // e.g., "FloatAddInst"
        std::string mnemonic; // e.g., "v_add_f32"

        // Optional fields (inherit from format if not specified)
        std::string                    format; // e.g., "VOP3"
        int                            cycle   = 1; // Default cost
        int                            latency = 1; // Default latency
        std::vector<OperandSpec>       operands; // Operand specifications
        std::vector<OperandWidthEntry> operandWidths; // Register width/type for verifier
        std::vector<std::string>       flags; // Instruction-specific flags
        std::vector<std::string>
            logical; // Logical IR names mapping to this mnemonic: .logical = {"A", "B"} or .logical = "A"
        std::string              unit; // Override format unit
        std::string              pipeline; // Override format pipeline
        std::vector<std::string> modifiers; // Override/restrict format modifiers

        // After format inheritance applied
        std::vector<std::string> finalFlags; // Merged format + instance flags
        std::string              finalUnit;
        std::string              finalPipeline;
        std::vector<std::string> finalModifiers;
    };

    //==========================================================================
    // PARSER
    //==========================================================================

    class DefTParser
    {
    public:
        DefTParser(const std::string& arch)
            : archName_(arch)
        {
            arch_.name = arch;
        }

        // Parse format definitions file
        bool parseFormats(const std::string& formatFile)
        {
            std::ifstream ifs(formatFile);
            if(!ifs)
            {
                std::cerr << "Error: Cannot open format file: " << formatFile << "\n";
                return false;
            }

            std::string content((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());

            return parseFormatContent(content);
        }

        // Parse instruction definitions file
        bool parseInstructions(const std::string& instFile)
        {
            std::ifstream ifs(instFile);
            if(!ifs)
            {
                std::cerr << "Error: Cannot open instruction file: " << instFile << "\n";
                return false;
            }

            std::string content((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());

            return parseInstructionContent(content);
        }

        // Resolve a format by merging with its parent (e.g. MFMA -> VOP3P). Single level only.
        FormatDef getResolvedFormat(const std::string& name) const
        {
            auto it = formats_.find(name);
            if(it == formats_.end())
                return FormatDef{};
            FormatDef fmt = it->second;
            if(fmt.parent.empty())
                return fmt;
            auto pit = formats_.find(fmt.parent);
            if(pit == formats_.end())
                return fmt;
            const FormatDef& p = pit->second;
            // Child overrides where non-empty
            if(fmt.unit.empty())
                fmt.unit = p.unit;
            if(fmt.maxOperands == 0)
                fmt.maxOperands = p.maxOperands;
            if(fmt.modifiers.empty())
                fmt.modifiers = p.modifiers;
            if(fmt.encoding.empty())
                fmt.encoding = p.encoding;
            if(fmt.pipeline.empty())
                fmt.pipeline = p.pipeline;
            // Flags: parent first, then child (child adds MFMA etc.)
            fmt.flags = p.flags;
            fmt.flags.insert(fmt.flags.end(), it->second.flags.begin(), it->second.flags.end());
            return fmt;
        }

        // Apply format inheritance to all instructions
        void applyFormatDefaults()
        {
            for(auto& inst : instructions_)
            {
                if(inst.format.empty())
                {
                    // No format: use instruction's own fields so final* are always set
                    inst.finalFlags     = inst.flags;
                    inst.finalUnit      = inst.unit;
                    inst.finalPipeline  = inst.pipeline;
                    inst.finalModifiers = inst.modifiers;
                    continue;
                }

                FormatDef fmt = getResolvedFormat(inst.format);
                if(fmt.name.empty())
                {
                    std::cerr << "Warning: Unknown format '" << inst.format << "' for instruction '"
                              << inst.mnemonic << "'\n";
                    inst.finalFlags     = inst.flags;
                    inst.finalUnit      = inst.unit;
                    inst.finalPipeline  = inst.pipeline;
                    inst.finalModifiers = inst.modifiers;
                    continue;
                }

                // Merge flags: format flags + instance flags
                inst.finalFlags = fmt.flags;
                inst.finalFlags.insert(inst.finalFlags.end(), inst.flags.begin(), inst.flags.end());

                // Apply unit (instance overrides format)
                inst.finalUnit = inst.unit.empty() ? fmt.unit : inst.unit;

                // Apply pipeline (instance overrides format)
                inst.finalPipeline = inst.pipeline.empty() ? fmt.pipeline : inst.pipeline;

                // Apply modifiers (instance overrides format)
                inst.finalModifiers = inst.modifiers.empty() ? fmt.modifiers : inst.modifiers;
            }
        }

        // Get parsed data
        const ArchDef& getArch() const
        {
            return arch_;
        }
        const std::map<std::string, FormatDef>& getFormats() const
        {
            return formats_;
        }
        const std::vector<InstructionDef>& getInstructions() const
        {
            return instructions_;
        }

    private:
        std::string                      archName_;
        ArchDef                          arch_;
        std::map<std::string, FormatDef> formats_;
        std::vector<InstructionDef>      instructions_;

        bool parseFormatContent(const std::string& content)
        {
            // Parse DEF_ARCH(...) first (optional; overrides arch name and metadata)
            size_t defArchPos = content.find("DEF_ARCH(");
            if(defArchPos != std::string::npos)
            {
                size_t nameStart = defArchPos + 9;
                size_t nameEnd   = content.find(",", nameStart);
                if(nameEnd != std::string::npos)
                {
                    arch_.name = content.substr(nameStart, nameEnd - nameStart);
                    arch_.name.erase(0, arch_.name.find_first_not_of(" \t\n\r"));
                    arch_.name.erase(arch_.name.find_last_not_of(" \t\n\r") + 1);

                    size_t blockEnd = findMatchingParen(content, nameEnd);
                    if(blockEnd != std::string::npos)
                    {
                        std::string block = content.substr(nameEnd, blockEnd - nameEnd);
                        parseFieldInt(block, ".major", arch_.major);
                        parseFieldInt(block, ".minor", arch_.minor);
                        parseFieldInt(block, ".stepping", arch_.stepping);
                        parseFieldInt(block, ".wavefront", arch_.wavefront);
                        parseFieldInt(block, ".maxVGPR", arch_.maxVGPR);
                        parseFieldInt(block, ".maxSGPR", arch_.maxSGPR);
                        parseFieldInt(block, ".maxAGPR", arch_.maxAGPR);
                        parseFieldInt(block, ".defaultCycle", arch_.defaultCycle);
                        parseFieldInt(block, ".defaultLatency", arch_.defaultLatency);
                    }
                }
            }
            else
            {
                arch_.major          = 9;
                arch_.minor          = 4;
                arch_.stepping       = 0;
                arch_.wavefront      = 64;
                arch_.maxVGPR        = 256;
                arch_.maxSGPR        = 102;
                arch_.maxAGPR        = 0;
                arch_.defaultCycle   = 4;
                arch_.defaultLatency = 4;
            }

            // Parse DEF_FORMAT(...) blocks
            size_t pos = 0;
            while((pos = content.find("DEF_FORMAT(", pos)) != std::string::npos)
            {
                size_t nameStart = pos + 11; // After "DEF_FORMAT("
                size_t nameEnd   = content.find(",", nameStart);
                if(nameEnd == std::string::npos)
                    break;

                std::string formatName = content.substr(nameStart, nameEnd - nameStart);
                // Trim whitespace
                formatName.erase(0, formatName.find_first_not_of(" \t\n\r"));
                formatName.erase(formatName.find_last_not_of(" \t\n\r") + 1);

                FormatDef fmt;
                fmt.name = formatName;

                // Find the closing paren for this DEF_FORMAT
                size_t blockEnd = findMatchingParen(content, nameEnd);
                if(blockEnd == std::string::npos)
                {
                    std::cerr << "Error: No matching paren for DEF_FORMAT " << formatName << "\n";
                    return false;
                }

                std::string block = content.substr(nameEnd, blockEnd - nameEnd);

                // Parse fields (simple substring matching for proof-of-concept)
                parseField(block, ".parent", fmt.parent);
                parseField(block, ".unit", fmt.unit);
                parseFieldInt(block, ".maxOperands", fmt.maxOperands);
                parseFieldFlags(block, ".flags", fmt.flags);

                formats_[formatName] = fmt;
                pos                  = blockEnd;
            }

            std::cout << "Parsed " << formats_.size() << " formats\n";
            return true;
        }

        bool parseInstructionContent(const std::string& content)
        {
            // Minimal parser for proof-of-concept
            // Parse DEF_T(...) blocks
            size_t pos = 0;
            while((pos = content.find("DEF_T(", pos)) != std::string::npos)
            {
                size_t argsStart  = pos + 6; // After "DEF_T("
                size_t firstComma = content.find(",", argsStart);
                if(firstComma == std::string::npos)
                    break;

                // Parse: DEF_T(InstClass, "mnemonic", ...)
                std::string instClass = content.substr(argsStart, firstComma - argsStart);
                instClass.erase(0, instClass.find_first_not_of(" \t\n\r"));
                instClass.erase(instClass.find_last_not_of(" \t\n\r") + 1);

                size_t mnemonicStart = content.find("\"", firstComma);
                size_t mnemonicEnd   = content.find("\"", mnemonicStart + 1);
                if(mnemonicStart == std::string::npos || mnemonicEnd == std::string::npos)
                    break;

                std::string mnemonic
                    = content.substr(mnemonicStart + 1, mnemonicEnd - mnemonicStart - 1);

                InstructionDef inst;
                inst.instClass = instClass;
                inst.mnemonic  = mnemonic;

                // Find the closing paren
                size_t blockEnd = findMatchingParen(content, mnemonicEnd);
                if(blockEnd == std::string::npos)
                {
                    std::cerr << "Error: No matching paren for DEF_T " << mnemonic << "\n";
                    return false;
                }

                std::string block = content.substr(mnemonicEnd, blockEnd - mnemonicEnd);

                // Parse optional fields
                parseField(block, ".format", inst.format);
                parseFieldCost(block, ".cost", inst.cycle, inst.latency);
                parseFieldFlags(block, ".flags", inst.flags);
                parseFieldOperandWidths(block, inst.operandWidths);
                parseFieldLogical(block, inst.logical);

                instructions_.push_back(inst);
                pos = blockEnd;
            }

            std::cout << "Parsed " << instructions_.size() << " instructions\n";
            return true;
        }

        // Helper: Find matching closing paren
        size_t findMatchingParen(const std::string& content, size_t start)
        {
            int depth = 1;
            for(size_t i = start; i < content.size(); ++i)
            {
                if(content[i] == '(')
                    depth++;
                else if(content[i] == ')')
                {
                    depth--;
                    if(depth == 0)
                        return i;
                }
            }
            return std::string::npos;
        }

        // Helper: Parse simple field
        void parseField(const std::string& block, const std::string& fieldName, std::string& out)
        {
            size_t pos = block.find(fieldName);
            if(pos == std::string::npos)
                return;

            size_t eqPos = block.find("=", pos);
            if(eqPos == std::string::npos)
                return;

            size_t valStart = eqPos + 1;
            size_t valEnd   = block.find_first_of(",\n)", valStart);
            if(valEnd == std::string::npos)
                valEnd = block.size();

            out = block.substr(valStart, valEnd - valStart);
            out.erase(0, out.find_first_not_of(" \t\n\r"));
            out.erase(out.find_last_not_of(" \t\n\r,") + 1);
            // Strip surrounding quotes for string values (e.g. .logical = "VCmpEQF32")
            if(out.size() >= 2 && out.front() == '"' && out.back() == '"')
                out = out.substr(1, out.size() - 2);
        }

        // Helper: Parse integer field
        void parseFieldInt(const std::string& block, const std::string& fieldName, int& out)
        {
            std::string val;
            parseField(block, fieldName, val);
            if(!val.empty())
                out = std::stoi(val);
        }

        // Helper: Parse cost field (.cost = {cycle, latency})
        void parseFieldCost(const std::string& block,
                            const std::string& fieldName,
                            int&               cycle,
                            int&               latency)
        {
            size_t pos = block.find(fieldName);
            if(pos == std::string::npos)
                return;

            size_t lbrace = block.find("{", pos);
            size_t rbrace = block.find("}", lbrace);
            if(lbrace == std::string::npos || rbrace == std::string::npos)
                return;

            std::string costStr = block.substr(lbrace + 1, rbrace - lbrace - 1);
            size_t      comma   = costStr.find(",");
            if(comma != std::string::npos)
            {
                cycle   = std::stoi(costStr.substr(0, comma));
                latency = std::stoi(costStr.substr(comma + 1));
            }
        }

        // Helper: Parse flags field (.flags = {flag1, flag2})
        void parseFieldFlags(const std::string&        block,
                             const std::string&        fieldName,
                             std::vector<std::string>& flags)
        {
            size_t pos = block.find(fieldName);
            if(pos == std::string::npos)
                return;

            size_t lbrace = block.find("{", pos);
            size_t rbrace = block.find("}", lbrace);
            if(lbrace == std::string::npos || rbrace == std::string::npos)
                return;

            std::string flagsStr = block.substr(lbrace + 1, rbrace - lbrace - 1);

            // Split by comma
            size_t start = 0;
            while(start < flagsStr.size())
            {
                size_t end = flagsStr.find(",", start);
                if(end == std::string::npos)
                    end = flagsStr.size();

                std::string flag = flagsStr.substr(start, end - start);
                flag.erase(0, flag.find_first_not_of(" \t\n\r"));
                flag.erase(flag.find_last_not_of(" \t\n\r") + 1);

                if(!flag.empty())
                    flags.push_back(flag);

                start = end + 1;
            }
        }

        // Helper: Parse .logical = "X" or .logical = {"X", "Y"}
        void parseFieldLogical(const std::string& block, std::vector<std::string>& out)
        {
            size_t pos = block.find(".logical");
            if(pos == std::string::npos)
                return;

            size_t eqPos = block.find("=", pos);
            if(eqPos == std::string::npos)
                return;

            size_t valStart = eqPos + 1;
            valStart        = block.find_first_not_of(" \t\n\r", valStart);
            if(valStart == std::string::npos)
                return;

            if(block[valStart] == '{')
            {
                // .logical = {"X", "Y"}
                size_t rbrace = block.find("}", valStart);
                if(rbrace == std::string::npos)
                    return;
                std::string inner = block.substr(valStart + 1, rbrace - valStart - 1);
                size_t      start = 0;
                while(start < inner.size())
                {
                    size_t end = inner.find(",", start);
                    if(end == std::string::npos)
                        end = inner.size();
                    std::string s = inner.substr(start, end - start);
                    s.erase(0, s.find_first_not_of(" \t\n\r\""));
                    s.erase(s.find_last_not_of(" \t\n\r\"") + 1);
                    if(!s.empty())
                        out.push_back(s);
                    start = end + 1;
                }
            }
            else if(block[valStart] == '"')
            {
                // .logical = "X"
                size_t endQuote = block.find('"', valStart + 1);
                if(endQuote != std::string::npos)
                    out.push_back(block.substr(valStart + 1, endQuote - valStart - 1));
            }
        }

        // Helper: Parse .operand_widths = { {0, 4, false, S}, {1, 8, false, S} }
        void parseFieldOperandWidths(const std::string& block, std::vector<OperandWidthEntry>& out)
        {
            size_t pos = block.find(".operand_widths");
            if(pos == std::string::npos)
                return;

            size_t outerStart = block.find("{", pos);
            if(outerStart == std::string::npos)
                return;

            size_t i = outerStart + 1;
            while(i < block.size())
            {
                // Find next inner { for one entry
                size_t innerStart = block.find("{", i);
                if(innerStart == std::string::npos)
                    break;
                size_t innerEnd = block.find("}", innerStart);
                if(innerEnd == std::string::npos)
                    break;

                std::string       entry = block.substr(innerStart + 1, innerEnd - innerStart - 1);
                OperandWidthEntry e;
                // Parse: 0, 4, false, S (operandIndex, width, isDest, regType)
                size_t p      = 0;
                auto   skipWs = [&]() {
                    while(p < entry.size() && (entry[p] == ' ' || entry[p] == '\t'))
                        p++;
                };
                auto nextInt = [&]() {
                    skipWs();
                    size_t start = p;
                    while(p < entry.size() && (std::isdigit(entry[p]) || entry[p] == '-'))
                        p++;
                    return std::stoi(entry.substr(start, p - start));
                };
                auto nextBool = [&]() {
                    skipWs();
                    if(entry.compare(p, 5, "false") == 0)
                    {
                        p += 5;
                        return false;
                    }
                    if(entry.compare(p, 4, "true") == 0)
                    {
                        p += 4;
                        return true;
                    }
                    return false;
                };
                auto nextRegType = [&]() {
                    skipWs();
                    if(p < entry.size() && std::isalpha(entry[p]))
                    {
                        char c = entry[p++];
                        return c;
                    }
                    return 'S';
                };

                e.operandIndex = nextInt();
                skipWs();
                if(p < entry.size() && entry[p] == ',')
                    p++;
                e.width = nextInt();
                skipWs();
                if(p < entry.size() && entry[p] == ',')
                    p++;
                e.isDest = nextBool();
                skipWs();
                if(p < entry.size() && entry[p] == ',')
                    p++;
                e.regType = nextRegType();

                out.push_back(e);
                i = innerEnd + 1;
            }
        }
    };

    //==========================================================================
    // CODE GENERATORS
    //==========================================================================

    class InstructionCodeGen
    {
    public:
        InstructionCodeGen(const std::string& arch, const std::vector<InstructionDef>& instructions)
            : arch_(arch)
            , instructions_(instructions)
        {
        }

        // Generate cost table file
        bool generateCostTable(const std::string& outputPath)
        {
            std::ofstream out(outputPath);
            if(!out)
            {
                std::cerr << "Error: Cannot write " << outputPath << "\n";
                return false;
            }

            emitHeader(out, "Instruction Cost Table");
            out << "// Cost table: {mnemonic, cycle, latency}\n";
            out << "// Only non-default costs are listed (default: cycle=1, latency=1)\n\n";

            out << "constexpr InstructionCost " << arch_ << "_GENERATED_COSTS[] = {\n";

            for(const auto& inst : instructions_)
            {
                if(inst.cycle != 1 || inst.latency != 1) // Only emit non-default costs
                {
                    out << "    {\"" << inst.mnemonic << "\", " << inst.cycle << ", "
                        << inst.latency << "},\n";
                }
            }

            out << "};\n\n";
            out << "// Total generated instructions: " << instructions_.size() << "\n";
            return true;
        }

        // Map reg type char from .def to C++ RegType
        static std::string regTypeToCpp(char c)
        {
            switch(c)
            {
            case 'S':
                return "RegType::S";
            case 'V':
                return "RegType::V";
            case 'A':
                return "RegType::A";
            case 'M':
                return "RegType::M";
            default:
                return "RegType::S";
            }
        }

        // Generate operand requirements file (included by ArchInfo getMCIDTable())
        bool generateOperandRequirements(const std::string& outputPath)
        {
            std::ofstream out(outputPath);
            if(!out)
            {
                std::cerr << "Error: Cannot write " << outputPath << "\n";
                return false;
            }

            emitHeader(out, "Operand Requirements");
            out << "// Operand width/type requirements for IR verifier.\n";
            out << "// Included inside getMCIDTable(); defines instRequirements[].\n\n";

            std::vector<const InstructionDef*> withWidths;
            for(const auto& inst : instructions_)
            {
                if(!inst.operandWidths.empty())
                    withWidths.push_back(&inst);
            }

            if(withWidths.empty())
            {
                out << "// No instructions with .operand_widths in this arch\n";
                out << "static constexpr struct {\n";
                out << "    const char* mnemonic;\n";
                out << "    stinkytofu::span<const stinkytofu::HwInstDesc::OperandWidth> "
                       "requirements;\n";
                out << "} instRequirements[] = {};\n";
                return true;
            }

            for(const auto* inst : withWidths)
            {
                std::string arrayName = "operand_widths_";
                for(char c : inst->mnemonic)
                    arrayName += (c == '_' ? '_' : (char)std::tolower(c));

                out << "static constexpr stinkytofu::HwInstDesc::OperandWidth " << arrayName
                    << "[] = {\n";
                for(const auto& e : inst->operandWidths)
                {
                    out << "    {" << (int)e.operandIndex << ", " << (int)e.width << ", "
                        << (e.isDest ? "true" : "false") << ", " << regTypeToCpp(e.regType)
                        << "},\n";
                }
                out << "};\n\n";
            }

            out << "static constexpr struct {\n";
            out << "    const char* mnemonic;\n";
            out << "    stinkytofu::span<const stinkytofu::HwInstDesc::OperandWidth> "
                   "requirements;\n";
            out << "} instRequirements[] = {\n";
            for(size_t i = 0; i < withWidths.size(); ++i)
            {
                const auto* inst      = withWidths[i];
                std::string arrayName = "operand_widths_";
                for(char c : inst->mnemonic)
                    arrayName += (c == '_' ? '_' : (char)std::tolower(c));
                out << "    {\"" << inst->mnemonic << "\", " << arrayName << "}";
                out << (i + 1 < withWidths.size() ? ",\n" : "\n");
            }
            out << "};\n";

            return true;
        }

        // Generate initialization file (DEF_T calls)
        bool generateInitFile(const std::string& outputPath)
        {
            std::ofstream out(outputPath);
            if(!out)
            {
                std::cerr << "Error: Cannot write " << outputPath << "\n";
                return false;
            }

            emitHeader(out, "Instruction Initialization");
            out << "// Generated DEF_T initialization calls\n\n";

            for(const auto& inst : instructions_)
            {
                std::string flagArgs = flagsToInitArgs(inst.finalFlags);
                out << "DEF_T(\"" << inst.mnemonic << "\"";
                if(!flagArgs.empty())
                    out << ", " << flagArgs;
                out << ");\n";
            }

            out << "\n// Total generated instructions: " << instructions_.size() << "\n";
            return true;
        }

    private:
        static std::string flagsToInitArgs(const std::vector<std::string>& flags)
        {
            if(flags.empty())
                return "";
            std::ostringstream os;
            for(size_t i = 0; i < flags.size(); ++i)
                os << (i ? ", " : "") << "IF_" << flags[i];
            return os.str();
        }
        std::string                        arch_;
        const std::vector<InstructionDef>& instructions_;

        void emitHeader(std::ofstream& out, const std::string& description)
        {
            out << "//===----------------------------------------------------------------------===/"
                   "/\n"
                << "// Auto-generated " << description << " for " << arch_ << "\n"
                << "//\n"
                << "// DO NOT EDIT MANUALLY!\n"
                << "// Generated by tablegen --gen-instructions\n"
                << "//===----------------------------------------------------------------------===/"
                   "/\n\n";
        }
    };

    //==========================================================================
    // ISA .inc EMISSION (from .def only; no gfxisa dependency)
    //==========================================================================

    // Convert finalFlags to C++ makeFlagSet({ IF_XXX, ... }) content (IF_ prefix per flag)
    static std::string flagsToMakeFlagSetContent(const std::vector<std::string>& flags)
    {
        if(flags.empty())
            return "";
        std::ostringstream os;
        for(size_t i = 0; i < flags.size(); ++i)
            os << (i ? ", " : "") << "IF_" << flags[i];
        return os.str();
    }

    // Emit one arch's Isa.inc (opcode enum, MCIDTable, getArchOpcode, MnemonicToIsaOpcodeMap).
    // unifiedOpcodeMap: mnemonic -> global unified opcode index.
    static bool emitArchIsaFile(const std::string&                          arch,
                                const std::vector<InstructionDef>&          instructions,
                                const std::unordered_map<std::string, int>& unifiedOpcodeMap,
                                const std::string&                          outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }

        size_t maxMnemonicLen = 0;
        for(const auto& inst : instructions)
            maxMnemonicLen = std::max(maxMnemonicLen, inst.mnemonic.size());

        out << "//===----------------------------------------------------------------------===//\n"
            << "// Auto-generated ISA header for " << arch << " (from .def, tablegen_inst_gen)\n"
            << "// DO NOT EDIT MANUALLY! DO NOT USE #pragma once IN THIS FILE!\n"
            << "//===----------------------------------------------------------------------===//"
               "\n\n";

#define EMIT_GUARD(MACRO) out << "#ifdef " << (MACRO) << "\n#undef " << (MACRO) << "\n\n"

        // Opcode enumeration
        EMIT_GUARD("GET_ISAINFO_OPCODE_ENUMERATION");
        out << "enum " << arch << " : uint16_t {\n";
        for(size_t i = 0; i < instructions.size(); ++i)
            out << "  " << instructions[i].mnemonic << ", // " << i << "\n";
        out << "};\n\n";
        out << "#endif // GET_ISAINFO_OPCODE_ENUMERATION\n\n";

        // MCIDTable
        EMIT_GUARD("GET_ISAINFO_HWINSTDESC_TABLE");
        out << "// MCIDTable: operandWidths set by ArchInfo getMCIDTable()\n"
            << "static HwInstDesc MCIDTable[] = {\n";
        for(size_t i = 0; i < instructions.size(); ++i)
        {
            const auto& inst = instructions[i];
            int         uop  = 0;
            auto        it   = unifiedOpcodeMap.find(inst.mnemonic);
            if(it != unifiedOpcodeMap.end())
                uop = it->second;
            std::string flagStr = flagsToMakeFlagSetContent(inst.finalFlags);
            out << "  { " << std::setw(3) << i << ", " << std::setw(5) << uop << ", "
                << std::setw(3) << inst.cycle << ", " << std::setw(4) << inst.latency << ", "
                << "\"" << inst.mnemonic << "\", "
                << "makeFlagSet({" << (flagStr.empty() ? "" : flagStr) << "}), "
                << "{} },\n";
        }
        out << "};\n\n";
        out << "#endif // GET_ISAINFO_HWINSTDESC_TABLE\n\n";

        // getArchOpcode(unifiedOpcode) - binary search table
        EMIT_GUARD("GET_ISAINFO_UOP_MAPPINGS");
        // Sort by unified opcode for binary search
        std::vector<size_t> idx(instructions.size());
        for(size_t i = 0; i < instructions.size(); ++i)
            idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            int  ua = 0, ub = 0;
            auto ita = unifiedOpcodeMap.find(instructions[a].mnemonic);
            auto itb = unifiedOpcodeMap.find(instructions[b].mnemonic);
            if(ita != unifiedOpcodeMap.end())
                ua = ita->second;
            if(itb != unifiedOpcodeMap.end())
                ub = itb->second;
            return ua < ub;
        });
        out << "uint16_t get" << arch << "Opcode(uint16_t unifiedOpcode) {\n"
            << "    static constexpr uint16_t Table[][2] = {\n";
        for(size_t k = 0; k < idx.size(); ++k)
        {
            size_t i   = idx[k];
            int    uop = 0;
            auto   it  = unifiedOpcodeMap.find(instructions[i].mnemonic);
            if(it != unifiedOpcodeMap.end())
                uop = it->second;
            out << "      { " << uop << ", " << i << " }, // " << instructions[i].mnemonic << "\n";
        }
        out << "    };\n"
            << "    unsigned low = 0, high = " << (instructions.size() - 1) << ";\n"
            << "    while(low <= high) {\n"
            << "      unsigned mid = low + (high - low) / 2;\n"
            << "      if(Table[mid][0] == unifiedOpcode) return Table[mid][1];\n"
            << "      if(Table[mid][0] < unifiedOpcode) low = mid + 1; else high = mid - 1;\n"
            << "    }\n"
            << "    return GFX::INVALID;\n"
            << "}\n\n";
        out << "#endif // GET_ISAINFO_UOP_MAPPINGS\n\n";

        // MnemonicToIsaOpcodeMap
        EMIT_GUARD("GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS");
        out << "static const std::unordered_map<std::string, IsaOpcode> MnemonicToIsaOpcodeMap = "
               "{\n";
        for(size_t i = 0; i < instructions.size(); ++i)
            out << "  {\"" << instructions[i].mnemonic << "\", " << i << "},\n";
        out << "};\n\n";
        out << "#endif // GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS\n\n";

#undef EMIT_GUARD
        return true;
    }

    // Emit hardware/gfxIsa.inc (GFX unified opcode enum)
    static bool emitGfxIsaFile(const std::vector<std::string>& unifiedMnemonics,
                               const std::string&              outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        out << "//===----------------------------------------------------------------------===//\n"
            << "// Auto-generated unified opcodes (from .def, tablegen_inst_gen)\n"
            << "// DO NOT EDIT MANUALLY! DO NOT USE #pragma once IN THIS FILE!\n"
            << "//===----------------------------------------------------------------------===//"
               "\n\n";
        out << "#ifdef GET_ISAINFO_UNIFIED_OPCODES\n"
            << "#undef GET_ISAINFO_UNIFIED_OPCODES\n\n";
        out << "enum GFX : uint16_t {\n";
        for(size_t i = 0; i < unifiedMnemonics.size(); ++i)
            out << "  " << unifiedMnemonics[i] << ", // " << i << "\n";
        out << "};\n\n#endif // GET_ISAINFO_UNIFIED_OPCODES\n\n";
        return true;
    }

    // Emit GfxXXX.hpp (ArchInfo class) - replaces GfxArch.hpp.in
    static bool emitArchHeader(const ArchDef& arch, const std::string& outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        out << "/* ************************************************************************\n"
            << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n"
            << " *\n"
            << " * Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            << " * of this software and associated documentation files (the \"Software\"), to "
               "deal\n"
            << " * in the Software without restriction, including without limitation the rights\n"
            << " * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
            << " * copies of the Software, and to permit persons to whom the Software is\n"
            << " * furnished to do so, subject to the following conditions:\n"
            << " *\n"
            << " * THE SOFTWARE IS PROVIDED \"AS IS\" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
            << " * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
            << " * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n"
            << " * ************************************************************************ */\n"
            << "#pragma once\n\n"
            << "#include \"stinkytofu/hardware/ArchHelper.hpp\"\n"
            << "#include \"stinkytofu/ir/asm/StinkyAsmIR.hpp\"\n\n"
            << "#include <mutex>\n\n"
            << "// Auto-generated from Formats.def + Instructions.def - do not edit manually.\n\n"
            << "namespace\n{\n\n"
            << "#define GET_ISAINFO_UOP_MAPPINGS\n"
            << "#include \"hardware/" << arch.name << "Isa.inc\"\n\n"
            << "}\n\n"
            << "using namespace stinkytofu;\n\n"
            << "// clang-format off\n"
            << "struct " << arch.name << "ArchInfo : public ArchHelper::ArchInfo\n"
            << "{\n"
            << "    " << arch.name << "ArchInfo()\n"
            << "        : ArchInfo(" << arch.major << ", " << arch.minor << ", " << arch.stepping
            << ", " << arch.wavefront << " /* waveFrontSize */)\n"
            << "    {\n"
            << "    }\n\n"
            << "    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override\n"
            << "    {\n"
            << "        return get" << arch.name << "Opcode(unifiedOpcode);\n"
            << "    }\n\n"
            << "    const HwInstDesc* getMCIDTable() const override\n"
            << "    {\n"
            << "#define GET_ISAINFO_HWINSTDESC_TABLE\n"
            << "#include \"hardware/" << arch.name << "Isa.inc\"\n"
            << "#include \"hardware/generated/" << arch.name << "_operands.inc\"\n\n"
            << "        static std::once_flag once;\n"
            << "        std::call_once(once, [] {\n"
            << "            for(const auto& req : instRequirements)\n"
            << "            {\n"
            << "                for(size_t i = 0; i < sizeof(MCIDTable) / sizeof(MCIDTable[0]); "
               "++i)\n"
            << "                {\n"
            << "                    if(MCIDTable[i].mnemonic && std::string(MCIDTable[i].mnemonic) "
               "== req.mnemonic)\n"
            << "                    {\n"
            << "                        const_cast<HwInstDesc&>(MCIDTable[i]).operandWidths = "
               "req.requirements;\n"
            << "                        break;\n"
            << "                    }\n"
            << "                }\n"
            << "            }\n"
            << "        });\n\n"
            << "        return MCIDTable;\n"
            << "    }\n\n"
            << "    const std::unordered_map<std::string, uint16_t>& getMnemonicToIsaOpcodeMap() "
               "const override\n"
            << "    {\n"
            << "#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS\n"
            << "#include \"hardware/" << arch.name << "Isa.inc\"\n"
            << "        return MnemonicToIsaOpcodeMap;\n"
            << "    }\n"
            << "};\n"
            << "// clang-format on\n";
        return true;
    }

    // Emit GfxXXX_block.inc (defineGfxXXXInsts) - replaces GfxArchDefines_block.inc.in
    static bool emitArchDefinesBlock(const ArchDef& arch, const std::string& outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        out << "// clang-format off\n"
            << "// Per-arch block: costs + define" << arch.name << "Insts().\n"
            << "namespace {\n"
            << "#include \"hardware/generated/" << arch.name << "_costs.inc\"\n"
            << "}\n\n"
            << "void define" << arch.name << "Insts(GpuArch& registry)\n"
            << "{\n"
            << "    registry.setWaveFrontSize(" << arch.wavefront << ");\n"
            << "    GpuArch::RegisterLimits limits;\n"
            << "    limits.maxVGPR = " << arch.maxVGPR << ";\n"
            << "    limits.maxSGPR = " << arch.maxSGPR << ";\n"
            << "    limits.maxAGPR = " << arch.maxAGPR << ";\n"
            << "    if(limits.maxVGPR == 0 || limits.maxSGPR == 0)\n"
            << "    {\n"
            << "        std::cerr << \"FATAL: " << arch.name
            << " must define maxVGPR, maxSGPR > 0 in Formats.def (maxAGPR may be 0)\\n\";\n"
            << "        return;\n"
            << "    }\n"
            << "    registry.setRegisterLimits(limits);\n"
            << "    // Instruction definitions: generated from " << arch.name
            << "Instructions.def\n"
            << "#include \"hardware/generated/" << arch.name << "_init.inc\"\n"
            << "    registry.setDefaultCosts(" << arch.defaultCycle << ", " << arch.defaultLatency
            << ");\n"
            << "    for(const auto& cost : " << arch.name << "_GENERATED_COSTS)\n"
            << "    {\n"
            << "        registry.setInstructionCost(cost.opcode, cost.cycle, cost.latency);\n"
            << "    }\n"
            << "    if(!registry.applyInstructionCosts())\n"
            << "    {\n"
            << "        std::cerr << \"FATAL: Failed to apply instruction costs for " << arch.name
            << "\\n\";\n"
            << "        return;\n"
            << "    }\n"
            << "}\n"
            << "// clang-format on\n";
        return true;
    }

    // Emit GfxArchDefines.cpp - replaces GfxArchDefines.cpp.in
    static bool emitGfxArchDefinesCpp(const std::vector<std::string>& archs,
                                      const std::string&              outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        out << "/* ************************************************************************\n"
            << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n"
            << " * ...\n"
            << " * ************************************************************************ */\n\n"
            << "#include <iostream>\n\n"
            << "#include \"gfx/InstDefDSL.hpp\"\n"
            << "#include \"gfx/InstructionCost.hpp\"\n\n"
            << "// clang-format off\n"
            << "namespace stinkytofu\n"
            << "{\n";
        for(const auto& arch : archs)
            out << "    #include \"hardware/generated/" << arch << "_block.inc\"\n";
        out << "}\n"
            << "// clang-format on\n";
        return true;
    }

    // Emit GfxLogicalMaps.cpp - setGfxXXXLogicalToArchMap from .logical in DEF_T
    static bool emitGfxLogicalMapsCpp(
        const std::map<std::string, std::vector<InstructionDef>>& archInstructions,
        const std::vector<std::string>&                           archs,
        const std::string&                                        outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        out << "/* ************************************************************************\n"
            << " * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.\n"
            << " * Auto-generated from .logical in Instructions.def - do not edit manually.\n"
            << " * ************************************************************************ */\n\n"
            << "#include <string>\n"
            << "#include <unordered_map>\n\n"
            << "#include \"gfx/InstDefDSL.hpp\"\n\n"
            << "namespace stinkytofu\n{\n";

        for(const auto& arch : archs)
        {
            auto it = archInstructions.find(arch);
            if(it == archInstructions.end())
                continue;
            const auto& insts = it->second;
            out << "\nvoid set" << arch << "LogicalToArchMap(GpuArch& registry)\n{\n"
                << "    std::unordered_map<std::string, std::string> logicalToHwInstMap = {\n";
            size_t count = 0;
            for(const auto& inst : insts)
            {
                for(const auto& logicalName : inst.logical)
                {
                    if(!logicalName.empty())
                    {
                        if(count > 0)
                            out << ",\n";
                        out << "        {\"" << logicalName << "\", \"" << inst.mnemonic << "\"}";
                        count++;
                    }
                }
            }
            out << "\n    };\n"
                << "    registry.setLogicalToArchMap(std::move(logicalToHwInstMap));\n"
                << "}\n";
        }
        out << "}\n";
        return true;
    }

    // Emit ArchHelper_includes.inc
    static bool emitArchHelperIncludes(const std::vector<std::string>& archs,
                                       const std::string&              outputPath)
    {
        std::ofstream out(outputPath);
        if(!out)
        {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        out << "/* Architecture-specific ArchInfo headers. */\n";
        for(const auto& arch : archs)
        {
            std::string archUpper = arch;
            for(char& c : archUpper)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            out << "#ifdef STINKYTOFU_ARCH_" << archUpper << "\n"
                << "#include \"" << arch << ".hpp\"\n"
                << "#endif\n\n";
        }
        return true;
    }

    //==========================================================================
    // MAIN ENTRY POINT
    //==========================================================================

    // Normalize arch to match .def filenames (e.g. gfx1250 -> Gfx1250)
    static std::string normalizeArch(const std::string& arch)
    {
        if(arch.empty())
            return arch;
        std::string s = arch;
        if(s.size() >= 1)
            s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
        return s;
    }

    bool genInstructions(const std::string& arch,
                         const std::string& inputDir,
                         const std::string& outputDir)
    {
        std::string normArch = normalizeArch(arch);
        std::cout << "Generating instruction metadata for " << normArch << "...\n";

        // Construct file paths: inputDir is base (e.g. hardware/src/gfx), .def files in arch subdir
        std::string formatFile = inputDir + "/" + normArch + "/" + normArch + "Formats.def";
        std::string instFile   = inputDir + "/" + normArch + "/" + normArch + "Instructions.def";
        std::string outputBase = outputDir + "/" + normArch;

        // Parse format definitions
        DefTParser parser(normArch);
        if(!parser.parseFormats(formatFile))
        {
            std::cerr << "Error: Failed to parse format file\n";
            return false;
        }

        // Parse instruction definitions
        if(!parser.parseInstructions(instFile))
        {
            std::cerr << "Error: Failed to parse instruction file\n";
            return false;
        }

        // Apply format inheritance
        parser.applyFormatDefaults();

        // Generate output files
        InstructionCodeGen codegen(normArch, parser.getInstructions());

        bool success = true;
        success &= codegen.generateCostTable(outputBase + "_costs.inc");
        success &= codegen.generateOperandRequirements(outputBase + "_operands.inc");
        success &= codegen.generateInitFile(outputBase + "_init.inc");

        if(success)
        {
            std::cout << "Successfully generated instruction metadata for " << normArch << "\n";
        }

        return success;
    }

    // Generate for all archs from .def and emit ISA .inc (so full tablegen does not need gfxisa for ISA).
    // Single run: *.def -> costs, init, operands, *Isa.inc, gfxIsa.inc, GfxXXX.hpp, *_block.inc, GfxArchDefines.cpp -> one build.
    bool genAllInstructions(const std::string& inputDir, const std::string& outputDir)
    {
        const std::vector<std::string> archs = {"Gfx1250", "Gfx942", "Gfx950"};

        std::map<std::string, std::vector<InstructionDef>> archInstructions;
        std::map<std::string, ArchDef>                     archDefs;
        for(const std::string& arch : archs)
        {
            std::string formatFile = inputDir + "/" + arch + "/" + arch + "Formats.def";
            std::string instFile   = inputDir + "/" + arch + "/" + arch + "Instructions.def";
            DefTParser  parser(arch);
            if(!parser.parseFormats(formatFile))
            {
                std::cerr << "Error: Failed to parse format file " << formatFile << "\n";
                return false;
            }
            if(!parser.parseInstructions(instFile))
            {
                std::cerr << "Error: Failed to parse instruction file " << instFile << "\n";
                return false;
            }
            parser.applyFormatDefaults();
            archInstructions[arch] = parser.getInstructions();
            archDefs[arch]         = parser.getArch();
        }

        // Build unified mnemonic list (sorted, + LABEL, INVALID) and map mnemonic -> index
        std::set<std::string> allMnemonics;
        for(const auto& p : archInstructions)
            for(const auto& inst : p.second)
                allMnemonics.insert(inst.mnemonic);
        std::vector<std::string> unifiedList(allMnemonics.begin(), allMnemonics.end());
        std::sort(unifiedList.begin(), unifiedList.end());
        unifiedList.push_back("LABEL");
        unifiedList.push_back("INVALID");
        std::unordered_map<std::string, int> unifiedOpcodeMap;
        for(size_t i = 0; i < unifiedList.size(); ++i)
            unifiedOpcodeMap[unifiedList[i]] = static_cast<int>(i);

        std::filesystem::path outPath(outputDir);
        if(!outPath.is_absolute())
            outPath = std::filesystem::absolute(outPath);
        std::filesystem::path hwDir      = outPath / "hardware";
        std::filesystem::path genDir     = hwDir / "generated";
        std::filesystem::path archHdrDir = hwDir / "arch_headers";
        std::error_code       ec;
        std::filesystem::create_directories(hwDir, ec);
        std::filesystem::create_directories(genDir, ec);
        std::filesystem::create_directories(archHdrDir, ec);
        if(!std::filesystem::is_directory(genDir) || !std::filesystem::is_directory(hwDir))
        {
            std::cerr << "Error: Output directories missing or not usable: " << genDir.string()
                      << " " << hwDir.string() << "\n";
            return false;
        }

        bool success = true;
        for(const std::string& arch : archs)
        {
            const std::vector<InstructionDef>& insts = archInstructions[arch];
            const ArchDef&                     ad    = archDefs[arch];
            InstructionCodeGen                 codegen(arch, insts);
            success &= codegen.generateCostTable((genDir / (arch + "_costs.inc")).string());
            success &= codegen.generateOperandRequirements(
                (genDir / (arch + "_operands.inc")).string());
            success &= codegen.generateInitFile((genDir / (arch + "_init.inc")).string());
            success &= emitArchIsaFile(
                arch, insts, unifiedOpcodeMap, (hwDir / (arch + "Isa.inc")).string());
            success &= emitArchHeader(ad, (archHdrDir / (arch + ".hpp")).string());
            success &= emitArchDefinesBlock(ad, (genDir / (arch + "_block.inc")).string());
        }
        success &= emitGfxIsaFile(unifiedList, (hwDir / "gfxIsa.inc").string());
        success &= emitGfxArchDefinesCpp(archs, (genDir / "GfxArchDefines.cpp").string());
        success &= emitGfxLogicalMapsCpp(
            archInstructions, archs, (genDir / "GfxLogicalMaps.cpp").string());
        success &= emitArchHelperIncludes(archs, (archHdrDir / "ArchHelper_includes.inc").string());

        if(success)
            std::cout << "Successfully generated instruction metadata and ISA for all archs\n";
        return success;
    }

} // namespace stinkytofu
