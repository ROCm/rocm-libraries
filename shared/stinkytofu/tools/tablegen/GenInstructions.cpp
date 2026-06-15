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
 * Usage: tablegen --gen-instructions --arch=gfx1250 --input-dir=hardware/src/gfx
 * --output-dir=<build>/hardware (per-arch .def in GfxXXX/ subdirs)
 */

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

//==========================================================================
// DATA STRUCTURES
//==========================================================================

// Operand specification
struct OperandSpec {
    bool isDest = false;
    int width = 1;                        // Register width (number of consecutive registers)
    std::vector<std::string> valueTypes;  // e.g., {"Register", "LiteralInt"}
    std::vector<std::string> regTypes;    // e.g., {"VGPR", "SGPR"}

    // Empty vectors mean "any type"
    bool isAnyValueType() const {
        return valueTypes.empty();
    }
    bool isAnyRegType() const {
        return regTypes.empty();
    }
};

// HWREG name table entry parsed from DEF_HWREG(NAME, ID) lines.
struct HwRegEntry {
    std::string name;  // e.g. "HW_REG_WAVE_MODE"
    int id = 0;        // numeric slot, 0..63
};

// Architecture metadata (from DEF_ARCH in Formats.def)
struct ArchDef {
    std::string name;
    int major = 0;
    int minor = 0;
    int stepping = 0;
    int wavefront = 64;
    int maxVGPR = 256;
    int maxSGPR = 102;
    int maxAGPR = 0;
    int totalVgprPerSimd = 0;
    int vgprAllocGranule = 0;
    int defaultCycle = 4;
    int defaultLatency = 4;
    // ECC presence: D16 VMEM zero-fills the non-data half and True16 VALU does
    // a HW RMW at full-DWORD granularity.
    int d16Writes32BitVgpr = 0;
    std::vector<HwRegEntry> hwRegs;
};

// Operand field description.
//
// Full entry (in .fields or .operand_fields):
//   {Pos, EncodeField, Type, Size [, Options...]}
//   e.g. {D0, vdata, vgpr, 32, RW}
//   e.g. {D0, vdst, vcc, M64}        — M64 in size position implies 64
//
//   Pos          D0, D1, ... (dest) or S0, S1, ... (src)
//   EncodeField  hardware encoding field (vdata, simm16, ssrc0, ...)
//   Type         operand type           (vgpr, sreg, label, wait_alu, ...)
//   Size         field size in bits (16, 32, 64, 128, ...) or M64
//   Options      RW (read-write), M64 (64-bit lane mask, truncatable in wave32)
//
// Partial override (in .operand_fields only) — inherits from format .fields
// and selectively overrides individual properties by name:
//   {S0, .size=64}              — override size only
//   {S0, .type=wait_alu}        — override type only
//   {D0, .size=64, .type=vgpr}  — override both
struct OperandFieldEntry {
    bool isDest = false;
    bool isReadWrite = false;
    bool isM64 = false;
    std::string encodeField;  // "vdata", "vaddr", "rsrc", "soffset", "simm16", ...
    std::string fieldType;    // "vgpr", "sreg", "label", "wait_alu", ...
    int sizeBits = 0;
    std::string positionKey;  // "D0", "S0", etc. — non-empty for partial overrides
};

// Format definition (provides defaults for instructions)
// Hardware formats (VOP1, VOP3P, MUBUF, SMRD, etc.) are defined first; opcode-family
// formats (e.g. MFMA on CDNA) can inherit from a hardware format via .parent.
struct FormatDef {
    std::string name;
    std::string parent;     // e.g. "VOP3P" for MFMA (inherits encoding, etc.)
    std::string microcode;  // microcode format (e.g. "MC_VOP3P", "MC_SMEM")
    std::string unit;
    std::vector<int> encoding;
    int maxOperands = 0;
    int cycle = 0;  // 0 = not specified (inherit from parent or arch default)
    int latency = 0;
    int coIssueWindow = -1;  // -1 = not specified (inherit from parent); VALU co-issue window
    std::vector<std::string> flags;

    // Default operand field descriptions for instructions using this format
    std::vector<OperandFieldEntry> fields;

    // Promoted (wider) encoding format name (e.g., "VOP3" for a VOP2 format)
    std::string promotedFormat;

    // Operand field descriptions for the promoted encoding
    std::vector<OperandFieldEntry> altFields;
};

// Cost override: when modifier matches (e.g. MatrixFmtModifiers(FP4, FP4)), use (cycle, latency)
struct CostOverrideEntry {
    std::string modifierType;       // e.g. "MatrixFmtModifiers"
    std::vector<std::string> args;  // e.g. {"FP4", "FP4"}
    int cycle = 1;
    int latency = 1;
};

// Instruction definition
struct InstructionDef {
    std::string instClass;  // e.g., "FloatAddInst"
    std::string mnemonic;   // e.g., "v_add_f32"
    int lineNumber = 0;     // 1-based source line for error reporting

    // Optional fields (inherit from format if not specified)
    std::string format;  // e.g., "VOP3"
    int cycle = 0;       // 0 = not specified (inherit from format, then arch default)
    int latency = 0;
    int coIssueWindow = -1;  // -1 = not specified (inherit from format); VALU co-issue window
    std::vector<CostOverrideEntry> costOverrides;     // modifier-keyed overrides
    std::vector<OperandSpec> operands;                // Operand specifications
    std::vector<OperandFieldEntry> operandFields;     // Operand field descriptions
    std::vector<OperandFieldEntry> altOperandFields;  // Promoted encoding field overrides
    std::vector<std::string> flags;                   // Instruction-specific flags
    std::vector<std::string> logical;  // Logical IR names mapping to this mnemonic: .logical =
                                       // {"A", "B"} or .logical = "A"
    std::string unit;                  // Override format unit

    // After format inheritance applied
    std::vector<std::string> finalFlags;  // Merged format + instance flags
    std::string finalUnit;
    std::string finalMicrocode;                          // Microcode format (e.g. "MC_VOP3P")
    int finalEncoding = 0;                               // Encoding size in bits
    std::vector<OperandFieldEntry> finalOperandFields;   // After format inheritance
    std::string finalPromotedFormat;                     // Promoted encoding format
    std::vector<OperandFieldEntry> finalPromotedFields;  // Promoted encoding fields
    int encodingBits = 32;  // From format .encoding (bits); sizeInBytes = encodingBits/8
};

//==========================================================================
// HELPERS
//==========================================================================

// Return indices into `v` sorted by `key(v[i])`. Lets a generator iterate the
// input vector in sorted order without mutating it, then emit a sorted table.
template <typename T, typename KeyFn>
static std::vector<size_t> sortedIndices(const std::vector<T>& v, KeyFn key) {
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return key(v[a]) < key(v[b]); });
    return idx;
}

//==========================================================================
// PARSER
//==========================================================================

class DefTParser {
   public:
    DefTParser(const std::string& arch) : archName_(arch) {
        arch_.name = arch;
    }

    // Parse format definitions file
    bool parseFormats(const std::string& formatFile) {
        std::ifstream ifs(formatFile);
        if (!ifs) {
            std::cerr << "Error: Cannot open format file: " << formatFile << "\n";
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());

        if (!parseFormatContent(content, formatFile)) return false;
        std::cout << std::filesystem::path(formatFile).filename().string() << ": parsed "
                  << formats_.size() << " formats\n";
        return true;
    }

    // Parse instruction definitions file
    bool parseInstructions(const std::string& instFile) {
        std::ifstream ifs(instFile);
        if (!ifs) {
            std::cerr << "Error: Cannot open instruction file: " << instFile << "\n";
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());

        lastInstFile_ = instFile;
        size_t prevSize = instructions_.size();
        if (!parseBatchContent(content, instFile)) return false;
        if (!parseInstructionContent(content, instFile)) return false;
        // Stable-sort newly added instructions by line number so that
        // DEF_BATCH and DEF_T entries appear in document order.
        std::stable_sort(instructions_.begin() + prevSize, instructions_.end(),
                         [](const InstructionDef& a, const InstructionDef& b) {
                             return a.lineNumber < b.lineNumber;
                         });
        std::cout << std::filesystem::path(instFile).filename().string() << ": parsed "
                  << instructions_.size() << " instructions\n";
        return true;
    }

    // Resolve a format by merging with its parent chain recursively.
    FormatDef getResolvedFormat(const std::string& name) const {
        auto it = formats_.find(name);
        if (it == formats_.end()) return FormatDef{};
        FormatDef fmt = it->second;
        if (fmt.parent.empty()) return fmt;
        // Recursively resolve the parent first so grandparent fields propagate.
        FormatDef p = getResolvedFormat(fmt.parent);
        // Child overrides where non-empty
        if (fmt.microcode.empty()) fmt.microcode = p.microcode;
        if (fmt.unit.empty()) fmt.unit = p.unit;
        if (fmt.maxOperands == 0) fmt.maxOperands = p.maxOperands;
        if (fmt.encoding.empty()) fmt.encoding = p.encoding;
        if (fmt.cycle == 0) fmt.cycle = p.cycle;
        if (fmt.latency == 0) fmt.latency = p.latency;
        if (fmt.coIssueWindow < 0) fmt.coIssueWindow = p.coIssueWindow;
        // Flags: parent first, then child (child adds MFMA etc.)
        fmt.flags = p.flags;
        fmt.flags.insert(fmt.flags.end(), it->second.flags.begin(), it->second.flags.end());
        // Fields: child overrides parent if non-empty
        if (fmt.fields.empty()) fmt.fields = p.fields;
        // Promoted format: child overrides parent if non-empty
        if (fmt.promotedFormat.empty()) fmt.promotedFormat = p.promotedFormat;
        if (fmt.altFields.empty()) fmt.altFields = p.altFields;
        return fmt;
    }

    // Apply format inheritance to all instructions
    bool applyFormatDefaults() {
        for (auto& inst : instructions_) {
            if (inst.format.empty()) {
                // No format: use instruction's own fields so final* are always set
                inst.finalFlags = inst.flags;
                inst.finalUnit = inst.unit;
                inst.finalOperandFields = inst.operandFields;
                if (inst.cycle == 0) inst.cycle = arch_.defaultCycle;
                if (inst.latency == 0) inst.latency = arch_.defaultLatency;
                if (inst.coIssueWindow < 0) inst.coIssueWindow = 0;
                continue;
            }

            FormatDef fmt = getResolvedFormat(inst.format);
            if (fmt.name.empty()) {
                if (!lastInstFile_.empty() && inst.lineNumber > 0)
                    std::cerr << lastInstFile_ << ":" << inst.lineNumber << ": ";
                std::cerr << "Error: Unknown format '" << inst.format << "' for instruction '"
                          << inst.mnemonic << "'\n";
                return false;
            }

            // Merge flags: format flags + instance flags
            inst.finalFlags = fmt.flags;
            inst.finalFlags.insert(inst.finalFlags.end(), inst.flags.begin(), inst.flags.end());

            // Apply unit (instance overrides format)
            inst.finalUnit = inst.unit.empty() ? fmt.unit : inst.unit;

            // Apply microcode format (from format definition)
            inst.finalMicrocode = fmt.microcode;

            // Apply encoding (first element of encoding vector)
            if (!fmt.encoding.empty()) inst.finalEncoding = fmt.encoding[0];

            // Apply cost: explicit instruction cost > format cost > arch default
            if (inst.cycle == 0 && fmt.cycle > 0) inst.cycle = fmt.cycle;
            if (inst.latency == 0 && fmt.latency > 0) inst.latency = fmt.latency;
            if (inst.cycle == 0) inst.cycle = arch_.defaultCycle;
            if (inst.latency == 0) inst.latency = arch_.defaultLatency;
            if (inst.coIssueWindow < 0 && fmt.coIssueWindow >= 0)
                inst.coIssueWindow = fmt.coIssueWindow;
            if (inst.coIssueWindow < 0) inst.coIssueWindow = 0;
            if (inst.coIssueWindow > 0xFFFF) {
                std::cerr << "error: " << inst.mnemonic << ": .coissue value 0x" << std::hex
                          << inst.coIssueWindow << " exceeds uint16_t range\n";
                return false;
            }

            // Apply operand fields: instruction overrides format
            if (!inst.operandFields.empty()) {
                bool isPartial = !inst.operandFields[0].positionKey.empty();
                if (isPartial && !fmt.fields.empty()) {
                    // Merge: start from format defaults, apply named overrides
                    inst.finalOperandFields = fmt.fields;
                    int destIdx = 0, srcIdx = 0;
                    std::unordered_map<std::string, size_t> keyToIdx;
                    for (size_t fi = 0; fi < inst.finalOperandFields.size(); ++fi) {
                        std::string key = inst.finalOperandFields[fi].isDest
                                              ? "D" + std::to_string(destIdx++)
                                              : "S" + std::to_string(srcIdx++);
                        keyToIdx[key] = fi;
                    }
                    for (const auto& ov : inst.operandFields) {
                        auto it = keyToIdx.find(ov.positionKey);
                        if (it != keyToIdx.end()) {
                            if (ov.sizeBits != 0)
                                inst.finalOperandFields[it->second].sizeBits = ov.sizeBits;
                            if (!ov.fieldType.empty())
                                inst.finalOperandFields[it->second].fieldType = ov.fieldType;
                        }
                    }
                } else {
                    inst.finalOperandFields = inst.operandFields;
                }
            } else if (!fmt.fields.empty())
                inst.finalOperandFields = fmt.fields;

            // Apply promoted format and alt fields
            inst.finalPromotedFormat = fmt.promotedFormat;

            if (!inst.altOperandFields.empty()) {
                bool isAltPartial = !inst.altOperandFields[0].positionKey.empty();
                if (isAltPartial && !fmt.altFields.empty()) {
                    inst.finalPromotedFields = fmt.altFields;
                    int destIdx = 0, srcIdx = 0;
                    std::unordered_map<std::string, size_t> keyToIdx;
                    for (size_t fi = 0; fi < inst.finalPromotedFields.size(); ++fi) {
                        std::string key = inst.finalPromotedFields[fi].isDest
                                              ? "D" + std::to_string(destIdx++)
                                              : "S" + std::to_string(srcIdx++);
                        keyToIdx[key] = fi;
                    }
                    for (const auto& ov : inst.altOperandFields) {
                        auto it2 = keyToIdx.find(ov.positionKey);
                        if (it2 != keyToIdx.end()) {
                            if (ov.sizeBits != 0)
                                inst.finalPromotedFields[it2->second].sizeBits = ov.sizeBits;
                            if (!ov.fieldType.empty())
                                inst.finalPromotedFields[it2->second].fieldType = ov.fieldType;
                        }
                    }
                } else {
                    inst.finalPromotedFields = inst.altOperandFields;
                }
            } else if (!fmt.altFields.empty()) {
                // Start from format alt_fields, apply same size overrides
                // from operandFields (sizes are instruction-level, not encoding-specific)
                inst.finalPromotedFields = fmt.altFields;
                if (!inst.operandFields.empty() && !inst.operandFields[0].positionKey.empty()) {
                    int destIdx = 0, srcIdx = 0;
                    std::unordered_map<std::string, size_t> keyToIdx;
                    for (size_t fi = 0; fi < inst.finalPromotedFields.size(); ++fi) {
                        std::string key = inst.finalPromotedFields[fi].isDest
                                              ? "D" + std::to_string(destIdx++)
                                              : "S" + std::to_string(srcIdx++);
                        keyToIdx[key] = fi;
                    }
                    for (const auto& ov : inst.operandFields) {
                        auto it2 = keyToIdx.find(ov.positionKey);
                        if (it2 != keyToIdx.end()) {
                            if (ov.sizeBits != 0)
                                inst.finalPromotedFields[it2->second].sizeBits = ov.sizeBits;
                        }
                    }
                }
            }
        }
        return true;
    }

    // Get parsed data
    const ArchDef& getArch() const {
        return arch_;
    }
    const std::map<std::string, FormatDef>& getFormats() const {
        return formats_;
    }
    const std::vector<InstructionDef>& getInstructions() const {
        return instructions_;
    }

   private:
    std::string archName_;
    ArchDef arch_;
    std::map<std::string, FormatDef> formats_;
    std::vector<InstructionDef> instructions_;
    std::string lastInstFile_;  // for error reporting in applyFormatDefaults

    static int getLineNumber(const std::string& content, size_t pos) {
        int line = 1;
        for (size_t i = 0; i < pos && i < content.size(); ++i)
            if (content[i] == '\n') ++line;
        return line;
    }

    bool parseFormatContent(const std::string& content,
                            const std::string& formatFile = std::string()) {
        // Parse DEF_ARCH(...) first (optional; overrides arch name and metadata)
        size_t defArchPos = content.find("DEF_ARCH(");
        if (defArchPos != std::string::npos) {
            size_t nameStart = defArchPos + 9;
            size_t nameEnd = content.find(',', nameStart);
            if (nameEnd != std::string::npos) {
                arch_.name = content.substr(nameStart, nameEnd - nameStart);
                arch_.name.erase(0, arch_.name.find_first_not_of(" \t\n\r"));
                arch_.name.erase(arch_.name.find_last_not_of(" \t\n\r") + 1);

                size_t blockEnd = findMatchingParen(content, nameEnd);
                if (blockEnd != std::string::npos) {
                    std::string block = content.substr(nameEnd, blockEnd - nameEnd);
                    parseFieldInt(block, ".major", arch_.major);
                    parseFieldInt(block, ".minor", arch_.minor);
                    parseFieldInt(block, ".stepping", arch_.stepping);
                    parseFieldInt(block, ".wavefront", arch_.wavefront);
                    parseFieldInt(block, ".maxVGPR", arch_.maxVGPR);
                    parseFieldInt(block, ".maxSGPR", arch_.maxSGPR);
                    parseFieldInt(block, ".maxAGPR", arch_.maxAGPR);
                    parseFieldInt(block, ".totalVgprPerSimd", arch_.totalVgprPerSimd);
                    parseFieldInt(block, ".vgprAllocGranule", arch_.vgprAllocGranule);
                    parseFieldInt(block, ".defaultCycle", arch_.defaultCycle);
                    parseFieldInt(block, ".defaultLatency", arch_.defaultLatency);
                    parseFieldInt(block, ".d16Writes32BitVgpr", arch_.d16Writes32BitVgpr);
                }
            }
        } else {
            arch_.major = 9;
            arch_.minor = 4;
            arch_.stepping = 0;
            arch_.wavefront = 64;
            arch_.maxVGPR = 256;
            arch_.maxSGPR = 102;
            arch_.maxAGPR = 0;
            arch_.defaultCycle = 4;
            arch_.defaultLatency = 4;
        }

        // Parse DEF_HWREG(NAME, ID) — one HWREG entry per line, terse form.
        // Lookup table for s_setreg/s_getreg is emitted from these per arch.
        {
            size_t hwregPos = 0;
            while ((hwregPos = content.find("DEF_HWREG(", hwregPos)) != std::string::npos) {
                size_t argsStart = hwregPos + 10;  // After "DEF_HWREG("
                size_t argsEnd = content.find(')', argsStart);
                if (argsEnd == std::string::npos) {
                    int line = getLineNumber(content, argsStart);
                    if (!formatFile.empty()) std::cerr << formatFile << ":" << line << ": ";
                    std::cerr << "Error: DEF_HWREG(...): missing closing ')'.\n";
                    return false;
                }
                std::string args = content.substr(argsStart, argsEnd - argsStart);
                size_t comma = args.find(',');
                if (comma == std::string::npos) {
                    int line = getLineNumber(content, argsStart);
                    if (!formatFile.empty()) std::cerr << formatFile << ":" << line << ": ";
                    std::cerr << "Error: DEF_HWREG(NAME, ID): missing ',' between name and id.\n";
                    return false;
                }
                HwRegEntry entry;
                entry.name = args.substr(0, comma);
                entry.name.erase(0, entry.name.find_first_not_of(" \t\n\r"));
                entry.name.erase(entry.name.find_last_not_of(" \t\n\r") + 1);
                std::string idStr = args.substr(comma + 1);
                idStr.erase(0, idStr.find_first_not_of(" \t\n\r"));
                idStr.erase(idStr.find_last_not_of(" \t\n\r") + 1);
                char* end = nullptr;
                long parsed = std::strtol(idStr.c_str(), &end, 0);
                if (idStr.empty() || end != idStr.c_str() + idStr.size()) {
                    int line = getLineNumber(content, argsStart);
                    if (!formatFile.empty()) std::cerr << formatFile << ":" << line << ": ";
                    std::cerr << "Error: DEF_HWREG(" << entry.name << ", " << idStr
                              << "): id is not a valid integer.\n";
                    return false;
                }
                entry.id = static_cast<int>(parsed);
                arch_.hwRegs.push_back(std::move(entry));
                hwregPos = argsEnd + 1;
            }
        }

        // Parse DEF_FORMAT(...) blocks
        size_t pos = 0;
        while ((pos = content.find("DEF_FORMAT(", pos)) != std::string::npos) {
            size_t nameStart = pos + 11;  // After "DEF_FORMAT("
            size_t firstComma = content.find(',', nameStart);
            size_t firstDot = content.find('.', nameStart);
            // Comma must appear before any field (.xxx = ...)
            if (firstDot != std::string::npos &&
                (firstComma == std::string::npos || firstDot < firstComma)) {
                size_t end = firstDot;
                while (end > nameStart && (content[end - 1] == ' ' || content[end - 1] == '\t'))
                    --end;
                std::string formatName = content.substr(nameStart, end - nameStart);
                formatName.erase(0, formatName.find_first_not_of(" \t\n\r"));
                formatName.erase(formatName.find_last_not_of(" \t\n\r") + 1);
                int line = getLineNumber(content, nameStart);
                if (!formatFile.empty()) std::cerr << formatFile << ":" << line << ": ";
                std::cerr << "Error: DEF_FORMAT(" << formatName
                          << "): missing ',' after format name.\n";
                return false;
            }
            if (firstComma == std::string::npos) {
                int line = getLineNumber(content, nameStart);
                if (!formatFile.empty()) std::cerr << formatFile << ":" << line << ": ";
                std::cerr << "Error: DEF_FORMAT(...): missing ',' after format name.\n";
                return false;
            }
            size_t nameEnd = firstComma;

            std::string formatName = content.substr(nameStart, nameEnd - nameStart);
            formatName.erase(0, formatName.find_first_not_of(" \t\n\r"));
            formatName.erase(formatName.find_last_not_of(" \t\n\r") + 1);

            FormatDef fmt;
            fmt.name = formatName;

            // Find the closing paren for this DEF_FORMAT
            size_t blockEnd = findMatchingParen(content, nameEnd);
            if (blockEnd == std::string::npos) {
                int line = getLineNumber(content, nameStart);
                if (!formatFile.empty()) std::cerr << formatFile << ":" << line << ": ";
                std::cerr << "Error: No matching paren for DEF_FORMAT " << formatName << "\n";
                return false;
            }

            std::string block = content.substr(nameEnd, blockEnd - nameEnd);

            // Parse fields (simple substring matching for proof-of-concept)
            parseField(block, ".parent", fmt.parent);
            parseField(block, ".microcode", fmt.microcode);
            parseField(block, ".unit", fmt.unit);
            parseFieldInt(block, ".maxOperands", fmt.maxOperands);
            parseFieldCost(block, ".cost", fmt.cycle, fmt.latency);
            parseFieldIntAuto(block, ".coissue", fmt.coIssueWindow);
            parseFieldEncoding(block, ".encoding", fmt.encoding);
            parseFieldFlags(block, ".flags", fmt.flags);
            parseFieldOperandFields(block, ".fields", fmt.fields);
            parseField(block, ".promotedFormat", fmt.promotedFormat);
            parseFieldOperandFields(block, ".alt_fields", fmt.altFields);

            formats_[formatName] = fmt;
            pos = blockEnd;
        }

        return true;
    }

    // Parse DEF_BATCH(...) blocks.
    //
    // Syntax — shared header fields + entries using DEF_T argument syntax:
    //   DEF_BATCH(.format = FMT, [.flags = {F1, F2},]
    //       [.microcode = U, .unit = X, .encoding = {N},]
    //       [.maxOperands = N,]
    //       ClassName, "mnemonic",
    //       ClassName, "mnemonic", .logical = "LogicalName",
    //       ClassName, "mnemonic", .logical = {"L1", "L2"}, .flags = {Extra},
    //   )
    //
    // Each entry uses the same field syntax as DEF_T arguments (without the
    // DEF_T() wrapper).  Shared header fields are inherited by all entries;
    // per-entry fields override or extend them (flags are additive).
    //
    // The header supports all FormatDef fields. If format-level fields beyond
    // .format/.flags are given, an anonymous format is created automatically.
    bool parseBatchContent(const std::string& content,
                           const std::string& instFile = std::string()) {
        size_t pos = 0;
        int batchIdx = 0;
        while ((pos = content.find("DEF_BATCH(", pos)) != std::string::npos) {
            int defLine = getLineNumber(content, pos);
            size_t argsStart = pos + 10;  // After "DEF_BATCH("

            size_t blockEnd = findMatchingParen(content, argsStart);
            if (blockEnd == std::string::npos) {
                if (!instFile.empty()) std::cerr << instFile << ":" << defLine << ": ";
                std::cerr << "Error: No matching paren for DEF_BATCH\n";
                return false;
            }

            std::string body = content.substr(argsStart, blockEnd - argsStart);

            // --- Find header portion (before the first entry) ---
            // Entries start with Identifier, "mnemonic". Header fields start
            // with '.'. We extract the header text so shared field parsing
            // doesn't accidentally pick up per-entry fields.
            std::vector<size_t> tmpStarts;
            findBatchEntries(body, tmpStarts);
            std::string header = tmpStarts.empty() ? body : body.substr(0, tmpStarts[0]);

            // --- Parse shared header fields ---
            std::string hdrFormat;
            std::vector<std::string> hdrFlags;
            std::string hdrMicrocode;
            std::string hdrUnit;
            std::string hdrEncodingStr;
            int hdrMaxOperands = 0;

            parseField(header, ".format", hdrFormat);
            parseFieldFlags(header, ".flags", hdrFlags);
            parseField(header, ".microcode", hdrMicrocode);
            parseField(header, ".unit", hdrUnit);
            parseField(header, ".encoding", hdrEncodingStr);
            parseFieldInt(header, ".maxOperands", hdrMaxOperands);

            bool hasFormatFields = !hdrMicrocode.empty() || !hdrUnit.empty() ||
                                   !hdrEncodingStr.empty() || hdrMaxOperands != 0;

            std::string effectiveFormat = hdrFormat;

            if (hasFormatFields) {
                FormatDef anonFmt;
                anonFmt.name =
                    "__batch_" + std::to_string(defLine) + "_" + std::to_string(batchIdx++);
                anonFmt.parent = hdrFormat;
                if (!hdrMicrocode.empty()) anonFmt.microcode = hdrMicrocode;
                if (!hdrUnit.empty()) anonFmt.unit = hdrUnit;
                if (!hdrEncodingStr.empty()) {
                    std::string enc = hdrEncodingStr;
                    enc.erase(std::remove(enc.begin(), enc.end(), '{'), enc.end());
                    enc.erase(std::remove(enc.begin(), enc.end(), '}'), enc.end());
                    enc.erase(0, enc.find_first_not_of(" \t"));
                    enc.erase(enc.find_last_not_of(" \t") + 1);
                    if (!enc.empty()) anonFmt.encoding.push_back(std::stoi(enc));
                }
                anonFmt.maxOperands = hdrMaxOperands;
                anonFmt.flags = hdrFlags;

                formats_[anonFmt.name] = anonFmt;
                effectiveFormat = anonFmt.name;
                hdrFlags.clear();
            }

            if (effectiveFormat.empty()) {
                if (!instFile.empty()) std::cerr << instFile << ":" << defLine << ": ";
                std::cerr << "Error: DEF_BATCH needs .format or format-level fields\n";
                return false;
            }

            // Reuse entry positions found during header extraction.
            std::vector<size_t>& entryStarts = tmpStarts;

            if (entryStarts.empty()) {
                if (!instFile.empty()) std::cerr << instFile << ":" << defLine << ": ";
                std::cerr << "Error: DEF_BATCH contains no entries\n";
                return false;
            }

            // --- Parse each entry ---
            for (size_t ei = 0; ei < entryStarts.size(); ++ei) {
                size_t entryPos = entryStarts[ei];
                size_t entryEnd = (ei + 1 < entryStarts.size()) ? entryStarts[ei + 1] : body.size();

                // Extract ClassName
                size_t idEnd = body.find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", entryPos);
                std::string instClass = body.substr(entryPos, idEnd - entryPos);

                // Expect comma after ClassName
                size_t commaPos = body.find_first_not_of(" \t", idEnd);
                if (commaPos == std::string::npos || body[commaPos] != ',') {
                    int entryLine = defLine + getLineNumber(body.substr(0, entryPos), 0) - 1;
                    if (!instFile.empty()) std::cerr << instFile << ":" << entryLine << ": ";
                    std::cerr << "Error: Expected ',' after class name '" << instClass
                              << "' in DEF_BATCH\n";
                    return false;
                }

                // Expect "mnemonic" after comma
                size_t qStart = body.find('"', commaPos + 1);
                if (qStart == std::string::npos || qStart >= entryEnd) {
                    int entryLine = defLine + getLineNumber(body.substr(0, entryPos), 0) - 1;
                    if (!instFile.empty()) std::cerr << instFile << ":" << entryLine << ": ";
                    std::cerr << "Error: Expected '\"mnemonic\"' after '" << instClass
                              << ",' in DEF_BATCH\n";
                    return false;
                }
                size_t qEnd = body.find('"', qStart + 1);
                if (qEnd == std::string::npos) {
                    int entryLine = defLine + getLineNumber(body.substr(0, entryPos), 0) - 1;
                    if (!instFile.empty()) std::cerr << instFile << ":" << entryLine << ": ";
                    std::cerr << "Error: Unterminated mnemonic string for '" << instClass
                              << "' in DEF_BATCH\n";
                    return false;
                }
                std::string mnem = body.substr(qStart + 1, qEnd - qStart - 1);

                // Everything between the mnemonic close-quote and the next
                // entry (or end of batch) is this entry's optional fields.
                std::string entryFields = body.substr(qEnd + 1, entryEnd - qEnd - 1);

                InstructionDef inst;
                inst.instClass = instClass;
                inst.mnemonic = mnem;
                inst.lineNumber = defLine + getLineNumber(body.substr(0, entryPos), 0) - 1;
                inst.format = effectiveFormat;

                // Start with shared flags, then append per-entry flags
                inst.flags = hdrFlags;
                std::vector<std::string> entryFlags;
                parseFieldFlags(entryFields, ".flags", entryFlags);
                inst.flags.insert(inst.flags.end(), entryFlags.begin(), entryFlags.end());

                // Parse per-entry optional fields (same helpers as DEF_T)
                parseFieldCost(entryFields, ".cost", inst.cycle, inst.latency);
                parseFieldIntAuto(entryFields, ".coissue", inst.coIssueWindow);
                parseFieldCostOverrides(entryFields, inst.costOverrides);
                parseFieldOperandFields(entryFields, ".operand_fields", inst.operandFields);
                parseFieldOperandFields(entryFields, ".alt_operand_fields", inst.altOperandFields);
                parseFieldLogical(entryFields, inst.logical);

                // Validate: per-entry must NOT redefine .format (use header)
                std::string entryFmt;
                parseField(entryFields, ".format", entryFmt);
                if (!entryFmt.empty()) {
                    if (!instFile.empty()) std::cerr << instFile << ":" << inst.lineNumber << ": ";
                    std::cerr << "Error: Entry '" << instClass
                              << "' in DEF_BATCH must not specify .format "
                              << "(use the batch header instead)\n";
                    return false;
                }

                instructions_.push_back(inst);
            }

            pos = blockEnd + 1;
        }
        return true;
    }

    // Find the start positions of all entries inside a DEF_BATCH body.
    // An entry is: Identifier , "mnemonic" [, .field = ...]
    // Header fields start with '.' and are skipped.
    void findBatchEntries(const std::string& body, std::vector<size_t>& starts) const {
        for (size_t i = 0; i < body.size(); ++i) {
            char c = body[i];

            // Skip whitespace and commas
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',') continue;

            // Skip // comments
            if (c == '/' && i + 1 < body.size() && body[i + 1] == '/') {
                size_t eol = body.find('\n', i);
                i = (eol == std::string::npos) ? body.size() : eol;
                continue;
            }

            // Skip .field = value (header or per-entry fields)
            if (c == '.') {
                i = skipDotField(body, i);
                continue;
            }

            // Potential entry: Identifier , "mnemonic"
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                size_t idEnd = i + 1;
                while (
                    idEnd < body.size() &&
                    (std::isalnum(static_cast<unsigned char>(body[idEnd])) || body[idEnd] == '_'))
                    idEnd++;

                size_t afterId = body.find_first_not_of(" \t", idEnd);
                if (afterId != std::string::npos && body[afterId] == ',') {
                    size_t afterComma = body.find_first_not_of(" \t", afterId + 1);
                    if (afterComma != std::string::npos && body[afterComma] == '"') {
                        starts.push_back(i);
                        // Advance past the mnemonic to find per-entry fields
                        size_t qEnd = body.find('"', afterComma + 1);
                        i = (qEnd != std::string::npos) ? qEnd : idEnd;
                        continue;
                    }
                }
                i = idEnd - 1;
            }
        }
    }

    // Skip a .field = value expression, returning position of its last char.
    size_t skipDotField(const std::string& body, size_t dotPos) const {
        size_t i = dotPos + 1;
        // Skip field name
        while (i < body.size() &&
               (std::isalnum(static_cast<unsigned char>(body[i])) || body[i] == '_'))
            i++;
        // Skip whitespace and '='
        i = body.find_first_not_of(" \t", i);
        if (i == std::string::npos) return body.size() - 1;
        if (body[i] == '=') {
            i = body.find_first_not_of(" \t", i + 1);
            if (i == std::string::npos) return body.size() - 1;
        }
        // If value starts with '{', skip to matching '}'
        if (body[i] == '{') {
            size_t close = findMatchingBrace(body, i);
            return (close != std::string::npos) ? close : body.size() - 1;
        }
        // Otherwise skip to next comma or newline
        size_t end = body.find_first_of(",\n", i);
        return (end != std::string::npos) ? end - 1 : body.size() - 1;
    }

    bool parseInstructionContent(const std::string& content,
                                 const std::string& instFile = std::string()) {
        // Minimal parser for proof-of-concept
        // Parse DEF_T(...) blocks
        size_t pos = 0;
        while ((pos = content.find("DEF_T(", pos)) != std::string::npos) {
            int defLine = getLineNumber(content, pos);
            size_t argsStart = pos + 6;  // After "DEF_T("
            size_t firstComma = content.find(',', argsStart);
            if (firstComma == std::string::npos) break;

            // Parse: DEF_T(InstClass, "mnemonic", ...)
            std::string instClass = content.substr(argsStart, firstComma - argsStart);
            instClass.erase(0, instClass.find_first_not_of(" \t\n\r"));
            instClass.erase(instClass.find_last_not_of(" \t\n\r") + 1);

            size_t mnemonicStart = content.find('"', firstComma);
            size_t mnemonicEnd = content.find('"', mnemonicStart + 1);
            if (mnemonicStart == std::string::npos || mnemonicEnd == std::string::npos) break;

            std::string mnemonic =
                content.substr(mnemonicStart + 1, mnemonicEnd - mnemonicStart - 1);

            InstructionDef inst;
            inst.instClass = instClass;
            inst.mnemonic = mnemonic;
            inst.lineNumber = defLine;

            // Find the closing paren
            size_t blockEnd = findMatchingParen(content, mnemonicEnd);
            if (blockEnd == std::string::npos) {
                if (!instFile.empty()) std::cerr << instFile << ":" << defLine << ": ";
                std::cerr << "Error: No matching paren for DEF_T " << mnemonic << "\n";
                return false;
            }

            std::string block = content.substr(mnemonicEnd, blockEnd - mnemonicEnd);

            // Parse optional fields
            parseField(block, ".format", inst.format);
            parseFieldCost(block, ".cost", inst.cycle, inst.latency);
            parseFieldIntAuto(block, ".coissue", inst.coIssueWindow);
            parseFieldCostOverrides(block, inst.costOverrides);
            parseFieldFlags(block, ".flags", inst.flags);
            parseFieldOperandFields(block, ".operand_fields", inst.operandFields);
            parseFieldOperandFields(block, ".alt_operand_fields", inst.altOperandFields);
            parseFieldLogical(block, inst.logical);

            instructions_.push_back(inst);
            pos = blockEnd;
        }

        return true;
    }

    // Helper: Find matching closing paren
    size_t findMatchingParen(const std::string& content, size_t start) {
        int depth = 1;
        for (size_t i = start; i < content.size(); ++i) {
            if (content[i] == '(')
                depth++;
            else if (content[i] == ')') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return std::string::npos;
    }

    // Helper: Parse simple field
    void parseField(const std::string& block, const std::string& fieldName, std::string& out) {
        size_t pos = block.find(fieldName);
        if (pos == std::string::npos) return;

        size_t eqPos = block.find('=', pos);
        if (eqPos == std::string::npos) return;

        size_t valStart = eqPos + 1;
        size_t valEnd = block.find_first_of(",\n)", valStart);
        if (valEnd == std::string::npos) valEnd = block.size();

        out = block.substr(valStart, valEnd - valStart);
        out.erase(0, out.find_first_not_of(" \t\n\r"));
        out.erase(out.find_last_not_of(" \t\n\r,") + 1);
        // Strip surrounding quotes for string values (e.g. .logical = "VCmpEQF32")
        if (out.size() >= 2 && out.front() == '"' && out.back() == '"')
            out = out.substr(1, out.size() - 2);
    }

    // Helper: Parse integer field
    void parseFieldInt(const std::string& block, const std::string& fieldName, int& out) {
        std::string val;
        parseField(block, fieldName, val);
        if (!val.empty()) out = std::stoi(val);
    }

    // Helper: Parse integer field with auto base detection (handles 0x hex prefix)
    void parseFieldIntAuto(const std::string& block, const std::string& fieldName, int& out) {
        std::string val;
        parseField(block, fieldName, val);
        if (!val.empty()) out = std::stoi(val, nullptr, 0);
    }

    // Parse .encoding = {32} or .encoding = {64} (instruction size in bits)
    void parseFieldEncoding(const std::string& block, const std::string& fieldName,
                            std::vector<int>& out) {
        size_t pos = block.find(fieldName);
        if (pos == std::string::npos) return;
        size_t lbrace = block.find('{', pos);
        size_t rbrace = block.find('}', lbrace);
        if (lbrace == std::string::npos || rbrace == std::string::npos) return;
        std::string inner = block.substr(lbrace + 1, rbrace - lbrace - 1);
        out.clear();
        size_t start = 0;
        while (start < inner.size()) {
            size_t end = inner.find(',', start);
            if (end == std::string::npos) end = inner.size();
            std::string num = inner.substr(start, end - start);
            num.erase(0, num.find_first_not_of(" \t\n\r"));
            num.erase(num.find_last_not_of(" \t\n\r") + 1);
            if (!num.empty()) out.push_back(std::stoi(num));
            start = end + 1;
        }
    }

    // Helper: Parse cost field (.cost = {cycle, latency})
    void parseFieldCost(const std::string& block, const std::string& fieldName, int& cycle,
                        int& latency) {
        size_t pos = block.find(fieldName);
        if (pos == std::string::npos) return;

        size_t lbrace = block.find('{', pos);
        size_t rbrace = block.find('}', lbrace);
        if (lbrace == std::string::npos || rbrace == std::string::npos) return;

        std::string costStr = block.substr(lbrace + 1, rbrace - lbrace - 1);
        size_t comma = costStr.find(',');
        if (comma != std::string::npos) {
            cycle = std::stoi(costStr.substr(0, comma));
            latency = std::stoi(costStr.substr(comma + 1));
        }
    }

    // Helper: Find matching '}' for '{' at start (skips nested braces)
    size_t findMatchingBrace(const std::string& s, size_t start) const {
        int depth = 1;
        for (size_t i = start + 1; i < s.size(); ++i) {
            if (s[i] == '{')
                depth++;
            else if (s[i] == '}') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return std::string::npos;
    }

    // Helper: Parse .costOverrides = { { MatrixFmtModifiers(FP4, FP4), 6, 24 }, ... }
    void parseFieldCostOverrides(const std::string& block, std::vector<CostOverrideEntry>& out) {
        size_t pos = block.find(".costOverrides");
        if (pos == std::string::npos) return;
        size_t eqPos = block.find('=', pos);
        if (eqPos == std::string::npos) return;
        size_t lbrace = block.find('{', eqPos);
        if (lbrace == std::string::npos) return;
        size_t rbrace = findMatchingBrace(block, lbrace);
        if (rbrace == std::string::npos) return;
        std::string content = block.substr(lbrace + 1, rbrace - lbrace - 1);
        size_t entryStart = 0;
        while (entryStart < content.size()) {
            size_t entryL = content.find('{', entryStart);
            if (entryL == std::string::npos) break;
            size_t entryR = findMatchingBrace(content, entryL);
            if (entryR == std::string::npos) break;
            std::string entry = content.substr(entryL + 1, entryR - entryL - 1);
            // entry is "MatrixFmtModifiers(FP4, FP4), 6, 24"
            size_t sep = entry.find("),");
            if (sep == std::string::npos) {
                entryStart = entryR + 1;
                continue;
            }
            std::string modPart = entry.substr(0, sep + 1);  // include ')'
            std::string costPart = entry.substr(sep + 2);
            size_t lparen = modPart.find('(');
            if (lparen == std::string::npos) {
                entryStart = entryR + 1;
                continue;
            }
            size_t rparen = findMatchingParen(modPart, lparen + 1);
            if (rparen == std::string::npos) {
                entryStart = entryR + 1;
                continue;
            }
            CostOverrideEntry e;
            e.modifierType = modPart.substr(0, lparen);
            e.modifierType.erase(0, e.modifierType.find_first_not_of(" \t\n\r"));
            e.modifierType.erase(e.modifierType.find_last_not_of(" \t\n\r") + 1);
            std::string argsStr = modPart.substr(lparen + 1, rparen - lparen - 1);
            size_t as = 0;
            while (as < argsStr.size()) {
                size_t ac = argsStr.find(',', as);
                if (ac == std::string::npos) ac = argsStr.size();
                std::string arg = argsStr.substr(as, ac - as);
                arg.erase(0, arg.find_first_not_of(" \t\n\r"));
                arg.erase(arg.find_last_not_of(" \t\n\r") + 1);
                if (!arg.empty()) e.args.push_back(arg);
                as = ac + 1;
            }
            costPart.erase(0, costPart.find_first_not_of(" \t\n\r"));
            size_t cc = costPart.find(',');
            if (cc != std::string::npos) {
                e.cycle = std::stoi(costPart.substr(0, cc));
                e.latency = std::stoi(costPart.substr(cc + 1));
            }
            out.push_back(e);
            entryStart = entryR + 1;
        }
    }

    // Helper: Parse flags field (.flags = {flag1, flag2})
    void parseFieldFlags(const std::string& block, const std::string& fieldName,
                         std::vector<std::string>& flags) {
        size_t pos = block.find(fieldName);
        if (pos == std::string::npos) return;

        size_t lbrace = block.find('{', pos);
        size_t rbrace = block.find('}', lbrace);
        if (lbrace == std::string::npos || rbrace == std::string::npos) return;

        std::string flagsStr = block.substr(lbrace + 1, rbrace - lbrace - 1);

        // Split by comma
        size_t start = 0;
        while (start < flagsStr.size()) {
            size_t end = flagsStr.find(',', start);
            if (end == std::string::npos) end = flagsStr.size();

            std::string flag = flagsStr.substr(start, end - start);
            flag.erase(0, flag.find_first_not_of(" \t\n\r"));
            flag.erase(flag.find_last_not_of(" \t\n\r") + 1);

            if (!flag.empty()) flags.push_back(flag);

            start = end + 1;
        }
    }

    // Helper: Parse .logical = "X" or .logical = {"X", "Y"}
    void parseFieldLogical(const std::string& block, std::vector<std::string>& out) {
        size_t pos = block.find(".logical");
        if (pos == std::string::npos) return;

        size_t eqPos = block.find('=', pos);
        if (eqPos == std::string::npos) return;

        size_t valStart = eqPos + 1;
        valStart = block.find_first_not_of(" \t\n\r", valStart);
        if (valStart == std::string::npos) return;

        if (block[valStart] == '{') {
            // .logical = {"X", "Y"}
            size_t rbrace = block.find('}', valStart);
            if (rbrace == std::string::npos) return;
            std::string inner = block.substr(valStart + 1, rbrace - valStart - 1);
            size_t start = 0;
            while (start < inner.size()) {
                size_t end = inner.find(',', start);
                if (end == std::string::npos) end = inner.size();
                std::string s = inner.substr(start, end - start);
                s.erase(0, s.find_first_not_of(" \t\n\r\""));
                s.erase(s.find_last_not_of(" \t\n\r\"") + 1);
                if (!s.empty()) out.push_back(s);
                start = end + 1;
            }
        } else if (block[valStart] == '"') {
            // .logical = "X"
            size_t endQuote = block.find('"', valStart + 1);
            if (endQuote != std::string::npos)
                out.push_back(block.substr(valStart + 1, endQuote - valStart - 1));
        }
    }

    // Parse .fields or .operand_fields:
    //
    // Full entry (format-level .fields or instruction-level .operand_fields):
    //   {Pos, EncodeField, Name, Type, Size [, Options...]}
    //   e.g.  {D0, vdata, vgpr_d, vgpr, 32, RW}
    //   e.g.  {S0, simm16, label, label, 16}
    //
    // Partial override (.operand_fields only — merges with format .fields):
    //   {S0, .type=wait_alu}          — override type
    //   {D0, .size=64}                — override size
    //   {D0, .size=64, .type=vgpr}    — override both
    size_t findMatchingBrace(const std::string& s, size_t openPos) {
        int depth = 1;
        for (size_t i = openPos + 1; i < s.size(); ++i) {
            if (s[i] == '{')
                depth++;
            else if (s[i] == '}') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return std::string::npos;
    }

    void parseFieldOperandFields(const std::string& block, const std::string& fieldKey,
                                 std::vector<OperandFieldEntry>& out) {
        // Search for fieldKey, ensuring it's not a substring of a longer key
        // (e.g., ".operand_fields" must not match ".alt_operand_fields")
        size_t pos = 0;
        while (pos < block.size()) {
            pos = block.find(fieldKey, pos);
            if (pos == std::string::npos) return;
            if (pos > 0 && (std::isalnum(block[pos - 1]) || block[pos - 1] == '_')) {
                pos += fieldKey.size();
                continue;
            }
            break;
        }

        size_t eqPos = block.find('=', pos);
        if (eqPos == std::string::npos) return;
        size_t outerStart = block.find('{', eqPos);
        if (outerStart == std::string::npos) return;
        size_t outerEnd = findMatchingBrace(block, outerStart);
        if (outerEnd == std::string::npos) return;

        size_t i = outerStart + 1;
        while (i < outerEnd) {
            size_t innerStart = block.find('{', i);
            if (innerStart == std::string::npos || innerStart >= outerEnd) break;
            size_t innerEnd = block.find('}', innerStart);
            if (innerEnd == std::string::npos || innerEnd >= outerEnd) break;

            std::string entry = block.substr(innerStart + 1, innerEnd - innerStart - 1);

            // Tokenize by comma
            std::vector<std::string> tokens;
            size_t ts = 0;
            while (ts < entry.size()) {
                size_t te = entry.find(',', ts);
                if (te == std::string::npos) te = entry.size();
                std::string tok = entry.substr(ts, te - ts);
                tok.erase(0, tok.find_first_not_of(" \t\n\r"));
                tok.erase(tok.find_last_not_of(" \t\n\r") + 1);
                if (!tok.empty()) tokens.push_back(tok);
                ts = te + 1;
            }

            if (tokens.size() >= 4 && tokens[1].find('.') == std::string::npos) {
                // Full entry: {D0, vdata, vgpr, 32 [, RW|M64]}
                // M64 can appear as the size field (implies 64) or as an option.
                OperandFieldEntry e;
                e.isDest = (!tokens[0].empty() && tokens[0][0] == 'D');
                e.encodeField = tokens[1];
                e.fieldType = tokens[2];
                if (tokens[3] == "M64") {
                    e.sizeBits = 64;
                    e.isM64 = true;
                } else {
                    e.sizeBits = std::stoi(tokens[3]);
                }
                for (size_t ti = 4; ti < tokens.size(); ++ti) {
                    if (tokens[ti] == "RW")
                        e.isReadWrite = true;
                    else if (tokens[ti] == "M64")
                        e.isM64 = true;
                }
                out.push_back(e);
            } else if (tokens.size() >= 2 && tokens[1].find('.') != std::string::npos) {
                // Named-field partial override: {S0, .size=64, .type=wait_alu}
                OperandFieldEntry e;
                e.positionKey = tokens[0];
                for (size_t ti = 1; ti < tokens.size(); ++ti) {
                    size_t eq = tokens[ti].find('=');
                    if (eq == std::string::npos) continue;
                    std::string key = tokens[ti].substr(0, eq);
                    std::string val = tokens[ti].substr(eq + 1);
                    if (key == ".size")
                        e.sizeBits = std::stoi(val);
                    else if (key == ".type")
                        e.fieldType = val;
                }
                out.push_back(e);
            }

            i = innerEnd + 1;
        }
    }
};

//==========================================================================
// CODE GENERATORS
//==========================================================================

class InstructionCodeGen {
   public:
    InstructionCodeGen(const std::string& arch, const std::vector<InstructionDef>& instructions,
                       const ArchDef& archDef)
        : arch_(arch), instructions_(instructions), archDef_(archDef) {}

    // Generate cost table file
    bool generateCostTable(const std::string& outputPath) {
        std::ofstream out(outputPath);
        if (!out) {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }

        emitHeader(out, "Instruction Cost Table");
        out << "// Cost table: {mnemonic, cycle, latency}\n";
        out << "// Only non-default costs are listed (default: cycle=" << archDef_.defaultCycle
            << ", latency=" << archDef_.defaultLatency << ")\n\n";

        out << "constexpr InstructionCost " << arch_ << "_GENERATED_COSTS[] = {\n";

        for (const auto& inst : instructions_) {
            if (inst.cycle != archDef_.defaultCycle || inst.latency != archDef_.defaultLatency) {
                out << "    {\"" << inst.mnemonic << "\", " << inst.cycle << ", " << inst.latency
                    << "},\n";
            }
        }

        out << "};\n\n";

        // Emit cost overrides keyed by modifier (e.g. MatrixFmtModifiers(a, b)); runtime uses for
        // modifier-dependent cost
        std::vector<const InstructionDef*> withOverrides;
        for (const auto& inst : instructions_) {
            if (!inst.costOverrides.empty()) withOverrides.push_back(&inst);
        }
        if (!withOverrides.empty()) {
            out << "// Cost overrides: when modifier matches (e.g. MatrixFmtModifiers), use "
                   "(cycle, latency).\n"
                   "// fmtA/fmtB values match stinkytofu::MatrixFmt enum (LLVM "
                   "WMMA::MatrixFMT).\n";
            out << "struct InstructionCostOverrideMatrixFmt { const char* mnemonic; uint8_t "
                   "fmtA; uint8_t fmtB; uint16_t cycle; uint16_t latency; };\n";
            out << "constexpr InstructionCostOverrideMatrixFmt " << arch_
                << "_COST_OVERRIDES_MATRIX_FMT[] = {\n";
            for (const auto* inst : withOverrides) {
                for (const auto& ov : inst->costOverrides) {
                    if (ov.modifierType != "MatrixFmtModifiers" || ov.args.size() < 2) continue;
                    // Map .def format names to MatrixFmt enum values (LLVM WMMA::MatrixFMT)
                    auto toFmt = [](const std::string& s) -> int {
                        if (s == "FP8") return 0;
                        if (s == "BF8") return 1;
                        if (s == "FP6") return 2;
                        if (s == "BF6") return 3;
                        if (s == "FP4") return 4;
                        return 0xFF;  // NONE
                    };
                    out << "    {\"" << inst->mnemonic << "\", " << toFmt(ov.args[0]) << ", "
                        << toFmt(ov.args[1]) << ", " << ov.cycle << ", " << ov.latency << "},\n";
                }
            }
            out << "};\n\n";
        }

        out << "// Total generated instructions: " << instructions_.size() << "\n";
        return true;
    }

    static std::string encodeFieldToCpp(const std::string& f) {
        if (f == "vdata") return "EncodeField::vdata";
        if (f == "vaddr") return "EncodeField::vaddr";
        if (f == "rsrc") return "EncodeField::rsrc";
        if (f == "soffset") return "EncodeField::soffset";
        if (f == "vdst") return "EncodeField::vdst";
        if (f == "vsrc") return "EncodeField::vsrc";
        if (f == "saddr") return "EncodeField::saddr";
        if (f == "addr") return "EncodeField::addr";
        if (f == "data0") return "EncodeField::data0";
        if (f == "data1") return "EncodeField::data1";
        if (f == "src0") return "EncodeField::src0";
        if (f == "src1") return "EncodeField::src1";
        if (f == "src2") return "EncodeField::src2";
        if (f == "scale_src0") return "EncodeField::scale_src0";
        if (f == "scale_src1") return "EncodeField::scale_src1";
        if (f == "sdst") return "EncodeField::sdst";
        if (f == "ssrc1") return "EncodeField::ssrc1";
        if (f == "literal") return "EncodeField::literal";
        if (f == "simm16") return "EncodeField::simm16";
        if (f == "simm32") return "EncodeField::simm32";
        if (f == "ssrc0") return "EncodeField::ssrc0";
        if (f == "sdata") return "EncodeField::sdata";
        if (f == "ioffset") return "EncodeField::ioffset";
        if (f == "sbase") return "EncodeField::sbase";
        if (f == "vsrc1") return "EncodeField::vsrc1";
        if (f == "vaddr0") return "EncodeField::vaddr0";
        if (f == "vaddr1") return "EncodeField::vaddr1";
        if (f == "vaddr2") return "EncodeField::vaddr2";
        if (f == "vaddr3") return "EncodeField::vaddr3";
        return "EncodeField::None";
    }

    static std::string fieldTypeToCpp(const std::string& t) {
        if (t == "vgpr") return "FieldType::vgpr";
        if (t == "sreg") return "FieldType::sreg";
        if (t == "sreg_m0") return "FieldType::sreg_m0";
        if (t == "label") return "FieldType::label";
        if (t == "simm16") return "FieldType::simm16";
        if (t == "simm32") return "FieldType::simm32";
        if (t == "simm24") return "FieldType::simm24";
        if (t == "simm5") return "FieldType::simm5";
        if (t == "sdst") return "FieldType::sdst";
        if (t == "ssrc") return "FieldType::ssrc";
        if (t == "hwreg") return "FieldType::hwreg";
        if (t == "delay") return "FieldType::delay";
        if (t == "set_vgpr_msb") return "FieldType::set_vgpr_msb";
        if (t == "sleep") return "FieldType::sleep";
        if (t == "ssrc_barrier_id") return "FieldType::ssrc_barrier_id";
        if (t == "src_vgpr") return "FieldType::src_vgpr";
        if (t == "src_vgpr_or_inline") return "FieldType::src_vgpr_or_inline";
        if (t == "src_simple") return "FieldType::src_simple";
        if (t == "wait_alu") return "FieldType::wait_alu";
        if (t == "wait_mem_ds") return "FieldType::wait_mem_ds";
        if (t == "smem_offset") return "FieldType::smem_offset";
        if (t == "smem_offset_nok") return "FieldType::smem_offset_nok";
        if (t == "src") return "FieldType::src";
        if (t == "vcc") return "FieldType::vcc";
        if (t == "exec") return "FieldType::exec";
        if (t == "sgpr") return "FieldType::sgpr";
        return "FieldType::None";
    }

    static std::string mnemonicToArraySuffix(const std::string& mnemonic) {
        std::string s;
        for (char c : mnemonic) s += (c == '_' ? '_' : (char)std::tolower(c));
        return s;
    }

    // Generate operand requirements file (included by ArchInfo getMCIDTable())
    bool generateOperandRequirements(const std::string& outputPath) {
        std::ofstream out(outputPath);
        if (!out) {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }

        emitHeader(out, "Operand Requirements");
        out << "// Operand field requirements for IR verifier.\n";
        out << "// Included inside getMCIDTable(); defines "
               "instFieldRequirements[].\n\n";

        // --- Operand field description arrays ---
        constexpr int kFieldSizeBitsMax = (1 << 12) - 1;  // 4095

        std::vector<const InstructionDef*> withFields;
        for (const auto& inst : instructions_) {
            if (!inst.finalOperandFields.empty()) withFields.push_back(&inst);
        }

        if (withFields.empty()) {
            out << "// No instructions with operand field descriptions in this arch\n";
            out << "static constexpr struct {\n";
            out << "    const char* mnemonic;\n";
            out << "    stinkytofu::span<const stinkytofu::HwInstDesc::OperandFieldDesc> "
                   "fields;\n";
            out << "} instFieldRequirements[] = {};\n";
        } else {
            for (const auto* inst : withFields) {
                std::string arrayName = "operand_fields_" + mnemonicToArraySuffix(inst->mnemonic);
                out << "static constexpr stinkytofu::HwInstDesc::OperandFieldDesc " << arrayName
                    << "[] = {\n";
                for (const auto& e : inst->finalOperandFields) {
                    if (e.sizeBits > kFieldSizeBitsMax) {
                        std::cerr << "error: " << inst->mnemonic
                                  << " operand sizeBits=" << e.sizeBits << " exceeds 12-bit limit ("
                                  << kFieldSizeBitsMax << ")\n";
                        return false;
                    }
                    out << "    {" << encodeFieldToCpp(e.encodeField) << ", "
                        << fieldTypeToCpp(e.fieldType) << ", " << e.sizeBits << ", "
                        << (e.isDest ? 1 : 0) << ", " << (e.isReadWrite ? 1 : 0) << ", "
                        << (e.isM64 ? 1 : 0) << "},\n";
                }
                out << "};\n\n";
            }

            out << "static constexpr struct {\n";
            out << "    const char* mnemonic;\n";
            out << "    stinkytofu::span<const stinkytofu::HwInstDesc::OperandFieldDesc> "
                   "fields;\n";
            out << "} instFieldRequirements[] = {\n";
            for (size_t i = 0; i < withFields.size(); ++i) {
                const auto* inst = withFields[i];
                std::string arrayName = "operand_fields_" + mnemonicToArraySuffix(inst->mnemonic);
                out << "    {\"" << inst->mnemonic << "\", " << arrayName << "}";
                out << (i + 1 < withFields.size() ? ",\n" : "\n");
            }
            out << "};\n";
        }

        // --- Promoted (alt) operand field description arrays ---
        std::vector<const InstructionDef*> withPromoted;
        for (const auto& inst : instructions_) {
            if (!inst.finalPromotedFields.empty()) withPromoted.push_back(&inst);
        }

        out << "\n";
        if (withPromoted.empty()) {
            out << "// No instructions with promoted operand field descriptions in this arch\n";
            out << "static constexpr struct {\n";
            out << "    const char* mnemonic;\n";
            out << "    stinkytofu::MicrocodeFormat promotedFormat;\n";
            out << "    stinkytofu::span<const stinkytofu::HwInstDesc::OperandFieldDesc> "
                   "fields;\n";
            out << "} instPromotedFieldRequirements[] = {};\n";
        } else {
            for (const auto* inst : withPromoted) {
                std::string arrayName = "promoted_fields_" + mnemonicToArraySuffix(inst->mnemonic);
                out << "static constexpr stinkytofu::HwInstDesc::OperandFieldDesc " << arrayName
                    << "[] = {\n";
                for (const auto& e : inst->finalPromotedFields) {
                    if (e.sizeBits > kFieldSizeBitsMax) {
                        std::cerr << "error: " << inst->mnemonic
                                  << " promoted operand sizeBits=" << e.sizeBits
                                  << " exceeds 12-bit limit (" << kFieldSizeBitsMax << ")\n";
                        return false;
                    }
                    out << "    {" << encodeFieldToCpp(e.encodeField) << ", "
                        << fieldTypeToCpp(e.fieldType) << ", " << e.sizeBits << ", "
                        << (e.isDest ? 1 : 0) << ", " << (e.isReadWrite ? 1 : 0) << ", "
                        << (e.isM64 ? 1 : 0) << "},\n";
                }
                out << "};\n\n";
            }

            out << "static constexpr struct {\n";
            out << "    const char* mnemonic;\n";
            out << "    stinkytofu::MicrocodeFormat promotedFormat;\n";
            out << "    stinkytofu::span<const stinkytofu::HwInstDesc::OperandFieldDesc> "
                   "fields;\n";
            out << "} instPromotedFieldRequirements[] = {\n";
            for (size_t i = 0; i < withPromoted.size(); ++i) {
                const auto* inst = withPromoted[i];
                std::string arrayName = "promoted_fields_" + mnemonicToArraySuffix(inst->mnemonic);
                std::string pfmt = inst->finalPromotedFormat.empty()
                                       ? "MicrocodeFormat::NONE"
                                       : "MicrocodeFormat::" + inst->finalPromotedFormat;
                out << "    {\"" << inst->mnemonic << "\", " << "stinkytofu::" << pfmt << ", "
                    << arrayName << "}";
                out << (i + 1 < withPromoted.size() ? ",\n" : "\n");
            }
            out << "};\n";
        }

        return true;
    }

    // Generate initialization file (DEF_T calls)
    bool generateInitFile(const std::string& outputPath) {
        std::ofstream out(outputPath);
        if (!out) {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }

        emitHeader(out, "Instruction Initialization");
        out << "// Generated DEF_T initialization calls\n\n";

        for (const auto& inst : instructions_) {
            std::string flagArgs = flagsToInitArgs(inst.finalFlags);
            out << "DEF_T(\"" << inst.mnemonic << "\"";
            if (!flagArgs.empty()) out << ", " << flagArgs;
            out << ");\n";
        }

        out << "\n// Total generated instructions: " << instructions_.size() << "\n";
        return true;
    }

    // Emit two per-arch sorted (name, id) tables and the two binary-search lookup
    // functions that consume them. HwReg.cpp dispatches on arch to nameToId<Arch>
    // / idToName<Arch>.
    //
    //   kHwregByName_<Arch>[]   — sorted by name; backs nameToId<Arch>
    //   kHwregById_<Arch>[]     — sorted by id;   backs idToName<Arch>
    //   nameToId<Arch>(name, out) — std::lower_bound + equality check
    //   idToName<Arch>(id)        — std::lower_bound + equality check
    //
    // Each table is guarded by a static_assert(is_sorted) so any future drift fails to compile.
    bool generateHwregTable(const std::string& outputPath) {
        std::ofstream out(outputPath);
        if (!out) {
            std::cerr << "Error: Cannot write " << outputPath << "\n";
            return false;
        }
        emitHeader(out, "HWREG Name Tables");
        out << "// HWREG (name, id) rows + per-arch lookup functions for "
               "s_setreg / s_getreg.\n"
            << "// Included inside HwReg.cpp at namespace stinkytofu::HwReg scope.\n\n";

        auto emitTable = [&](const std::string& suffix, const std::vector<size_t>& idx,
                             const std::string& sortKey) {
            const std::string symbol = "kHwreg" + suffix + "_" + arch_;
            out << "static constexpr ::stinkytofu::HwReg::NamedId " << symbol << "[] = {\n";
            for (size_t k : idx) {
                const auto& e = archDef_.hwRegs[k];
                out << "    {\"" << e.name << "\", static_cast<::stinkytofu::HwReg::Id>(" << e.id
                    << ")},\n";
            }
            out << "};\n";
            out << "static_assert(std::is_sorted(std::begin(" << symbol << "), std::end(" << symbol
                << "),\n"
                << "    [](const ::stinkytofu::HwReg::NamedId& a,\n"
                << "       const ::stinkytofu::HwReg::NamedId& b) {\n"
                << "        return " << sortKey << ";\n"
                << "    }),\n"
                << "    \"" << symbol << " must be sorted by " << suffix << "\");\n\n";
        };

        auto byName = sortedIndices(archDef_.hwRegs, [](const HwRegEntry& e) { return e.name; });
        emitTable("ByName", byName, "a.name < b.name");

        auto byId = sortedIndices(archDef_.hwRegs, [](const HwRegEntry& e) { return e.id; });
        emitTable("ById", byId, "static_cast<uint16_t>(a.id) < static_cast<uint16_t>(b.id)");

        // nameToId<Arch>: binary-search the by-name table.
        out << "inline bool nameToId" << arch_
            << "(std::string_view name, ::stinkytofu::HwReg::Id& out) {\n"
            << "    const auto* end = std::end(kHwregByName_" << arch_ << ");\n"
            << "    const auto* it = std::lower_bound(\n"
            << "        std::begin(kHwregByName_" << arch_ << "), end, name,\n"
            << "        [](const ::stinkytofu::HwReg::NamedId& e, std::string_view target) {\n"
            << "            return e.name < target;\n"
            << "        });\n"
            << "    if (it != end && it->name == name) {\n"
            << "        out = it->id;\n"
            << "        return true;\n"
            << "    }\n"
            << "    return false;\n"
            << "}\n\n";

        // idToName<Arch>: binary-search the by-id table.
        out << "inline std::string_view idToName" << arch_ << "(uint16_t id) {\n"
            << "    const auto* end = std::end(kHwregById_" << arch_ << ");\n"
            << "    const auto* it = std::lower_bound(\n"
            << "        std::begin(kHwregById_" << arch_ << "), end, id,\n"
            << "        [](const ::stinkytofu::HwReg::NamedId& e, uint16_t target) {\n"
            << "            return static_cast<uint16_t>(e.id) < target;\n"
            << "        });\n"
            << "    if (it != end && static_cast<uint16_t>(it->id) == id) return it->name;\n"
            << "    return {};\n"
            << "}\n\n";

        return true;
    }

   private:
    static std::string flagsToInitArgs(const std::vector<std::string>& flags) {
        if (flags.empty()) return "";
        std::ostringstream os;
        for (size_t i = 0; i < flags.size(); ++i) os << (i ? ", " : "") << "IF_" << flags[i];
        return os.str();
    }
    std::string arch_;
    const std::vector<InstructionDef>& instructions_;
    ArchDef archDef_;

    void emitHeader(std::ofstream& out, const std::string& description) {
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

// Convert microcode string (e.g. "MC_VOP3P") to C++ enum literal "MicrocodeFormat::MC_VOP3P"
static std::string microcodeToCpp(const std::string& microcode) {
    if (microcode.empty()) return "MicrocodeFormat::NONE";
    return "MicrocodeFormat::" + microcode;
}

// Convert unit string (e.g. "VALU") to C++ enum literal "ExecUnit::VALU"
static std::string unitToCpp(const std::string& unit) {
    if (unit.empty()) return "ExecUnit::NONE";
    return "ExecUnit::" + unit;
}

// Convert finalFlags to C++ makeFlagSet({ IF_XXX, ... }) content (IF_ prefix per flag)
static std::string flagsToMakeFlagSetContent(const std::vector<std::string>& flags) {
    if (flags.empty()) return "";
    std::ostringstream os;
    for (size_t i = 0; i < flags.size(); ++i) os << (i ? ", " : "") << "IF_" << flags[i];
    return os.str();
}

// Emit one arch's Isa.inc (opcode enum, MCIDTable, getArchOpcode, MnemonicToIsaOpcodeMap).
// unifiedOpcodeMap: mnemonic -> global unified opcode index.
static bool emitArchIsaFile(const std::string& arch,
                            const std::vector<InstructionDef>& instructions,
                            const std::unordered_map<std::string, int>& unifiedOpcodeMap,
                            const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
        std::cerr << "Error: Cannot write " << outputPath << "\n";
        return false;
    }

    size_t maxMnemonicLen = 0;
    for (const auto& inst : instructions)
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
    for (size_t i = 0; i < instructions.size(); ++i)
        out << "  " << instructions[i].mnemonic << ", // " << i << "\n";
    out << "};\n\n";
    out << "#endif // GET_ISAINFO_OPCODE_ENUMERATION\n\n";

    // MCIDTable (HwInstDesc: isaOpcode, unifiedOpcode, issue, latency, mnemonic, flags, microcode,
    // encoding bits, unit, operandFields placeholder)
    EMIT_GUARD("GET_ISAINFO_HWINSTDESC_TABLE");
    out << "// MCIDTable: operandFields set by ArchInfo getMCIDTable()\n"
        << "static HwInstDesc MCIDTable[] = {\n";
    for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& inst = instructions[i];
        int uop = 0;
        auto it = unifiedOpcodeMap.find(inst.mnemonic);
        if (it != unifiedOpcodeMap.end()) uop = it->second;
        std::string flagStr = flagsToMakeFlagSetContent(inst.finalFlags);
        out << "  { " << std::setw(3) << i << ", " << std::setw(5) << uop << ", " << std::setw(3)
            << inst.cycle << ", " << std::setw(4) << inst.latency << ", " << "0x" << std::hex
            << std::setfill('0') << std::setw(4)
            << (inst.coIssueWindow >= 0 ? inst.coIssueWindow : 0) << std::dec << std::setfill(' ')
            << ", " << "\"" << inst.mnemonic << "\", " << "makeFlagSet({"
            << (flagStr.empty() ? "" : flagStr) << "}), " << microcodeToCpp(inst.finalMicrocode)
            << ", " << inst.finalEncoding << ", " << unitToCpp(inst.finalUnit) << ", " << "{} },\n";
    }
    out << "};\n\n";
    out << "#endif // GET_ISAINFO_HWINSTDESC_TABLE\n\n";

    // getArchOpcode(unifiedOpcode) - binary search table
    EMIT_GUARD("GET_ISAINFO_UOP_MAPPINGS");
    // Sort by unified opcode for binary search
    std::vector<size_t> idx(instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        int ua = 0, ub = 0;
        auto ita = unifiedOpcodeMap.find(instructions[a].mnemonic);
        auto itb = unifiedOpcodeMap.find(instructions[b].mnemonic);
        if (ita != unifiedOpcodeMap.end()) ua = ita->second;
        if (itb != unifiedOpcodeMap.end()) ub = itb->second;
        return ua < ub;
    });
    out << "uint16_t get" << arch << "Opcode(uint16_t unifiedOpcode) {\n"
        << "    static constexpr uint16_t Table[][2] = {\n";
    for (size_t k = 0; k < idx.size(); ++k) {
        size_t i = idx[k];
        int uop = 0;
        auto it = unifiedOpcodeMap.find(instructions[i].mnemonic);
        if (it != unifiedOpcodeMap.end()) uop = it->second;
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
    for (size_t i = 0; i < instructions.size(); ++i)
        out << "  {\"" << instructions[i].mnemonic << "\", " << i << "},\n";
    out << "};\n\n";
    out << "#endif // GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS\n\n";

#undef EMIT_GUARD
    return true;
}

// Emit hardware/gfxIsa.inc (GFX unified opcode enum)
static bool emitGfxIsaFile(const std::vector<std::string>& unifiedMnemonics,
                           const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
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
    for (size_t i = 0; i < unifiedMnemonics.size(); ++i)
        out << "  " << unifiedMnemonics[i] << ", // " << i << "\n";
    out << "};\n\n#endif // GET_ISAINFO_UNIFIED_OPCODES\n\n";
    return true;
}

// Emit GfxXXX.hpp (ArchInfo class) - replaces GfxArch.hpp.in
static bool emitArchHeader(const ArchDef& arch, const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
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
        << ", " << arch.wavefront << " /* waveFrontSize */" << ", " << arch.totalVgprPerSimd
        << " /* totalVgprPerSimd */" << ", " << arch.vgprAllocGranule
        << " /* vgprAllocGranule */)\n"
        << "    {\n"
        << "    }\n\n"
        << "    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override\n"
        << "    {\n"
        << "        return get" << arch.name << "Opcode(unifiedOpcode);\n"
        << "    }\n\n"
        << "    bool hasD16Writes32BitVgpr() const override\n"
        << "    {\n"
        << "        return " << (arch.d16Writes32BitVgpr ? "true" : "false") << ";\n"
        << "    }\n\n"
        << "    const HwInstDesc* getMCIDTable() const override\n"
        << "    {\n"
        << "#define GET_ISAINFO_HWINSTDESC_TABLE\n"
        << "#include \"hardware/" << arch.name << "Isa.inc\"\n"
        << "#include \"hardware/generated/" << arch.name << "_operands.inc\"\n\n"
        << "        static std::once_flag once;\n"
        << "        std::call_once(once, [] {\n"
        << "            for(const auto& req : instFieldRequirements)\n"
        << "            {\n"
        << "                for(size_t i = 0; i < sizeof(MCIDTable) / sizeof(MCIDTable[0]); "
           "++i)\n"
        << "                {\n"
        << "                    if(MCIDTable[i].mnemonic && std::string(MCIDTable[i].mnemonic) "
           "== req.mnemonic)\n"
        << "                    {\n"
        << "                        const_cast<HwInstDesc&>(MCIDTable[i]).operandFields = "
           "req.fields;\n"
        << "                        break;\n"
        << "                    }\n"
        << "                }\n"
        << "            }\n"
        << "            for(const auto& req : instPromotedFieldRequirements)\n"
        << "            {\n"
        << "                for(size_t i = 0; i < sizeof(MCIDTable) / sizeof(MCIDTable[0]); "
           "++i)\n"
        << "                {\n"
        << "                    if(MCIDTable[i].mnemonic && std::string(MCIDTable[i].mnemonic) "
           "== req.mnemonic)\n"
        << "                    {\n"
        << "                        const_cast<HwInstDesc&>(MCIDTable[i]).promotedFormat = "
           "req.promotedFormat;\n"
        << "                        const_cast<HwInstDesc&>(MCIDTable[i]).promotedFields = "
           "req.fields;\n"
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
static bool emitArchDefinesBlock(const ArchDef& arch, const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
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
        << "    // Instruction definitions: generated from " << arch.name << "Instructions.def\n"
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
                                  const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
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
    for (const auto& arch : archs)
        out << "    #include \"hardware/generated/" << arch << "_block.inc\"\n";
    out << "}\n"
        << "// clang-format on\n";
    return true;
}

// Emit GfxLogicalMaps.cpp - setGfxXXXLogicalToArchMap from .logical in DEF_T
static bool emitGfxLogicalMapsCpp(
    const std::map<std::string, std::vector<InstructionDef>>& archInstructions,
    const std::vector<std::string>& archs, const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
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

    for (const auto& arch : archs) {
        auto it = archInstructions.find(arch);
        if (it == archInstructions.end()) continue;
        const auto& insts = it->second;
        out << "\nvoid set" << arch << "LogicalToArchMap(GpuArch& registry)\n{\n"
            << "    std::unordered_map<std::string, std::string> logicalToHwInstMap = {\n";
        size_t count = 0;
        for (const auto& inst : insts) {
            for (const auto& logicalName : inst.logical) {
                if (!logicalName.empty()) {
                    if (count > 0) out << ",\n";
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
                                   const std::string& outputPath) {
    std::ofstream out(outputPath);
    if (!out) {
        std::cerr << "Error: Cannot write " << outputPath << "\n";
        return false;
    }
    out << "/* Architecture-specific ArchInfo headers. */\n";
    for (const auto& arch : archs) {
        std::string archUpper = arch;
        for (char& c : archUpper)
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
static std::string normalizeArch(const std::string& arch) {
    if (arch.empty()) return arch;
    std::string s = arch;
    if (s.size() >= 1) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    return s;
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
bool genInstructions(const std::string& arch, const std::string& inputDir,
                     const std::string& outputDir) {
    std::string normArch = normalizeArch(arch);
    std::cout << "Generating instruction metadata for " << normArch << "...\n";

    // Construct file paths: inputDir is base (e.g. hardware/src/gfx), .def files in arch subdir
    std::string formatFile = inputDir + "/" + normArch + "/" + normArch + "Formats.def";
    std::string instFile = inputDir + "/" + normArch + "/" + normArch + "Instructions.def";
    std::string outputBase = outputDir + "/" + normArch;

    // Parse format definitions
    DefTParser parser(normArch);
    if (!parser.parseFormats(formatFile)) {
        std::cerr << "Error: Failed to parse format file\n";
        return false;
    }

    // Parse instruction definitions
    if (!parser.parseInstructions(instFile)) {
        std::cerr << "Error: Failed to parse instruction file\n";
        return false;
    }

    // Apply format inheritance
    if (!parser.applyFormatDefaults()) return false;

    // Generate output files
    InstructionCodeGen codegen(normArch, parser.getInstructions(), parser.getArch());

    bool success = true;
    success &= codegen.generateCostTable(outputBase + "_costs.inc");
    success &= codegen.generateOperandRequirements(outputBase + "_operands.inc");
    success &= codegen.generateInitFile(outputBase + "_init.inc");
    success &= codegen.generateHwregTable(outputBase + "_hwreg.inc");

    if (success) {
        std::cout << "Successfully generated instruction metadata for " << normArch << "\n";
    }

    return success;
}

// Generate for all archs from .def and emit ISA .inc (so full tablegen does not need gfxisa for
// ISA). Single run: *.def -> costs, init, operands, *Isa.inc, gfxIsa.inc, GfxXXX.hpp, *_block.inc,
// GfxArchDefines.cpp -> one build.
// NOLINTNEXTLINE(misc-use-internal-linkage)
bool genAllInstructions(const std::string& inputDir, const std::string& outputDir) {
    const std::vector<std::string> archs = {"Gfx1250"};

    std::map<std::string, std::vector<InstructionDef>> archInstructions;
    std::map<std::string, ArchDef> archDefs;
    for (const std::string& arch : archs) {
        // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
        std::string formatFile = inputDir + "/" + arch + "/" + arch + "Formats.def";
        // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
        std::string instFile = inputDir + "/" + arch + "/" + arch + "Instructions.def";
        DefTParser parser(arch);
        if (!parser.parseFormats(formatFile)) {
            std::cerr << "Error: Failed to parse format file " << formatFile << "\n";
            return false;
        }
        if (!parser.parseInstructions(instFile)) {
            std::cerr << "Error: Failed to parse instruction file " << instFile << "\n";
            return false;
        }
        if (!parser.applyFormatDefaults()) return false;
        archInstructions[arch] = parser.getInstructions();
        archDefs[arch] = parser.getArch();
    }

    // Build unified mnemonic list (sorted, + LABEL, INVALID) and map mnemonic -> index
    std::set<std::string> allMnemonics;
    for (const auto& p : archInstructions)
        for (const auto& inst : p.second) allMnemonics.insert(inst.mnemonic);
    std::vector<std::string> unifiedList(allMnemonics.begin(), allMnemonics.end());
    std::sort(unifiedList.begin(), unifiedList.end());
    unifiedList.push_back("FENCE");
    unifiedList.push_back("LABEL");
    unifiedList.push_back("PHI");
    unifiedList.push_back("INVALID");
    std::unordered_map<std::string, int> unifiedOpcodeMap;
    for (size_t i = 0; i < unifiedList.size(); ++i)
        unifiedOpcodeMap[unifiedList[i]] = static_cast<int>(i);

    std::filesystem::path outPath(outputDir);
    if (!outPath.is_absolute()) outPath = std::filesystem::absolute(outPath);
    std::filesystem::path hwDir = outPath / "hardware";
    std::filesystem::path genDir = hwDir / "generated";
    std::filesystem::path archHdrDir = hwDir / "arch_headers";
    std::error_code ec;
    std::filesystem::create_directories(hwDir, ec);
    std::filesystem::create_directories(genDir, ec);
    std::filesystem::create_directories(archHdrDir, ec);
    if (!std::filesystem::is_directory(genDir) || !std::filesystem::is_directory(hwDir)) {
        std::cerr << "Error: Output directories missing or not usable: " << genDir.string() << " "
                  << hwDir.string() << "\n";
        return false;
    }

    bool success = true;
    for (const std::string& arch : archs) {
        const std::vector<InstructionDef>& insts = archInstructions[arch];
        const ArchDef& ad = archDefs[arch];
        InstructionCodeGen codegen(arch, insts, ad);
        success &= codegen.generateCostTable((genDir / (arch + "_costs.inc")).string());
        success &=
            codegen.generateOperandRequirements((genDir / (arch + "_operands.inc")).string());
        success &= codegen.generateInitFile((genDir / (arch + "_init.inc")).string());
        success &= codegen.generateHwregTable((genDir / (arch + "_hwreg.inc")).string());
        success &=
            emitArchIsaFile(arch, insts, unifiedOpcodeMap, (hwDir / (arch + "Isa.inc")).string());
        success &= emitArchHeader(ad, (archHdrDir / (arch + ".hpp")).string());
        success &= emitArchDefinesBlock(ad, (genDir / (arch + "_block.inc")).string());
    }
    success &= emitGfxIsaFile(unifiedList, (hwDir / "gfxIsa.inc").string());
    success &= emitGfxArchDefinesCpp(archs, (genDir / "GfxArchDefines.cpp").string());
    success &=
        emitGfxLogicalMapsCpp(archInstructions, archs, (genDir / "GfxLogicalMaps.cpp").string());
    success &= emitArchHelperIncludes(archs, (archHdrDir / "ArchHelper_includes.inc").string());

    if (success) std::cout << "Successfully generated instruction metadata and ISA for all archs\n";
    return success;
}

}  // namespace stinkytofu
