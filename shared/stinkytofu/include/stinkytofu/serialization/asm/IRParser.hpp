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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Diagnostic.hpp"

namespace stinkytofu {

/// Parsed instruction node from IR text format.
/// This is a lightweight structure that holds the parsed instruction data
/// before it's converted to StinkyInstruction or used by other components.
/// Parsed modifier: attrKey (e.g. "mod.ds") -> field name -> value string
using ParsedModifierDict =
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

struct ParsedInstruction {
    std::string opcodeStr;
    std::vector<StinkyRegister> destRegs;
    std::vector<StinkyRegister> srcRegs;
    int issueCycles;
    int latencyCycles;
    ParsedModifierDict modifiers;  // mod.X = { field = value, ... }
    bool isLabel;                  // true if this represents a label

    ParsedInstruction(const std::string& opcode, bool label = false)
        : opcodeStr(opcode), issueCycles(0), latencyCycles(0), isLabel(label) {}

    ParsedInstruction(std::string&& opcode, bool label = false)
        : opcodeStr(std::move(opcode)), issueCycles(0), latencyCycles(0), isLabel(label) {}
};

/// Parsed basic block: blockId + instructions + successor block IDs
struct ParsedBlock {
    std::string blockId;
    std::vector<std::unique_ptr<ParsedInstruction>> instructions;
    std::vector<std::string> successorIds;  // ^blockId references
};

/// Parsed function: name + blocks (Function -> BasicBlock -> IRBase hierarchy)
struct ParsedFunction {
    std::string funcName;
    std::vector<std::unique_ptr<ParsedBlock>> blocks;
};

/// Result of parsing IR source, including instructions and diagnostics.
struct ParseResult {
    std::vector<Diagnostic> diagnostics;

    /// Parsed function (hierarchical or flat). Flat format has a single block "entry".
    std::unique_ptr<ParsedFunction> parsedFunction;

    /// Instructions from first block (flat format). Empty if no blocks.
    std::vector<ParsedInstruction> getInstructions() const;

    /// Check if any errors occurred during parsing.
    bool hasErrors() const {
        for (const auto& diag : diagnostics) {
            if (diag.getLevel() == Diagnostic::Level::Error) {
                return true;
            }
        }
        return false;
    }

    /// Get count of errors (excluding warnings).
    size_t errorCount() const {
        size_t count = 0;
        for (const auto& diag : diagnostics) {
            if (diag.getLevel() == Diagnostic::Level::Error) {
                count++;
            }
        }
        return count;
    }
};

/// Parses a StinkyTofu IR source string and returns instructions with diagnostic information.
/// @param sourceStr The IR source text to parse.
/// @return A ParseResult containing parsed instructions and any diagnostics (errors/warnings).
STINKYTOFU_EXPORT ParseResult parseSourceStringWithDiagnostics(const std::string& sourceStr);

/// Result of parsing multiple functions from an IR source string.
struct MultiParseResult {
    std::vector<std::unique_ptr<ParsedFunction>> functions;
    std::vector<Diagnostic> diagnostics;

    bool hasErrors() const {
        for (const auto& diag : diagnostics)
            if (diag.getLevel() == Diagnostic::Level::Error) return true;
        return false;
    }
};

/// Parse all st.func definitions from a source string.
STINKYTOFU_EXPORT MultiParseResult
parseAllSourceStringsWithDiagnostics(const std::string& sourceStr);

}  // namespace stinkytofu
