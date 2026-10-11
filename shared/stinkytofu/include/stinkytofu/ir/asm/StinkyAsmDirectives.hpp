/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <cstdint>
#include <map>
#include <ostream>
#include <string>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
/**
 * @brief Assembly directive kinds (low-level IR)
 *
 * These are REAL assembly directives that appear in assembly output.
 * Similar to how StinkyInstruction has an opcode kind.
 */
enum class AsmDirectiveKind {
    SET,       // .set symbol, value
    IF,        // .if condition
    ELSE,      // .else
    ELSEIF,    // .elseif condition
    ENDIF,     // .endif
    MACRO,     // .macro name
    ENDM,      // .endm
    INCLUDE,   // .include "file"
    SECTION,   // .section name
    TEXT,      // .text
    DATA,      // .data
    BSS,       // .bss
    ALIGN,     // .align n
    SKIP,      // .skip n
    BYTE,      // .byte value
    WORD,      // .word value
    LONG,      // .long value
    QUAD,      // .quad value
    ASCII,     // .ascii "string"
    ASCIZ,     // .asciz "string"
    GLOBAL,    // .global symbol
    LOCAL,     // .local symbol
    EQU,       // .equ name, value
    TEXTBLOCK  // Raw text block (comments)
};

/**
 * @brief Assembly directive struct (low-level IR)
 *
 * Similar to StinkyInstruction - a simple struct with data.
 * No virtual functions, no class hierarchy.
 *
 * Factory functions in StinkyTofu.cpp create these.
 * Emitter handles emission based on kind.
 */
struct AsmDirective : public IRBase {
    friend class IRBase;

    AsmDirectiveKind kind;
    std::string name;  // ".set", ".if", etc.
    std::string comment;

    // Flexible data storage (like StinkyInstruction's modifiers)
    std::map<std::string, std::string> params;

    // Common fields
    std::string symbol;     // For .set, .equ
    std::string value;      // For .set, .equ
    std::string condition;  // For .if, .elseif
    std::string filename;   // For .include
    int64_t intValue;       // For .align, .skip, etc.

   private:
    AsmDirective() : IRBase(IRType::StinkyAsmDirective), kind(AsmDirectiveKind::SET), intValue(0) {}

    ~AsmDirective() override = default;

   public:
    // Implement IRBase::dump()
    void dump(std::ostream& out) const override {
        out << name;
        if (!symbol.empty()) out << " " << symbol;
        if (!value.empty()) out << ", " << value;
        if (!condition.empty()) out << " " << condition;
        if (!comment.empty()) out << "  // " << comment;
    }

    static bool classof(const IRBase* ir) {
        return ir->getType() == IRType::StinkyAsmDirective;
    }

    IRBase* clone() const override {
        auto* copied = new AsmDirective();
        copied->kind = kind;
        copied->name = name;
        copied->comment = comment;
        copied->params = params;
        copied->symbol = symbol;
        copied->value = value;
        copied->condition = condition;
        copied->filename = filename;
        copied->intValue = intValue;
        return copied;
    }
};

/// True for a TEXTBLOCK that may hold a machine instruction no pass can see
/// (an unknown mnemonic, or a known one kept verbatim): anything other than
/// blank lines, comments, directives and labels. Wait and hazard passes must
/// treat it as reading and writing everything.
inline bool isOpaqueInstructionText(const IRBase* ir) {
    const auto* d = dyn_cast<AsmDirective>(ir);
    if (d == nullptr || d->kind != AsmDirectiveKind::TEXTBLOCK) return false;
    // Strip /* */, //, ; and line-leading # comments, then look at what is left
    // line by line.
    std::string code;
    const std::string& text = d->value;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text.compare(i, 2, "/*") == 0) {
            const size_t end = text.find("*/", i + 2);
            if (end == std::string::npos) break;
            i = end + 1;
            code += ' ';
        } else if (text.compare(i, 2, "//") == 0 || text[i] == ';' ||
                   (text[i] == '#' &&
                    code.find_first_not_of(" \t\r", code.rfind('\n') + 1) == std::string::npos)) {
            const size_t eol = text.find('\n', i);
            if (eol == std::string::npos) break;
            i = eol - 1;
        } else {
            code += text[i];
        }
    }
    // A code-object metadata block is YAML data, not code.
    const size_t first = code.find_first_not_of(" \t\r\n");
    if (first != std::string::npos && code.compare(first, 16, ".amdgpu_metadata") == 0)
        return false;
    size_t pos = 0;
    while (pos < code.size()) {
        size_t eol = code.find('\n', pos);
        if (eol == std::string::npos) eol = code.size();
        const size_t b = code.find_first_not_of(" \t\r", pos);
        if (b != std::string::npos && b < eol) {
            const size_t e = code.find_last_not_of(" \t\r", eol - 1);
            const bool directive = code[b] == '.' && code.find(':', b) >= eol;
            const bool label = code[e] == ':';
            if (!directive && !label) return true;
        }
        pos = eol + 1;
    }
    return false;
}

}  // namespace stinkytofu
