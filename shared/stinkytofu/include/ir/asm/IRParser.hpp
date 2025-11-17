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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "ir/asm/StinkyAsmIR.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace stinkytofu
{
    /// Diagnostic message for parser errors.
    class Diagnostic
    {
    public:
        enum class Level
        {
            Error,
            Warning,
            Note
        };

    private:
        Level       level;
        std::string message;
        unsigned    line;
        unsigned    column;

    public:
        Diagnostic(Level lvl, std::string msg, unsigned l, unsigned c)
            : level(lvl)
            , message(std::move(msg))
            , line(l)
            , column(c)
        {
        }

        Level getLevel() const
        {
            return level;
        }
        const std::string& getMessage() const
        {
            return message;
        }
        unsigned getLine() const
        {
            return line;
        }
        unsigned getColumn() const
        {
            return column;
        }

        std::string format() const
        {
            std::ostringstream oss;
            oss << line << ":" << column << ": ";
            switch(level)
            {
            case Level::Error:
                oss << "error: ";
                break;
            case Level::Warning:
                oss << "warning: ";
                break;
            case Level::Note:
                oss << "note: ";
                break;
            }
            oss << message;
            return oss.str();
        }
    };

    /// Parsed instruction node from IR text format.
    /// This is a lightweight structure that holds the parsed instruction data
    /// before it's converted to StinkyInstruction or used by other components.
    struct ParsedInstruction
    {
        std::string                 opcodeStr;
        std::vector<StinkyRegister> destRegs;
        std::vector<StinkyRegister> srcRegs;
        int                         issueCycles;
        int                         latencyCycles;
        bool                        isLabel; // true if this represents a label

        ParsedInstruction(const std::string& opcode, bool label = false)
            : opcodeStr(opcode)
            , issueCycles(0)
            , latencyCycles(0)
            , isLabel(label)
        {
        }

        ParsedInstruction(std::string&& opcode, bool label = false)
            : opcodeStr(std::move(opcode))
            , issueCycles(0)
            , latencyCycles(0)
            , isLabel(label)
        {
        }
    };

    /// Parses a StinkyTofu IR source string and returns a vector of parsed instructions.
    /// @param sourceStr The IR source text to parse.
    /// @return A vector of unique pointers to ParsedInstruction objects representing the parsed instructions.
    std::vector<std::unique_ptr<ParsedInstruction>> parseSourceString(const std::string& sourceStr);

} // namespace stinkytofu
