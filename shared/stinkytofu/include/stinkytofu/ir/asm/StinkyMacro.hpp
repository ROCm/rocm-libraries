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

#include <map>
#include <string>
#include <vector>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
/**
 * @brief Macro instruction struct (high-level IR)
 *
 * Similar to StinkyInstruction - a simple struct with data.
 * No virtual functions, no class hierarchy.
 *
 * Factory functions in StinkyTofu.cpp create these.
 * Expander functions (free functions) handle expansion based on kind.
 */
struct MacroInstruction : public IRBase {
    friend class IRBase;

    std::string name;
    std::string comment;

    // Operands (like StinkyInstruction's srcRegs/destRegs)
    std::vector<StinkyRegister> operands;

    // Flexible parameter storage
    std::map<std::string, std::string> params;

    // Common parameters
    uint32_t divisor;      // For division macros
    uint32_t sizeBytes;    // For DSInit, ArgumentLoader
    uint32_t offsetBytes;  // For ArgumentLoader
    uint32_t value;        // For DSInit
    std::string label;     // For branch macros

   private:
    MacroInstruction()
        : IRBase(IRType::StinkyTofu), divisor(0), sizeBytes(0), offsetBytes(0), value(0) {}

    ~MacroInstruction() = default;

   public:
    // Implement IRBase::dump()
    void dump(std::ostream& out) const override {
        out << "%macro." << name << "( " << operands.size() << " operands )";
        if (!comment.empty()) out << "  // " << comment;
    }
};

}  // namespace stinkytofu
