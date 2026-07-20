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

#include <iosfwd>
#include <sstream>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyMacro.hpp"

namespace stinkytofu {
class StinkyAsmModule;

// AsmPrinter configuration options
struct AsmPrinterOptions {
    // Indentation for nested structures
    int indent = 2;
};

class STINKYTOFU_EXPORT AsmPrinter {
   public:
    AsmPrinter(std::ostream& os, const AsmPrinterOptions& options = AsmPrinterOptions())
        : os(os), options(options) {}

    void print(const StinkyRegister& reg);
    // Print a single StinkyInstruction (op line: dest = "st.mnemonic"(...) { ... })
    void print(const StinkyInstruction& inst);

    void print(const AsmDirective& directive);

    // Print an entire Function: st.func @name() { ^block: ... }
    void print(const Function& function);

    // Print an entire Module: st.module @name { st.func ... }
    void print(const StinkyAsmModule& module);

    // ^block_id: then body, then Successors/goto line
    void printBlock(const BasicBlock& bb, size_t blockIndex);

    // Dispatch by IRType: StinkyInstruction, AsmDirective, LogicalIR
    void printIR(const IRBase& ir);

   private:
    void printFunction(const Function& function, int baseIndent);
    void printBlock(const BasicBlock& bb, size_t blockIndex, int baseIndent);
    void printIR(const IRBase& ir, int baseIndent);
    void printRegister(const StinkyRegister& reg);
    void printInstruction(const StinkyInstruction& inst, int baseIndent);
    void printDirective(const AsmDirective& directive, int baseIndent);
    void printSuccessorsLine(const BasicBlock& bb, int baseIndent);

    /// Print modifier as structured dict: { key = value, ... }. Returns true if printed.
    bool printModifierAsDict(const Modifier& mod);

    std::ostream& os;
    AsmPrinterOptions options;
};

// Utility functions for quick printing
inline std::string toString(const StinkyRegister& reg) {
    std::ostringstream oss;
    AsmPrinter printer(oss, AsmPrinterOptions());
    printer.print(reg);
    return oss.str();
}

inline std::string toString(const StinkyInstruction& inst,
                            const AsmPrinterOptions& options = AsmPrinterOptions()) {
    std::ostringstream oss;
    AsmPrinter printer(oss, options);
    printer.print(inst);
    return oss.str();
}

inline std::string toString(const Function& function,
                            const AsmPrinterOptions& options = AsmPrinterOptions()) {
    std::ostringstream oss;
    AsmPrinter printer(oss, options);
    printer.print(function);
    return oss.str();
}

inline std::string toString(const StinkyAsmModule& module,
                            const AsmPrinterOptions& options = AsmPrinterOptions()) {
    std::ostringstream oss;
    AsmPrinter printer(oss, options);
    printer.print(module);
    return oss.str();
}

inline std::ostream& operator<<(std::ostream& os, const Function& function) {
    AsmPrinter printer(os, AsmPrinterOptions());
    printer.print(function);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const StinkyAsmModule& module) {
    AsmPrinter printer(os, AsmPrinterOptions());
    printer.print(module);
    return os;
}

// Stream operator overloads for convenient printing
inline std::ostream& operator<<(std::ostream& os, const StinkyRegister& reg) {
    AsmPrinter printer(os, AsmPrinterOptions());
    printer.print(reg);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const AsmDirective& directive) {
    AsmPrinter printer(os, AsmPrinterOptions());
    printer.print(directive);
    return os;
}

}  // namespace stinkytofu
