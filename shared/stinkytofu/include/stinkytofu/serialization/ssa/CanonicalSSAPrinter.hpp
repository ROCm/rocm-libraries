/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"

namespace stinkytofu {

struct CanonicalSSAPrinterOptions {
    unsigned indent = 2;

    /// Print the physical register origin of every defined value.
    bool printProvenance = true;

    /// Print the exact reverse-use list of every defined value.
    bool printUses = false;

    /// Print the original physical instruction as a trailing comment.
    bool printPhysicalInstruction = true;
};

/// Prints a canonical SSA graph in a deterministic, diagnostic-only form.
///
/// This is not the physical `.stir` format and is not accepted by the parser;
/// use AsmPrinter for the physical instruction stream. Every atomic register
/// unit is printed explicitly so partial definitions stay visible.
///
/// The printer is defensive rather than authoritative: invalid IDs and
/// references print as markers instead of being dereferenced, so a malformed
/// graph can still be inspected. Callers that need a valid graph should run
/// verifyCanonicalSSA() first.
class STINKYTOFU_EXPORT CanonicalSSAPrinter {
   public:
    explicit CanonicalSSAPrinter(std::ostream& os, const CanonicalSSAPrinterOptions& options = {});

    void print(const Function& function, const CanonicalSSA& ssa);

    /// Prints the placeholder for a function that has no graph, so a dump of a
    /// function nobody could lift is still a checkable artifact.
    void printMissing(const Function& function);

   private:
    void buildNames(const Function& function);

    std::string valueRef(SSAValueID id) const;
    std::string phiRef(SSAPhiID id) const;
    std::string blockRef(const BasicBlock* block) const;
    std::string instructionRef(const StinkyInstruction* instruction) const;
    std::string bindingText(const SSAOperandBinding& binding) const;
    std::string originListText(const std::vector<SSAOperandBinding>& bindings) const;
    std::string physicalText(const StinkyInstruction& instruction) const;
    std::string useListText(const SSAValue& value) const;

    void printInitialValues();
    void printBlock(const BasicBlock& block);
    void printPhi(const SSAPhi& phi);
    void printInstruction(const StinkyInstruction& instruction);
    void printUnprintedValues();
    void line(unsigned depth, const std::string& text);

    std::ostream& os_;
    CanonicalSSAPrinterOptions options_;

    // Per-print state; the printer is not reentrant.
    const CanonicalSSA* ssa_ = nullptr;
    std::unordered_map<const BasicBlock*, std::string> blockNames_;
    std::unordered_map<const BasicBlock*, uint32_t> blockOrder_;
    std::unordered_map<const StinkyInstruction*, uint32_t> instructionOrder_;
    std::vector<bool> printedValues_;
};

/// Convenience wrapper returning the dump as a string.
STINKYTOFU_EXPORT std::string canonicalSSAToString(const Function& function,
                                                   const CanonicalSSA& ssa,
                                                   const CanonicalSSAPrinterOptions& options = {});

}  // namespace stinkytofu
