/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated software files (the "Software"), to deal
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace stinkytofu {
class Function;

/// Resolved value and how many SET directives defined the symbol (program order).
struct AsmSetSymbolInfo {
    int64_t value = 0;
    unsigned definitionCount = 0;
    /// False when the right-hand side could not be resolved to a number, in
    /// which case `value` is 0 and means nothing. Without this a `.set FOO,
    /// UNDEF` is indistinguishable from `.set FOO, 0`, and a consumer that
    /// trusts the value silently reads the wrong register.
    bool resolved = false;
};

/// Walk all basic blocks in program order and collect `.set` symbol definitions
/// from `AsmDirective` (SET) IR. Resolves values that are decimal/hex literals
/// or refer to earlier symbols (no expression arithmetic). Output values are
/// signed int64; large unsigned 32-bit constants (e.g. 0xffffffff) are stored
/// as their 64-bit magnitude.
///
/// When a symbol is defined more than once, `definitionCount` reflects that and
/// `value` is the last definition (flat map semantics).
void collectAsmSetSymbolValues(const Function& func, std::unordered_map<std::string, int64_t>& out);

/// Like `collectAsmSetSymbolValues`, but also reports per-symbol definition
/// multiplicity for consumers that must treat redefined symbols as unresolvable.
void collectAsmSetSymbolInfo(const Function& func,
                             std::unordered_map<std::string, AsmSetSymbolInfo>& out);

/// One operand whose numeric index disagreed with its symbolic name, and the
/// index it was given instead.
struct SymbolicOperandFix {
    std::string symbol;
    uint32_t fromIdx = 0;
    uint32_t toIdx = 0;
};

/// Give every symbolically-named register operand the index its name resolves
/// to. Identity in the IR is the `(type, idx)` pair, so an operand left with a
/// placeholder index reads as a deliberate access to an unintended register.
///
/// Resolution is positional: each operand resolves against the `.set` bindings
/// in force where it appears, so a symbol redefined mid-function reads the way
/// the assembler would. Unresolvable names are left untouched.
///
/// By default, operands with an unresolved placeholder index (bit 31 set) are
/// treated as virtual registers and skipped: a name cannot contradict an index
/// that is not yet assigned. When \p resolveUnresolvedSentinel is true, the pass
/// instead targets *exactly* the all-ones placeholder index (0xFFFFFFFF, i.e.
/// rocisa regIdx == -1 widened to uint32) and resolves only those, leaving every
/// other operand -- already-resolved native operands and real virtual registers
/// (small low-bit index with bit 31 set) -- untouched. This mode exists for the
/// gfx1250 logical-IR migration: a mixed-tree logical leaf is converted from
/// rocisa eagerly at build time, before rocisa's convertTextVariablesToRegisters
/// resolves symbolic SGPR names, so the leaf's named sgpr operands reach the asm
/// IR with that placeholder and must be resolved here to match the native path.
///
/// Returns the number of indices changed, appending a description of each to
/// \p fixes.
size_t resolveSymbolicOperands(Function& func, std::vector<SymbolicOperandFix>& fixes,
                               bool resolveUnresolvedSentinel = false);

/// If \p name matches a collected `.set` symbol, set \p outInt32 to the value
/// narrowed like a 32-bit immediate (unsigned wrap for values outside int32
/// range) and return true. Used by instruction size / literal accounting.
bool tryResolveAsmSetSymbolToInt32(const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                   const std::string& name, int32_t& outInt32);

/// Print resolved `.set` map for debugging (one line per symbol: decimal int64
/// and u32 hex).
void dumpAsmSetSymbolMap(std::ostream& os, const std::unordered_map<std::string, int64_t>& map);
}  // namespace stinkytofu
