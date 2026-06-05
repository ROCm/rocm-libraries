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

namespace stinkytofu {
class Function;

/// Walk all basic blocks in program order and collect `.set` symbol definitions
/// from `AsmDirective` (SET) IR. Resolves values that are decimal/hex literals
/// or refer to earlier symbols (no expression arithmetic). Output values are
/// signed int64; large unsigned 32-bit constants (e.g. 0xffffffff) are stored
/// as their 64-bit magnitude.
void collectAsmSetSymbolValues(const Function& func, std::unordered_map<std::string, int64_t>& out);

/// If \p name matches a collected `.set` symbol, set \p outInt32 to the value
/// narrowed like a 32-bit immediate (unsigned wrap for values outside int32
/// range) and return true. Used by instruction size / literal accounting.
bool tryResolveAsmSetSymbolToInt32(const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                   const std::string& name, int32_t& outInt32);

/// Print resolved `.set` map for debugging (one line per symbol: decimal int64
/// and u32 hex).
void dumpAsmSetSymbolMap(std::ostream& os, const std::unordered_map<std::string, int64_t>& map);
}  // namespace stinkytofu
