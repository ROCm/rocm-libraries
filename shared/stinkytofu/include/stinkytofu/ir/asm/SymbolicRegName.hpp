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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace stinkytofu {

/// One endpoint of a symbolic register name, e.g. `vgprFoo+4` or bare `sgprGSU`.
struct SymbolicRegEndpoint {
    std::string base;
    std::vector<int64_t> terms;
};

/// Parsed symbolic register name carried on `StinkyRegister::literalValue`.
struct ParsedSymbolicRegName {
    SymbolicRegEndpoint start;
    /// When set, the name is an explicit range `left:right` (RawAsmParser form).
    std::optional<SymbolicRegEndpoint> rangeEnd;
};

/// Parse rocisa / RawAsmParser symbolic name shapes. Returns nullopt on failure.
std::optional<ParsedSymbolicRegName> parseSymbolicRegName(const std::string& name);

/// Resolve a single endpoint against a `.set` map: `baseValue + sum(terms)`.
std::optional<int64_t> resolveSymbolicRegEndpoint(
    const SymbolicRegEndpoint& endpoint, const std::unordered_map<std::string, int64_t>& setMap);

/// Resolve a full symbolic name to the index the assembler would use for the
/// operand base. For ranges, \p regNum must match `right - left + 1`.
std::optional<int64_t> resolveNamedIndex(const std::string& name,
                                         const std::unordered_map<std::string, int64_t>& setMap,
                                         uint16_t regNum = 1);

}  // namespace stinkytofu
