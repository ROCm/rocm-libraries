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
#include "stinkytofu/ir/asm/SymbolicRegName.hpp"

#include <cctype>
#include <charconv>
#include <string_view>

namespace stinkytofu {
namespace {

bool isIdentifierChar(unsigned char c, bool first) {
    if (first) return std::isalpha(c) || c == '_' || c == '.';
    return std::isalnum(c) || c == '_' || c == '.';
}

bool parseSignedTerm(std::string_view text, int64_t& out) {
    if (text.empty()) return false;
    int64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const std::from_chars_result parsed = std::from_chars(begin, end, value);
    if (parsed.ptr != end) return false;
    out = value;
    return true;
}

std::optional<SymbolicRegEndpoint> parseEndpoint(std::string_view text) {
    if (text.empty()) return std::nullopt;

    size_t split = text.size();
    for (size_t i = 1; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '+' || c == '-') {
            split = i;
            break;
        }
    }

    SymbolicRegEndpoint endpoint;
    endpoint.base = std::string(text.substr(0, split));
    if (endpoint.base.empty()) return std::nullopt;
    if (!isIdentifierChar(static_cast<unsigned char>(endpoint.base.front()), /*first=*/true))
        return std::nullopt;
    for (size_t i = 1; i < endpoint.base.size(); ++i) {
        if (!isIdentifierChar(static_cast<unsigned char>(endpoint.base[i]), /*first=*/false))
            return std::nullopt;
    }

    std::string_view remainder = text.substr(split);
    while (!remainder.empty()) {
        const char sign = remainder.front();
        if (sign != '+' && sign != '-') return std::nullopt;
        remainder.remove_prefix(1);
        size_t termEnd = 0;
        while (termEnd < remainder.size() &&
               std::isdigit(static_cast<unsigned char>(remainder[termEnd])))
            ++termEnd;
        if (termEnd == 0) return std::nullopt;
        int64_t term = 0;
        if (!parseSignedTerm(remainder.substr(0, termEnd), term)) return std::nullopt;
        if (sign == '-') term = -term;
        endpoint.terms.push_back(term);
        remainder.remove_prefix(termEnd);
    }
    return endpoint;
}

}  // namespace

std::optional<ParsedSymbolicRegName> parseSymbolicRegName(const std::string& name) {
    if (name.empty()) return std::nullopt;

    ParsedSymbolicRegName parsed;
    const size_t colon = name.find(':');
    if (colon != std::string::npos) {
        auto start = parseEndpoint(std::string_view(name).substr(0, colon));
        auto end = parseEndpoint(std::string_view(name).substr(colon + 1));
        if (!start.has_value() || !end.has_value()) return std::nullopt;
        parsed.start = *start;
        parsed.rangeEnd = *end;
        return parsed;
    }

    auto start = parseEndpoint(name);
    if (!start.has_value()) return std::nullopt;
    parsed.start = *start;
    return parsed;
}

std::optional<int64_t> resolveSymbolicRegEndpoint(
    const SymbolicRegEndpoint& endpoint, const std::unordered_map<std::string, int64_t>& setMap) {
    auto it = setMap.find(endpoint.base);
    if (it == setMap.end()) return std::nullopt;
    int64_t value = it->second;
    for (int64_t term : endpoint.terms) value += term;
    return value;
}

std::optional<int64_t> resolveNamedIndex(const std::string& name,
                                         const std::unordered_map<std::string, int64_t>& setMap,
                                         uint16_t regNum) {
    const std::optional<ParsedSymbolicRegName> parsed = parseSymbolicRegName(name);
    if (!parsed.has_value()) return std::nullopt;

    const std::optional<int64_t> start = resolveSymbolicRegEndpoint(parsed->start, setMap);
    if (!start.has_value()) return std::nullopt;

    if (!parsed->rangeEnd.has_value()) return start;

    const std::optional<int64_t> end = resolveSymbolicRegEndpoint(*parsed->rangeEnd, setMap);
    if (!end.has_value()) return std::nullopt;
    if (*end < *start) return std::nullopt;
    const int64_t width = *end - *start + 1;
    if (width <= 0 || static_cast<uint16_t>(width) != regNum) return std::nullopt;
    return start;
}

}  // namespace stinkytofu
