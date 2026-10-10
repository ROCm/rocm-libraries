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

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace stinkytofu {

/// Non-throwing number parsing. stinkytofu builds with -fno-exceptions, so a
/// std::stoi/stod on bad input terminates the process; use these instead.

/// The whole of \p text as an integer of type T in \p base (a leading '+' or, for
/// signed T, '-' is accepted; no 0x prefix). nullopt on junk or out of range.
template <typename T>
std::optional<T> parseInteger(std::string_view text, int base = 10) {
    static_assert(std::is_integral_v<T>);
    if (!text.empty() && text.front() == '+') text.remove_prefix(1);
    if (text.empty() || text.front() == '+') return std::nullopt;
    T value{};
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value, base);
    if (ec != std::errc() || ptr != end) return std::nullopt;
    return value;
}

/// The whole of \p text as a finite double. Underflow (a subnormal such as
/// 1e-310, or a value that rounds to 0) is accepted; overflow is not.
inline std::optional<double> parseDouble(std::string_view text) {
    if (text.empty()) return std::nullopt;
    const std::string owned(text);
    char* end = nullptr;
    const double value = std::strtod(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size()) return std::nullopt;
    if (!std::isfinite(value)) return std::nullopt;
    return value;
}

}  // namespace stinkytofu
