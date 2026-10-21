// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Reading a pass's comma-separated argument list.
//
// stinkytofu-opt hands each pass the words of `--PassName=arg1,arg2` as a
// vector: bare flags, and `key=value` pairs. These helpers answer the three
// questions a pass asks of that vector.

#include <string>
#include <string_view>
#include <vector>

namespace stinkytofu {

/// True when \p args contains the bare flag \p flag (case-sensitive, exact).
inline bool hasPassArg(const std::vector<std::string>& args, std::string_view flag) {
    for (const std::string& arg : args)
        if (arg == flag) return true;
    return false;
}

/// The value of `key=value` in \p args, or \p defaultValue when the key is absent.
inline std::string passArgValue(const std::vector<std::string>& args, std::string_view key,
                                std::string defaultValue = {}) {
    const std::string prefix = std::string(key) + "=";
    for (const std::string& arg : args) {
        if (arg.starts_with(prefix)) return arg.substr(prefix.size());
    }
    return defaultValue;
}

/// The items of \p value, a list joined with '+'. Empty items are dropped.
inline std::vector<std::string> splitPlusList(std::string_view value) {
    std::vector<std::string> items;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t plus = value.find('+', start);
        const std::string_view item = value.substr(
            start, plus == std::string_view::npos ? std::string_view::npos : plus - start);
        if (!item.empty()) items.emplace_back(item);
        if (plus == std::string_view::npos) break;
        start = plus + 1;
    }
    return items;
}

/// Every value of `key=value` in \p args, in order.
///
/// Arguments are comma-separated, so a comma cannot also separate items inside
/// one value. A list is therefore written either as repeated keys
/// (`rules=A,rules=B`) or joined with '+' (`rules=A+B`); both are accepted.
inline std::vector<std::string> passArgValues(const std::vector<std::string>& args,
                                              std::string_view key) {
    const std::string prefix = std::string(key) + "=";
    std::vector<std::string> values;
    for (const std::string& arg : args) {
        if (!arg.starts_with(prefix)) continue;
        for (std::string& item : splitPlusList(std::string_view(arg).substr(prefix.size())))
            values.push_back(std::move(item));
    }
    return values;
}

}  // namespace stinkytofu
