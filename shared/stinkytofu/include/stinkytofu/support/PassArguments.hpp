// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Reading a pass's comma-separated argument list.
//
// stinkytofu-opt hands each pass the words of `--PassName=arg1,arg2` as a
// vector: bare flags, and `key=value` pairs. A value that is itself a list
// joins its items with '+', since a comma already separates the arguments.

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

/// The items of \p text, separated by \p separator. Empty items are dropped.
inline std::vector<std::string> splitList(std::string_view text, char separator) {
    std::vector<std::string> items;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t next = text.find(separator, start);
        const std::string_view item = text.substr(
            start, next == std::string_view::npos ? std::string_view::npos : next - start);
        if (!item.empty()) items.emplace_back(item);
        if (next == std::string_view::npos) break;
        start = next + 1;
    }
    return items;
}

}  // namespace stinkytofu
