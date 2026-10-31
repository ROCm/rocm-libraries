// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Internal COMGR filename rules, independent of the host running the classifier.
 * Mirrors runtime._sdk_comgr payload ownership and runtime_coexistence mapped names. */
#ifndef ROCKE_LOWER_LLVM_COMPILER_LIBRARY_NAME_H
#define ROCKE_LOWER_LLVM_COMPILER_LIBRARY_NAME_H

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

namespace ckc
{
inline bool comgr_library_name(std::string name, bool windows, bool mapped)
{
    if(windows)
    {
        if(mapped)
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
        static const std::regex payload("amd_comgr[0-9]*\\.dll");
        static const std::regex module("amd_comgr(_?[0-9]+)?\\.dll");
        return std::regex_match(name, mapped ? module : payload);
    }
    static const std::regex payload("libamd_comgr\\.so(\\.[0-9]+)*");
    return std::regex_match(name, payload);
}
} // namespace ckc

#endif // ROCKE_LOWER_LLVM_COMPILER_LIBRARY_NAME_H
