// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// UmdPathParse.hpp - shared parsing of a sigil-stripped UMD variable path
// (RFC 0018 A.4), used by both the runtime resolver (BindingContext) and the
// compile-time resolver (UmdCompiler). The two MUST agree on how a path splits
// into namespace root and field, so the logic lives in one place rather than
// being duplicated where it could drift.

#include <cstddef>
#include <string>
#include <string_view>

namespace hip_kernel_provider_common::umd::path
{

// Split a path into its first segment (`root`) and the remainder (`rest`). A
// leading '[' subscript stays in `rest` so a bare-tensor path like `q` and a
// subscripted `q.dims[0]` both split on the first '.'/'['.
inline void splitRoot(const std::string& path, std::string& root, std::string& rest)
{
    const std::size_t p = path.find_first_of(".[");
    if(p == std::string::npos)
    {
        root = path;
        rest.clear();
    }
    else if(path[p] == '.')
    {
        root = path.substr(0, p);
        rest = path.substr(p + 1);
    }
    else
    {
        root = path.substr(0, p);
        rest = path.substr(p); // keep '[' for the subscript
    }
}

// Parse `<prefix>[N]` -> N where N is a non-negative decimal integer. Returns
// false when `rest` is not exactly that form (wrong prefix, missing/misplaced
// brackets, empty, non-numeric, negative, or trailing characters).
inline bool parseSubscript(const std::string& rest, std::string_view prefix, std::size_t& idx)
{
    const std::size_t plen = prefix.size();
    if(rest.size() < plen + 2 || rest.compare(0, plen, prefix) != 0 || rest[plen] != '[')
    {
        return false;
    }
    const std::size_t close = rest.find(']', plen + 1);
    if(close == std::string::npos || close != rest.size() - 1)
    {
        return false;
    }
    const std::string num = rest.substr(plen + 1, close - plen - 1);
    if(num.empty())
    {
        return false;
    }
    std::size_t consumed = 0;
    long value = 0;
    try
    {
        value = std::stol(num, &consumed);
    }
    catch(const std::exception&)
    {
        return false;
    }
    if(consumed != num.size() || value < 0)
    {
        return false;
    }
    idx = static_cast<std::size_t>(value);
    return true;
}

// If `s` ends with the reserved `.present` suffix, strip it and return true;
// otherwise leave `s` unchanged and return false.
inline bool stripPresentSuffix(std::string& s)
{
    static constexpr std::string_view K_PRESENT = ".present";
    if(s.size() > K_PRESENT.size()
       && s.compare(s.size() - K_PRESENT.size(), K_PRESENT.size(), K_PRESENT) == 0)
    {
        s.resize(s.size() - K_PRESENT.size());
        return true;
    }
    return false;
}

// True when `rest` begins with `<prefix>[`, i.e. it is (well-formed or not) a
// subscript of that prefix. Used to distinguish a subscript attempt from an
// unrelated field before validating it with parseSubscript.
inline bool isSubscriptOf(const std::string& rest, std::string_view prefix)
{
    return rest.size() > prefix.size() && rest.compare(0, prefix.size(), prefix) == 0
           && rest[prefix.size()] == '[';
}

} // namespace hip_kernel_provider_common::umd::path
