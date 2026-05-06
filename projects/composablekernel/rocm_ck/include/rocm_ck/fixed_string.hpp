// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — FixedString. No runtime, no CK deps.
//
// A fixed-capacity string that satisfies C++20 structural type requirements,
// so it can be used as an NTTP (non-type template parameter) member.
//
// std::string and std::string_view are non-structural — they can't appear in
// template parameter lists. FixedString fills that gap: a constexpr-friendly,
// trivially comparable string with a compile-time length check.
//
// The capacity is a template parameter so each use site documents its limit:
//   FixedString<16> name("bias");   // tensor names: 15 chars max

#pragma once

#include <cstddef>
#include <string_view>

namespace rocm_ck {

template <std::size_t MaxLen>
struct FixedString
{
    char data[MaxLen]{};
    int len = 0;

    constexpr FixedString() = default;

    constexpr FixedString(std::string_view sv) : len(static_cast<int>(sv.size()))
    {
        if(sv.size() > MaxLen - 1)
            throw "FixedString: input exceeds capacity";
        for(int i = 0; i < len; ++i)
            data[i] = sv[i];
    }

    constexpr bool operator==(std::string_view sv) const
    {
        if(len != static_cast<int>(sv.size()))
            return false;
        for(int i = 0; i < len; ++i)
            if(data[i] != sv[i])
                return false;
        return true;
    }

    constexpr bool operator==(const FixedString&) const  = default;
    constexpr auto operator<=>(const FixedString&) const = default;
};

} // namespace rocm_ck
