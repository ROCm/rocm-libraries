// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string>

#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    inline Generator<std::string> EscapeComment(std::string comment, int indent)
    {
        auto isblank = [](auto x) { return std::isblank(x); };
        if(comment.empty() || std::all_of(comment.begin(), comment.end(), isblank))
            co_return;

        std::string prefix;
        for(int i = 0; i < indent; i++)
        {
            prefix += " ";
        }
        prefix += "// ";

        size_t beginIndex = 0;
        for(size_t idx = 0; idx < comment.size(); idx++)
        {
            if(comment[idx] == '\n')
            {
                auto n = (idx + 1) - beginIndex;
                co_yield(prefix + comment.substr(beginIndex, n));
                beginIndex = idx + 1;
            }
        }

        if(beginIndex < comment.size())
        {
            auto n = comment.size() - beginIndex;
            co_yield(prefix + comment.substr(beginIndex, n));
        }
    }
}
