// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <spdlog/spdlog.h>

template <>
struct fmt::formatter<std::vector<int64_t>> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(const std::vector<int64_t>& vec, FormatContext& ctx) const
    {
        std::string result = "[";
        for(size_t i = 0; i < vec.size(); ++i)
        {
            if(i > 0)
            {
                result += ", ";
            }
            result += std::to_string(vec[i]);
        }
        result += "]";
        return fmt::formatter<std::string>::format(result, ctx);
    }
};