// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_builder_shared.hpp"

std::size_t FirstDifference(const std::string& a, const std::string& b)
{
    for(auto i = 0; i < min(a.size(), b.size()); i++)
    {
        if(a[i] != b[i])
        {
            return i;
        }
    }

    if(a.size() == b.size())
    {
        return a.size();
    }

    return min(a.size(), b.size());
}
