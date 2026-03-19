// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <vector>

#pragma once

namespace sdpa_kernel_provider
{

// Helper to construct vector from a list of non-copyable types when initializer lists can't work
template <typename T, typename... Args>
std::vector<T> makeVector(Args&&... args)
{
    std::vector<T> v;
    v.reserve(sizeof...(args));
    (v.push_back(std::forward<Args>(args)), ...);
    return v;
}

}
