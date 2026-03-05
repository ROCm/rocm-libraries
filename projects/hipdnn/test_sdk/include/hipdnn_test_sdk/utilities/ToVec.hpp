// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <vector>

namespace hipdnn_tests
{

template <typename T, std::size_t N>
std::vector<T> toVec(const std::array<T, N>& arr)
{
    return {arr.begin(), arr.end()};
}

} // namespace hipdnn_tests
