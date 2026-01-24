// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <type_traits>

#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_test_sdk::utilities
{

template <typename T>
constexpr double getEpsilon()
{
    if constexpr(std::is_same_v<T, half>)
    {
        // 2^-10 = 0.0009765625
        return 0.0009765625;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        // 2^-7 = 0.0078125
        return 0.0078125;
    }
    else
    {
        static_assert(std::numeric_limits<T>::is_specialized,
                      "Type not supported and std::numeric_limits not specialized");
        return static_cast<double>(std::numeric_limits<T>::epsilon());
    }
}

} // namespace hipdnn_test_sdk::utilities
