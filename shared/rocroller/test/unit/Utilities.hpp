// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <common/Utilities.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef TEST
#undef TEST // Rely on TEST_F and TEST_P instead
#endif

MATCHER_P(HasHipSuccess, p, "")
{
    auto result = arg;
    if(result != hipSuccess)
    {
        std::string msg = hipGetErrorString(result);
        *result_listener << msg;
    }
    return result == hipSuccess;
}

template <typename T>
requires(std::is_same_v<std::tuple_element_t<0, T>, rocRoller::GPUArchitectureTarget>) static ::
    testing::internal::ParamGenerator<T> filterValidDataTypeScaleTypeParams(
        ::testing::internal::ParamGenerator<T>&& inputParamGenerator)
{
    std::vector<T> filtered;
    for(auto const& inputParam : inputParamGenerator)
    {
        const auto& params = std::get<1>(inputParam);

        const auto& typeA      = std::get<0>(params);
        const auto& typeB      = std::get<1>(params);
        const auto& scaleTypeA = std::get<2>(params);
        const auto& scaleTypeB = std::get<3>(params);

        if(isValidDataTypeScaleTypeCombination(typeA, typeB, scaleTypeA, scaleTypeB))
        {
            filtered.push_back(inputParam);
        }
    }

    return ::testing::ValuesIn(filtered);
}
