// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace hipdnn_test_sdk::detail
{

/**
    * @brief Safely test a type cast
    *
    * This function tests a type cast to ensure it throws an exception on overflow/underflow.
    *
    * @tparam TargetType The type to cast to
    * @tparam SourceType The type to cast from
    * @param value The value to cast
    * @return The cast value
    */
template <typename TargetType, typename SourceType>
inline TargetType safeTestTypeCast(SourceType value)
{
    static_assert(std::numeric_limits<std::remove_cv_t<SourceType>>::is_specialized,
                  "safeTestTypeCast: SourceType must define numeric_limits");
    static_assert(std::numeric_limits<std::remove_cv_t<TargetType>>::is_specialized,
                  "safeTestTypeCast: TargetType must define numeric_limits");

    const auto src = static_cast<double>(value);

    // If SourceType is not integral, treat it as floating-like and reject NaN/Inf.
    if constexpr(!std::is_integral_v<std::remove_cv_t<SourceType>>)
    {
        if(!std::isfinite(src))
        {
            throw std::out_of_range("safeTestTypeCast: non-finite source value");
        }
    }

    const auto lo = static_cast<double>(std::numeric_limits<TargetType>::lowest());
    const auto hi = static_cast<double>(std::numeric_limits<TargetType>::max());
    if(src < lo || src > hi)
    {
        throw std::out_of_range("safeTestTypeCast: value out of representable range");
    }

    return static_cast<TargetType>(src);
}

} // namespace hipdnn_test_sdk::detail
