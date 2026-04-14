// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace rocRoller
{
    struct E8M0;
    template <typename T>
    concept CScaleType = std::is_same_v<T, E8M0>;

    template <CScaleType T>
    inline float scaleToFloat(T scale)
    {
        return static_cast<float>(scale);
    }

    template <CScaleType T>
    inline T floatToScale(float value)
    {
        return T(value);
    }
} // namespace rocRoller
