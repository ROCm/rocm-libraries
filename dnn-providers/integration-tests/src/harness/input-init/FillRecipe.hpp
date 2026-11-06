// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>

namespace hipdnn_integration_tests
{

struct FillRecipe
{
    enum class Kind
    {
        FREE,
        FIXED,
    };

    static constexpr float K_DEFAULT_LO = -1.0f;
    static constexpr float K_DEFAULT_HI = 1.0f;

    Kind kind = Kind::FREE;
    float lo = K_DEFAULT_LO;
    float hi = K_DEFAULT_HI;
    float value = 0.0f;

    static FillRecipe free(float lo, float hi)
    {
        FillRecipe f;
        f.kind = Kind::FREE;
        f.lo = lo;
        f.hi = hi;
        return f;
    }
    static FillRecipe fixed(float v)
    {
        FillRecipe f;
        f.kind = Kind::FIXED;
        f.value = v;
        return f;
    }

    /// Largest magnitude a draw from this recipe can have.
    float maxMagnitude() const
    {
        if(kind == Kind::FIXED)
        {
            return std::fabs(value);
        }
        return std::max(std::fabs(lo), std::fabs(hi));
    }

    friend bool operator==(const FillRecipe& a, const FillRecipe& b)
    {
        return a.kind == b.kind && a.lo == b.lo && a.hi == b.hi && a.value == b.value;
    }
    friend bool operator!=(const FillRecipe& a, const FillRecipe& b)
    {
        return !(a == b);
    }
};

} // namespace hipdnn_integration_tests
