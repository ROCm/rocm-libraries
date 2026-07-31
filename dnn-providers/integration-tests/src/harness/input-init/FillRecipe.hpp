// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipdnn_integration_tests
{

struct FillRecipe
{
    enum class Kind
    {
        FREE,
        FIXED,
        STRUCTURED,
        DERIVED,
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
    static FillRecipe structured()
    {
        FillRecipe f;
        f.kind = Kind::STRUCTURED;
        return f;
    }
    static FillRecipe derived()
    {
        FillRecipe f;
        f.kind = Kind::DERIVED;
        return f;
    }
};

} // namespace hipdnn_integration_tests
