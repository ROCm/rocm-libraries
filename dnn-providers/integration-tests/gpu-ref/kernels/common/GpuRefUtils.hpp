// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<double>
{
    static constexpr double maxVal = 1.7976931348623157e+308;
    static constexpr double minVal = -1.7976931348623157e+308;
};

template <>
struct NumericLimits<float>
{
    static constexpr float maxVal = 3.402823466e+38f;
    static constexpr float minVal = -3.402823466e+38f;
};

template <>
struct NumericLimits<_Float16>
{
    static constexpr _Float16 maxVal = static_cast<_Float16>(65504.0);
    static constexpr _Float16 minVal = static_cast<_Float16>(-65504.0);
};

template <>
struct NumericLimits<__bf16>
{
    static constexpr __bf16 maxVal = static_cast<__bf16>(0x1.fep+127);
    static constexpr __bf16 minVal = static_cast<__bf16>(-0x1.fep+127);
};
