// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipdnn_sdk
{
namespace test_utilities
{
namespace pointwise
{

struct Add
{
    template <typename X0, typename X1>
    auto operator()(const X0& x0, const X1& x1) const -> decltype(x0 + x1)
    {
        return x0 + x1;
    }
};

struct Subtract
{
    template <typename X0, typename X1>
    auto operator()(const X0& x0, const X1& x1) const -> decltype(x0 - x1)
    {
        return x0 - x1;
    }
};

} // namespace pointwise
} // namespace test_utilities
} // namespace hipdnn_sdk
