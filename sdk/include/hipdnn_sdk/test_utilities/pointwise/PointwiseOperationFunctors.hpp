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
    template <typename Y, typename X0, typename X1>
    void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = static_cast<Y>(x0 + x1);
    }
};

struct Subtract
{
    template <typename Y, typename X0, typename X1>
    void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = static_cast<Y>(x0 - x1);
    }
};

} // namespace pointwise
} // namespace test_utilities
} // namespace hipdnn_sdk
