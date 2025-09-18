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

struct Multiply
{
    template <typename Y, typename X0, typename X1>
    void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = static_cast<Y>(x0 * x1);
    }
};

// Unary operations
struct Identity
{
    template <typename Y, typename X>
    void operator()(Y& y, const X& x) const
    {
        y = static_cast<Y>(x);
    }
};

struct Negate
{
    template <typename Y, typename X>
    void operator()(Y& y, const X& x) const
    {
        y = static_cast<Y>(-x);
    }
};

struct Abs
{
    template <typename Y, typename X>
    void operator()(Y& y, const X& x) const
    {
        y = static_cast<Y>(x < 0 ? -x : x);
    }
};

// Ternary operation example for future extensibility
struct BinarySelect
{
    template <typename Y, typename Condition, typename X0, typename X1>
    void operator()(Y& y, const Condition& cond, const X0& x0, const X1& x1) const
    {
        y = static_cast<Y>(cond != 0 ? x0 : x1);
    }
};

} // namespace pointwise
} // namespace test_utilities
} // namespace hipdnn_sdk
