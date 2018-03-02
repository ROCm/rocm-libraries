// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
#define HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_

#include "../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

struct Equality
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

struct Inequality
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template <class EqualityOp>
struct InequalityWrapper
{
    EqualityOp op;

    HIPCUB_HOST_DEVICE inline
    InequalityWrapper(EqualityOp op) : op(op) {}

    template<class T>
    HIPCUB_HOST_DEVICE inline
    bool operator()(const T &a, const T &b)
    {
        return !op(a, b);
    }
};

struct Sum
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

struct Max
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? b : a;
    }
};

struct Min
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? a : b;
    }
};

struct ArgMax
{
    template<
        class Key,
        class Value
    >
    HIPCUB_HOST_DEVICE inline
    constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a,
               const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

struct ArgMin
{
    template<
        class Key,
        class Value
    >
    HIPCUB_HOST_DEVICE inline
    constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a,
               const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
