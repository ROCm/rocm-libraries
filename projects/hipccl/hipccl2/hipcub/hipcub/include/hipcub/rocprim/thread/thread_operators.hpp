/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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
