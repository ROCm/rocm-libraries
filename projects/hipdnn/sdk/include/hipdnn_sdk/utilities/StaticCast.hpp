// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#pragma once

namespace hipdnn_sdk::utilities
{

namespace detail
{
template <class T>
struct CastTo
{
    template <class S>
    T from(S value)
    {
        return static_cast<T>(value);
    }
};

template <>
struct CastTo<hip_bfloat16>
{
    template <class T>
    hip_bfloat16 from(T value)
    {
        return static_cast<hip_bfloat16>(value);
    }

    template <>
    hip_bfloat16 from<double>(double value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    template <>
    hip_bfloat16 from<int>(int value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    template <>
    hip_bfloat16 from<unsigned int>(unsigned int value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    template <>
    hip_bfloat16 from<long>(long value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    template <>
    hip_bfloat16 from<unsigned long>(unsigned long value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }
};

template <>
struct CastTo<half>
{
    template <class T>
    half from(T value)
    {
        return static_cast<half>(value);
    }

    template <>
    half from<int>(int value)
    {
        return static_cast<half>(static_cast<float>(value));
    }

    template <>
    half from<unsigned int>(unsigned int value)
    {
        return static_cast<half>(static_cast<float>(value));
    }

    template <>
    half from<long>(long value)
    {
        return static_cast<half>(static_cast<float>(value));
    }

    template <>
    half from<unsigned long>(unsigned long value)
    {
        return static_cast<half>(static_cast<float>(value));
    }
};

}

template <class S, class T>
S staticCast(T value)
{
    return detail::CastTo<S>{}.from(value);
}
}
