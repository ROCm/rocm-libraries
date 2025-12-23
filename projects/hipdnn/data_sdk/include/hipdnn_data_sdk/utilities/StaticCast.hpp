// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace hipdnn_data_sdk::utilities
{

namespace detail
{
template <class T>
struct CastTo
{
    template <class S>
    static T from(S value)
    {
        return static_cast<T>(value);
    }
};

template <>
struct CastTo<hip_bfloat16>
{
    template <class T>
    static hip_bfloat16 from(T value)
    {
        return static_cast<hip_bfloat16>(value);
    }

    static hip_bfloat16 from(double value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    static hip_bfloat16 from(int value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    static hip_bfloat16 from(unsigned int value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    static hip_bfloat16 from(int64_t value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }

    static hip_bfloat16 from(uint64_t value)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(value));
    }
};

template <>
struct CastTo<half>
{
    template <class T>
    static half from(T value)
    {
        return static_cast<half>(value);
    }

    static half from(int value)
    {
        return static_cast<half>(static_cast<float>(value));
    }

    static half from(unsigned int value)
    {
        return static_cast<half>(static_cast<float>(value));
    }

    static half from(int64_t value)
    {
        return static_cast<half>(static_cast<float>(value));
    }

    static half from(uint64_t value)
    {
        return static_cast<half>(static_cast<float>(value));
    }
};

} // namespace detail

template <class S, class T>
S staticCast(T value)
{
    return detail::CastTo<S>::from(value);
}

} // namespace hipdnn_data_sdk::utilities
