// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN

#ifndef HIPCUB_TEST_UTILS_ARGMINMAX_HPP
#define HIPCUB_TEST_UTILS_ARGMINMAX_HPP

#include <hipcub/thread/thread_operators.hpp>
#include <type_traits>

/**
 * \brief Arg max functor - Because NVIDIA's hipcub::ArgMax doesn't work with bfloat16 (HOST-SIDE)
 */
struct ArgMax
{
    template<typename OffsetT,
             class T,
             std::enable_if_t<std::is_same<T, test_utils::half>::value
                                  || std::is_same<T, test_utils::bfloat16>::value,
                              bool>
             = true>
    HIPCUB_HOST_DEVICE __forceinline__ hipcub::KeyValuePair<OffsetT, T>
                                       operator()(const hipcub::KeyValuePair<OffsetT, T>& a,
                   const hipcub::KeyValuePair<OffsetT, T>& b) const
    {
        const hipcub::KeyValuePair<OffsetT, float> native_a(a.key, a.value);
        const hipcub::KeyValuePair<OffsetT, float> native_b(b.key, b.value);

        if((native_b.value > native_a.value)
           || ((native_a.value == native_b.value) && (native_b.key < native_a.key)))
            return b;
        return a;
    }
};
/**
 * \brief Arg min functor - Because NVIDIA's hipcub::ArgMin doesn't work with bfloat16 (HOST-SIDE)
 */
struct ArgMin
{
    template<typename OffsetT,
             class T,
             std::enable_if_t<std::is_same<T, test_utils::half>::value
                                  || std::is_same<T, test_utils::bfloat16>::value,
                              bool>
             = true>
    HIPCUB_HOST_DEVICE __forceinline__ hipcub::KeyValuePair<OffsetT, T>
                                       operator()(const hipcub::KeyValuePair<OffsetT, T>& a,
                   const hipcub::KeyValuePair<OffsetT, T>& b) const
    {
        const hipcub::KeyValuePair<OffsetT, float> native_a(a.key, a.value);
        const hipcub::KeyValuePair<OffsetT, float> native_b(b.key, b.value);

        if((native_b.value < native_a.value)
           || ((native_a.value == native_b.value) && (native_b.key < native_a.key)))
            return b;
        return a;
    }
};

// Maximum to operator selector
template<typename T>
struct ArgMaxSelector
{
    typedef hipcub::ArgMax type;
};

template<>
struct ArgMaxSelector<test_utils::half>
{
    typedef ArgMax type;
};

template<>
struct ArgMaxSelector<test_utils::bfloat16>
{
    typedef ArgMax type;
};

// Minimum to operator selector
template<typename T>
struct ArgMinSelector
{
    typedef hipcub::ArgMin type;
};

#ifdef __HIP_PLATFORM_NVIDIA__
template<>
struct ArgMinSelector<test_utils::half>
{
    typedef ArgMin type;
};

template<>
struct ArgMinSelector<test_utils::bfloat16>
{
    typedef ArgMin type;
};
#endif

#endif //HIPCUB_TEST_UTILS_ARGMINMAX_HPP
