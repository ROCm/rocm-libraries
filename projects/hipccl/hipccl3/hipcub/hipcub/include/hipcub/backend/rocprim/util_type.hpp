/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
#define HIPCUB_ROCPRIM_UTIL_TYPE_HPP_

#include <type_traits>

#include "../../config.hpp"

#include <rocprim/detail/various.hpp>

BEGIN_HIPCUB_NAMESPACE

using NullType = ::rocprim::empty_type;

template<bool B, typename T, typename F>
struct If
{
    using Type = typename std::conditional<B, T, F>::type;
};

template<typename T>
struct IsPointer
{
    static constexpr bool VALUE = std::is_pointer<T>::value;
};

template<typename T>
struct IsVolatile
{
    static constexpr bool VALUE = std::is_volatile<T>::value;
};

template<typename T>
struct RemoveQualifiers
{
    using Type = typename std::remove_cv<T>::type;
};

template<int N>
struct PowerOfTwo
{
    static constexpr bool VALUE = ::rocprim::detail::is_power_of_two<N>();
};

namespace detail
{

template<int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2Impl
{
    static constexpr int VALUE = Log2Impl<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template<int N, int COUNT>
struct Log2Impl<N, 0, COUNT>
{
    static constexpr int VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1;
};

} // end of detail namespace

template<int N>
struct Log2
{
    static_assert(N != 0, "The logarithm of zero is undefined");
    static constexpr int VALUE = detail::Log2Impl<N>::VALUE;
};

template<typename T>
struct DoubleBuffer
{
    T * d_buffers[2];

    int selector;

    HIPCUB_HOST_DEVICE inline
    DoubleBuffer()
    {
        selector = 0;
        d_buffers[0] = nullptr;
        d_buffers[1] = nullptr;
    }

    HIPCUB_HOST_DEVICE inline
    DoubleBuffer(T * d_current, T * d_alternate)
    {
        selector = 0;
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    HIPCUB_HOST_DEVICE inline
    T * Current()
    {
        return d_buffers[selector];
    }

    HIPCUB_HOST_DEVICE inline
    T * Alternate()
    {
        return d_buffers[selector ^ 1];
    }
};

template<
    class Key,
    class Value
>
using KeyValuePair = ::rocprim::key_value_pair<Key, Value>;

namespace detail
{

template<typename T>
inline
::rocprim::double_buffer<T> to_double_buffer(DoubleBuffer<T>& source)
{
    return ::rocprim::double_buffer<T>(source.Current(), source.Alternate());
}

template<typename T>
inline
void update_double_buffer(DoubleBuffer<T>& target, ::rocprim::double_buffer<T>& source)
{
    if(target.Current() != source.current())
    {
        target.selector ^= 1;
    }
}

}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
