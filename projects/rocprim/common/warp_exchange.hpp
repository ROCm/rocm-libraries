// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef COMMON_WARP_EXCHANGE_HPP_
#define COMMON_WARP_EXCHANGE_HPP_

#include <rocprim/config.hpp>

namespace common
{

struct BlockedToStripedOp
{
    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&input_data)[ItemsPerThread],
                    T (&output_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage) const
    {
        warp_exchange.blocked_to_striped(input_data, output_data, storage);
    }

    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage) const
    {
        warp_exchange.blocked_to_striped(thread_data, thread_data, storage);
    }
};

struct BlockedToStripedShuffleOp
{
    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&input_data)[ItemsPerThread],
                    T (&output_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/) const
    {
        warp_exchange.blocked_to_striped_shuffle(input_data, output_data);
    }

    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/) const
    {
        warp_exchange.blocked_to_striped_shuffle(thread_data, thread_data);
    }
};

struct StripedToBlockedOp
{
    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&input_data)[ItemsPerThread],
                    T (&output_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage) const
    {
        warp_exchange.striped_to_blocked(input_data, output_data, storage);
    }

    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage) const
    {
        warp_exchange.striped_to_blocked(thread_data, thread_data, storage);
    }
};

struct StripedToBlockedShuffleOp
{
    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&input_data)[ItemsPerThread],
                    T (&output_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/) const
    {
        warp_exchange.striped_to_blocked_shuffle(input_data, output_data);
    }

    template<typename warp_exchange_type, typename T, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/) const
    {
        warp_exchange.striped_to_blocked_shuffle(thread_data, thread_data);
    }
};

} // namespace common

#endif // COMMON_WARP_EXCHANGE_HPP_
