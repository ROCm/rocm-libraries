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

#ifndef ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
#define ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_

// HC API
#include <hcc/hc.hpp>
#include <hcc/hc_short_vector.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_exchange
{
    // Select warp size
    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

    // Minimize LDS bank conflicts for power-of-two strides, i.e. when items accessed
    // using `thread_id * ItemsPerThread` pattern where ItemsPerThread is power of two
    // (all exchanges from/to blocked).
    static constexpr bool has_bank_conflicts =
        ItemsPerThread >= 2 && ::rocprim::detail::is_power_of_two(ItemsPerThread);
    static constexpr unsigned int banks = 32;
    static constexpr unsigned int extra_items =
        has_bank_conflicts ? (BlockSize * ItemsPerThread / banks) : 0;

public:

    struct storage_type
    {
        T buffer[BlockSize * ItemsPerThread + extra_items];
    };

    template<class U>
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        blocked_to_striped(input, output, storage);
    }

    template<class U>
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer[index(flat_id * ItemsPerThread + i)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[index(i * BlockSize + flat_id)];
        }
    }

    template<class U>
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        striped_to_blocked(input, output, storage);
    }

    template<class U>
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer[index(i * BlockSize + flat_id)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    template<class U>
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        blocked_to_warp_striped(input, output, storage);
    }

    template<class U>
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage) [[hc]]
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int offset = warp_id * items_per_warp;

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer[index(offset + lane_id * ItemsPerThread + i)] = input[i];
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[index(offset + i * current_warp_size + lane_id)];
        }
    }

    template<class U>
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        warp_striped_to_blocked(input, output, storage);
    }

    template<class U>
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage) [[hc]]
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int offset = warp_id * items_per_warp;

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer[index(offset + i * current_warp_size + lane_id)] = input[i];
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[index(offset + lane_id * ItemsPerThread + i)];
        }
    }

    template<class U, class Offset>
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        scatter_to_blocked(input, output, ranks, storage);
    }

    template<class U, class Offset>
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage.buffer[index(rank)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    template<class U, class Offset>
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        scatter_to_striped(input, output, ranks, storage);
    }

    template<class U, class Offset>
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage.buffer[rank] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[i * BlockSize + flat_id];
        }
    }

    template<class U, class Offset>
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        scatter_to_striped_guarded(input, output, ranks, storage);
    }

    template<class U, class Offset>
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(rank >= 0)
            {
                storage.buffer[rank] = input[i];
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[i * BlockSize + flat_id];
        }
    }

    template<class U, class Offset, class ValidFlag>
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread]) [[hc]]
    {
        tile_static storage_type storage;
        scatter_to_striped_flagged(input, output, ranks, is_valid, storage);
    }

    template<class U, class Offset, class ValidFlag>
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread],
                                    storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(is_valid[i])
            {
                storage.buffer[rank] = input[i];
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage.buffer[i * BlockSize + flat_id];
        }
    }

private:

    unsigned int get_current_warp_size() const [[hc]]
    {
        const unsigned int warp_id = ::rocprim::warp_id();
        return (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;
    }

    // Change index to minimize LDS bank conflicts if necessary
    unsigned int index(unsigned int n) [[hc]]
    {
        // Move every 32-bank wide "row" (32 banks * 4 bytes) by one item
        return has_bank_conflicts ? (n + n / banks) : n;
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
