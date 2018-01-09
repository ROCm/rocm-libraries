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

#ifndef ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
#define ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Trait checks if FlagOp has 3 arguments (a, b, b_index)
template<class T, class FlagOp, class Decl = decltype(&FlagOp::operator())>
struct with_b_index_arg
    : std::integral_constant<bool,
        std::is_same<Decl, bool (FlagOp::*)(const T&, const T&, unsigned int) const>::value ||
        std::is_same<Decl, bool (FlagOp::*)(const T&, const T&, unsigned int)>::value ||
        std::is_same<Decl, bool (FlagOp::*)(T, T, unsigned int) const>::value ||
        std::is_same<Decl, bool (FlagOp::*)(T, T, unsigned int)>::value
    >
{ };

// Wrapping function that allows to call FlagOp of any of these signatures:
// with b_index (a, b, b_index) or without it (a, b).
template<class T, class FlagOp>
typename std::enable_if<with_b_index_arg<T, FlagOp>::value, bool>::type
apply(FlagOp flag_op, const T& a, const T& b, unsigned int b_index) [[hc]]
{
    return flag_op(a, b, b_index);
}

template<class T, class FlagOp>
typename std::enable_if<!with_b_index_arg<T, FlagOp>::value, bool>::type
apply(FlagOp flag_op, const T& a, const T& b, unsigned int) [[hc]]
{
    return flag_op(a, b);
}

} // end namespace detail

template<
    class T,
    unsigned int BlockSize
>
class block_discontinuity
{

public:

    struct storage_type
    {
        T first_items[BlockSize];
        T last_items[BlockSize];
    };

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage) [[hc]]
    {
        flag_impl<true, false, false, false>(
            head_flags, /* ignored: */ input[0], /* ignored: */ head_flags, /* ignored: */ input[0],
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op) [[hc]]
    {
        tile_static storage_type storage;
        flag_heads(head_flags, input, flag_op, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    T tile_predecessor_item,
                    storage_type& storage) [[hc]]
    {
        flag_impl<true, true, false, false>(
            head_flags, tile_predecessor_item, /* ignored: */ head_flags, /* ignored: */ input[0],
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    T tile_predecessor_item) [[hc]]
    {
        tile_static storage_type storage;
        flag_heads(head_flags, input, flag_op, tile_predecessor_item, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage) [[hc]]
    {
        flag_impl<false, false, true, false>(
            /* ignored: */ tail_flags, /* ignored: */ input[0], tail_flags, /* ignored: */ input[0],
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op) [[hc]]
    {
        tile_static storage_type storage;
        flag_tails(tail_flags, input, flag_op, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    T tile_successor_item,
                    storage_type& storage) [[hc]]
    {
        flag_impl<false, false, true, true>(
            /* ignored: */ tail_flags, /* ignored: */ input[0], tail_flags, tile_successor_item,
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    T tile_successor_item) [[hc]]
    {
        tile_static storage_type storage;
        flag_tails(tail_flags, input, flag_op, tile_successor_item, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage) [[hc]]
    {
        flag_impl<true, false, true, false>(
            head_flags, /* ignored: */ input[0], tail_flags, /* ignored: */ input[0],
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op) [[hc]]
    {
        tile_static storage_type storage;
        flag_heads_and_tails(head_flags, tail_flags, input, flag_op, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage) [[hc]]
    {
        flag_impl<true, false, true, true>(
            head_flags, /* ignored: */ input[0], tail_flags, tile_successor_item,
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op) [[hc]]
    {
        tile_static storage_type storage;
        flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, flag_op, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage) [[hc]]
    {
        flag_impl<true, true, true, false>(
            head_flags, tile_predecessor_item, tail_flags, /* ignored: */ input[0],
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op) [[hc]]
    {
        tile_static storage_type storage;
        flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, flag_op, storage);
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage) [[hc]]
    {
        flag_impl<true, true, true, true>(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item,
            input, flag_op, storage
        );
    }

    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op) [[hc]]
    {
        tile_static storage_type storage;
        flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item,
            input, flag_op, storage
        );
    }

private:

    template<
        bool WithHeads,
        bool WithTilePredecessor,
        bool WithTails,
        bool WithTileSuccessor,
        unsigned int ItemsPerThread,
        class Flag,
        class FlagOp
    >
    void flag_impl(Flag (&head_flags)[ItemsPerThread],
                   T tile_predecessor_item,
                   Flag (&tail_flags)[ItemsPerThread],
                   T tile_successor_item,
                   const T (&input)[ItemsPerThread],
                   FlagOp flag_op,
                   storage_type& storage) [[hc]]
    {
        static_assert(std::is_integral<Flag>::value, "Flag must be integral type");

        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        // Copy input items for rare cases when input and head_flags/tail_flags are the same arrays
        // (in other cases it does not affect performance)
        T items[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            items[i] = input[i];
        }

        if(WithHeads)
        {
            storage.last_items[flat_id] = items[ItemsPerThread - 1];
        }
        if(WithTails)
        {
            storage.first_items[flat_id] = items[0];
        }
        ::rocprim::syncthreads();

        if(WithHeads)
        {
            if(WithTilePredecessor)
            {
                const T predecessor_item = (flat_id == 0)
                    ? tile_predecessor_item
                    : storage.last_items[flat_id - 1];
                head_flags[0] = detail::apply(flag_op, predecessor_item, items[0], flat_id * ItemsPerThread);
            }
            else
            {
                head_flags[0] = (flat_id == 0)
                    ? Flag(true) // The first item in the block is always flagged
                    : detail::apply(flag_op, storage.last_items[flat_id - 1], items[0], flat_id * ItemsPerThread);
            }

            for(unsigned int i = 1; i < ItemsPerThread; i++)
            {
                head_flags[i] = detail::apply(flag_op, items[i - 1], items[i], flat_id * ItemsPerThread + i);
            }
        }
        if(WithTails)
        {
            for(unsigned int i = 0; i < ItemsPerThread - 1; i++)
            {
                tail_flags[i] = detail::apply(flag_op, items[i], items[i + 1], flat_id * ItemsPerThread + i + 1);
            }

            if(WithTileSuccessor)
            {
                const T successor_item = (flat_id == BlockSize - 1)
                    ? tile_successor_item
                    : storage.first_items[flat_id + 1];
                tail_flags[ItemsPerThread - 1] = detail::apply(
                    flag_op, items[ItemsPerThread - 1], successor_item,
                    flat_id * ItemsPerThread + ItemsPerThread
                );
            }
            else
            {
                tail_flags[ItemsPerThread - 1] = (flat_id == BlockSize - 1)
                    ? Flag(true) // The last item in the block is always flagged
                    : detail::apply(
                        flag_op, items[ItemsPerThread - 1], storage.first_items[flat_id + 1],
                        flat_id * ItemsPerThread + ItemsPerThread
                    );
            }
        }
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
